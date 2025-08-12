/*
 * reduction_sync.cu
 * Reduction 패턴과 동기화
 * 효율적인 병렬 합계 계산
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// v0: 단순 Reduction (비효율적)
__global__ void reductionV0(float *input, float *output, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 데이터 로드
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Reduction - 비효율적 divergent 패턴
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// v1: Sequential Addressing (Bank Conflict 없음)
__global__ void reductionV1(float *input, float *output, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 데이터 로드
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Sequential addressing - no bank conflicts
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// v2: 첫 단계에서 두 개 로드 (메모리 대역폭 활용)
__global__ void reductionV2(float *input, float *output, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    // 각 스레드가 2개 요소 로드 및 첫 reduction
    float mySum = (idx < n) ? input[idx] : 0.0f;
    if (idx + blockDim.x < n) {
        mySum += input[idx + blockDim.x];
    }
    sdata[tid] = mySum;
    __syncthreads();
    
    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// v3: Warp-level optimization (마지막 warp는 동기화 불필요)
__global__ void reductionV3(float *input, float *output, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    // 데이터 로드
    float mySum = (idx < n) ? input[idx] : 0.0f;
    if (idx + blockDim.x < n) {
        mySum += input[idx + blockDim.x];
    }
    sdata[tid] = mySum;
    __syncthreads();
    
    // Reduction - warp 크기(32)보다 큰 경우만 동기화
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Warp 내에서는 동기화 불필요 (SIMT 특성)
    if (tid < 32) {
        volatile float *smem = sdata;
        if (blockDim.x >= 64) smem[tid] += smem[tid + 32];
        if (blockDim.x >= 32) smem[tid] += smem[tid + 16];
        if (blockDim.x >= 16) smem[tid] += smem[tid + 8];
        if (blockDim.x >= 8) smem[tid] += smem[tid + 4];
        if (blockDim.x >= 4) smem[tid] += smem[tid + 2];
        if (blockDim.x >= 2) smem[tid] += smem[tid + 1];
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// v4: Template 버전 (컴파일 타임 최적화)
template <int BLOCK_SIZE>
__global__ void reductionV4(float *input, float *output, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * (BLOCK_SIZE * 2) + threadIdx.x;
    
    // 데이터 로드
    float mySum = (idx < n) ? input[idx] : 0.0f;
    if (idx + BLOCK_SIZE < n) {
        mySum += input[idx + BLOCK_SIZE];
    }
    sdata[tid] = mySum;
    __syncthreads();
    
    // Unrolled reduction
    if (BLOCK_SIZE >= 512) {
        if (tid < 256) sdata[tid] += sdata[tid + 256];
        __syncthreads();
    }
    if (BLOCK_SIZE >= 256) {
        if (tid < 128) sdata[tid] += sdata[tid + 128];
        __syncthreads();
    }
    if (BLOCK_SIZE >= 128) {
        if (tid < 64) sdata[tid] += sdata[tid + 64];
        __syncthreads();
    }
    
    // Warp-level unrolling
    if (tid < 32) {
        volatile float *smem = sdata;
        if (BLOCK_SIZE >= 64) smem[tid] += smem[tid + 32];
        if (BLOCK_SIZE >= 32) smem[tid] += smem[tid + 16];
        smem[tid] += smem[tid + 8];
        smem[tid] += smem[tid + 4];
        smem[tid] += smem[tid + 2];
        smem[tid] += smem[tid + 1];
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// CPU reduction for verification
float cpuReduction(float *data, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum;
}

// 성능 테스트 함수
template <typename KernelFunc>
float testReduction(KernelFunc kernel, const char *name, 
                    float *d_input, float *d_output, 
                    int n, int blockSize) {
    
    int gridSize = (n + blockSize - 1) / blockSize;
    if (name[strlen(name)-1] >= '2') {  // v2 이상은 각 블록이 2배 처리
        gridSize = (n + blockSize * 2 - 1) / (blockSize * 2);
    }
    
    size_t sharedMemSize = blockSize * sizeof(float);
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    // Warm-up
    kernel<<<gridSize, blockSize, sharedMemSize>>>(d_input, d_output, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 실제 측정
    const int iterations = 100;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        kernel<<<gridSize, blockSize, sharedMemSize>>>(d_input, d_output, n);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    return ms / iterations;
}

int main() {
    printf("=== Reduction 패턴과 동기화 ===\n\n");
    
    // 데이터 크기
    const int n = 1 << 20;  // 1M elements
    const int blockSize = 256;
    size_t size = n * sizeof(float);
    
    printf("데이터 크기: %d elements (%.2f MB)\n", n, size / (1024.0f * 1024.0f));
    printf("블록 크기: %d threads\n\n", blockSize);
    
    // 메모리 할당
    float *h_input = (float*)malloc(size);
    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, size));
    CHECK_CUDA(cudaMalloc(&d_output, n * sizeof(float)));  // 충분한 공간
    
    // 데이터 초기화
    for (int i = 0; i < n; i++) {
        h_input[i] = 1.0f;  // 간단한 테스트를 위해 모두 1
    }
    CHECK_CUDA(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
    
    // CPU 결과 (검증용)
    auto cpuStart = std::chrono::high_resolution_clock::now();
    float cpuSum = cpuReduction(h_input, n);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    auto cpuTime = std::chrono::duration_cast<std::chrono::microseconds>(cpuEnd - cpuStart).count() / 1000.0f;
    
    printf("CPU 결과: %.0f (시간: %.3f ms)\n\n", cpuSum, cpuTime);
    
    // 각 버전 테스트
    printf("GPU Reduction 성능 비교:\n");
    printf("%-20s %-15s %-15s %-15s\n", "Version", "Time (ms)", "Bandwidth (GB/s)", "Speedup");
    printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" 
           "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" 
           "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" 
           "=" "=" "=" "=" "=\n");
    
    // v0: Basic
    float timeV0 = testReduction(reductionV0, "v0", d_input, d_output, n, blockSize);
    float bandwidthV0 = (size / 1e9) / (timeV0 / 1000);
    printf("%-20s %-15.3f %-15.2f %-15.2fx\n", 
           "v0 (Basic)", timeV0, bandwidthV0, cpuTime / timeV0);
    
    // v1: Sequential
    float timeV1 = testReduction(reductionV1, "v1", d_input, d_output, n, blockSize);
    float bandwidthV1 = (size / 1e9) / (timeV1 / 1000);
    printf("%-20s %-15.3f %-15.2f %-15.2fx\n", 
           "v1 (Sequential)", timeV1, bandwidthV1, cpuTime / timeV1);
    
    // v2: Two loads
    float timeV2 = testReduction(reductionV2, "v2", d_input, d_output, n, blockSize);
    float bandwidthV2 = (size / 1e9) / (timeV2 / 1000);
    printf("%-20s %-15.3f %-15.2f %-15.2fx\n", 
           "v2 (Two loads)", timeV2, bandwidthV2, cpuTime / timeV2);
    
    // v3: Warp opt
    float timeV3 = testReduction(reductionV3, "v3", d_input, d_output, n, blockSize);
    float bandwidthV3 = (size / 1e9) / (timeV3 / 1000);
    printf("%-20s %-15.3f %-15.2f %-15.2fx\n", 
           "v3 (Warp opt)", timeV3, bandwidthV3, cpuTime / timeV3);
    
    // v4: Template
    float timeV4 = testReduction(reductionV4<256>, "v4", d_input, d_output, n, blockSize);
    float bandwidthV4 = (size / 1e9) / (timeV4 / 1000);
    printf("%-20s %-15.3f %-15.2f %-15.2fx\n", 
           "v4 (Template)", timeV4, bandwidthV4, cpuTime / timeV4);
    
    // 개선 비율
    printf("\n최적화 효과:\n");
    printf("  v0 → v1: %.2fx (Sequential addressing)\n", timeV0 / timeV1);
    printf("  v1 → v2: %.2fx (Two loads per thread)\n", timeV1 / timeV2);
    printf("  v2 → v3: %.2fx (Warp optimization)\n", timeV2 / timeV3);
    printf("  v3 → v4: %.2fx (Template unrolling)\n", timeV3 / timeV4);
    printf("  Total:   %.2fx (v0 → v4)\n", timeV0 / timeV4);
    
    // Reduction 팁
    printf("\nReduction 최적화 팁:\n");
    printf("  1. Sequential addressing으로 bank conflict 제거\n");
    printf("  2. 각 스레드가 여러 요소 처리\n");
    printf("  3. Warp 내에서는 동기화 생략\n");
    printf("  4. Template으로 loop unrolling\n");
    printf("  5. Shared memory 사용 최소화\n");
    
    // 메모리 해제
    free(h_input);
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    
    printf("\n프로그램 완료!\n");
    return 0;
}