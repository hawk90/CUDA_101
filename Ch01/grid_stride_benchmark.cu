/*
 * grid_stride_benchmark.cu
 * Grid-Stride Loop 패턴 벤치마크
 * GTX 1070 (CC 6.1) / Jetson Orin Nano (CC 8.7) 호환
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

// 일반적인 커널 (스레드당 1개 요소)
__global__ void normalKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 간단한 계산
        data[idx] = sqrtf(data[idx]) + sinf(data[idx]);
    }
}

// Grid-Stride Loop 커널
__global__ void gridStrideKernel(float *data, int n) {
    // 시작 인덱스
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 전체 그리드의 스레드 수 (stride)
    int stride = gridDim.x * blockDim.x;
    
    // Grid-Stride Loop
    for (int i = idx; i < n; i += stride) {
        data[i] = sqrtf(data[i]) + sinf(data[i]);
    }
}

// 성능 측정 함수
float measureKernelTime(void (*kernel)(float*, int), 
                        dim3 grid, dim3 block, 
                        float *d_data, int n, 
                        int iterations = 100) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    // Warm-up
    kernel<<<grid, block>>>(d_data, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 실제 측정
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        kernel<<<grid, block>>>(d_data, n);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    return milliseconds / iterations;  // 평균 시간
}

int main() {
    printf("=== Grid-Stride Loop 벤치마크 ===\n\n");
    
    // GPU 정보 출력
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("SM 개수: %d\n", prop.multiProcessorCount);
    printf("Block당 최대 스레드: %d\n", prop.maxThreadsPerBlock);
    printf("Grid당 최대 블록: %d\n\n", prop.maxGridSize[0]);
    
    // 다양한 데이터 크기로 테스트
    uint64_t dataSizes[] = {
        1000,        // 1K
        10000,       // 10K
        100000,      // 100K
        1000000,     // 1M
        10000000,    // 10M
        100000000,   // 100M
        1000000000,  // 1B
        10000000000,  // 10B
        10000000000,  // 100B   
        100000000000,  // 1T
    };
    
    printf("데이터 크기별 성능 비교\n");
    printf("%-12s %-20s %-20s %-15s %-10s\n", 
           "Size", "Normal (ms)", "Grid-Stride (ms)", "Speedup", "Efficiency");
    printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" 
           "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" 
           "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" 
           "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=\n");
    
    for (int test = 0; test < 6; test++) {
        int N = dataSizes[test];
        size_t size = N * sizeof(float);
        
        // 메모리 할당
        float *h_data = (float*)malloc(size);
        float *d_data;
        CHECK_CUDA(cudaMalloc(&d_data, size));
        
        // 데이터 초기화
        for (int i = 0; i < N; i++) {
            h_data[i] = (float)(i % 100);
        }
        CHECK_CUDA(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
        
        // 구성 설정
        int blockSize = 256;
        
        // Normal 커널: 필요한 만큼 블록 생성
        int normalGridSize = (N + blockSize - 1) / blockSize;
        // 하드웨어 한계 체크
        if (normalGridSize > prop.maxGridSize[0]) {
            printf("%-12d 스킵 (Grid 크기 초과: %d > %d)\n", 
                   N, normalGridSize, prop.maxGridSize[0]);
            free(h_data);
            CHECK_CUDA(cudaFree(d_data));
            continue;
        }
        
        // Grid-Stride 커널: SM 수에 맞춰 최적 블록 수 설정
        int gridStrideGridSize = prop.multiProcessorCount * 2;  // SM당 2블록
        
        // 성능 측정
        float normalTime = measureKernelTime(normalKernel, 
                                            dim3(normalGridSize), 
                                            dim3(blockSize), 
                                            d_data, N);
        
        float gridStrideTime = measureKernelTime(gridStrideKernel, 
                                                dim3(gridStrideGridSize), 
                                                dim3(blockSize), 
                                                d_data, N);
        
        float speedup = normalTime / gridStrideTime;
        
        // 효율성 계산 (이론적 최대 대비)
        float normalBlocks = (float)normalGridSize;
        float gridStrideBlocks = (float)gridStrideGridSize;
        float efficiency = (gridStrideBlocks / normalBlocks) * speedup * 100.0f;
        
        printf("%-12d %-20.3f %-20.3f %-15.2fx %-10.1f%%\n",
               N, normalTime, gridStrideTime, speedup, efficiency);
        
        // 메모리 해제
        free(h_data);
        CHECK_CUDA(cudaFree(d_data));
    }

    int blockSize = 256;
    
    // Grid-Stride Loop 장점 설명
    printf("\n=== Grid-Stride Loop 장점 ===\n");
    printf("1. 커널 실행 오버헤드 감소\n");
    printf("   - Normal: N/blockSize 번 커널 호출 필요\n");
    printf("   - Grid-Stride: 1번 커널 호출\n\n");
    
    printf("2. 더 나은 SM 활용\n");
    printf("   - 모든 SM을 일관되게 활용\n");
    printf("   - Load balancing 자동화\n\n");
    
    printf("3. 큰 데이터셋 처리 가능\n");
    printf("   - Grid 크기 한계 극복\n");
    printf("   - 10억 개 이상 요소 처리 가능\n\n");
    
    // 실제 사용 예제
    printf("=== 실제 사용 예제 ===\n");
    const int LARGE_N = 1000000000;  // 10억 개
    printf("10억 개 요소 처리 시뮬레이션:\n");
    
    // Normal 방식
    int normalGridNeeded = (LARGE_N + blockSize - 1) / blockSize;
    printf("Normal 방식:\n");
    printf("  필요한 블록 수: %d\n", normalGridNeeded);
    if (normalGridNeeded > prop.maxGridSize[0]) {
        printf("  상태: ❌ 불가능 (최대 %d 블록)\n", prop.maxGridSize[0]);
    }
    
    // Grid-Stride 방식
    printf("\nGrid-Stride 방식:\n");
    printf("  사용 블록 수: %d (SM * 2)\n", prop.multiProcessorCount * 2);
    printf("  블록당 반복: ~%d 회\n", 
           LARGE_N / (prop.multiProcessorCount * 2 * blockSize));
    printf("  상태: ✓ 가능\n");
    
    // 코드 비교
    printf("\n=== 코드 패턴 비교 ===\n");
    printf("Normal Pattern:\n");
    printf("  int idx = blockIdx.x * blockDim.x + threadIdx.x;\n");
    printf("  if (idx < n) {\n");
    printf("      data[idx] = process(data[idx]);\n");
    printf("  }\n\n");
    
    printf("Grid-Stride Pattern:\n");
    printf("  int idx = blockIdx.x * blockDim.x + threadIdx.x;\n");
    printf("  int stride = gridDim.x * blockDim.x;\n");
    printf("  for (int i = idx; i < n; i += stride) {\n");
    printf("      data[i] = process(data[i]);\n");
    printf("  }\n");
    
    printf("\n프로그램 완료!\n");
    return 0;
}