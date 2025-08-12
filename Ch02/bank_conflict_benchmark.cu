/*
 * bank_conflict_benchmark.cu
 * Shared Memory Bank Conflict 분석 및 최적화
 * 32개 뱅크 구조와 충돌 회피 기법
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

#define BANK_SIZE 32  // CUDA shared memory는 32개 뱅크

// Bank conflict 있는 접근 (Stride = 32)
__global__ void withBankConflict(float *output) {
    __shared__ float sdata[1024];
    int tid = threadIdx.x;
    
    // 초기화
    sdata[tid] = tid * 1.0f;
    __syncthreads();
    
    // Bank conflict 발생: 모든 스레드가 같은 뱅크 접근
    // tid=0,32,64,96... 모두 bank 0 접근
    float sum = 0.0f;
    for (int i = 0; i < 32; i++) {
        int idx = (tid * 32 + i) % 1024;  // Stride = 32
        sum += sdata[idx];  // 32-way bank conflict!
    }
    
    output[tid] = sum;
}

// Bank conflict 없는 접근 (Sequential)
__global__ void withoutBankConflict(float *output) {
    __shared__ float sdata[1024];
    int tid = threadIdx.x;
    
    // 초기화
    sdata[tid] = tid * 1.0f;
    __syncthreads();
    
    // No bank conflict: Sequential access
    float sum = 0.0f;
    for (int i = 0; i < 32; i++) {
        int idx = (tid + i * blockDim.x) % 1024;
        sum += sdata[idx];  // No conflict
    }
    
    output[tid] = sum;
}

// Padding으로 bank conflict 해결
__global__ void withPadding(float *output) {
    // 33개씩 padding하여 bank conflict 회피
    __shared__ float sdata[32][33];  // 32x33 instead of 32x32
    int tid = threadIdx.x;
    int row = tid / 32;
    int col = tid % 32;
    
    // 초기화
    if (tid < 32 * 32) {
        sdata[row][col] = tid * 1.0f;
    }
    __syncthreads();
    
    // Column-wise access (원래는 conflict 발생)
    // Padding 덕분에 conflict 없음
    float sum = 0.0f;
    if (col < 32) {
        for (int i = 0; i < 32; i++) {
            sum += sdata[i][col];  // No conflict with padding
        }
    }
    
    if (tid < 32 * 32) {
        output[tid] = sum;
    }
}

// 2D 배열 전치 - Bank conflict 있음
__global__ void transposeWithConflict(float *output, float *input, int width) {
    __shared__ float tile[32][32];
    
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    
    // Load tile (coalesced)
    if (x < width && y < width) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }
    __syncthreads();
    
    // Transpose indices
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    
    // Store transposed (bank conflicts in shared memory read)
    if (x < width && y < width) {
        output[y * width + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// 2D 배열 전치 - Bank conflict 해결 (Padding)
__global__ void transposeWithoutConflict(float *output, float *input, int width) {
    __shared__ float tile[32][33];  // Padding 추가
    
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    
    // Load tile (coalesced)
    if (x < width && y < width) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }
    __syncthreads();
    
    // Transpose indices
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    
    // Store transposed (no bank conflicts due to padding)
    if (x < width && y < width) {
        output[y * width + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// Bank conflict 패턴 분석
__global__ void analyzeBankConflicts() {
    __shared__ float sdata[1024];
    int tid = threadIdx.x;
    
    if (tid == 0) {
        printf("\n=== Bank Conflict 패턴 분석 ===\n");
        printf("Shared memory는 %d개 뱅크로 구성\n", BANK_SIZE);
        printf("4-byte word 기준, 연속 4바이트씩 다른 뱅크\n\n");
        
        printf("주소와 뱅크 매핑:\n");
        for (int i = 0; i < 8; i++) {
            printf("sdata[%d] → Bank %d\n", i, i % BANK_SIZE);
        }
        printf("...\n");
        printf("sdata[32] → Bank 0 (다시 Bank 0!)\n");
        printf("sdata[33] → Bank 1\n\n");
        
        printf("Conflict 예시:\n");
        printf("Thread 0: sdata[0]  → Bank 0\n");
        printf("Thread 1: sdata[32] → Bank 0 (Conflict!)\n");
        printf("Thread 2: sdata[64] → Bank 0 (Conflict!)\n");
    }
}

// 성능 측정 함수
float measureKernel(void (*kernel)(float*), const char *name, int blockSize) {
    float *d_output;
    size_t size = blockSize * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_output, size));
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    // Warm-up
    kernel<<<1, blockSize>>>(d_output);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 측정
    const int iterations = 1000;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        kernel<<<1, blockSize>>>(d_output);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    return ms / iterations;
}

int main() {
    printf("=== Shared Memory Bank Conflict 벤치마크 ===\n\n");
    
    // GPU 정보
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("Shared memory per block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("Shared memory banks: 32 (4-byte mode)\n\n");
    
    // Bank conflict 분석
    analyzeBankConflicts<<<1, 1>>>();
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 1. 기본 Bank Conflict 테스트
    printf("\n1. 기본 Bank Conflict 테스트\n");
    printf("%-30s %-15s %-15s\n", "Test", "Time (ms)", "Relative");
    printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" 
           "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" 
           "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=\n");
    
    float timeWith = measureKernel(withBankConflict, "With conflict", 256);
    float timeWithout = measureKernel(withoutBankConflict, "Without conflict", 256);
    float timePadding = measureKernel(withPadding, "With padding", 256);
    
    printf("%-30s %-15.4f %-15s\n", "With bank conflict", timeWith, "1.00x");
    printf("%-30s %-15.4f %-15.2fx\n", "Without bank conflict", timeWithout, timeWith/timeWithout);
    printf("%-30s %-15.4f %-15.2fx\n", "With padding", timePadding, timeWith/timePadding);
    
    // 2. Matrix Transpose 테스트
    printf("\n2. Matrix Transpose 테스트 (1024x1024)\n");
    
    const int matrixSize = 1024;
    size_t matrixBytes = matrixSize * matrixSize * sizeof(float);
    
    float *h_input = (float*)malloc(matrixBytes);
    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, matrixBytes));
    CHECK_CUDA(cudaMalloc(&d_output, matrixBytes));
    
    // 초기화
    for (int i = 0; i < matrixSize * matrixSize; i++) {
        h_input[i] = (float)i;
    }
    CHECK_CUDA(cudaMemcpy(d_input, h_input, matrixBytes, cudaMemcpyHostToDevice));
    
    dim3 grid(matrixSize/32, matrixSize/32);
    dim3 block(32, 32);
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    // With conflict
    CHECK_CUDA(cudaEventRecord(start));
    transposeWithConflict<<<grid, block>>>(d_output, d_input, matrixSize);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float transposeTimeWith = 0;
    CHECK_CUDA(cudaEventElapsedTime(&transposeTimeWith, start, stop));
    
    // Without conflict
    CHECK_CUDA(cudaEventRecord(start));
    transposeWithoutConflict<<<grid, block>>>(d_output, d_input, matrixSize);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float transposeTimeWithout = 0;
    CHECK_CUDA(cudaEventElapsedTime(&transposeTimeWithout, start, stop));
    
    printf("%-30s %-15s %-15s\n", "Method", "Time (ms)", "Bandwidth (GB/s)");
    printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" 
           "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" 
           "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=\n");
    
    float bandwidth1 = (2.0f * matrixBytes / 1e9) / (transposeTimeWith / 1000);
    float bandwidth2 = (2.0f * matrixBytes / 1e9) / (transposeTimeWithout / 1000);
    
    printf("%-30s %-15.3f %-15.2f\n", "With bank conflict", transposeTimeWith, bandwidth1);
    printf("%-30s %-15.3f %-15.2f\n", "Without bank conflict", transposeTimeWithout, bandwidth2);
    printf("Speedup: %.2fx\n", transposeTimeWith / transposeTimeWithout);
    
    // 3. Bank Conflict 회피 전략
    printf("\n3. Bank Conflict 회피 전략:\n");
    printf("   a) Padding: __shared__ float data[32][33]\n");
    printf("   b) Permutation: 접근 패턴 변경\n");
    printf("   c) Reordering: 데이터 레이아웃 변경\n");
    printf("   d) Broadcasting: 모든 스레드가 같은 주소 읽기는 OK\n");
    
    // 4. Bank Conflict 체크리스트
    printf("\n4. Bank Conflict 체크리스트:\n");
    printf("   ✓ Stride가 32의 배수인가?\n");
    printf("   ✓ 2D 배열의 column-wise 접근인가?\n");
    printf("   ✓ Transpose나 permutation 연산인가?\n");
    printf("   ✓ Padding으로 해결 가능한가?\n");
    
    // 메모리 해제
    free(h_input);
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    printf("\n프로그램 완료!\n");
    return 0;
}