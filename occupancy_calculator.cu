/*
 * occupancy_calculator.cu
 * CUDA Occupancy 계산 및 최적화
 * GTX 1070 (CC 6.1) / Jetson Orin Nano (CC 8.7) 호환
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_occupancy.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// 다양한 리소스 사용 커널들
__global__ void minimalKernel(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] = idx * 2.0f;
}

__global__ void sharedMemoryKernel(float *data) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Shared memory 사용
    sdata[tid] = data[idx];
    __syncthreads();
    
    // 간단한 reduction
    if (tid < 128) sdata[tid] += sdata[tid + 128];
    __syncthreads();
    if (tid < 64) sdata[tid] += sdata[tid + 64];
    __syncthreads();
    if (tid < 32) sdata[tid] += sdata[tid + 32];
    __syncthreads();
    
    if (tid == 0) {
        data[blockIdx.x] = sdata[0];
    }
}

__global__ void registerHeavyKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 많은 레지스터 사용
    float a = data[idx];
    float b = a * 2.0f;
    float c = b + 3.0f;
    float d = c * 4.0f;
    float e = d - 5.0f;
    float f = e / 6.0f;
    float g = f + 7.0f;
    float h = g * 8.0f;
    float i = h - 9.0f;
    float j = i / 10.0f;
    
    data[idx] = a + b + c + d + e + f + g + h + i + j;
}

// Occupancy 정보 출력 함수
void printOccupancyInfo(const char* kernelName, 
                        int blockSize, 
                        size_t dynamicSharedMem,
                        void* kernel) {
    
    // 최대 활성 블록 계산
    int maxActiveBlocks;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocks, kernel, blockSize, dynamicSharedMem));
    
    // GPU 속성 가져오기
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    
    // Occupancy 계산
    int maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;
    int activeThreads = maxActiveBlocks * blockSize;
    float occupancy = (float)activeThreads / (float)maxThreadsPerSM * 100.0f;
    
    // 결과 출력
    printf("%-25s | Block: %4d | Shared: %6zu B | ", 
           kernelName, blockSize, dynamicSharedMem);
    printf("Max Blocks/SM: %2d | Active Threads/SM: %4d | Occupancy: %5.1f%%\n",
           maxActiveBlocks, activeThreads, occupancy);
}

// 최적 블록 크기 찾기
void findOptimalBlockSize() {
    printf("\n=== 최적 블록 크기 찾기 ===\n");
    
    // 커널별 최적 블록 크기 계산
    int minGridSize, blockSize;
    
    // Minimal Kernel
    CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(
        &minGridSize, &blockSize, minimalKernel, 0, 0));
    printf("Minimal Kernel:\n");
    printf("  권장 블록 크기: %d\n", blockSize);
    printf("  최소 그리드 크기: %d\n", minGridSize);
    
    // Shared Memory Kernel
    CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(
        &minGridSize, &blockSize, sharedMemoryKernel, 0, 0));
    printf("\nShared Memory Kernel:\n");
    printf("  권장 블록 크기: %d\n", blockSize);
    printf("  최소 그리드 크기: %d\n", minGridSize);
    
    // Register Heavy Kernel
    CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(
        &minGridSize, &blockSize, registerHeavyKernel, 0, 0));
    printf("\nRegister Heavy Kernel:\n");
    printf("  권장 블록 크기: %d\n", blockSize);
    printf("  최소 그리드 크기: %d\n", minGridSize);
}

// 블록 크기별 Occupancy 분석
void analyzeOccupancy() {
    printf("\n=== 블록 크기별 Occupancy 분석 ===\n");
    printf("%-25s | %-11s | %-12s | %-13s | %-17s | %-10s\n",
           "Kernel", "Block Size", "Shared Mem", "Max Blocks/SM", 
           "Active Threads/SM", "Occupancy");
    printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" 
           "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" 
           "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" 
           "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" 
           "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" 
           "=" "=" "=" "=" "=" "=\n");
    
    int blockSizes[] = {64, 128, 256, 512, 1024};
    int numSizes = sizeof(blockSizes) / sizeof(int);
    
    // Minimal Kernel 분석
    for (int i = 0; i < numSizes; i++) {
        printOccupancyInfo("Minimal Kernel", blockSizes[i], 0, 
                          (void*)minimalKernel);
    }
    
    printf("\n");
    
    // Shared Memory Kernel 분석 (동적 shared memory 사용)
    for (int i = 0; i < numSizes; i++) {
        size_t sharedMem = blockSizes[i] * sizeof(float);
        printOccupancyInfo("Shared Memory Kernel", blockSizes[i], sharedMem,
                          (void*)sharedMemoryKernel);
    }
    
    printf("\n");
    
    // Register Heavy Kernel 분석
    for (int i = 0; i < numSizes; i++) {
        printOccupancyInfo("Register Heavy Kernel", blockSizes[i], 0,
                          (void*)registerHeavyKernel);
    }
}

// 리소스 제한 분석
void analyzeResourceLimits() {
    printf("\n=== 리소스 제한 분석 ===\n");
    
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    
    printf("GPU: %s (CC %d.%d)\n\n", prop.name, prop.major, prop.minor);
    
    printf("SM당 리소스 한계:\n");
    printf("  최대 스레드: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  최대 블록: %d\n", prop.maxBlocksPerMultiProcessor);
    printf("  레지스터: %d (32-bit)\n", prop.regsPerMultiprocessor);
    printf("  Shared Memory: %zu KB\n", prop.sharedMemPerMultiprocessor / 1024);
    
    printf("\n블록당 리소스 한계:\n");
    printf("  최대 스레드: %d\n", prop.maxThreadsPerBlock);
    printf("  최대 Shared Memory: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("  최대 레지스터: %d\n", prop.regsPerBlock);
    
    // Occupancy 제한 요소 계산
    printf("\n제한 요소별 최대 블록 수 (블록 크기 = 256):\n");
    int blockSize = 256;
    
    // 스레드 제한
    int maxBlocksByThreads = prop.maxThreadsPerMultiProcessor / blockSize;
    printf("  스레드 제한: %d blocks\n", maxBlocksByThreads);
    
    // 블록 제한
    printf("  블록 수 제한: %d blocks\n", prop.maxBlocksPerMultiProcessor);
    
    // Shared Memory 제한 (16KB per block 가정)
    size_t sharedMemPerBlock = 16 * 1024;
    int maxBlocksBySharedMem = prop.sharedMemPerMultiprocessor / sharedMemPerBlock;
    printf("  Shared Memory 제한 (16KB/block): %d blocks\n", maxBlocksBySharedMem);
    
    // 레지스터 제한 (64 registers per thread 가정)
    int regsPerThread = 64;
    int maxBlocksByRegs = prop.regsPerMultiprocessor / (regsPerThread * blockSize);
    printf("  레지스터 제한 (64 regs/thread): %d blocks\n", maxBlocksByRegs);
    
    // 실제 제한 요소 판단
    int actualMaxBlocks = maxBlocksByThreads;
    const char* limitingFactor = "Threads";
    
    if (prop.maxBlocksPerMultiProcessor < actualMaxBlocks) {
        actualMaxBlocks = prop.maxBlocksPerMultiProcessor;
        limitingFactor = "Block count";
    }
    if (maxBlocksBySharedMem < actualMaxBlocks) {
        actualMaxBlocks = maxBlocksBySharedMem;
        limitingFactor = "Shared Memory";
    }
    if (maxBlocksByRegs < actualMaxBlocks) {
        actualMaxBlocks = maxBlocksByRegs;
        limitingFactor = "Registers";
    }
    
    printf("\n실제 최대 블록: %d (제한 요소: %s)\n", actualMaxBlocks, limitingFactor);
    float occupancy = (float)(actualMaxBlocks * blockSize) / 
                     (float)prop.maxThreadsPerMultiProcessor * 100.0f;
    printf("예상 Occupancy: %.1f%%\n", occupancy);
}

// 성능 vs Occupancy 테스트
void testPerformanceVsOccupancy() {
    printf("\n=== 성능 vs Occupancy 테스트 ===\n");
    
    const int N = 10000000;  // 10M elements
    size_t size = N * sizeof(float);
    
    float *h_data = (float*)malloc(size);
    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, size));
    
    // 초기화
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)i;
    }
    CHECK_CUDA(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
    
    printf("\n%-12s %-12s %-12s %-12s\n", 
           "Block Size", "Occupancy", "Time (ms)", "Throughput (GB/s)");
    printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" 
           "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" 
           "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=\n");
    
    int blockSizes[] = {32, 64, 128, 256, 512, 1024};
    
    for (int i = 0; i < 6; i++) {
        int blockSize = blockSizes[i];
        int gridSize = (N + blockSize - 1) / blockSize;
        
        // Occupancy 계산
        int maxActiveBlocks;
        CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxActiveBlocks, minimalKernel, blockSize, 0));
        
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
        float occupancy = (float)(maxActiveBlocks * blockSize) / 
                         (float)prop.maxThreadsPerMultiProcessor * 100.0f;
        
        // 성능 측정
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        
        // Warm-up
        minimalKernel<<<gridSize, blockSize>>>(d_data);
        CHECK_CUDA(cudaDeviceSynchronize());
        
        CHECK_CUDA(cudaEventRecord(start));
        for (int iter = 0; iter < 100; iter++) {
            minimalKernel<<<gridSize, blockSize>>>(d_data);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        
        float milliseconds = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        milliseconds /= 100;  // 평균
        
        // Throughput 계산
        float throughput = (2.0f * size) / (milliseconds * 1e6);  // GB/s
        
        printf("%-12d %-12.1f%% %-12.3f %-12.2f\n",
               blockSize, occupancy, milliseconds, throughput);
        
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }
    
    free(h_data);
    CHECK_CUDA(cudaFree(d_data));
}

int main() {
    printf("=== CUDA Occupancy 계산기 ===\n");
    
    // GPU 정보
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (CC %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("SM 개수: %d\n", prop.multiProcessorCount);
    
    // 분석 실행
    findOptimalBlockSize();
    analyzeOccupancy();
    analyzeResourceLimits();
    testPerformanceVsOccupancy();
    
    // Occupancy 최적화 팁
    printf("\n=== Occupancy 최적화 팁 ===\n");
    printf("1. 높은 Occupancy가 항상 최고 성능은 아님\n");
    printf("2. Memory-bound 커널: Occupancy 중요\n");
    printf("3. Compute-bound 커널: ILP가 더 중요할 수 있음\n");
    printf("4. 레지스터 사용 제한: --maxrregcount 컴파일 옵션\n");
    printf("5. Shared Memory 동적 할당으로 유연성 확보\n");
    printf("6. 50%% 이상 Occupancy면 일반적으로 충분\n");
    
    printf("\n프로그램 완료!\n");
    return 0;
}