/*
 * exec_config_test.cu
 * 다양한 실행 구성 테스트
 * GTX 1070 (CC 6.1) / Jetson Orin Nano (CC 8.7) 호환
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

// 간단한 커널
__global__ void simpleKernel(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = idx * 2;
    }
}

// 다양한 블록 크기 테스트
void testBlockSizes(int n) {
    printf("\n=== 블록 크기별 성능 테스트 (N=%d) ===\n", n);
    
    int *d_data;
    size_t size = n * sizeof(int);
    CHECK_CUDA(cudaMalloc(&d_data, size));
    
    // 테스트할 블록 크기들 (32의 배수가 효율적)
    int blockSizes[] = {32, 64, 128, 256, 512, 1024};
    int numTests = sizeof(blockSizes) / sizeof(int);
    
    printf("%-15s %-15s %-15s %-15s %-15s\n", 
           "Block Size", "Grid Size", "Total Threads", "Time (ms)", "Bandwidth (GB/s)");
    printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" 
           "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" 
           "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" 
           "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=\n");
    
    for (int i = 0; i < numTests; i++) {
        int blockSize = blockSizes[i];
        int gridSize = (n + blockSize - 1) / blockSize;
        
        // 시간 측정
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        
        // Warm-up
        simpleKernel<<<gridSize, blockSize>>>(d_data, n);
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // 실제 측정
        CHECK_CUDA(cudaEventRecord(start));
        for (int iter = 0; iter < 100; iter++) {
            simpleKernel<<<gridSize, blockSize>>>(d_data, n);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        
        float milliseconds = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        milliseconds /= 100;  // 평균
        
        // Bandwidth 계산 (read + write)
        float bandwidth = (2.0f * size) / (milliseconds * 1e6);
        
        printf("%-15d %-15d %-15d %-15.3f %-15.2f\n",
               blockSize, gridSize, gridSize * blockSize, milliseconds, bandwidth);
        
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }
    
    CHECK_CUDA(cudaFree(d_data));
}

// 2D/3D 구성 테스트
__global__ void kernel2D(int *data, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        data[idx] = x + y;
    }
}

void test2DConfiguration() {
    printf("\n=== 2D 실행 구성 테스트 ===\n");
    
    const int width = 1920;
    const int height = 1080;
    const int n = width * height;
    
    int *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, n * sizeof(int)));
    
    // 다양한 2D 블록 구성 테스트
    struct Config2D {
        dim3 block;
        const char* name;
    } configs[] = {
        {{16, 16}, "16x16 (256 threads)"},
        {{32, 16}, "32x16 (512 threads)"},
        {{32, 32}, "32x32 (1024 threads)"},
        {{64, 16}, "64x16 (1024 threads)"}
    };
    
    printf("%-25s %-20s %-15s %-15s\n", 
           "Block Config", "Grid Config", "Time (ms)", "Occupancy");
    printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" 
           "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" 
           "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" 
           "=" "=" "=" "=" "=" "=" "=" "=" "=" "=\n");
    
    for (int i = 0; i < 4; i++) {
        dim3 block = configs[i].block;
        dim3 grid((width + block.x - 1) / block.x, 
                  (height + block.y - 1) / block.y);
        
        // 시간 측정
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        
        CHECK_CUDA(cudaEventRecord(start));
        kernel2D<<<grid, block>>>(d_data, width, height);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        
        float milliseconds = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        
        // Occupancy 계산
        int maxActiveBlocks;
        CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxActiveBlocks, kernel2D, block.x * block.y, 0));
        
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
        
        float occupancy = (float)(maxActiveBlocks * block.x * block.y) / 
                         (float)prop.maxThreadsPerMultiProcessor * 100.0f;
        
        char gridStr[50];
        sprintf(gridStr, "%dx%d", grid.x, grid.y);
        
        printf("%-25s %-20s %-15.3f %-15.1f%%\n",
               configs[i].name, gridStr, milliseconds, occupancy);
        
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }
    
    CHECK_CUDA(cudaFree(d_data));
}

// 동적 구성 계산
void testDynamicConfiguration() {
    printf("\n=== 동적 실행 구성 계산 ===\n");
    
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    
    printf("GPU 속성:\n");
    printf("  SM 개수: %d\n", prop.multiProcessorCount);
    printf("  SM당 최대 스레드: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  SM당 최대 블록: %d\n", prop.maxBlocksPerMultiProcessor);
    printf("  워프 크기: %d\n", prop.warpSize);
    
    // 다양한 데이터 크기에 대한 최적 구성 계산
    int dataSizes[] = {1000, 10000, 100000, 1000000, 10000000};
    
    printf("\n데이터 크기별 권장 구성:\n");
    printf("%-12s %-15s %-15s %-20s\n", 
           "Data Size", "Block Size", "Grid Size", "Strategy");
    printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" 
           "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" 
           "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=\n");
    
    for (int i = 0; i < 5; i++) {
        int n = dataSizes[i];
        int blockSize, gridSize;
        const char* strategy;
        
        if (n < 10000) {
            // 작은 데이터: 작은 블록
            blockSize = 128;
            gridSize = (n + blockSize - 1) / blockSize;
            strategy = "Small blocks";
        } else if (n < 1000000) {
            // 중간 데이터: 표준 블록
            blockSize = 256;
            gridSize = (n + blockSize - 1) / blockSize;
            strategy = "Standard blocks";
        } else {
            // 큰 데이터: Grid-Stride
            blockSize = 256;
            gridSize = prop.multiProcessorCount * 2;
            strategy = "Grid-stride loop";
        }
        
        printf("%-12d %-15d %-15d %-20s\n", n, blockSize, gridSize, strategy);
    }
}

// 하드웨어 한계 테스트
void testHardwareLimits() {
    printf("\n=== 하드웨어 한계 테스트 ===\n");
    
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    
    // 1. 최대 블록 크기 테스트
    printf("\n1. 블록 크기 한계:\n");
    
    // 1D 최대
    {
        dim3 block(prop.maxThreadsPerBlock);
        dim3 grid(1);
        simpleKernel<<<grid, block>>>(nullptr, 0);
        cudaError_t err = cudaGetLastError();
        printf("   1D: %d threads - %s\n", 
               block.x, (err == cudaSuccess) ? "✓" : cudaGetErrorString(err));
    }
    
    // 2D 최대
    {
        int size = (int)sqrt(prop.maxThreadsPerBlock);
        dim3 block(size, size);
        dim3 grid(1);
        kernel2D<<<grid, block>>>(nullptr, 0, 0);
        cudaError_t err = cudaGetLastError();
        printf("   2D: %dx%d (%d threads) - %s\n", 
               size, size, size*size, (err == cudaSuccess) ? "✓" : cudaGetErrorString(err));
    }
    
    // 너무 큰 블록
    {
        dim3 block(prop.maxThreadsPerBlock + 1);
        dim3 grid(1);
        simpleKernel<<<grid, block>>>(nullptr, 0);
        cudaError_t err = cudaGetLastError();
        printf("   Over limit: %d threads - %s\n", 
               block.x, cudaGetErrorString(err));
    }
    
    // 2. 그리드 크기 한계
    printf("\n2. 그리드 크기 한계:\n");
    printf("   최대 1D 그리드: %d blocks\n", prop.maxGridSize[0]);
    printf("   최대 2D 그리드: %d x %d blocks\n", 
           prop.maxGridSize[0], prop.maxGridSize[1]);
    printf("   최대 3D 그리드: %d x %d x %d blocks\n", 
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
}

int main() {
    printf("=== CUDA 실행 구성 테스트 ===\n");
    
    // GPU 정보
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (CC %d.%d)\n", prop.name, prop.major, prop.minor);
    
    // 테스트 실행
    testBlockSizes(1000000);        // 100만 개 요소
    test2DConfiguration();           // 2D 구성
    testDynamicConfiguration();      // 동적 구성
    testHardwareLimits();           // 하드웨어 한계
    
    printf("\n프로그램 완료!\n");
    return 0;
}