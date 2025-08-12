/*
 * matmul_optimized.cu
 * Optimized Matrix Multiplication with Tiling
 * Shared memory optimization for better performance
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
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

#define TILE_SIZE 16  // Tile size for shared memory

// Naive matrix multiplication
__global__ void matmulNaive(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Tiled matrix multiplication with shared memory
__global__ void matmulTiled(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile from A
        if (row < M && t * TILE_SIZE + tx < K) {
            tileA[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        } else {
            tileA[ty][tx] = 0.0f;
        }
        
        // Load tile from B
        if (col < N && t * TILE_SIZE + ty < K) {
            tileB[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            tileB[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[ty][k] * tileB[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Optimized with bank conflict avoidance
__global__ void matmulOptimized(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE + 1];  // Padding to avoid bank conflicts
    __shared__ float tileB[TILE_SIZE][TILE_SIZE + 1];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Coalesced loading
        int aRow = row;
        int aCol = t * TILE_SIZE + tx;
        int bRow = t * TILE_SIZE + ty;
        int bCol = col;
        
        tileA[ty][tx] = (aRow < M && aCol < K) ? A[aRow * K + aCol] : 0.0f;
        tileB[ty][tx] = (bRow < K && bCol < N) ? B[bRow * N + bCol] : 0.0f;
        
        __syncthreads();
        
        // Compute with loop unrolling
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[ty][k] * tileB[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// 2x2 thread tile for better instruction-level parallelism
#define THREAD_TILE_SIZE 2
__global__ void matmul2x2ThreadTile(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Each thread computes 2x2 output tile
    int baseRow = blockIdx.y * TILE_SIZE + ty * THREAD_TILE_SIZE;
    int baseCol = blockIdx.x * TILE_SIZE + tx * THREAD_TILE_SIZE;
    
    float sum[THREAD_TILE_SIZE][THREAD_TILE_SIZE] = {0};
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Collaborative loading (each thread loads multiple elements)
        for (int i = 0; i < THREAD_TILE_SIZE; i++) {
            for (int j = 0; j < THREAD_TILE_SIZE; j++) {
                int tileRow = ty * THREAD_TILE_SIZE + i;
                int tileCol = tx * THREAD_TILE_SIZE + j;
                
                if (tileRow < TILE_SIZE && tileCol < TILE_SIZE) {
                    int aRow = blockIdx.y * TILE_SIZE + tileRow;
                    int aCol = t * TILE_SIZE + tileCol;
                    tileA[tileRow][tileCol] = 
                        (aRow < M && aCol < K) ? A[aRow * K + aCol] : 0.0f;
                    
                    int bRow = t * TILE_SIZE + tileRow;
                    int bCol = blockIdx.x * TILE_SIZE + tileCol;
                    tileB[tileRow][tileCol] = 
                        (bRow < K && bCol < N) ? B[bRow * N + bCol] : 0.0f;
                }
            }
        }
        
        __syncthreads();
        
        // Compute 2x2 output tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            for (int i = 0; i < THREAD_TILE_SIZE; i++) {
                for (int j = 0; j < THREAD_TILE_SIZE; j++) {
                    int row = ty * THREAD_TILE_SIZE + i;
                    int col = tx * THREAD_TILE_SIZE + j;
                    if (row < TILE_SIZE && col < TILE_SIZE) {
                        sum[i][j] += tileA[row][k] * tileB[k][col];
                    }
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write 2x2 results
    for (int i = 0; i < THREAD_TILE_SIZE; i++) {
        for (int j = 0; j < THREAD_TILE_SIZE; j++) {
            int row = baseRow + i;
            int col = baseCol + j;
            if (row < M && col < N) {
                C[row * N + col] = sum[i][j];
            }
        }
    }
}

// CPU matrix multiplication for verification
void cpuMatmul(float *A, float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Verify results
bool verifyResults(float *gpu, float *cpu, int M, int N) {
    const float epsilon = 1e-3f;  // Higher tolerance for accumulated errors
    for (int i = 0; i < M * N; i++) {
        if (fabs(gpu[i] - cpu[i]) > epsilon) {
            if (i < 10) {  // Print first few mismatches
                printf("Mismatch at %d: GPU=%.6f, CPU=%.6f\n", 
                       i, gpu[i], cpu[i]);
            }
            return false;
        }
    }
    return true;
}

// Performance test function
float testMatmul(void (*kernel)(float*, float*, float*, int, int, int),
                const char *name, float *d_A, float *d_B, float *d_C,
                int M, int N, int K, dim3 grid, dim3 block) {
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    // Warm-up
    kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Measure
    const int iterations = 10;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
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
    printf("=== Optimized Matrix Multiplication ===\n\n");
    
    // Test sizes
    const int sizes[] = {128, 256, 512, 1024};
    const int numSizes = sizeof(sizes) / sizeof(int);
    
    for (int s = 0; s < numSizes; s++) {
        int M = sizes[s];
        int N = sizes[s];
        int K = sizes[s];
        
        printf("Matrix size: %dx%d x %dx%d = %dx%d\n", M, K, K, N, M, N);
        printf("------------------------------------------------\n");
        
        size_t sizeA = M * K * sizeof(float);
        size_t sizeB = K * N * sizeof(float);
        size_t sizeC = M * N * sizeof(float);
        
        // Allocate host memory
        float *h_A = (float*)malloc(sizeA);
        float *h_B = (float*)malloc(sizeB);
        float *h_C = (float*)malloc(sizeC);
        float *h_cpu = (float*)malloc(sizeC);
        
        // Initialize matrices
        for (int i = 0; i < M * K; i++) h_A[i] = (float)(rand() % 10) / 10.0f;
        for (int i = 0; i < K * N; i++) h_B[i] = (float)(rand() % 10) / 10.0f;
        
        // CPU reference (only for small matrices)
        float cpuTime = 0;
        if (M <= 512) {
            auto cpuStart = std::chrono::high_resolution_clock::now();
            cpuMatmul(h_A, h_B, h_cpu, M, N, K);
            auto cpuEnd = std::chrono::high_resolution_clock::now();
            cpuTime = std::chrono::duration_cast<std::chrono::microseconds>
                      (cpuEnd - cpuStart).count() / 1000.0f;
        }
        
        // Allocate device memory
        float *d_A, *d_B, *d_C;
        CHECK_CUDA(cudaMalloc(&d_A, sizeA));
        CHECK_CUDA(cudaMalloc(&d_B, sizeB));
        CHECK_CUDA(cudaMalloc(&d_C, sizeC));
        
        // Copy to device
        CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));
        
        // Configure kernel launch
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, 
                  (M + TILE_SIZE - 1) / TILE_SIZE);
        
        // Test different implementations
        printf("\nPerformance Results:\n");
        printf("%-25s %-12s %-12s %-12s\n", 
               "Implementation", "Time (ms)", "GFLOPS", "Speedup");
        printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" 
               "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" 
               "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" 
               "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "\n");
        
        // Calculate FLOPS
        double flops = 2.0 * M * N * K;
        
        // CPU baseline
        if (cpuTime > 0) {
            double cpuGflops = flops / (cpuTime * 1e6);
            printf("%-25s %-12.3f %-12.2f %-12s\n", 
                   "CPU", cpuTime, cpuGflops, "1.00x");
        }
        
        // Naive implementation
        float naiveTime = testMatmul(matmulNaive, "Naive", 
                                     d_A, d_B, d_C, M, N, K, grid, block);
        double naiveGflops = flops / (naiveTime * 1e6);
        
        CHECK_CUDA(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));
        bool naiveCorrect = (cpuTime > 0) ? verifyResults(h_C, h_cpu, M, N) : true;
        
        printf("%-25s %-12.3f %-12.2f %-12.2fx %s\n", 
               "Naive", naiveTime, naiveGflops, 
               cpuTime > 0 ? cpuTime/naiveTime : 0, 
               naiveCorrect ? "" : "[FAIL]");
        
        // Tiled implementation
        float tiledTime = testMatmul(matmulTiled, "Tiled", 
                                     d_A, d_B, d_C, M, N, K, grid, block);
        double tiledGflops = flops / (tiledTime * 1e6);
        
        CHECK_CUDA(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));
        bool tiledCorrect = (cpuTime > 0) ? verifyResults(h_C, h_cpu, M, N) : true;
        
        printf("%-25s %-12.3f %-12.2f %-12.2fx %s\n", 
               "Tiled (Shared Memory)", tiledTime, tiledGflops, 
               cpuTime > 0 ? cpuTime/tiledTime : 0,
               tiledCorrect ? "" : "[FAIL]");
        
        // Optimized implementation
        float optTime = testMatmul(matmulOptimized, "Optimized", 
                                  d_A, d_B, d_C, M, N, K, grid, block);
        double optGflops = flops / (optTime * 1e6);
        
        CHECK_CUDA(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));
        bool optCorrect = (cpuTime > 0) ? verifyResults(h_C, h_cpu, M, N) : true;
        
        printf("%-25s %-12.3f %-12.2f %-12.2fx %s\n", 
               "Optimized (No Conflicts)", optTime, optGflops, 
               cpuTime > 0 ? cpuTime/optTime : 0,
               optCorrect ? "" : "[FAIL]");
        
        // cuBLAS comparison
        // cublasHandle_t handle;
        // cublasCreate(&handle);
        
        // const float alpha = 1.0f;
        // const float beta = 0.0f;
        
        // cudaEvent_t start, stop;
        // CHECK_CUDA(cudaEventCreate(&start));
        // CHECK_CUDA(cudaEventCreate(&stop));
        
        // // Warm-up
        // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        //            N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
        
        // CHECK_CUDA(cudaEventRecord(start));
        // for (int i = 0; i < 10; i++) {
        //     cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        //                N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
        // }
        // CHECK_CUDA(cudaEventRecord(stop));
        // CHECK_CUDA(cudaEventSynchronize(stop));
        
        // float cublasTime = 0;
        // CHECK_CUDA(cudaEventElapsedTime(&cublasTime, start, stop));
        // cublasTime /= 10;
        
        // double cublasGflops = flops / (cublasTime * 1e6);
        // printf("%-25s %-12.3f %-12.2f %-12.2fx\n", 
        //        "cuBLAS", cublasTime, cublasGflops, 
        //        cpuTime > 0 ? cpuTime/cublasTime : 0);
        
        // cublasDestroy(handle);
        
        // Performance analysis
        printf("\nOptimization Impact:\n");
        printf("  Naive → Tiled: %.2fx speedup\n", naiveTime / tiledTime);
        printf("  Tiled → Optimized: %.2fx speedup\n", tiledTime / optTime);
        // printf("  Our best vs cuBLAS: %.1f%% efficiency\n", 
        //        (cublasTime / optTime) * 100);
        
        // Memory bandwidth analysis
        float bandwidth = (sizeA + sizeB + sizeC) / (optTime * 1e6);  // GB/s
        printf("  Memory bandwidth: %.2f GB/s\n", bandwidth);
        printf("  Compute intensity: %.2f FLOP/byte\n", 
               flops / (sizeA + sizeB + sizeC));
        
        printf("\n");
        
        // Cleanup
        free(h_A);
        free(h_B);
        free(h_C);
        free(h_cpu);
        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_B));
        CHECK_CUDA(cudaFree(d_C));
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }
    
    // Optimization techniques summary
    printf("\n=== Matrix Multiplication Optimization Techniques ===\n");
    printf("1. Tiling: Reuse data in shared memory\n");
    printf("2. Padding: Avoid bank conflicts (add +1 to tile dimension)\n");
    printf("3. Coalescing: Ensure consecutive threads access consecutive memory\n");
    printf("4. Loop unrolling: Reduce loop overhead\n");
    printf("5. Thread coarsening: Each thread computes multiple outputs\n");
    printf("6. Prefetching: Hide memory latency\n");
    printf("7. Tensor cores: Use specialized hardware (Volta+)\n");
    
    printf("\nProgram completed!\n");
    return 0;
}