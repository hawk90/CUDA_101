/*
 * scan_hillis_steele.cu
 * Hillis-Steele Parallel Scan Algorithm
 * Work-inefficient but depth-efficient inclusive scan
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

// Hillis-Steele inclusive scan for one block
__global__ void hillisSteleInclusiveScan(float *data, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data
    if (idx < n) {
        sdata[tid] = data[idx];
    } else {
        sdata[tid] = 0.0f;
    }
    __syncthreads();
    
    // Hillis-Steele scan
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        float temp = 0.0f;
        if (tid >= stride) {
            temp = sdata[tid - stride];
        }
        __syncthreads();
        
        if (tid >= stride) {
            sdata[tid] += temp;
        }
        __syncthreads();
    }
    
    // Write result
    if (idx < n) {
        data[idx] = sdata[tid];
    }
}

// Optimized version with reduced synchronization
__global__ void hillisSteleOptimized(float *data, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data with coalescing
    if (idx < n) {
        sdata[tid] = data[idx];
    } else {
        sdata[tid] = 0.0f;
    }
    
    // Double buffer approach
    __shared__ float buffer[512];  // Assuming max block size 512
    __syncthreads();
    
    // Hillis-Steele with alternating buffers
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid >= stride) {
            buffer[tid] = sdata[tid] + sdata[tid - stride];
        } else {
            buffer[tid] = sdata[tid];
        }
        __syncthreads();
        
        // Swap buffers
        sdata[tid] = buffer[tid];
        __syncthreads();
    }
    
    // Write result
    if (idx < n) {
        data[idx] = sdata[tid];
    }
}

// Multi-block scan with Hillis-Steele
__global__ void multiBlockScan(float *data, float *blockSums, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;
    
    // Load data
    if (idx < n) {
        sdata[tid] = data[idx];
    } else {
        sdata[tid] = 0.0f;
    }
    __syncthreads();
    
    // Perform scan within block
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        float temp = 0.0f;
        if (tid >= stride) {
            temp = sdata[tid - stride];
        }
        __syncthreads();
        
        if (tid >= stride) {
            sdata[tid] += temp;
        }
        __syncthreads();
    }
    
    // Save block sum
    if (tid == blockDim.x - 1) {
        blockSums[bid] = sdata[tid];
    }
    
    // Write results
    if (idx < n) {
        data[idx] = sdata[tid];
    }
}

// Add block sums to each block
__global__ void addBlockSums(float *data, float *blockSums, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (blockIdx.x > 0 && idx < n) {
        data[idx] += blockSums[blockIdx.x - 1];
    }
}

// CPU scan for verification
void cpuInclusiveScan(float *data, int n) {
    for (int i = 1; i < n; i++) {
        data[i] += data[i-1];
    }
}

// Verify results
bool verifyResults(float *gpu, float *cpu, int n) {
    const float epsilon = 1e-5f;
    for (int i = 0; i < n; i++) {
        if (fabs(gpu[i] - cpu[i]) > epsilon) {
            printf("Mismatch at index %d: GPU=%.6f, CPU=%.6f\n", 
                   i, gpu[i], cpu[i]);
            return false;
        }
    }
    return true;
}

// Performance test function
float testScan(void (*kernel)(float*, int), const char *name, 
               float *d_data, float *h_data, int n, int blockSize) {
    
    // Copy original data to device
    CHECK_CUDA(cudaMemcpy(d_data, h_data, n * sizeof(float), 
                          cudaMemcpyHostToDevice));
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    int gridSize = (n + blockSize - 1) / blockSize;
    size_t sharedMemSize = blockSize * sizeof(float);
    
    // Warm-up
    kernel<<<gridSize, blockSize, sharedMemSize>>>(d_data, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Reset data
    CHECK_CUDA(cudaMemcpy(d_data, h_data, n * sizeof(float), 
                          cudaMemcpyHostToDevice));
    
    // Measure
    const int iterations = 100;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        kernel<<<gridSize, blockSize, sharedMemSize>>>(d_data, n);
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
    printf("=== Hillis-Steele Parallel Scan ===\n\n");
    
    // Test parameters
    const int blockSize = 256;
    const int testSizes[] = {256, 1024, 4096, 16384, 65536};
    const int numTests = sizeof(testSizes) / sizeof(int);
    
    printf("Block size: %d threads\n\n", blockSize);
    
    for (int test = 0; test < numTests; test++) {
        int n = testSizes[test];
        size_t size = n * sizeof(float);
        
        printf("Test size: %d elements (%.2f KB)\n", n, size / 1024.0f);
        printf("-----------------------------------------\n");
        
        // Allocate host memory
        float *h_data = (float*)malloc(size);
        float *h_result = (float*)malloc(size);
        float *h_cpu = (float*)malloc(size);
        
        // Initialize data
        for (int i = 0; i < n; i++) {
            h_data[i] = 1.0f;  // Simple test pattern
            h_cpu[i] = h_data[i];
        }
        
        // CPU reference
        auto cpuStart = std::chrono::high_resolution_clock::now();
        cpuInclusiveScan(h_cpu, n);
        auto cpuEnd = std::chrono::high_resolution_clock::now();
        float cpuTime = std::chrono::duration_cast<std::chrono::microseconds>
                        (cpuEnd - cpuStart).count() / 1000.0f;
        
        // Allocate device memory
        float *d_data;
        CHECK_CUDA(cudaMalloc(&d_data, size));
        
        // Test basic Hillis-Steele
        if (n <= blockSize) {
            float timeBasic = testScan(hillisSteleInclusiveScan, "Basic", 
                                       d_data, h_data, n, blockSize);
            
            // Get result
            CHECK_CUDA(cudaMemcpy(h_result, d_data, size, cudaMemcpyDeviceToHost));
            
            bool correct = verifyResults(h_result, h_cpu, n);
            printf("Basic Hillis-Steele: %.3f ms - %s\n", 
                   timeBasic, correct ? "PASS" : "FAIL");
        }
        
        // Test optimized version
        if (n <= blockSize) {
            float timeOpt = testScan(hillisSteleOptimized, "Optimized", 
                                    d_data, h_data, n, blockSize);
            
            // Get result
            CHECK_CUDA(cudaMemcpy(h_result, d_data, size, cudaMemcpyDeviceToHost));
            
            bool correct = verifyResults(h_result, h_cpu, n);
            printf("Optimized Hillis-Steele: %.3f ms - %s\n", 
                   timeOpt, correct ? "PASS" : "FAIL");
        }
        
        // Multi-block scan for larger arrays
        if (n > blockSize) {
            int numBlocks = (n + blockSize - 1) / blockSize;
            float *d_blockSums;
            CHECK_CUDA(cudaMalloc(&d_blockSums, numBlocks * sizeof(float)));
            
            // Copy data
            CHECK_CUDA(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
            
            cudaEvent_t start, stop;
            CHECK_CUDA(cudaEventCreate(&start));
            CHECK_CUDA(cudaEventCreate(&stop));
            
            CHECK_CUDA(cudaEventRecord(start));
            
            // Step 1: Scan within blocks
            multiBlockScan<<<numBlocks, blockSize, blockSize * sizeof(float)>>>
                          (d_data, d_blockSums, n);
            
            // Step 2: Scan block sums (simplified - single block)
            if (numBlocks <= blockSize) {
                hillisSteleInclusiveScan<<<1, numBlocks, numBlocks * sizeof(float)>>>
                                         (d_blockSums, numBlocks);
            }
            
            // Step 3: Add block sums
            addBlockSums<<<numBlocks, blockSize>>>(d_data, d_blockSums, n);
            
            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaEventSynchronize(stop));
            
            float ms = 0;
            CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
            
            // Get result
            CHECK_CUDA(cudaMemcpy(h_result, d_data, size, cudaMemcpyDeviceToHost));
            
            bool correct = verifyResults(h_result, h_cpu, n);
            printf("Multi-block scan: %.3f ms - %s\n", 
                   ms, correct ? "PASS" : "FAIL");
            
            CHECK_CUDA(cudaFree(d_blockSums));
            CHECK_CUDA(cudaEventDestroy(start));
            CHECK_CUDA(cudaEventDestroy(stop));
        }
        
        printf("CPU reference: %.3f ms\n", cpuTime);
        
        // Calculate metrics
        float bandwidth = (2.0f * size / 1e9) / (cpuTime / 1000);  // GB/s
        printf("Memory bandwidth: %.2f GB/s\n", bandwidth);
        
        // Work complexity analysis
        int depth = (int)ceil(log2f(n));
        int work = n * depth;
        printf("Algorithm complexity: O(%d) work, O(%d) depth\n", work, depth);
        
        printf("\n");
        
        // Cleanup
        free(h_data);
        free(h_result);
        free(h_cpu);
        CHECK_CUDA(cudaFree(d_data));
    }
    
    // Algorithm characteristics
    printf("\n=== Hillis-Steele Characteristics ===\n");
    printf("• Work complexity: O(n log n) - not work-efficient\n");
    printf("• Depth complexity: O(log n) - depth-efficient\n");
    printf("• Memory access: Coalesced within warps\n");
    printf("• Synchronization: log(n) barriers\n");
    printf("• Best for: Small arrays that fit in shared memory\n");
    printf("• Trade-off: More work for less depth\n");
    
    printf("\nProgram completed!\n");
    return 0;
}