/*
 * stream_compaction.cu
 * Stream Compaction using Parallel Scan
 * Removes unwanted elements from arrays efficiently
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>
#include <random>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// Predicate function - keep only positive values
__device__ bool keepElement(float value) {
    return value > 0.0f;
}

// Step 1: Mark valid elements
__global__ void markValidElements(float *input, int *flags, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        flags[idx] = keepElement(input[idx]) ? 1 : 0;
    }
}

// Step 2: Exclusive scan on flags (simplified single-block version)
__global__ void exclusiveScan(int *data, int n) {
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    
    // Load data
    if (tid < n) {
        sdata[tid] = data[tid];
    } else {
        sdata[tid] = 0;
    }
    __syncthreads();
    
    // Up-sweep
    int offset = 1;
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            sdata[bi] += sdata[ai];
        }
        offset *= 2;
    }
    
    // Clear last element
    if (tid == 0) sdata[n - 1] = 0;
    
    // Down-sweep
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            int temp = sdata[ai];
            sdata[ai] = sdata[bi];
            sdata[bi] += temp;
        }
    }
    __syncthreads();
    
    // Write back
    if (tid < n) {
        data[tid] = sdata[tid];
    }
}

// Step 3: Scatter valid elements
__global__ void scatterElements(float *input, float *output, 
                                int *flags, int *addresses, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n && flags[idx]) {
        output[addresses[idx]] = input[idx];
    }
}

// Combined kernel for better performance
__global__ void streamCompactionOptimized(float *input, float *output, 
                                         int *count, int n) {
    extern __shared__ int sharedMem[];
    int *sFlags = sharedMem;
    int *sAddresses = &sharedMem[blockDim.x];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Step 1: Load and mark valid elements
    int flag = 0;
    float value = 0.0f;
    if (idx < n) {
        value = input[idx];
        flag = keepElement(value) ? 1 : 0;
    }
    sFlags[tid] = flag;
    __syncthreads();
    
    // Step 2: Exclusive scan within block
    // Simplified scan for block size
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int temp = 0;
        if (tid >= stride) {
            temp = sFlags[tid - stride];
        }
        __syncthreads();
        if (tid >= stride) {
            sFlags[tid] += temp;
        }
        __syncthreads();
    }
    
    // Save addresses
    sAddresses[tid] = sFlags[tid];
    __syncthreads();
    
    // Convert to exclusive scan
    if (tid > 0) {
        sAddresses[tid] = sFlags[tid - 1];
    } else {
        sAddresses[tid] = 0;
    }
    __syncthreads();
    
    // Step 3: Scatter to output
    if (idx < n && flag) {
        output[blockIdx.x * blockDim.x + sAddresses[tid]] = value;
    }
    
    // Count valid elements in this block
    if (tid == blockDim.x - 1) {
        atomicAdd(count, sFlags[tid]);
    }
}

// Warp-level stream compaction using ballot
__global__ void streamCompactionWarp(float *input, float *output, 
                                     int *globalOffset, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;
    
    __shared__ int warpOffsets[32];  // Max 32 warps per block
    
    // Load element and check predicate
    float value = 0.0f;
    bool valid = false;
    if (idx < n) {
        value = input[idx];
        valid = keepElement(value);
    }
    
    // Warp-level voting
    unsigned mask = __ballot_sync(0xffffffff, valid);
    int warpCount = __popc(mask);
    int prefix = __popc(mask & ((1u << laneId) - 1));
    
    // First thread in warp reserves space
    if (laneId == 0) {
        warpOffsets[warpId] = atomicAdd(globalOffset, warpCount);
    }
    __syncthreads();
    
    // Write compacted data
    if (valid) {
        output[warpOffsets[warpId] + prefix] = value;
    }
}

// CPU stream compaction for verification
int cpuStreamCompaction(float *input, float *output, int n) {
    int count = 0;
    for (int i = 0; i < n; i++) {
        if (input[i] > 0.0f) {
            output[count++] = input[i];
        }
    }
    return count;
}

// Verify compacted results
bool verifyCompaction(float *gpu, float *cpu, int count, int cpuCount) {
    if (count != cpuCount) {
        printf("Count mismatch: GPU=%d, CPU=%d\n", count, cpuCount);
        return false;
    }
    
    const float epsilon = 1e-6f;
    for (int i = 0; i < count; i++) {
        if (fabs(gpu[i] - cpu[i]) > epsilon) {
            printf("Value mismatch at %d: GPU=%.6f, CPU=%.6f\n", 
                   i, gpu[i], cpu[i]);
            return false;
        }
    }
    return true;
}

int main() {
    printf("=== Stream Compaction using Parallel Scan ===\n\n");
    
    // Test parameters
    const int testSizes[] = {1024, 4096, 16384, 65536, 262144};
    const int numTests = sizeof(testSizes) / sizeof(int);
    const float sparsity = 0.3f;  // 30% of elements will be kept
    
    printf("Sparsity: %.1f%% elements kept\n\n", sparsity * 100);
    
    for (int test = 0; test < numTests; test++) {
        int n = testSizes[test];
        size_t size = n * sizeof(float);
        
        printf("Input size: %d elements (%.2f MB)\n", n, size / (1024.0f * 1024.0f));
        printf("------------------------------------------------\n");
        
        // Allocate host memory
        float *h_input = (float*)malloc(size);
        float *h_output = (float*)malloc(size);
        float *h_cpu = (float*)malloc(size);
        
        // Initialize input with mixed positive/negative values
        std::mt19937 gen(42);
        std::uniform_real_distribution<> dis(-1.0, 1.0);
        
        int expectedCount = 0;
        for (int i = 0; i < n; i++) {
            h_input[i] = dis(gen);
            if (h_input[i] > 0.0f) expectedCount++;
        }
        
        printf("Expected output size: %d elements (%.1f%%)\n", 
               expectedCount, (float)expectedCount / n * 100);
        
        // CPU reference
        auto cpuStart = std::chrono::high_resolution_clock::now();
        int cpuCount = cpuStreamCompaction(h_input, h_cpu, n);
        auto cpuEnd = std::chrono::high_resolution_clock::now();
        float cpuTime = std::chrono::duration_cast<std::chrono::microseconds>
                        (cpuEnd - cpuStart).count() / 1000.0f;
        
        // Allocate device memory
        float *d_input, *d_output;
        int *d_flags, *d_addresses, *d_count;
        CHECK_CUDA(cudaMalloc(&d_input, size));
        CHECK_CUDA(cudaMalloc(&d_output, size));
        CHECK_CUDA(cudaMalloc(&d_flags, n * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_addresses, n * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_count, sizeof(int)));
        
        // Copy input to device
        CHECK_CUDA(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
        
        // Method 1: Three-step approach
        int blockSize = 256;
        int gridSize = (n + blockSize - 1) / blockSize;
        
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        
        CHECK_CUDA(cudaEventRecord(start));
        
        // Step 1: Mark valid elements
        markValidElements<<<gridSize, blockSize>>>(d_input, d_flags, n);
        
        // Step 2: Exclusive scan (simplified for small arrays)
        if (n <= 1024) {
            exclusiveScan<<<1, n, n * sizeof(int)>>>(d_flags, n);
        }
        CHECK_CUDA(cudaMemcpy(d_addresses, d_flags, n * sizeof(int), 
                              cudaMemcpyDeviceToDevice));
        
        // Step 3: Scatter
        scatterElements<<<gridSize, blockSize>>>(d_input, d_output, 
                                                 d_flags, d_addresses, n);
        
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        
        float threeStepTime = 0;
        CHECK_CUDA(cudaEventElapsedTime(&threeStepTime, start, stop));
        
        // Get result count
        int gpuCount = 0;
        CHECK_CUDA(cudaMemcpy(&gpuCount, &d_addresses[n-1], sizeof(int), 
                              cudaMemcpyDeviceToHost));
        int lastFlag = 0;
        CHECK_CUDA(cudaMemcpy(&lastFlag, &d_flags[n-1], sizeof(int), 
                              cudaMemcpyDeviceToHost));
        gpuCount += lastFlag;
        
        // Verify three-step result
        CHECK_CUDA(cudaMemcpy(h_output, d_output, gpuCount * sizeof(float), 
                              cudaMemcpyDeviceToHost));
        bool threeStepCorrect = verifyCompaction(h_output, h_cpu, gpuCount, cpuCount);
        
        // Method 2: Optimized single kernel
        CHECK_CUDA(cudaMemset(d_count, 0, sizeof(int)));
        CHECK_CUDA(cudaMemset(d_output, 0, size));
        
        CHECK_CUDA(cudaEventRecord(start));
        streamCompactionOptimized<<<gridSize, blockSize, 2 * blockSize * sizeof(int)>>>
                                 (d_input, d_output, d_count, n);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        
        float optimizedTime = 0;
        CHECK_CUDA(cudaEventElapsedTime(&optimizedTime, start, stop));
        
        // Method 3: Warp-level compaction
        CHECK_CUDA(cudaMemset(d_count, 0, sizeof(int)));
        CHECK_CUDA(cudaMemset(d_output, 0, size));
        
        CHECK_CUDA(cudaEventRecord(start));
        streamCompactionWarp<<<gridSize, blockSize>>>(d_input, d_output, d_count, n);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        
        float warpTime = 0;
        CHECK_CUDA(cudaEventElapsedTime(&warpTime, start, stop));
        
        // Performance comparison
        printf("\nPerformance Results:\n");
        printf("%-25s %-12s %-12s %-12s\n", 
               "Method", "Time (ms)", "Throughput", "Speedup");
        printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" 
               "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" 
               "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" 
               "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "\n");
        
        float cpuThroughput = n / (cpuTime * 1e6);  // Elements/us
        float threeStepThroughput = n / (threeStepTime * 1e6);
        float optimizedThroughput = n / (optimizedTime * 1e6);
        float warpThroughput = n / (warpTime * 1e6);
        
        printf("%-25s %-12.3f %-12.2f %-12s\n", 
               "CPU Sequential", cpuTime, cpuThroughput, "1.00x");
        printf("%-25s %-12.3f %-12.2f %-12.2fx %s\n", 
               "Three-step Scan", threeStepTime, threeStepThroughput, 
               cpuTime/threeStepTime, threeStepCorrect ? "" : "[FAIL]");
        printf("%-25s %-12.3f %-12.2f %-12.2fx\n", 
               "Optimized Single Kernel", optimizedTime, optimizedThroughput, 
               cpuTime/optimizedTime);
        printf("%-25s %-12.3f %-12.2f %-12.2fx\n", 
               "Warp-level Ballot", warpTime, warpThroughput, 
               cpuTime/warpTime);
        
        // Compression ratio
        float compressionRatio = (float)cpuCount / n;
        float bandwidthUsed = (size + cpuCount * sizeof(float)) / (1e9 * warpTime / 1000);
        
        printf("\nCompression Statistics:\n");
        printf("  Input size: %d elements\n", n);
        printf("  Output size: %d elements\n", cpuCount);
        printf("  Compression ratio: %.2f:1\n", 1.0f / compressionRatio);
        printf("  Space saved: %.1f%%\n", (1 - compressionRatio) * 100);
        printf("  Effective bandwidth: %.2f GB/s\n", bandwidthUsed);
        
        printf("\n");
        
        // Cleanup
        free(h_input);
        free(h_output);
        free(h_cpu);
        CHECK_CUDA(cudaFree(d_input));
        CHECK_CUDA(cudaFree(d_output));
        CHECK_CUDA(cudaFree(d_flags));
        CHECK_CUDA(cudaFree(d_addresses));
        CHECK_CUDA(cudaFree(d_count));
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }
    
    // Algorithm applications
    printf("\n=== Stream Compaction Applications ===\n");
    printf("• Sparse matrix operations\n");
    printf("• Particle simulations (remove dead particles)\n");
    printf("• Ray tracing (active ray compaction)\n");
    printf("• Database operations (WHERE clauses)\n");
    printf("• Graph algorithms (active vertex lists)\n");
    printf("• Image processing (pixel filtering)\n");
    
    printf("\n=== Optimization Strategies ===\n");
    printf("1. Warp-level primitives (__ballot_sync, __popc)\n");
    printf("2. Shared memory for intra-block compaction\n");
    printf("3. Multi-level scan for large arrays\n");
    printf("4. Predicate fusion to avoid multiple passes\n");
    printf("5. Adaptive algorithms based on sparsity\n");
    
    printf("\nProgram completed!\n");
    return 0;
}