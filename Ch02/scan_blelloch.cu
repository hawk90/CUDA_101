/*
 * scan_blelloch.cu
 * Blelloch (Work-Efficient) Parallel Scan Algorithm
 * Two-phase approach: up-sweep and down-sweep
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

// Blelloch exclusive scan for power-of-2 arrays
__global__ void blellochExclusiveScan(float *data, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int offset = 1;
    
    // Load data into shared memory
    sdata[2*tid] = (2*tid < n) ? data[2*tid] : 0.0f;
    sdata[2*tid+1] = (2*tid+1 < n) ? data[2*tid+1] : 0.0f;
    
    // Build sum tree (up-sweep)
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2*tid + 1) - 1;
            int bi = offset * (2*tid + 2) - 1;
            sdata[bi] += sdata[ai];
        }
        offset *= 2;
    }
    
    // Clear the last element
    if (tid == 0) {
        sdata[n - 1] = 0;
    }
    
    // Traverse down tree and build scan (down-sweep)
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2*tid + 1) - 1;
            int bi = offset * (2*tid + 2) - 1;
            float temp = sdata[ai];
            sdata[ai] = sdata[bi];
            sdata[bi] += temp;
        }
    }
    __syncthreads();
    
    // Write results back
    if (2*tid < n) data[2*tid] = sdata[2*tid];
    if (2*tid+1 < n) data[2*tid+1] = sdata[2*tid+1];
}

// Optimized Blelloch with bank conflict avoidance
__global__ void blellochOptimized(float *data, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int offset = 1;
    
    // Add padding to avoid bank conflicts
    int ai = tid;
    int bi = tid + (n/2);
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
    
    // Load with padding
    sdata[ai + bankOffsetA] = (ai < n) ? data[ai] : 0.0f;
    sdata[bi + bankOffsetB] = (bi < n) ? data[bi] : 0.0f;
    
    // Up-sweep phase
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2*tid + 1) - 1;
            int bi = offset * (2*tid + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            
            sdata[bi] += sdata[ai];
        }
        offset *= 2;
    }
    
    // Clear last element
    if (tid == 0) {
        int idx = n - 1 + CONFLICT_FREE_OFFSET(n - 1);
        sdata[idx] = 0;
    }
    
    // Down-sweep phase
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2*tid + 1) - 1;
            int bi = offset * (2*tid + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            
            float temp = sdata[ai];
            sdata[ai] = sdata[bi];
            sdata[bi] += temp;
        }
    }
    __syncthreads();
    
    // Write back with padding
    if (ai < n) data[ai] = sdata[ai + bankOffsetA];
    if (bi < n) data[bi] = sdata[bi + bankOffsetB];
}

// Helper macro for bank conflict avoidance
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)

// Inclusive scan version (adds identity element)
__global__ void blellochInclusiveScan(float *data, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 2 + tid;
    
    // Load two elements per thread
    sdata[tid] = (idx < n) ? data[idx] : 0.0f;
    sdata[tid + blockDim.x] = (idx + blockDim.x < n) ? data[idx + blockDim.x] : 0.0f;
    __syncthreads();
    
    // Perform exclusive scan
    int offset = 1;
    
    // Up-sweep
    for (int d = blockDim.x; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2*tid + 1) - 1;
            int bi = offset * (2*tid + 2) - 1;
            if (bi < 2*blockDim.x) {
                sdata[bi] += sdata[ai];
            }
        }
        offset *= 2;
    }
    
    // Save and clear last element
    float lastElement = 0;
    if (tid == 0) {
        lastElement = sdata[2*blockDim.x - 1];
        sdata[2*blockDim.x - 1] = 0;
    }
    
    // Down-sweep
    for (int d = 1; d < 2*blockDim.x; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2*tid + 1) - 1;
            int bi = offset * (2*tid + 2) - 1;
            if (bi < 2*blockDim.x) {
                float temp = sdata[ai];
                sdata[ai] = sdata[bi];
                sdata[bi] += temp;
            }
        }
    }
    __syncthreads();
    
    // Convert to inclusive by shifting
    if (tid > 0) {
        float temp = sdata[tid - 1];
        __syncthreads();
        sdata[tid] = temp + ((tid < n) ? data[idx] : 0.0f);
    }
    __syncthreads();
    
    // Write back
    if (idx < n) data[idx] = sdata[tid];
    if (idx + blockDim.x < n) data[idx + blockDim.x] = sdata[tid + blockDim.x];
}

// CPU scan for verification
void cpuExclusiveScan(float *data, int n) {
    float prev = data[0];
    data[0] = 0;
    for (int i = 1; i < n; i++) {
        float temp = data[i];
        data[i] = prev + data[i-1];
        prev = temp;
    }
}

void cpuInclusiveScan(float *data, int n) {
    for (int i = 1; i < n; i++) {
        data[i] += data[i-1];
    }
}

// Verify results
bool verifyResults(float *gpu, float *cpu, int n, const char *name) {
    const float epsilon = 1e-5f;
    int errors = 0;
    for (int i = 0; i < n; i++) {
        if (fabs(gpu[i] - cpu[i]) > epsilon) {
            if (errors < 5) {  // Print first 5 errors
                printf("%s - Mismatch at %d: GPU=%.6f, CPU=%.6f\n", 
                       name, i, gpu[i], cpu[i]);
            }
            errors++;
        }
    }
    return errors == 0;
}

// Performance comparison
void compareAlgorithms() {
    printf("\n=== Algorithm Comparison ===\n");
    printf("%-20s %-15s %-15s %-15s\n", 
           "Algorithm", "Work", "Depth", "Efficiency");
    printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" 
           "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" 
           "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" 
           "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "\n");
    
    printf("%-20s %-15s %-15s %-15s\n", 
           "Sequential", "O(n)", "O(n)", "Baseline");
    printf("%-20s %-15s %-15s %-15s\n", 
           "Hillis-Steele", "O(n log n)", "O(log n)", "Work-inefficient");
    printf("%-20s %-15s %-15s %-15s\n", 
           "Blelloch", "O(n)", "O(log n)", "Work-efficient");
    printf("%-20s %-15s %-15s %-15s\n", 
           "Kogge-Stone", "O(n log n)", "O(log n)", "Simple");
}

int main() {
    printf("=== Blelloch (Work-Efficient) Parallel Scan ===\n\n");
    
    // Test parameters
    const int testSizes[] = {128, 256, 512, 1024, 2048};
    const int numTests = sizeof(testSizes) / sizeof(int);
    
    for (int test = 0; test < numTests; test++) {
        int n = testSizes[test];
        size_t size = n * sizeof(float);
        
        printf("Test size: %d elements\n", n);
        printf("----------------------------------------\n");
        
        // Allocate host memory
        float *h_data = (float*)malloc(size);
        float *h_result = (float*)malloc(size);
        float *h_cpu = (float*)malloc(size);
        
        // Initialize test data
        for (int i = 0; i < n; i++) {
            h_data[i] = 1.0f;  // Simple pattern for testing
            h_cpu[i] = h_data[i];
        }
        
        // CPU reference (exclusive scan)
        auto cpuStart = std::chrono::high_resolution_clock::now();
        cpuExclusiveScan(h_cpu, n);
        auto cpuEnd = std::chrono::high_resolution_clock::now();
        float cpuTime = std::chrono::duration_cast<std::chrono::microseconds>
                        (cpuEnd - cpuStart).count() / 1000.0f;
        
        // Allocate device memory
        float *d_data;
        CHECK_CUDA(cudaMalloc(&d_data, size));
        
        // Test Blelloch exclusive scan
        CHECK_CUDA(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
        
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        
        int blockSize = n / 2;  // Blelloch needs n/2 threads for n elements
        size_t sharedMemSize = n * sizeof(float);
        
        // Warm-up
        blellochExclusiveScan<<<1, blockSize, sharedMemSize>>>(d_data, n);
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // Reset and measure
        CHECK_CUDA(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
        
        const int iterations = 100;
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < iterations; i++) {
            blellochExclusiveScan<<<1, blockSize, sharedMemSize>>>(d_data, n);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        
        float ms = 0;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        float gpuTime = ms / iterations;
        
        // Get result and verify
        CHECK_CUDA(cudaMemcpy(h_result, d_data, size, cudaMemcpyDeviceToHost));
        bool correct = verifyResults(h_result, h_cpu, n, "Blelloch");
        
        // Performance metrics
        printf("Blelloch Exclusive Scan:\n");
        printf("  GPU Time: %.3f ms\n", gpuTime);
        printf("  CPU Time: %.3f ms\n", cpuTime);
        printf("  Speedup: %.2fx\n", cpuTime / gpuTime);
        printf("  Result: %s\n", correct ? "PASS" : "FAIL");
        
        // Calculate efficiency metrics
        float bandwidth = (2.0f * size / 1e9) / (gpuTime / 1000);
        int workComplexity = n;  // O(n) work
        int depthComplexity = (int)ceil(log2f(n));  // O(log n) depth
        
        printf("  Bandwidth: %.2f GB/s\n", bandwidth);
        printf("  Work: O(%d), Depth: O(%d)\n", workComplexity, depthComplexity);
        printf("  Efficiency: %.2f%%\n", 
               (float)workComplexity / (n * depthComplexity) * 100);
        
        printf("\n");
        
        // Cleanup
        free(h_data);
        free(h_result);
        free(h_cpu);
        CHECK_CUDA(cudaFree(d_data));
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }
    
    // Compare algorithms
    compareAlgorithms();
    
    // Implementation notes
    printf("\n=== Implementation Notes ===\n");
    printf("• Two-phase algorithm: up-sweep and down-sweep\n");
    printf("• Work-efficient: O(n) operations\n");
    printf("• Requires power-of-2 array sizes (padding needed)\n");
    printf("• Uses n/2 threads for n elements\n");
    printf("• Bank conflicts can impact performance\n");
    printf("• Best for large arrays where work efficiency matters\n");
    
    printf("\n=== Optimization Strategies ===\n");
    printf("1. Padding for bank conflict avoidance\n");
    printf("2. Multiple elements per thread\n");
    printf("3. Warp-level primitives for small reductions\n");
    printf("4. Hybrid approaches for non-power-of-2 sizes\n");
    
    printf("\nProgram completed!\n");
    return 0;
}