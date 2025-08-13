// cuda_utils.h - CUDA utility functions

#pragma once

#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <iostream>
#include <string>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                     << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

// cuDNN error checking macro
#define CUDNN_CHECK(call) \
    do { \
        cudnnStatus_t status = call; \
        if (status != CUDNN_STATUS_SUCCESS) { \
            std::cerr << "cuDNN error at " << __FILE__ << ":" << __LINE__ \
                     << " - " << cudnnGetErrorString(status) << std::endl; \
            exit(1); \
        } \
    } while(0)

// cuBLAS error checking macro
#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ \
                     << " - " << status << std::endl; \
            exit(1); \
        } \
    } while(0)

// Memory management helpers
template<typename T>
class CudaMemory {
private:
    T* d_ptr;
    size_t size;
    
public:
    CudaMemory(size_t n) : size(n) {
        CUDA_CHECK(cudaMalloc(&d_ptr, sizeof(T) * n));
    }
    
    ~CudaMemory() {
        if (d_ptr) {
            cudaFree(d_ptr);
        }
    }
    
    T* get() { return d_ptr; }
    const T* get() const { return d_ptr; }
    size_t get_size() const { return size; }
    
    void copy_from_host(const T* h_ptr) {
        CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, sizeof(T) * size, cudaMemcpyHostToDevice));
    }
    
    void copy_to_host(T* h_ptr) const {
        CUDA_CHECK(cudaMemcpy(h_ptr, d_ptr, sizeof(T) * size, cudaMemcpyDeviceToHost));
    }
    
    void set_zero() {
        CUDA_CHECK(cudaMemset(d_ptr, 0, sizeof(T) * size));
    }
};

// Kernel launch configuration helper
struct LaunchConfig {
    dim3 blocks;
    dim3 threads;
    size_t shared_mem;
    cudaStream_t stream;
    
    LaunchConfig(int total_threads, int block_size = 256) 
        : shared_mem(0), stream(0) {
        threads = dim3(block_size);
        blocks = dim3((total_threads + block_size - 1) / block_size);
    }
    
    LaunchConfig(dim3 grid, dim3 block, size_t shared = 0, cudaStream_t s = 0)
        : blocks(grid), threads(block), shared_mem(shared), stream(s) {}
};

// Device properties query
class DeviceInfo {
private:
    cudaDeviceProp prop;
    int device_id;
    
public:
    DeviceInfo(int id = 0) : device_id(id) {
        CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
    }
    
    void print_info() const {
        std::cout << "Device " << device_id << ": " << prop.name << std::endl;
        std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  SMs: " << prop.multiProcessorCount << std::endl;
        std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Max blocks per SM: " << prop.maxBlocksPerMultiProcessor << std::endl;
        std::cout << "  Shared memory per block: " << prop.sharedMemPerBlock << " bytes" << std::endl;
        std::cout << "  Total global memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    }
    
    int get_sm_count() const { return prop.multiProcessorCount; }
    int get_max_threads_per_block() const { return prop.maxThreadsPerBlock; }
    size_t get_shared_mem_per_block() const { return prop.sharedMemPerBlock; }
};

// Stream wrapper
class CudaStream {
private:
    cudaStream_t stream;
    
public:
    CudaStream() {
        CUDA_CHECK(cudaStreamCreate(&stream));
    }
    
    ~CudaStream() {
        cudaStreamDestroy(stream);
    }
    
    cudaStream_t get() { return stream; }
    
    void synchronize() {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
};

// Event timer
class CudaTimer {
private:
    cudaEvent_t start, stop;
    
public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }
    
    ~CudaTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    void start_timer(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(start, stream));
    }
    
    float stop_timer(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        return milliseconds;
    }
};