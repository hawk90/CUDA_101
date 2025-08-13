#ifndef TENSOR_H
#define TENSOR_H

#include <cuda_runtime.h>
#include <vector>
#include <memory>

class TensorFloat {
private:
    float* data_;
    std::vector<int> shape_;
    size_t size_;
    bool on_device_;

public:
    // Constructor for N-dimensional tensor
    TensorFloat(std::initializer_list<int> shape) : shape_(shape) {
        size_ = 1;
        for (int dim : shape_) {
            size_ *= dim;
        }
        cudaMalloc(&data_, size_ * sizeof(float));
        on_device_ = true;
    }
    
    // Constructor with dimensions
    TensorFloat(int n, int c, int h, int w) 
        : shape_({n, c, h, w}) {
        size_ = n * c * h * w;
        cudaMalloc(&data_, size_ * sizeof(float));
        on_device_ = true;
    }
    
    ~TensorFloat() {
        if (data_ && on_device_) {
            cudaFree(data_);
        }
    }
    
    // Move constructor
    TensorFloat(TensorFloat&& other) noexcept 
        : data_(other.data_), shape_(std::move(other.shape_)), 
          size_(other.size_), on_device_(other.on_device_) {
        other.data_ = nullptr;
    }
    
    // Move assignment
    TensorFloat& operator=(TensorFloat&& other) noexcept {
        if (this != &other) {
            if (data_ && on_device_) {
                cudaFree(data_);
            }
            data_ = other.data_;
            shape_ = std::move(other.shape_);
            size_ = other.size_;
            on_device_ = other.on_device_;
            other.data_ = nullptr;
        }
        return *this;
    }
    
    // Accessors
    float* data() { return data_; }
    const float* data() const { return data_; }
    size_t size() const { return size_; }
    const std::vector<int>& shape() const { return shape_; }
    
    int dim(int i) const { return shape_[i]; }
    int batch() const { return shape_[0]; }
    int channels() const { return shape_[1]; }
    int height() const { return shape_[2]; }
    int width() const { return shape_[3]; }
    
    // Reshape
    void reshape(std::initializer_list<int> new_shape) {
        size_t new_size = 1;
        for (int dim : new_shape) {
            new_size *= dim;
        }
        if (new_size != size_) {
            throw std::runtime_error("Reshape size mismatch");
        }
        shape_ = new_shape;
    }
    
    // Copy to host
    std::vector<float> to_host() const {
        std::vector<float> host_data(size_);
        cudaMemcpy(host_data.data(), data_, size_ * sizeof(float), 
                   cudaMemcpyDeviceToHost);
        return host_data;
    }
    
    // Copy from host
    void from_host(const std::vector<float>& host_data) {
        if (host_data.size() != size_) {
            throw std::runtime_error("Size mismatch");
        }
        cudaMemcpy(data_, host_data.data(), size_ * sizeof(float), 
                   cudaMemcpyHostToDevice);
    }
};

#endif // TENSOR_H