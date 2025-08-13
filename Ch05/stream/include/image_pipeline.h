// image_pipeline.h - Image processing pipeline class

#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <functional>

enum class ColorSpace {
    RGB,
    YUV,
    HSV,
    BGR
};

// Forward declarations of kernel functions
extern "C" {
    void launch_sobel(unsigned char* data, int width, int height, cudaStream_t stream);
    void launch_resize(unsigned char* src, unsigned char* dst, 
                      int src_w, int src_h, int dst_w, int dst_h, cudaStream_t stream);
    void launch_gaussian_blur(unsigned char* data, int width, int height, 
                             float sigma, cudaStream_t stream);
    void launch_color_conversion(unsigned char* data, int width, int height,
                                ColorSpace from, ColorSpace to, cudaStream_t stream);
}

class ImagePipeline {
public:
    struct Image {
        unsigned char* data;
        int width, height, channels;
        
        Image(int w, int h, int c) : width(w), height(h), channels(c) {
            cudaMalloc(&data, w * h * c);
        }
        
        ~Image() {
            if (data) cudaFree(data);
        }
        
        // Move constructor
        Image(Image&& other) noexcept 
            : data(other.data), width(other.width), 
              height(other.height), channels(other.channels) {
            other.data = nullptr;
        }
        
        // Move assignment
        Image& operator=(Image&& other) noexcept {
            if (this != &other) {
                if (data) cudaFree(data);
                data = other.data;
                width = other.width;
                height = other.height;
                channels = other.channels;
                other.data = nullptr;
            }
            return *this;
        }
        
        // Copy constructor (deep copy)
        Image(const Image& other) 
            : width(other.width), height(other.height), channels(other.channels) {
            size_t size = width * height * channels;
            cudaMalloc(&data, size);
            cudaMemcpy(data, other.data, size, cudaMemcpyDeviceToDevice);
        }
    };

private:
    std::vector<std::function<void(Image&)>> stages_;
    cudaStream_t stream_;
    
public:
    ImagePipeline() {
        cudaStreamCreate(&stream_);
    }
    
    ~ImagePipeline() {
        cudaStreamDestroy(stream_);
    }
    
    void addStage(std::function<void(Image&)> stage) {
        stages_.push_back(stage);
    }
    
    void addGaussianBlur(float sigma) {
        stages_.push_back([this, sigma](Image& img) {
            launch_gaussian_blur(img.data, img.width, img.height, sigma, stream_);
        });
    }
    
    void addSobelEdgeDetection() {
        stages_.push_back([this](Image& img) {
            launch_sobel(img.data, img.width, img.height, stream_);
        });
    }
    
    void addColorConversion(ColorSpace from, ColorSpace to) {
        stages_.push_back([this, from, to](Image& img) {
            launch_color_conversion(img.data, img.width, img.height, from, to, stream_);
        });
    }
    
    void addResize(int new_width, int new_height) {
        stages_.push_back([this, new_width, new_height](Image& img) {
            Image resized(new_width, new_height, img.channels);
            launch_resize(img.data, resized.data, 
                         img.width, img.height, new_width, new_height, stream_);
            img = std::move(resized);
        });
    }
    
    void process(const Image& input, Image& output) {
        Image current(input);  // Deep copy
        
        for (auto& stage : stages_) {
            stage(current);
            cudaStreamSynchronize(stream_);
        }
        
        output = std::move(current);
    }
};