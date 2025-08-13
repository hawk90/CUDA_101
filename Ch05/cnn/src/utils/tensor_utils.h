// tensor_utils.h - Tensor manipulation utilities

#pragma once

#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include "cuda_utils.h"

// Tensor initialization strategies
class TensorInitializer {
public:
    // Xavier/Glorot initialization
    static void xavier_init(float* data, size_t size, int fan_in, int fan_out) {
        std::random_device rd;
        std::mt19937 gen(rd());
        float limit = sqrtf(6.0f / (fan_in + fan_out));
        std::uniform_real_distribution<float> dist(-limit, limit);
        
        for (size_t i = 0; i < size; i++) {
            data[i] = dist(gen);
        }
    }
    
    // He initialization
    static void he_init(float* data, size_t size, int fan_in) {
        std::random_device rd;
        std::mt19937 gen(rd());
        float std_dev = sqrtf(2.0f / fan_in);
        std::normal_distribution<float> dist(0.0f, std_dev);
        
        for (size_t i = 0; i < size; i++) {
            data[i] = dist(gen);
        }
    }
    
    // Uniform initialization
    static void uniform_init(float* data, size_t size, float min_val, float max_val) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(min_val, max_val);
        
        for (size_t i = 0; i < size; i++) {
            data[i] = dist(gen);
        }
    }
    
    // Zero initialization
    static void zero_init(float* data, size_t size) {
        std::fill_n(data, size, 0.0f);
    }
    
    // Constant initialization
    static void constant_init(float* data, size_t size, float value) {
        std::fill_n(data, size, value);
    }
};

// Tensor shape utilities
class TensorShape {
private:
    std::vector<int> dims;
    size_t total_size;
    
public:
    TensorShape(std::initializer_list<int> shape) : dims(shape) {
        total_size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
    }
    
    TensorShape(const std::vector<int>& shape) : dims(shape) {
        total_size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
    }
    
    size_t size() const { return total_size; }
    int ndims() const { return dims.size(); }
    int dim(int i) const { return dims[i]; }
    const std::vector<int>& get_dims() const { return dims; }
    
    // Reshape tensor
    TensorShape reshape(std::initializer_list<int> new_shape) const {
        std::vector<int> new_dims(new_shape);
        int infer_idx = -1;
        size_t known_size = 1;
        
        for (int i = 0; i < new_dims.size(); i++) {
            if (new_dims[i] == -1) {
                infer_idx = i;
            } else {
                known_size *= new_dims[i];
            }
        }
        
        if (infer_idx >= 0) {
            new_dims[infer_idx] = total_size / known_size;
        }
        
        return TensorShape(new_dims);
    }
    
    // Get strides for indexing
    std::vector<int> get_strides() const {
        std::vector<int> strides(dims.size());
        strides[dims.size() - 1] = 1;
        
        for (int i = dims.size() - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
        
        return strides;
    }
};

// Data augmentation utilities
class DataAugmentation {
public:
    // Random crop
    static void random_crop(
        const float* input, float* output,
        int in_h, int in_w, int out_h, int out_w, int channels
    ) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> h_dist(0, in_h - out_h);
        std::uniform_int_distribution<> w_dist(0, in_w - out_w);
        
        int h_offset = h_dist(gen);
        int w_offset = w_dist(gen);
        
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < out_h; h++) {
                for (int w = 0; w < out_w; w++) {
                    int in_idx = c * in_h * in_w + (h + h_offset) * in_w + (w + w_offset);
                    int out_idx = c * out_h * out_w + h * out_w + w;
                    output[out_idx] = input[in_idx];
                }
            }
        }
    }
    
    // Random horizontal flip
    static void random_flip(float* data, int height, int width, int channels, float prob = 0.5f) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dist(0.0, 1.0);
        
        if (dist(gen) < prob) {
            for (int c = 0; c < channels; c++) {
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width / 2; w++) {
                        int idx1 = c * height * width + h * width + w;
                        int idx2 = c * height * width + h * width + (width - 1 - w);
                        std::swap(data[idx1], data[idx2]);
                    }
                }
            }
        }
    }
    
    // Normalize with mean and std
    static void normalize(float* data, size_t size, float mean, float std) {
        for (size_t i = 0; i < size; i++) {
            data[i] = (data[i] - mean) / std;
        }
    }
};

// Metrics calculation
class Metrics {
public:
    // Calculate accuracy
    static float accuracy(const float* predictions, const int* labels, int batch_size, int num_classes) {
        int correct = 0;
        
        for (int i = 0; i < batch_size; i++) {
            int pred_class = 0;
            float max_prob = predictions[i * num_classes];
            
            for (int c = 1; c < num_classes; c++) {
                if (predictions[i * num_classes + c] > max_prob) {
                    max_prob = predictions[i * num_classes + c];
                    pred_class = c;
                }
            }
            
            if (pred_class == labels[i]) {
                correct++;
            }
        }
        
        return static_cast<float>(correct) / batch_size;
    }
    
    // Calculate top-k accuracy
    static float top_k_accuracy(const float* predictions, const int* labels, 
                                int batch_size, int num_classes, int k) {
        int correct = 0;
        
        for (int i = 0; i < batch_size; i++) {
            std::vector<std::pair<float, int>> probs;
            
            for (int c = 0; c < num_classes; c++) {
                probs.push_back({predictions[i * num_classes + c], c});
            }
            
            std::partial_sort(probs.begin(), probs.begin() + k, probs.end(),
                            std::greater<std::pair<float, int>>());
            
            for (int j = 0; j < k; j++) {
                if (probs[j].second == labels[i]) {
                    correct++;
                    break;
                }
            }
        }
        
        return static_cast<float>(correct) / batch_size;
    }
};