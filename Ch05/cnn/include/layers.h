#ifndef LAYERS_H
#define LAYERS_H

#include "tensor.h"
#include <memory>

// Base Layer interface
class Layer {
public:
    virtual ~Layer() = default;
    virtual TensorFloat forward(const TensorFloat& input) = 0;
    virtual TensorFloat backward(const TensorFloat& grad_output) = 0;
};

// Convolution Layer
class Conv2D : public Layer {
private:
    TensorFloat weights_;
    TensorFloat bias_;
    int in_channels_, out_channels_;
    int kernel_size_, stride_, padding_;
    
public:
    Conv2D(int in_ch, int out_ch, int k_size, int stride = 1, int pad = 0);
    TensorFloat forward(const TensorFloat& input) override;
    TensorFloat backward(const TensorFloat& grad_output) override;
};

// Batch Normalization
class BatchNorm : public Layer {
private:
    TensorFloat gamma_, beta_;
    TensorFloat running_mean_, running_var_;
    int num_features_;
    float momentum_, epsilon_;
    bool training_;
    
public:
    BatchNorm(int num_features, float momentum = 0.1f, float eps = 1e-5f);
    TensorFloat forward(const TensorFloat& input) override;
    TensorFloat backward(const TensorFloat& grad_output) override;
    void eval() { training_ = false; }
    void train() { training_ = true; }
};

// ReLU Activation
class ReLU : public Layer {
private:
    TensorFloat cached_input_;
    
public:
    TensorFloat forward(const TensorFloat& input) override;
    TensorFloat backward(const TensorFloat& grad_output) override;
};

// Max Pooling
class MaxPool2D : public Layer {
private:
    int pool_size_, stride_;
    TensorFloat max_indices_;
    
public:
    MaxPool2D(int pool_size, int stride);
    TensorFloat forward(const TensorFloat& input) override;
    TensorFloat backward(const TensorFloat& grad_output) override;
};

// Fully Connected Layer
class Linear : public Layer {
private:
    TensorFloat weights_, bias_;
    TensorFloat cached_input_;
    int in_features_, out_features_;
    
public:
    Linear(int in_features, int out_features);
    TensorFloat forward(const TensorFloat& input) override;
    TensorFloat backward(const TensorFloat& grad_output) override;
};

// Softmax Layer
class Softmax : public Layer {
private:
    TensorFloat cached_output_;
    
public:
    TensorFloat forward(const TensorFloat& input) override;
    TensorFloat backward(const TensorFloat& grad_output) override;
};

// Dropout Layer
class Dropout : public Layer {
private:
    float dropout_rate_;
    TensorFloat mask_;
    bool training_;
    
public:
    Dropout(float rate = 0.5f);
    TensorFloat forward(const TensorFloat& input) override;
    TensorFloat backward(const TensorFloat& grad_output) override;
    void eval() { training_ = false; }
    void train() { training_ = true; }
};

#endif // LAYERS_H