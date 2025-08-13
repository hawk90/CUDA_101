#ifndef NETWORK_H
#define NETWORK_H

#include "layers.h"
#include <vector>
#include <memory>
#include <string>

class SimpleCNN {
private:
    std::vector<std::unique_ptr<Layer>> layers_;
    bool training_;
    
public:
    SimpleCNN();
    
    // Add layers
    void add(std::unique_ptr<Layer> layer) {
        layers_.push_back(std::move(layer));
    }
    
    // Forward pass
    TensorFloat forward(const TensorFloat& input);
    
    // Backward pass
    TensorFloat backward(const TensorFloat& grad_output);
    
    // Training mode
    void train() {
        training_ = true;
        for (auto& layer : layers_) {
            if (auto* bn = dynamic_cast<BatchNorm*>(layer.get())) {
                bn->train();
            }
            if (auto* dropout = dynamic_cast<Dropout*>(layer.get())) {
                dropout->train();
            }
        }
    }
    
    // Evaluation mode
    void eval() {
        training_ = false;
        for (auto& layer : layers_) {
            if (auto* bn = dynamic_cast<BatchNorm*>(layer.get())) {
                bn->eval();
            }
            if (auto* dropout = dynamic_cast<Dropout*>(layer.get())) {
                dropout->eval();
            }
        }
    }
    
    // Save/Load model
    void save(const std::string& filepath);
    void load(const std::string& filepath);
    
    // Get number of parameters
    size_t num_parameters() const;
};

// Pre-defined architectures
class ResNet18 : public SimpleCNN {
public:
    ResNet18(int num_classes = 10);
};

class VGG16 : public SimpleCNN {
public:
    VGG16(int num_classes = 10);
};

// Loss functions
class CrossEntropyLoss {
private:
    TensorFloat cached_softmax_;
    
public:
    float forward(const TensorFloat& predictions, const TensorFloat& targets);
    TensorFloat backward();
};

// Optimizer
class SGD {
private:
    float learning_rate_;
    float momentum_;
    std::vector<TensorFloat> velocity_;
    
public:
    SGD(float lr = 0.01f, float momentum = 0.9f) 
        : learning_rate_(lr), momentum_(momentum) {}
    
    void step(std::vector<TensorFloat>& parameters, 
              const std::vector<TensorFloat>& gradients);
};

class Adam {
private:
    float learning_rate_;
    float beta1_, beta2_;
    float epsilon_;
    int t_;
    std::vector<TensorFloat> m_, v_;
    
public:
    Adam(float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, 
         float eps = 1e-8f)
        : learning_rate_(lr), beta1_(beta1), beta2_(beta2), 
          epsilon_(eps), t_(0) {}
    
    void step(std::vector<TensorFloat>& parameters, 
              const std::vector<TensorFloat>& gradients);
};

#endif // NETWORK_H