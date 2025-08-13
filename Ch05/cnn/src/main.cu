#include <iostream>
#include <chrono>
#include "../include/network.h"
#include "../include/tensor.h"

// Helper function to generate random data
void generate_random_data(TensorFloat& tensor) {
    std::vector<float> data(tensor.size());
    for (size_t i = 0; i < data.size(); i++) {
        data[i] = (float)rand() / RAND_MAX;
    }
    tensor.from_host(data);
}

// Training function
void train_epoch(SimpleCNN& model, 
                 const std::vector<TensorFloat>& train_data,
                 const std::vector<TensorFloat>& train_labels,
                 CrossEntropyLoss& criterion,
                 Adam& optimizer) {
    model.train();
    float total_loss = 0.0f;
    
    for (size_t i = 0; i < train_data.size(); i++) {
        // Forward pass
        TensorFloat output = model.forward(train_data[i]);
        float loss = criterion.forward(output, train_labels[i]);
        total_loss += loss;
        
        // Backward pass
        TensorFloat grad = criterion.backward();
        model.backward(grad);
        
        // Update weights
        // optimizer.step(model.parameters(), model.gradients());
    }
    
    printf("Average Loss: %.4f\n", total_loss / train_data.size());
}

// Evaluation function
float evaluate(SimpleCNN& model,
               const std::vector<TensorFloat>& test_data,
               const std::vector<TensorFloat>& test_labels) {
    model.eval();
    int correct = 0;
    int total = 0;
    
    for (size_t i = 0; i < test_data.size(); i++) {
        TensorFloat output = model.forward(test_data[i]);
        // Get predictions and calculate accuracy
        // ...
    }
    
    return (float)correct / total;
}

int main() {
    // Print GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Total Memory: %.2f GB\n\n", 
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    
    // Create model
    SimpleCNN model;
    
    // Build network architecture
    model.add(std::make_unique<Conv2D>(3, 32, 3, 1, 1));
    model.add(std::make_unique<BatchNorm>(32));
    model.add(std::make_unique<ReLU>());
    model.add(std::make_unique<MaxPool2D>(2, 2));
    
    model.add(std::make_unique<Conv2D>(32, 64, 3, 1, 1));
    model.add(std::make_unique<BatchNorm>(64));
    model.add(std::make_unique<ReLU>());
    model.add(std::make_unique<MaxPool2D>(2, 2));
    
    model.add(std::make_unique<Conv2D>(64, 128, 3, 1, 1));
    model.add(std::make_unique<BatchNorm>(128));
    model.add(std::make_unique<ReLU>());
    model.add(std::make_unique<MaxPool2D>(2, 2));
    
    // Add fully connected layers
    model.add(std::make_unique<Linear>(128 * 28 * 28, 256));
    model.add(std::make_unique<ReLU>());
    model.add(std::make_unique<Dropout>(0.5f));
    model.add(std::make_unique<Linear>(256, 10));
    model.add(std::make_unique<Softmax>());
    
    printf("Model created with %zu parameters\n", model.num_parameters());
    
    // Create dummy data for testing
    int batch_size = 32;
    int image_size = 224;
    TensorFloat input(batch_size, 3, image_size, image_size);
    generate_random_data(input);
    
    // Benchmark forward pass
    int num_iterations = 100;
    model.eval();
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        model.forward(input);
    }
    cudaDeviceSynchronize();
    
    // Timing
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; i++) {
        TensorFloat output = model.forward(input);
    }
    cudaDeviceSynchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    float avg_time = duration.count() / (float)num_iterations;
    printf("\nForward Pass Performance:\n");
    printf("Average time per batch: %.3f ms\n", avg_time);
    printf("Throughput: %.2f images/sec\n", (batch_size * 1000.0f) / avg_time);
    
    // Calculate FLOPS
    long long flops = 0;
    // Approximate FLOPs calculation for conv layers
    // Conv1: 3*32*3*3*222*222 * 2
    flops += 3LL * 32 * 9 * 222 * 222 * 2;
    // Conv2: 32*64*3*3*110*110 * 2
    flops += 32LL * 64 * 9 * 110 * 110 * 2;
    // Conv3: 64*128*3*3*54*54 * 2
    flops += 64LL * 128 * 9 * 54 * 54 * 2;
    
    float gflops = (flops * batch_size) / (avg_time * 1e6);
    printf("Performance: %.2f GFLOPS\n", gflops);
    
    return 0;
}