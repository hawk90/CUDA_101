// activation.cu - Activation functions for CNN

#include <cuda_runtime.h>
#include <math.h>

// ReLU activation
__global__ void relu_forward(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    output[idx] = fmaxf(0.0f, input[idx]);
}

// ReLU backward
__global__ void relu_backward(
    const float* grad_output, const float* input,
    float* grad_input, int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    grad_input[idx] = input[idx] > 0 ? grad_output[idx] : 0.0f;
}

// Leaky ReLU
__global__ void leaky_relu_forward(
    const float* input, float* output, float alpha, int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float x = input[idx];
    output[idx] = x > 0 ? x : alpha * x;
}

// ELU activation
__global__ void elu_forward(
    const float* input, float* output, float alpha, int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float x = input[idx];
    output[idx] = x > 0 ? x : alpha * (expf(x) - 1.0f);
}

// Sigmoid activation
__global__ void sigmoid_forward(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    output[idx] = 1.0f / (1.0f + expf(-input[idx]));
}

// Tanh activation
__global__ void tanh_forward(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    output[idx] = tanhf(input[idx]);
}

// Softmax activation
__global__ void softmax_forward(
    const float* input, float* output,
    int batch, int classes
) {
    int idx = blockIdx.x;
    if (idx >= batch) return;
    
    // Collaborative computation within block
    extern __shared__ float shared[];
    float* max_val = shared;
    float* sum_exp = &shared[1];
    
    // Find maximum for numerical stability
    if (threadIdx.x == 0) {
        max_val[0] = -FLT_MAX;
        for (int i = 0; i < classes; i++) {
            max_val[0] = fmaxf(max_val[0], input[idx * classes + i]);
        }
        sum_exp[0] = 0.0f;
    }
    __syncthreads();
    
    // Calculate exp and sum
    for (int i = threadIdx.x; i < classes; i += blockDim.x) {
        float exp_val = expf(input[idx * classes + i] - max_val[0]);
        output[idx * classes + i] = exp_val;
        atomicAdd(sum_exp, exp_val);
    }
    __syncthreads();
    
    // Normalize
    for (int i = threadIdx.x; i < classes; i += blockDim.x) {
        output[idx * classes + i] /= sum_exp[0];
    }
}

// GELU activation
__global__ void gelu_forward(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float x = input[idx];
    // Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float c1 = 0.7978845608f; // sqrt(2/pi)
    const float c2 = 0.044715f;
    
    float x3 = x * x * x;
    float tanh_arg = c1 * (x + c2 * x3);
    output[idx] = 0.5f * x * (1.0f + tanhf(tanh_arg));
}

// Swish/SiLU activation
__global__ void swish_forward(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float x = input[idx];
    output[idx] = x / (1.0f + expf(-x));
}

// Mish activation
__global__ void mish_forward(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float x = input[idx];
    float softplus = logf(1.0f + expf(x));
    output[idx] = x * tanhf(softplus);
}