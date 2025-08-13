// normalization.cu - Normalization layers for CNN

#include <cuda_runtime.h>
#include <math.h>

// Batch Normalization forward
__global__ void batch_norm_forward(
    const float* input, float* output,
    const float* gamma, const float* beta,
    const float* running_mean, const float* running_var,
    int batch, int channels, int height, int width,
    float eps, bool training
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int spatial_size = height * width;
    int total = batch * channels * spatial_size;
    if (idx >= total) return;
    
    // Calculate indices
    int c = (idx / spatial_size) % channels;
    
    float mean = running_mean[c];
    float var = running_var[c];
    
    // Normalize
    float x_norm = (input[idx] - mean) / sqrtf(var + eps);
    
    // Scale and shift
    output[idx] = gamma[c] * x_norm + beta[c];
}

// Layer Normalization
__global__ void layer_norm_forward(
    const float* input, float* output,
    const float* gamma, const float* beta,
    int batch, int features, float eps
) {
    int idx = blockIdx.x;
    if (idx >= batch) return;
    
    extern __shared__ float shared[];
    float* mean = shared;
    float* var = &shared[1];
    
    // Calculate mean
    if (threadIdx.x == 0) {
        mean[0] = 0.0f;
        for (int i = 0; i < features; i++) {
            mean[0] += input[idx * features + i];
        }
        mean[0] /= features;
        
        // Calculate variance
        var[0] = 0.0f;
        for (int i = 0; i < features; i++) {
            float diff = input[idx * features + i] - mean[0];
            var[0] += diff * diff;
        }
        var[0] /= features;
    }
    __syncthreads();
    
    // Normalize and apply affine transform
    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        float x_norm = (input[idx * features + i] - mean[0]) / sqrtf(var[0] + eps);
        output[idx * features + i] = gamma[i] * x_norm + beta[i];
    }
}

// Instance Normalization
__global__ void instance_norm_forward(
    const float* input, float* output,
    const float* gamma, const float* beta,
    int batch, int channels, int height, int width, float eps
) {
    // Each instance (batch, channel) pair is normalized independently
    int spatial_size = height * width;
    int idx = blockIdx.x;  // batch * channels
    if (idx >= batch * channels) return;
    
    int n = idx / channels;
    int c = idx % channels;
    
    // Calculate mean and variance for this instance
    float mean = 0.0f;
    float var = 0.0f;
    
    // First pass: mean
    for (int i = 0; i < spatial_size; i++) {
        int input_idx = (n * channels + c) * spatial_size + i;
        mean += input[input_idx];
    }
    mean /= spatial_size;
    
    // Second pass: variance
    for (int i = 0; i < spatial_size; i++) {
        int input_idx = (n * channels + c) * spatial_size + i;
        float diff = input[input_idx] - mean;
        var += diff * diff;
    }
    var /= spatial_size;
    
    // Normalize
    for (int i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        int input_idx = (n * channels + c) * spatial_size + i;
        float x_norm = (input[input_idx] - mean) / sqrtf(var + eps);
        output[input_idx] = gamma[c] * x_norm + beta[c];
    }
}

// Group Normalization
__global__ void group_norm_forward(
    const float* input, float* output,
    const float* gamma, const float* beta,
    int batch, int channels, int height, int width,
    int num_groups, float eps
) {
    int spatial_size = height * width;
    int channels_per_group = channels / num_groups;
    
    // Each block handles one group in one batch
    int idx = blockIdx.x;
    int n = idx / num_groups;
    int g = idx % num_groups;
    
    if (n >= batch) return;
    
    // Calculate mean and variance for this group
    float mean = 0.0f;
    float var = 0.0f;
    int group_size = channels_per_group * spatial_size;
    
    // First pass: mean
    for (int c = 0; c < channels_per_group; c++) {
        int channel_idx = g * channels_per_group + c;
        for (int i = 0; i < spatial_size; i++) {
            int input_idx = ((n * channels + channel_idx) * height * width) + i;
            mean += input[input_idx];
        }
    }
    mean /= group_size;
    
    // Second pass: variance
    for (int c = 0; c < channels_per_group; c++) {
        int channel_idx = g * channels_per_group + c;
        for (int i = 0; i < spatial_size; i++) {
            int input_idx = ((n * channels + channel_idx) * height * width) + i;
            float diff = input[input_idx] - mean;
            var += diff * diff;
        }
    }
    var /= group_size;
    
    // Normalize
    for (int c = 0; c < channels_per_group; c++) {
        int channel_idx = g * channels_per_group + c;
        for (int i = threadIdx.x; i < spatial_size; i += blockDim.x) {
            int input_idx = ((n * channels + channel_idx) * height * width) + i;
            float x_norm = (input[input_idx] - mean) / sqrtf(var + eps);
            output[input_idx] = gamma[channel_idx] * x_norm + beta[channel_idx];
        }
    }
}