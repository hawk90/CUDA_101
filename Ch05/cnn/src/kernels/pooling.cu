// pooling.cu - Pooling operations for CNN

#include <cuda_runtime.h>
#include <float.h>

// Max pooling kernel
__global__ void max_pool2d(
    const float* input, float* output,
    int batch, int channels, int in_h, int in_w,
    int pool_h, int pool_w, int stride_h, int stride_w
) {
    int out_h = (in_h - pool_h) / stride_h + 1;
    int out_w = (in_w - pool_w) / stride_w + 1;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * out_h * out_w;
    if (idx >= total) return;
    
    // Calculate indices
    int n = idx / (channels * out_h * out_w);
    int c = (idx / (out_h * out_w)) % channels;
    int oh = (idx / out_w) % out_h;
    int ow = idx % out_w;
    
    float max_val = -FLT_MAX;
    
    // Find maximum in pooling window
    for (int ph = 0; ph < pool_h; ph++) {
        for (int pw = 0; pw < pool_w; pw++) {
            int ih = oh * stride_h + ph;
            int iw = ow * stride_w + pw;
            
            if (ih < in_h && iw < in_w) {
                int input_idx = ((n * channels + c) * in_h + ih) * in_w + iw;
                max_val = fmaxf(max_val, input[input_idx]);
            }
        }
    }
    
    output[idx] = max_val;
}

// Average pooling kernel
__global__ void avg_pool2d(
    const float* input, float* output,
    int batch, int channels, int in_h, int in_w,
    int pool_h, int pool_w, int stride_h, int stride_w
) {
    int out_h = (in_h - pool_h) / stride_h + 1;
    int out_w = (in_w - pool_w) / stride_w + 1;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * out_h * out_w;
    if (idx >= total) return;
    
    // Calculate indices
    int n = idx / (channels * out_h * out_w);
    int c = (idx / (out_h * out_w)) % channels;
    int oh = (idx / out_w) % out_h;
    int ow = idx % out_w;
    
    float sum = 0.0f;
    int count = 0;
    
    // Calculate average in pooling window
    for (int ph = 0; ph < pool_h; ph++) {
        for (int pw = 0; pw < pool_w; pw++) {
            int ih = oh * stride_h + ph;
            int iw = ow * stride_w + pw;
            
            if (ih < in_h && iw < in_w) {
                int input_idx = ((n * channels + c) * in_h + ih) * in_w + iw;
                sum += input[input_idx];
                count++;
            }
        }
    }
    
    output[idx] = count > 0 ? sum / count : 0.0f;
}

// Global average pooling
__global__ void global_avg_pool(
    const float* input, float* output,
    int batch, int channels, int height, int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels;
    if (idx >= total) return;
    
    int n = idx / channels;
    int c = idx % channels;
    
    float sum = 0.0f;
    int size = height * width;
    
    // Sum all values in the spatial dimensions
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            int input_idx = ((n * channels + c) * height + h) * width + w;
            sum += input[input_idx];
        }
    }
    
    output[idx] = sum / size;
}

// Adaptive pooling to target size
__global__ void adaptive_avg_pool2d(
    const float* input, float* output,
    int batch, int channels, int in_h, int in_w,
    int out_h, int out_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * out_h * out_w;
    if (idx >= total) return;
    
    // Calculate indices
    int n = idx / (channels * out_h * out_w);
    int c = (idx / (out_h * out_w)) % channels;
    int oh = (idx / out_w) % out_h;
    int ow = idx % out_w;
    
    // Calculate input region
    int ih_start = (oh * in_h) / out_h;
    int ih_end = ((oh + 1) * in_h) / out_h;
    int iw_start = (ow * in_w) / out_w;
    int iw_end = ((ow + 1) * in_w) / out_w;
    
    float sum = 0.0f;
    int count = (ih_end - ih_start) * (iw_end - iw_start);
    
    for (int ih = ih_start; ih < ih_end; ih++) {
        for (int iw = iw_start; iw < iw_end; iw++) {
            int input_idx = ((n * channels + c) * in_h + ih) * in_w + iw;
            sum += input[input_idx];
        }
    }
    
    output[idx] = sum / count;
}