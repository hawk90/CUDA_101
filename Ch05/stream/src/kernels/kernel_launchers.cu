// kernel_launchers.cu - Kernel launcher implementations

#include "../../include/image_pipeline.h"
#include <cuda_runtime.h>

// Sobel edge detection kernel
__global__ void sobel_kernel(unsigned char* data, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1) return;
    
    // Simple Sobel implementation for grayscale
    int idx = y * width + x;
    
    // Sobel X and Y kernels
    float gx = 0, gy = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int sample_idx = (y + dy) * width + (x + dx);
            float val = data[sample_idx];
            
            // Sobel X weights
            if (dx != 0) gx += val * dx * (2 - abs(dy));
            // Sobel Y weights  
            if (dy != 0) gy += val * dy * (2 - abs(dx));
        }
    }
    
    float magnitude = sqrtf(gx * gx + gy * gy);
    data[idx] = fminf(magnitude, 255.0f);
}

// Resize kernel with bilinear interpolation
__global__ void resize_kernel(unsigned char* src, unsigned char* dst,
                              int src_w, int src_h, int dst_w, int dst_h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= dst_w || y >= dst_h) return;
    
    float src_x = x * (float)(src_w - 1) / (dst_w - 1);
    float src_y = y * (float)(src_h - 1) / (dst_h - 1);
    
    int x0 = floorf(src_x);
    int y0 = floorf(src_y);
    int x1 = min(x0 + 1, src_w - 1);
    int y1 = min(y0 + 1, src_h - 1);
    
    float fx = src_x - x0;
    float fy = src_y - y0;
    
    // Bilinear interpolation
    float p00 = src[y0 * src_w + x0];
    float p01 = src[y0 * src_w + x1];
    float p10 = src[y1 * src_w + x0];
    float p11 = src[y1 * src_w + x1];
    
    float result = (1 - fx) * (1 - fy) * p00 + fx * (1 - fy) * p01 +
                   (1 - fx) * fy * p10 + fx * fy * p11;
    
    dst[y * dst_w + x] = result;
}

// Gaussian blur kernel (simplified)
__global__ void gaussian_blur_kernel(unsigned char* data, int width, int height, float sigma) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Simple 3x3 Gaussian kernel
    int kernel_size = 3;
    int half_size = kernel_size / 2;
    
    float sum = 0.0f;
    float weight_sum = 0.0f;
    
    for (int dy = -half_size; dy <= half_size; dy++) {
        for (int dx = -half_size; dx <= half_size; dx++) {
            int sample_x = min(max(x + dx, 0), width - 1);
            int sample_y = min(max(y + dy, 0), height - 1);
            
            float dist_sq = dx * dx + dy * dy;
            float weight = expf(-dist_sq / (2 * sigma * sigma));
            
            sum += data[sample_y * width + sample_x] * weight;
            weight_sum += weight;
        }
    }
    
    // Write to temporary buffer in real implementation
    data[y * width + x] = sum / weight_sum;
}

// Color conversion kernel (RGB to YUV example)
__global__ void color_conversion_kernel(unsigned char* data, int width, int height,
                                       ColorSpace from, ColorSpace to) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    
    if (idx >= total) return;
    
    // Simplified RGB to YUV conversion
    if (from == ColorSpace::RGB && to == ColorSpace::YUV) {
        // Assuming packed RGB format
        int pixel_idx = idx * 3;
        float r = data[pixel_idx];
        float g = data[pixel_idx + 1];
        float b = data[pixel_idx + 2];
        
        // ITU-R BT.709 conversion
        float y = 0.2126f * r + 0.7152f * g + 0.0722f * b;
        float u = -0.0999f * r - 0.3360f * g + 0.4360f * b + 128;
        float v = 0.6150f * r - 0.5586f * g - 0.0563f * b + 128;
        
        data[pixel_idx] = fminf(fmaxf(y, 0), 255);
        data[pixel_idx + 1] = fminf(fmaxf(u, 0), 255);
        data[pixel_idx + 2] = fminf(fmaxf(v, 0), 255);
    }
}

// Launcher functions
extern "C" {
    void launch_sobel(unsigned char* data, int width, int height, cudaStream_t stream) {
        dim3 block(16, 16);
        dim3 grid((width + 15) / 16, (height + 15) / 16);
        sobel_kernel<<<grid, block, 0, stream>>>(data, width, height);
    }
    
    void launch_resize(unsigned char* src, unsigned char* dst,
                      int src_w, int src_h, int dst_w, int dst_h, cudaStream_t stream) {
        dim3 block(16, 16);
        dim3 grid((dst_w + 15) / 16, (dst_h + 15) / 16);
        resize_kernel<<<grid, block, 0, stream>>>(src, dst, src_w, src_h, dst_w, dst_h);
    }
    
    void launch_gaussian_blur(unsigned char* data, int width, int height,
                             float sigma, cudaStream_t stream) {
        dim3 block(16, 16);
        dim3 grid((width + 15) / 16, (height + 15) / 16);
        gaussian_blur_kernel<<<grid, block, 0, stream>>>(data, width, height, sigma);
    }
    
    void launch_color_conversion(unsigned char* data, int width, int height,
                                ColorSpace from, ColorSpace to, cudaStream_t stream) {
        int total = width * height;
        int block_size = 256;
        int grid_size = (total + block_size - 1) / block_size;
        color_conversion_kernel<<<grid_size, block_size, 0, stream>>>(
            data, width, height, from, to);
    }
}