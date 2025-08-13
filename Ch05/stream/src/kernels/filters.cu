#include <cuda_runtime.h>

#define FILTER_TILE_SIZE 16
#define FILTER_RADIUS 2

// Gaussian blur using shared memory
__global__ void gaussian_blur_kernel(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    const float* __restrict__ kernel,
    int width, int height, int channels,
    int kernel_size
) {
    extern __shared__ unsigned char sharedMem[];
    
    int radius = kernel_size / 2;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;
    
    // Load data into shared memory with halo
    int shared_width = blockDim.x + 2 * radius;
    int shared_idx = (ty + radius) * shared_width + (tx + radius);
    
    // Load center pixel
    if (x < width && y < height) {
        for (int c = 0; c < channels; c++) {
            sharedMem[shared_idx * channels + c] = 
                input[(y * width + x) * channels + c];
        }
    }
    
    // Load halo pixels
    if (tx < radius) {
        int halo_x = x - radius;
        if (halo_x >= 0 && y < height) {
            for (int c = 0; c < channels; c++) {
                sharedMem[(shared_idx - radius) * channels + c] = 
                    input[(y * width + halo_x) * channels + c];
            }
        }
    }
    
    __syncthreads();
    
    // Apply filter
    if (x < width && y < height) {
        for (int c = 0; c < channels; c++) {
            float sum = 0.0f;
            
            for (int ky = -radius; ky <= radius; ky++) {
                for (int kx = -radius; kx <= radius; kx++) {
                    int sy = ty + radius + ky;
                    int sx = tx + radius + kx;
                    
                    if (sy >= 0 && sy < blockDim.y + 2 * radius &&
                        sx >= 0 && sx < blockDim.x + 2 * radius) {
                        sum += sharedMem[(sy * shared_width + sx) * channels + c] * 
                               kernel[(ky + radius) * kernel_size + (kx + radius)];
                    }
                }
            }
            
            output[(y * width + x) * channels + c] = (unsigned char)sum;
        }
    }
}

// Sobel edge detection
__global__ void sobel_edge_kernel(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int width, int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x <= 0 || x >= width - 1 || y <= 0 || y >= height - 1) {
        if (x < width && y < height) {
            output[y * width + x] = 0;
        }
        return;
    }
    
    // Sobel X kernel: [-1, 0, 1; -2, 0, 2; -1, 0, 1]
    // Sobel Y kernel: [-1, -2, -1; 0, 0, 0; 1, 2, 1]
    
    float gx = 0.0f, gy = 0.0f;
    
    // Apply Sobel operators
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int idx = (y + dy) * width + (x + dx);
            float pixel = input[idx];
            
            // Sobel X
            if (dx != 0) {
                gx += pixel * dx * (2 - abs(dy));
            }
            
            // Sobel Y
            if (dy != 0) {
                gy += pixel * dy * (2 - abs(dx));
            }
        }
    }
    
    // Compute gradient magnitude
    float magnitude = sqrtf(gx * gx + gy * gy);
    magnitude = fminf(magnitude, 255.0f);
    
    output[y * width + x] = (unsigned char)magnitude;
}

// Bilateral filter for edge-preserving smoothing
__global__ void bilateral_filter_kernel(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    int width, int height, int channels,
    float sigma_spatial, float sigma_range
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int radius = (int)(sigma_spatial * 2);
    int idx = (y * width + x) * channels;
    
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        float weight_sum = 0.0f;
        float center_val = input[idx + c];
        
        for (int dy = -radius; dy <= radius; dy++) {
            for (int dx = -radius; dx <= radius; dx++) {
                int nx = x + dx;
                int ny = y + dy;
                
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    int nidx = (ny * width + nx) * channels + c;
                    float neighbor_val = input[nidx];
                    
                    // Spatial weight
                    float spatial_dist = sqrtf(dx * dx + dy * dy);
                    float spatial_weight = expf(-spatial_dist * spatial_dist / 
                                               (2.0f * sigma_spatial * sigma_spatial));
                    
                    // Range weight
                    float range_dist = fabsf(neighbor_val - center_val);
                    float range_weight = expf(-range_dist * range_dist / 
                                             (2.0f * sigma_range * sigma_range));
                    
                    float weight = spatial_weight * range_weight;
                    sum += neighbor_val * weight;
                    weight_sum += weight;
                }
            }
        }
        
        output[idx + c] = (unsigned char)(sum / weight_sum);
    }
}

// Wrapper functions
extern "C" {

void launchGaussianBlur(
    const unsigned char* input,
    unsigned char* output,
    const float* kernel,
    int width, int height, int channels,
    int kernel_size,
    cudaStream_t stream = 0
) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    
    int shared_size = (block.x + kernel_size - 1) * 
                     (block.y + kernel_size - 1) * 
                     channels * sizeof(unsigned char);
    
    gaussian_blur_kernel<<<grid, block, shared_size, stream>>>(
        input, output, kernel, width, height, channels, kernel_size
    );
}

void launchSobelEdge(
    const unsigned char* input,
    unsigned char* output,
    int width, int height,
    cudaStream_t stream = 0
) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    
    sobel_edge_kernel<<<grid, block, 0, stream>>>(
        input, output, width, height
    );
}

void launchBilateralFilter(
    const unsigned char* input,
    unsigned char* output,
    int width, int height, int channels,
    float sigma_spatial, float sigma_range,
    cudaStream_t stream = 0
) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    
    bilateral_filter_kernel<<<grid, block, 0, stream>>>(
        input, output, width, height, channels,
        sigma_spatial, sigma_range
    );
}

} // extern "C"