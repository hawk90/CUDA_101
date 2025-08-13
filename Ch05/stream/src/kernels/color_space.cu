#include <cuda_runtime.h>

// RGB to YUV conversion (ITU-R BT.709)
__global__ void rgb_to_yuv_kernel(
    const unsigned char* __restrict__ rgb,
    unsigned char* __restrict__ yuv,
    int width, int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = (y * width + x) * 3;
    
    float r = rgb[idx + 0] / 255.0f;
    float g = rgb[idx + 1] / 255.0f;
    float b = rgb[idx + 2] / 255.0f;
    
    // ITU-R BT.709 conversion
    float y_val = 0.2126f * r + 0.7152f * g + 0.0722f * b;
    float u_val = -0.09991f * r - 0.33609f * g + 0.436f * b + 0.5f;
    float v_val = 0.615f * r - 0.55861f * g - 0.05639f * b + 0.5f;
    
    yuv[idx + 0] = (unsigned char)(y_val * 255.0f);
    yuv[idx + 1] = (unsigned char)(u_val * 255.0f);
    yuv[idx + 2] = (unsigned char)(v_val * 255.0f);
}

// YUV to RGB conversion
__global__ void yuv_to_rgb_kernel(
    const unsigned char* __restrict__ yuv,
    unsigned char* __restrict__ rgb,
    int width, int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = (y * width + x) * 3;
    
    float y_val = yuv[idx + 0] / 255.0f;
    float u_val = yuv[idx + 1] / 255.0f - 0.5f;
    float v_val = yuv[idx + 2] / 255.0f - 0.5f;
    
    float r = y_val + 1.28033f * v_val;
    float g = y_val - 0.21482f * u_val - 0.38059f * v_val;
    float b = y_val + 2.12798f * u_val;
    
    // Clamp values
    r = fminf(fmaxf(r, 0.0f), 1.0f);
    g = fminf(fmaxf(g, 0.0f), 1.0f);
    b = fminf(fmaxf(b, 0.0f), 1.0f);
    
    rgb[idx + 0] = (unsigned char)(r * 255.0f);
    rgb[idx + 1] = (unsigned char)(g * 255.0f);
    rgb[idx + 2] = (unsigned char)(b * 255.0f);
}

// RGB to Grayscale
__global__ void rgb_to_gray_kernel(
    const unsigned char* __restrict__ rgb,
    unsigned char* __restrict__ gray,
    int width, int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int rgb_idx = (y * width + x) * 3;
    int gray_idx = y * width + x;
    
    // Luminance formula
    float luminance = 0.299f * rgb[rgb_idx + 0] + 
                     0.587f * rgb[rgb_idx + 1] + 
                     0.114f * rgb[rgb_idx + 2];
    
    gray[gray_idx] = (unsigned char)luminance;
}

// RGB to HSV conversion
__global__ void rgb_to_hsv_kernel(
    const unsigned char* __restrict__ rgb,
    unsigned char* __restrict__ hsv,
    int width, int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = (y * width + x) * 3;
    
    float r = rgb[idx + 0] / 255.0f;
    float g = rgb[idx + 1] / 255.0f;
    float b = rgb[idx + 2] / 255.0f;
    
    float max_val = fmaxf(r, fmaxf(g, b));
    float min_val = fminf(r, fminf(g, b));
    float delta = max_val - min_val;
    
    // Hue calculation
    float h;
    if (delta == 0) {
        h = 0;
    } else if (max_val == r) {
        h = 60.0f * fmodf((g - b) / delta, 6.0f);
    } else if (max_val == g) {
        h = 60.0f * ((b - r) / delta + 2.0f);
    } else {
        h = 60.0f * ((r - g) / delta + 4.0f);
    }
    
    if (h < 0) h += 360.0f;
    
    // Saturation
    float s = (max_val == 0) ? 0 : (delta / max_val);
    
    // Value
    float v = max_val;
    
    hsv[idx + 0] = (unsigned char)(h / 360.0f * 255.0f);
    hsv[idx + 1] = (unsigned char)(s * 255.0f);
    hsv[idx + 2] = (unsigned char)(v * 255.0f);
}

// Wrapper functions
extern "C" {

void launchRGBtoYUV(
    const unsigned char* rgb,
    unsigned char* yuv,
    int width, int height,
    cudaStream_t stream = 0
) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    
    rgb_to_yuv_kernel<<<grid, block, 0, stream>>>(
        rgb, yuv, width, height
    );
}

void launchYUVtoRGB(
    const unsigned char* yuv,
    unsigned char* rgb,
    int width, int height,
    cudaStream_t stream = 0
) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    
    yuv_to_rgb_kernel<<<grid, block, 0, stream>>>(
        yuv, rgb, width, height
    );
}

void launchRGBtoGray(
    const unsigned char* rgb,
    unsigned char* gray,
    int width, int height,
    cudaStream_t stream = 0
) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    
    rgb_to_gray_kernel<<<grid, block, 0, stream>>>(
        rgb, gray, width, height
    );
}

void launchRGBtoHSV(
    const unsigned char* rgb,
    unsigned char* hsv,
    int width, int height,
    cudaStream_t stream = 0
) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    
    rgb_to_hsv_kernel<<<grid, block, 0, stream>>>(
        rgb, hsv, width, height
    );
}

} // extern "C"