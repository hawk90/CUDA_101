#include <stdio.h>
#include <cuda_runtime.h>
#include "include/kernels/activation.h"
#include "include/kernels/convolution.h"


int main() {
    
    int inputSize = 1024;
    size_t bytes = size * sizeof(float);

    float *h_input;
    float *h_output;
    float *h_weight;
    float *h_bias;
    float *d_input;
    float *d_output;
    float *d_weight;
    float *d_bias;

    h_input = (float*)malloc(bytes);
    h_output = (float*)malloc(bytes);
    h_weight = (float*)malloc(bytes);
    h_bias = (float*)malloc(bytes);

    cudaMalloc((void**)&d_input, bytes);
    cudaMalloc((void**)&d_output, bytes);
    cudaMalloc((void**)&d_weight, bytes);
    cudaMalloc((void**)&d_bias, bytes);


    // Initialize input, weight, and bias
    for (int i = 0; i < inputSize; i++) {
        h_input[i] = (float)i;
        h_weight[i] = (float)(i + 1);
        h_bias[i] = 0.1f; // Example bias
    }
    
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, bytes, cudaMemcpyHostToDevice);

    // Define grid and block sizes
    int threadsPerBlock = 256;
    int blocksPerGrid = (inputSize + threadsPerBlock - 1) / threadsPerBlock;

    // Launch convolution kernel
    conv2d_forward<<<blocksPerGrid, threadsPerBlock>>>(
        d_input, d_weight, d_bias, d_output,
        1, 1, 1, 3, 3, 3, 3, 1, 1, 0, 0
    )

    // conv2d_tiled<<<blocksPerGrid, threadsPerBlock, 2 * threadsPerBlock * sizeof(float)>>>(
    //     d_input, d_weight, d_bias, d_output,
    //     1, 1, 1, 3, 3, 3, 3, 1, 1, 0
    // );

    cudaDeviceSynchronize();

    relu_forward<<<blocksPerGrid, threadsPerBlock>>>(
        d_output, d_output, inputSize
    );

    cudaDeviceSynchronize();

    memcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    free(h_input);
    free(h_output);
    free(h_weight);
    free(h_bias);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_weight);
    cudaFree(d_bias);

    
    return 0;
}