#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1024*1024
#define BLOCK_SIZE 256
#define NUM_STREAMS 4

// 간단한 벡터 덧셈 커널
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// 스트림을 사용하지 않는 버전
void withoutStreams(float *h_a, float *h_b, float *h_c, int size) {
    float *d_a, *d_b, *d_c;
    
    // GPU 메모리 할당
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    // 타이밍 이벤트
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    // 호스트에서 디바이스로 데이터 복사
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // 커널 실행
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vectorAdd<<<blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
    
    // 결과를 호스트로 복사
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Without streams: %.2f ms\n", milliseconds);
    
    // 정리
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// 스트림을 사용하는 버전
void withStreams(float *h_a, float *h_b, float *h_c, int size) {
    float *d_a, *d_b, *d_c;
    
    // GPU 메모리 할당
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    // 스트림 생성
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    // 타이밍 이벤트
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    // 각 스트림에서 처리할 데이터 크기
    int streamSize = N / NUM_STREAMS;
    int streamBytes = streamSize * sizeof(float);
    
    // 각 스트림에서 비동기적으로 작업 수행
    for (int i = 0; i < NUM_STREAMS; i++) {
        int offset = i * streamSize;
        
        // 비동기 메모리 복사 (H2D)
        cudaMemcpyAsync(&d_a[offset], &h_a[offset], streamBytes, 
                       cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(&d_b[offset], &h_b[offset], streamBytes, 
                       cudaMemcpyHostToDevice, streams[i]);
        
        // 커널 실행
        int blocks = (streamSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
        vectorAdd<<<blocks, BLOCK_SIZE, 0, streams[i]>>>
                 (&d_a[offset], &d_b[offset], &d_c[offset], streamSize);
        
        // 비동기 메모리 복사 (D2H)
        cudaMemcpyAsync(&h_c[offset], &d_c[offset], streamBytes, 
                       cudaMemcpyDeviceToHost, streams[i]);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("With %d streams: %.2f ms\n", NUM_STREAMS, milliseconds);
    
    // 정리
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Pinned Memory를 사용한 스트림 버전 (더 효율적)
void withPinnedMemoryStreams(float *h_a, float *h_b, float *h_c, int size) {
    float *d_a, *d_b, *d_c;
    float *pinned_a, *pinned_b, *pinned_c;
    
    // Pinned Memory 할당
    cudaHostAlloc(&pinned_a, size, cudaHostAllocDefault);
    cudaHostAlloc(&pinned_b, size, cudaHostAllocDefault);
    cudaHostAlloc(&pinned_c, size, cudaHostAllocDefault);
    
    // 데이터를 Pinned Memory로 복사
    memcpy(pinned_a, h_a, size);
    memcpy(pinned_b, h_b, size);
    
    // GPU 메모리 할당
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    // 스트림 생성
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    // 타이밍 이벤트
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    int streamSize = N / NUM_STREAMS;
    int streamBytes = streamSize * sizeof(float);
    
    for (int i = 0; i < NUM_STREAMS; i++) {
        int offset = i * streamSize;
        
        cudaMemcpyAsync(&d_a[offset], &pinned_a[offset], streamBytes, 
                       cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(&d_b[offset], &pinned_b[offset], streamBytes, 
                       cudaMemcpyHostToDevice, streams[i]);
        
        int blocks = (streamSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
        vectorAdd<<<blocks, BLOCK_SIZE, 0, streams[i]>>>
                 (&d_a[offset], &d_b[offset], &d_c[offset], streamSize);
        
        cudaMemcpyAsync(&pinned_c[offset], &d_c[offset], streamBytes, 
                       cudaMemcpyDeviceToHost, streams[i]);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // 결과를 일반 메모리로 복사
    memcpy(h_c, pinned_c, size);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("With pinned memory and %d streams: %.2f ms\n", NUM_STREAMS, milliseconds);
    
    // 정리
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }
    cudaFreeHost(pinned_a);
    cudaFreeHost(pinned_b);
    cudaFreeHost(pinned_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    int size = N * sizeof(float);
    
    // 호스트 메모리 할당
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);
    
    // 데이터 초기화
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)rand() / RAND_MAX;
        h_b[i] = (float)rand() / RAND_MAX;
    }
    
    printf("Vector size: %d elements\n", N);
    printf("Data size: %.2f MB\n\n", (float)size / (1024*1024));
    
    // 1. 스트림 없이 실행
    withoutStreams(h_a, h_b, h_c, size);
    
    // 2. 스트림 사용
    withStreams(h_a, h_b, h_c, size);
    
    // 3. Pinned Memory + 스트림 사용
    withPinnedMemoryStreams(h_a, h_b, h_c, size);
    
    printf("\n성능 향상 팁:\n");
    printf("- Pinned Memory 사용으로 메모리 전송 속도 향상\n");
    printf("- 여러 스트림으로 작업을 분할하여 동시 실행\n");
    printf("- 메모리 전송과 커널 실행을 오버랩\n");
    
    // 메모리 해제
    free(h_a);
    free(h_b);
    free(h_c);
    
    return 0;
}