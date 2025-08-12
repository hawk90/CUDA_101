/*
 * host_device_roles.cu
 * Host와 Device 역할 분담 예제
 * GTX 1070 (CC 6.1) / Jetson Orin Nano (CC 8.7) 호환
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// Device 역할: 병렬 계산 수행
__global__ void parallelComputation(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 간단한 병렬 계산: 각 요소 제곱
        data[idx] = data[idx] * data[idx];
    }
}

// Host 역할: 복잡한 로직과 분기
void hostLogic(float *data, int n) {
    // 1. 데이터 검증 (Host가 더 적합)
    for (int i = 0; i < 10 && i < n; i++) {
        if (data[i] < 0) {
            printf("Warning: Negative value at index %d\n", i);
        }
    }
    
    // 2. 통계 계산 (순차적 작업)
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    printf("Average: %.2f\n", sum / n);
}

int main() {
    printf("=== Host-Device 역할 분담 예제 ===\n\n");
    
    // 설정
    const int N = 1000000;  // 100만 개 데이터
    size_t size = N * sizeof(float);
    
    // Host 역할 1: 메모리 관리
    printf("[Host] 메모리 할당 중...\n");
    float *h_data = (float*)malloc(size);
    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, size));
    
    // Host 역할 2: 데이터 초기화 (I/O 작업)
    printf("[Host] 데이터 초기화 중...\n");
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)(rand() % 100) / 10.0f;
    }
    
    // Host 역할 3: 데이터 전송 관리
    printf("[Host] GPU로 데이터 전송 중...\n");
    auto start = std::chrono::high_resolution_clock::now();
    CHECK_CUDA(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    printf("      전송 시간: %.3f ms\n", duration.count() / 1000.0f);
    
    // Device 역할: 대규모 병렬 연산
    printf("\n[Device] GPU에서 병렬 계산 수행 중...\n");
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    // 커널 실행 시간 측정
    cudaEvent_t kernel_start, kernel_stop;
    CHECK_CUDA(cudaEventCreate(&kernel_start));
    CHECK_CUDA(cudaEventCreate(&kernel_stop));
    
    CHECK_CUDA(cudaEventRecord(kernel_start));
    parallelComputation<<<gridSize, blockSize>>>(d_data, N);
    CHECK_CUDA(cudaEventRecord(kernel_stop));
    
    CHECK_CUDA(cudaEventSynchronize(kernel_stop));
    float kernel_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop));
    printf("        계산 시간: %.3f ms\n", kernel_ms);
    printf("        사용된 블록: %d, 블록당 스레드: %d\n", gridSize, blockSize);
    
    // Host 역할 4: 결과 수신
    printf("\n[Host] GPU에서 결과 수신 중...\n");
    CHECK_CUDA(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));
    
    // Host 역할 5: 복잡한 로직 처리
    printf("[Host] 결과 분석 중...\n");
    hostLogic(h_data, N);
    
    // 역할 분담 요약
    printf("\n=== 역할 분담 요약 ===\n");
    printf("Host 담당:\n");
    printf("  - 메모리 관리 (malloc, cudaMalloc)\n");
    printf("  - 데이터 I/O\n");
    printf("  - 커널 실행 제어\n");
    printf("  - 복잡한 로직 및 분기\n");
    printf("  - 에러 처리\n");
    
    printf("\nDevice 담당:\n");
    printf("  - 대규모 병렬 연산\n");
    printf("  - SIMT 실행에 적합한 작업\n");
    printf("  - 데이터 병렬 처리\n");
    
    // 성능 비교 (CPU vs GPU)
    printf("\n=== 성능 비교 ===\n");
    
    // CPU 버전
    float *h_data_cpu = (float*)malloc(size);
    memcpy(h_data_cpu, h_data, size);
    
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        h_data_cpu[i] = h_data_cpu[i] * h_data_cpu[i];
    }
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    float cpu_ms = duration.count() / 1000.0f;
    
    printf("CPU 시간: %.3f ms\n", cpu_ms);
    printf("GPU 시간: %.3f ms (커널만)\n", kernel_ms);
    printf("속도 향상: %.2fx\n", cpu_ms / kernel_ms);
    
    // 정리
    free(h_data);
    free(h_data_cpu);
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaEventDestroy(kernel_start));
    CHECK_CUDA(cudaEventDestroy(kernel_stop));
    
    printf("\n프로그램 완료!\n");
    return 0;
}