/*
 * syncthreads_demo.cu
 * __syncthreads() 동기화 예제
 * 블록 내 스레드 동기화의 중요성 시연
 */

#include <stdio.h>
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

// 동기화 없는 커널 (잘못된 예제)
__global__ void withoutSync(int *data) {
    __shared__ int sharedData[256];
    int tid = threadIdx.x;
    
    // Phase 1: 공유 메모리에 쓰기
    sharedData[tid] = tid * 2;
    // ⚠️ 동기화 없음! - Race condition 발생 가능
    
    // Phase 2: 이웃 데이터 읽기
    int neighbor = (tid + 1) % blockDim.x;
    data[tid] = sharedData[neighbor];  // 이웃이 아직 쓰지 않았을 수 있음!
}

// 동기화 있는 커널 (올바른 예제)
__global__ void withSync(int *data) {
    __shared__ int sharedData[256];
    int tid = threadIdx.x;
    
    // Phase 1: 공유 메모리에 쓰기
    sharedData[tid] = tid * 2;
    
    __syncthreads();  // ✅ 모든 스레드가 쓰기 완료할 때까지 대기
    
    // Phase 2: 이웃 데이터 읽기
    int neighbor = (tid + 1) % blockDim.x;
    data[tid] = sharedData[neighbor];  // 이제 안전함!
}

// 조건부 동기화 - 데드락 예제
__global__ void conditionalSyncDeadlock(int *data) {
    int tid = threadIdx.x;
    
    // ⚠️ 위험: 조건부 __syncthreads()는 데드락 발생 가능
    if (tid < 128) {
        __syncthreads();  // 일부 스레드만 도달 -> 데드락!
    }
    
    data[tid] = tid;
}

// 올바른 조건부 처리
__global__ void conditionalSyncCorrect(int *data) {
    __shared__ int sharedData[256];
    int tid = threadIdx.x;
    
    // 모든 스레드가 동일한 경로 실행
    if (tid < 128) {
        sharedData[tid] = tid * 2;
    } else {
        sharedData[tid] = tid * 3;
    }
    
    __syncthreads();  // ✅ 모든 스레드가 도달
    
    data[tid] = sharedData[tid];
}

// 다단계 동기화 예제
__global__ void multiPhaseSync(float *output, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Phase 1: Global에서 Shared로 로드
    if (idx < n) {
        sdata[tid] = idx * 1.0f;
    } else {
        sdata[tid] = 0.0f;
    }
    __syncthreads();  // 동기화 1
    
    // Phase 2: 데이터 처리
    if (tid > 0 && tid < blockDim.x - 1) {
        float left = sdata[tid - 1];
        float center = sdata[tid];
        float right = sdata[tid + 1];
        sdata[tid] = (left + center + right) / 3.0f;  // 3-point average
    }
    __syncthreads();  // 동기화 2
    
    // Phase 3: 결과 저장
    if (idx < n) {
        output[idx] = sdata[tid];
    }
}

// 동기화 성능 측정
__global__ void syncPerformance(float *data, int iterations) {
    __shared__ float sharedData[256];
    int tid = threadIdx.x;
    
    for (int i = 0; i < iterations; i++) {
        // 작업 수행
        sharedData[tid] = data[tid] * 2.0f + 1.0f;
        
        __syncthreads();  // 동기화 오버헤드
        
        // 결과 사용
        data[tid] = sharedData[tid];
    }
}

int main() {
    printf("=== __syncthreads() 동기화 데모 ===\n\n");
    
    const int blockSize = 256;
    const int dataSize = blockSize;
    size_t size = dataSize * sizeof(int);
    
    // 메모리 할당
    int *h_data = (int*)malloc(size);
    int *h_result = (int*)malloc(size);
    int *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, size));
    
    // 1. 동기화 없는 경우 vs 있는 경우
    printf("1. 동기화 비교 테스트\n");
    
    // 동기화 없음
    withoutSync<<<1, blockSize>>>(d_data);
    CHECK_CUDA(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));
    
    // 동기화 있음
    withSync<<<1, blockSize>>>(d_data);
    CHECK_CUDA(cudaMemcpy(h_result, d_data, size, cudaMemcpyDeviceToHost));
    
    // 결과 비교
    int errors = 0;
    for (int i = 0; i < blockSize; i++) {
        int expected = ((i + 1) % blockSize) * 2;
        if (h_result[i] != expected) {
            errors++;
            if (errors <= 5) {  // 처음 5개만 출력
                printf("   오류: result[%d] = %d (예상: %d)\n", 
                       i, h_result[i], expected);
            }
        }
    }
    
    if (errors == 0) {
        printf("   ✅ 동기화로 모든 데이터 정확함\n");
    } else {
        printf("   ⚠️  동기화 없이 %d개 오류 발생 가능\n", errors);
    }
    
    // 2. 다단계 동기화 테스트
    printf("\n2. 다단계 동기화 테스트\n");
    
    const int n = 1024;
    float *h_float = (float*)malloc(n * sizeof(float));
    float *d_float;
    CHECK_CUDA(cudaMalloc(&d_float, n * sizeof(float)));
    
    // 초기화
    for (int i = 0; i < n; i++) {
        h_float[i] = (float)i;
    }
    CHECK_CUDA(cudaMemcpy(d_float, h_float, n * sizeof(float), 
                          cudaMemcpyHostToDevice));
    
    // 다단계 동기화 실행
    int gridSize = (n + blockSize - 1) / blockSize;
    size_t sharedMemSize = blockSize * sizeof(float);
    multiPhaseSync<<<gridSize, blockSize, sharedMemSize>>>(d_float, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    printf("   3-point averaging with multi-phase sync 완료\n");
    
    // 3. 동기화 오버헤드 측정
    printf("\n3. 동기화 오버헤드 측정\n");
    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    int iterations[] = {1, 10, 100, 1000};
    
    for (int i = 0; i < 4; i++) {
        CHECK_CUDA(cudaEventRecord(start));
        syncPerformance<<<1, blockSize>>>(d_float, iterations[i]);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        
        float ms = 0;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        
        printf("   %4d iterations: %.3f ms (%.3f μs/sync)\n",
               iterations[i], ms, (ms * 1000) / iterations[i]);
    }
    
    // 4. 동기화 규칙
    printf("\n4. __syncthreads() 규칙:\n");
    printf("   ✅ 모든 스레드가 도달해야 함\n");
    printf("   ✅ 블록 내에서만 동작\n");
    printf("   ✅ 조건문 내 사용 시 주의\n");
    printf("   ✅ 공유 메모리 사용 전후 필수\n");
    printf("   ⚠️  Global 메모리는 동기화 보장 안 됨\n");
    printf("   ⚠️  다른 블록과는 동기화 불가\n");
    
    // 5. 동기화 패턴
    printf("\n5. 일반적인 동기화 패턴:\n");
    printf("   1) Load → Sync → Process → Sync → Store\n");
    printf("   2) Write shared → Sync → Read shared\n");
    printf("   3) Reduction: 각 단계마다 Sync\n");
    printf("   4) Tiling: Tile 로드 후 Sync\n");
    
    // 메모리 해제
    free(h_data);
    free(h_result);
    free(h_float);
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_float));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    printf("\n프로그램 완료!\n");
    return 0;
}