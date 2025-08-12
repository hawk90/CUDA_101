/*
 * thread_id_calculator.cu
 * Thread ID 계산 및 데이터 매핑 예제
 * GTX 1070 (CC 6.1) / Jetson Orin Nano (CC 8.7) 호환
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// 1D Thread ID 계산 예제
__global__ void print1DThreadInfo() {
    // Global Thread ID 계산
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 처음 몇 개 스레드만 출력 (너무 많은 출력 방지)
    if (tid < 10) {
        printf("Block %d, Thread %d => Global Thread ID: %d\n", 
               blockIdx.x, threadIdx.x, tid);
    }
}

// 2D Thread ID 계산 예제
__global__ void print2DThreadInfo() {
    // 2D Global Thread ID 계산
    int tidX = blockIdx.x * blockDim.x + threadIdx.x;
    int tidY = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Grid 전체 크기
    int gridWidth = gridDim.x * blockDim.x;
    
    // 선형 인덱스 계산 (row-major)
    int linearIdx = tidY * gridWidth + tidX;
    
    // 처음 몇 개만 출력
    if (linearIdx < 10) {
        printf("Block(%d,%d) Thread(%d,%d) => Global(%d,%d) => Linear: %d\n",
               blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y,
               tidX, tidY, linearIdx);
    }
}

// 3D Thread ID 계산 예제
__global__ void print3DThreadInfo() {
    // 3D Global Thread ID 계산
    int tidX = blockIdx.x * blockDim.x + threadIdx.x;
    int tidY = blockIdx.y * blockDim.y + threadIdx.y;
    int tidZ = blockIdx.z * blockDim.z + threadIdx.z;
    
    // Grid 크기
    int gridWidth = gridDim.x * blockDim.x;
    int gridHeight = gridDim.y * blockDim.y;
    
    // 선형 인덱스 계산
    int linearIdx = tidZ * (gridWidth * gridHeight) + tidY * gridWidth + tidX;
    
    // 처음 몇 개만 출력
    if (linearIdx < 10) {
        printf("Block(%d,%d,%d) Thread(%d,%d,%d) => Global(%d,%d,%d) => Linear: %d\n",
               blockIdx.x, blockIdx.y, blockIdx.z,
               threadIdx.x, threadIdx.y, threadIdx.z,
               tidX, tidY, tidZ, linearIdx);
    }
}

// 실제 데이터 처리 예제: 벡터 더하기
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    // Thread ID 계산
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 범위 체크 (중요!)
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

// Thread ID 계산 퀴즈
__global__ void threadIdQuiz(int *results) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 각 스레드가 자신의 ID를 배열에 저장
    if (tid < 20) {
        results[tid] = tid;
    }
}

int main() {
    printf("=== Thread ID 계산 실습 ===\n\n");
    
    // 1. 1D Thread ID 예제
    printf("1. 1D Thread Configuration\n");
    printf("   Grid: 3 blocks, Block: 4 threads\n");
    printf("   Total threads: 3 * 4 = 12\n\n");
    
    dim3 grid1D(3);
    dim3 block1D(4);
    print1DThreadInfo<<<grid1D, block1D>>>();
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 2. 2D Thread ID 예제
    printf("\n2. 2D Thread Configuration\n");
    printf("   Grid: 2x2 blocks, Block: 3x3 threads\n");
    printf("   Total threads: (2*3) * (2*3) = 36\n\n");
    
    dim3 grid2D(2, 2);
    dim3 block2D(3, 3);
    print2DThreadInfo<<<grid2D, block2D>>>();
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 3. 3D Thread ID 예제
    printf("\n3. 3D Thread Configuration\n");
    printf("   Grid: 2x2x2 blocks, Block: 2x2x2 threads\n");
    printf("   Total threads: (2*2) * (2*2) * (2*2) = 64\n\n");
    
    dim3 grid3D(2, 2, 2);
    dim3 block3D(2, 2, 2);
    print3DThreadInfo<<<grid3D, block3D>>>();
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 4. Thread ID 계산 퀴즈
    printf("\n4. Thread ID 계산 퀴즈\n");
    printf("   문제: Block size = 256일 때, 다음 스레드의 Global ID는?\n");
    
    // 퀴즈 문제들
    struct {
        int blockId;
        int threadId;
        int answer;
    } quiz[] = {
        {0, 0, 0},      // 0 * 256 + 0 = 0
        {0, 255, 255},  // 0 * 256 + 255 = 255
        {1, 0, 256},    // 1 * 256 + 0 = 256
        {2, 100, 612},  // 2 * 256 + 100 = 612
        {3, 50, 818}    // 3 * 256 + 50 = 818
    };
    
    for (int i = 0; i < 5; i++) {
        printf("   Block %d, Thread %d = ? ", quiz[i].blockId, quiz[i].threadId);
        printf("(답: %d)\n", quiz[i].answer);
    }
    
    // 5. 실제 데이터 처리 예제
    printf("\n5. 실제 데이터 처리 예제\n");
    const int N = 1000;
    size_t size = N * sizeof(float);
    
    // 메모리 할당
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);
    float *d_a, *d_b, *d_c;
    
    CHECK_CUDA(cudaMalloc(&d_a, size));
    CHECK_CUDA(cudaMalloc(&d_b, size));
    CHECK_CUDA(cudaMalloc(&d_c, size));
    
    // 데이터 초기화
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }
    
    // GPU로 복사
    CHECK_CUDA(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));
    
    // 커널 실행 구성
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;  // 올림 나눗셈
    
    printf("   데이터 크기: %d\n", N);
    printf("   Block size: %d\n", blockSize);
    printf("   Grid size: %d\n", gridSize);
    printf("   총 스레드: %d (실제 필요: %d)\n", gridSize * blockSize, N);
    
    // 벡터 더하기 실행
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // 결과 복사
    CHECK_CUDA(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));
    
    // 결과 검증
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            correct = false;
            printf("   오류: h_c[%d] = %f (예상: %f)\n", 
                   i, h_c[i], h_a[i] + h_b[i]);
            break;
        }
    }
    
    if (correct) {
        printf("   ✓ 벡터 더하기 성공!\n");
    }
    
    // 6. Thread ID 매핑 테이블
    printf("\n6. Thread ID 매핑 참고표\n");
    printf("   +---------+-----------+----------------+\n");
    printf("   | BlockID | ThreadID  | Global ID      |\n");
    printf("   +---------+-----------+----------------+\n");
    printf("   | blockIdx| threadIdx | bid*bdim + tid |\n");
    printf("   +---------+-----------+----------------+\n");
    printf("   | 0       | 0         | 0              |\n");
    printf("   | 0       | 1         | 1              |\n");
    printf("   | 0       | bdim-1    | bdim-1         |\n");
    printf("   | 1       | 0         | bdim           |\n");
    printf("   | 1       | 1         | bdim+1         |\n");
    printf("   | ...     | ...       | ...            |\n");
    printf("   +---------+-----------+----------------+\n");
    
    // 메모리 해제
    free(h_a);
    free(h_b);
    free(h_c);
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    
    printf("\n프로그램 완료!\n");
    return 0;
}