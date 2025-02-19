#include <stdio.h>
#include <stdlib.h>

#define BLOCK_X			6
#define BLOCK_Y			8
#define BLOCKS			30
#define N			((BLOCKS) * (BLOCK_X) * (BLOCK_Y))

#define MAX_NUM			1000
#define RAND_ZERO_TO_ONE()	(((rand() % RAND_MAX) * 1.0) / RAND_MAX)
#define GEN_RAND_FLOAT()	(MAX_NUM * RAND_ZERO_TO_ONE())

// Kernel定义
__global__ void MatAdd(float *A, float *B, float *C) 
{ 
    int i = blockIdx.x * gridDim.x + blockIdx.y * gridDim.y +
	    threadIdx.x * blockDim.x + threadIdx.y;

    C[i] = A[i] + B[i]; 
}

int main() 
{
    int i;
    float *A, *B, *C;
    // Kernel 线程配置
    dim3 blockThreads(BLOCK_X, BLOCK_Y); 
    dim3 numBlocks(N / (blockThreads.x * blockThreads.y), N / blockThreads.y);

    cudaMallocManaged((void **)&A, N * sizeof(float));
    cudaMallocManaged((void **)&B, N * sizeof(float));
    cudaMallocManaged((void **)&C, N * sizeof(float));

    srand((unsigned int)time(NULL));
    for (i = 0; i < N; i++) {
        A[i] = GEN_RAND_FLOAT();
	B[i] = GEN_RAND_FLOAT();
    }

    // kernel调用
    MatAdd<<<numBlocks, blockThreads>>>(A, B, C); 

    // 同步device 保证结果能正确访问
    cudaDeviceSynchronize();

    for (i = 0; i < N; i++) {
        printf("[(), ()]%f + %f = %f\n", A[i], B[i], C[i]);
    }
    printf("\n");

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}
