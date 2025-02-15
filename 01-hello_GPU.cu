#include <stdio.h>

__global__ void myKernel() 
{
    printf("grid(%d/%d,%d/%d)block(%d/%d, %d/%d) Hello, world from the device!\n",
		blockIdx.x, gridDim.x, blockIdx.y, gridDim.y,
	       	threadIdx.x, blockDim.x, threadIdx.y, blockDim.y); 
} 

int main() 
{ 
    dim3 grid(3, 2);
    dim3 block(2, 2);

    myKernel<<<grid, block>>>();
    cudaError_t cudaError = cudaGetLastError();

    if(cudaError != cudaSuccess) {
        printf("cuda execute error: %s\n", cudaGetErrorString(cudaError));
        return -1;
    } else {
        printf("cuda execute success\n");
    }

    cudaDeviceSynchronize();

    return 0;
} 

