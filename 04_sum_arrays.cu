#include <cuda_runtime.h>
#include <stdio.h>

#include "freshman.h"

void sumArrays(float* a, float* b, float* res, const int size)
{
    // 一次循环进行4次元素加法
    for (int i = 0; i < size; i += 4)
    {
        res[i] = a[i] + b[i];
        res[i + 1] = a[i + 1] + b[i + 1];
        res[i + 2] = a[i + 2] + b[i + 2];
        res[i + 3] = a[i + 3] + b[i + 3];
    }
}

__global__ void sumArraysGPU(float* a, float* b, float* res)
{
    // int i=threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    res[i] = a[i] + b[i];
}

int main(int argc, char** argv)
{
    int dev = 0;
    cudaSetDevice(dev);

    int nElem = 1 << 14;
    printf("Vector size:%d\n", nElem);
    int nByte = sizeof(float) * nElem;

    // host data
    float* a_h = (float*)malloc(nByte);
    printf("%x\n", a_h);
    float* b_h = (float*)malloc(nByte);
    float* res_h = (float*)malloc(nByte);
    float* res_from_gpu_h = (float*)malloc(nByte);
    memset(res_h, 0, nByte);
    printf("%x\n", a_h);
    memset(res_from_gpu_h, 0, nByte);

    // device data
    float *a_d, *b_d, *res_d;

    // 为什么要传入二维指针：a_d这个指针是存储在主存上的, 之所以取a_d的地址,
    // 是为了将cudaMalloc在显存上获得的数组首地址赋值给a_d
    printf("%x\n", a_d);
    CHECK(cudaMalloc((float**)&a_d, nByte));
    printf("%x\n", a_d);

    CHECK(cudaMalloc((float**)&b_d, nByte));
    CHECK(cudaMalloc((float**)&res_d, nByte));

    initialData(a_h, nElem);
    initialData(b_h, nElem);

    CHECK(cudaMemcpy(a_d, a_h, nByte, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(b_d, b_h, nByte, cudaMemcpyHostToDevice));

    dim3 block(1024);
    dim3 grid(nElem / block.x);
    sumArraysGPU<<<grid, block>>>(a_d, b_d, res_d);
    printf("Execution configuration<<<%d,%d>>>\n", grid.x, block.x);

    CHECK(cudaMemcpy(res_from_gpu_h, res_d, nByte, cudaMemcpyDeviceToHost));
    sumArrays(a_h, b_h, res_h, nElem);

    checkResult(res_h, res_from_gpu_h, nElem);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(res_d);

    free(a_h);
    free(b_h);
    free(res_h);
    free(res_from_gpu_h);

    return 0;
}