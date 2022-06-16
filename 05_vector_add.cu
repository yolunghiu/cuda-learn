#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

using namespace std;

// 二：线程执行代码
// dim3 grid(1, 1, 1), block(length, 1, 1);
__global__ void vector_add1(float* vec1, float* vec2, float* vecres, int length)
{
    int tid = threadIdx.x;
    if (tid < length)
    {
        vecres[tid] = vec1[tid] + vec2[tid];
    }
}

// dim3 grid(16, 1, 1), block(1, 1, 1);
__global__ void vector_add2(float* vec1, float* vec2, float* vecres, int length)
{
    int tid = blockIdx.x;
    if (tid < length)
    {
        vecres[tid] = vec1[tid] + vec2[tid];
    }
}

// dim3 grid(1, 1, 1), block(4, 4, 1);
__global__ void vector_add3(float* vec1, float* vec2, float* vecres, int length)
{
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (tid < length)
    {
        vecres[tid] = vec1[tid] + vec2[tid];
    }
}

// dim3 grid(4, 1, 1), block(4, 1, 1);
__global__ void vector_add4(float* vec1, float* vec2, float* vecres, int length)
{
    int tid = blockIdx.x * gridDim.x + threadIdx.x;
    if (tid < length)
    {
        vecres[tid] = vec1[tid] + vec2[tid];
    }
}

// dim3 grid(2, 2, 1), block(2, 2, 1);
__global__ void vector_add5(float* vec1, float* vec2, float* vecres, int length)
{
    int tid =
        (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) +
        threadIdx.y * blockDim.y + threadIdx.x;
    if (tid < length)
    {
        vecres[tid] = vec1[tid] + vec2[tid];
    }
}

int main()
{
    const int length = 16;                  // 数组长度为16
    float a[length], b[length], c[length];  // host中的数组
    for (int i = 0; i < length; i++)
    {  // 初始赋值
        a[i] = b[i] = i;
    }
    float *a_device, *b_device, *c_device;  // device中的数组

    cudaMalloc((void**)&a_device, length * sizeof(float));  // 分配内存
    cudaMalloc((void**)&b_device, length * sizeof(float));
    cudaMalloc((void**)&c_device, length * sizeof(float));

    cudaMemcpy(a_device,
               a,
               length * sizeof(float),
               cudaMemcpyHostToDevice);  // 将host数组的值拷贝给device数组
    cudaMemcpy(b_device, b, length * sizeof(float), cudaMemcpyHostToDevice);

    // 一：参数配置
    //    dim3 grid(1, 1, 1), block(length, 1, 1);  // 设置参数
    //    dim3 grid(length, 1, 1), block(1, 1, 1);  // 设置参数
    //    dim3 grid(1, 1, 1), block(4, 4, 1);
    //    dim3 grid(4, 1, 1), block(4, 1, 1);
    dim3 grid(2, 2, 1), block(2, 2, 1);
    vector_add5<<<grid, block>>>(
        a_device, b_device, c_device, length);  // 启动kernel

    cudaMemcpy(c,
               c_device,
               length * sizeof(float),
               cudaMemcpyDeviceToHost);  // 将结果拷贝到host

    for (int i = 0; i < length; i++)
    {  // 打印出来方便观察
        cout << c[i] << " ";
    }
    
    return 0;
}
