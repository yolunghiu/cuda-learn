#include <stdio.h>

// __global__告诉编译器这个是个可以在设备上执行的核函数
__global__ void hello_world(void)
{
    printf("GPU: Hello world!\n");
}

int main(int argc, char** argv)
{
    printf("CPU: Hello world!\n");
    hello_world<<<1, 3>>>();

    // 等GPU执行完了，再退出主机线程
    // if no this line ,it can not output hello world from gpu
    cudaDeviceReset();
    return 0;
}