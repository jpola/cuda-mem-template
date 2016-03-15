#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>

#include "moving_average_cuda.hpp"
#include "memorytraverser.hpp"


//Kernel copy data from src --> dst with shift and proper AddresMode AM
//dst[i] = src[AM(i+shift)];
template<typename T, typename AM>
__global__ void Kernel(T* dst, T* src, const int size, const int shift, MemoryTraverser<T>* mt, AM am)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    if(x >= size) return;

    dst[x] = mt->get1D(src, (float)(x + shift), size, am);
}

//Each new kernel have to be template to pass functor
//instead of using tex, we have to pass src as an additional param
template<typename AM>
__global__ void moving_average_tr_kernel(float* dst, float* src, const int N, const int R,
                                         MemoryTraverser<float>* mt, AM am)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {

        float average = 0.f;

        for (int k = -R; k <= R; k++) {
            average = average + mt->get1D(src, (float)(tid - k + 0.5f)/(float)N, N, am);
        }

        dst[tid] = average / (2.f * (float)R + 1.f);
    }

}

void moving_average_tr(float *dst, float *src, const int N, const int R)
{
    //prepare data on device
    float* d_dst;
    float* d_src;

    cudaMalloc((void**)&d_dst, N * sizeof(float));
    cudaMalloc((void**)&d_src, N * sizeof(float));

    cudaMemcpy(d_src, src, N * sizeof(float), cudaMemcpyHostToDevice);

    MemoryTraverser<float> mt;
    mt.addressMode = cuAddressModeWrap;
    mt.filterMode = cuFilterModePoint;
    mt.normalized = 1;

    if (mt.addressMode == cuAddressModeWrap)
        moving_average_tr_kernel<<<iDivUp(N, 256), 256>>>(d_dst, d_src, N, R, &mt, Wrap());
    else
        moving_average_tr_kernel<<<iDivUp(N, 256), 256>>>(d_dst, d_src, N, R, &mt, Clamp());


    cudaMemcpy(dst, d_dst, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_dst);
    cudaFree(d_src);
}


int main(int argc, char *argv[])
{

    typedef float T;

    const int N = 10;
    T* host_memory = new T[N];

    T* h_cuda_output = new T[N];
    T* h_hip_output  = new T[N];

    std::iota(host_memory, host_memory + N, 0);


    const int shift = 4;

    //original cuda texture version
    moving_average_gpu(h_cuda_output, host_memory, N, shift);

    for(int i = 0; i < N; i++)
        std::cout << i << " " << h_cuda_output[i]<< std::endl;
    std::cout << std::endl;

    //alternative implementation
    moving_average_tr(h_hip_output, host_memory, N, shift);

    for(int i = 0; i < N; i++)
        std::cout << i << " " << h_hip_output[i]<< std::endl;

    delete [] host_memory;
    delete [] h_cuda_output;
    delete [] h_hip_output;

}
