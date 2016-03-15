#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>

#include "moving_average_cuda.hpp"
#include "memorytraverser.hpp"


//Kernel copy data from src --> dst with shift and proper AddresMode AM
//dst[i] = src[AM(i+shift)];
//template<typename T, typename AM>
//__global__ void Kernel(T* dst, T* src, const int size, const int shift, MemoryTraverser<T>* mt, AM am)
//{
//    int x = blockDim.x * blockIdx.x + threadIdx.x;

//    if(x >= size) return;

//    dst[x] = mt->get1D(src, (float)(x + shift), size, am);
//}

//Each new kernel have to be template to pass functor
//instead of using tex, we have to pass src as an additional param
template<typename AddressingModeFunctor>
__global__ void moving_average_tr_kernel(float* dst, float* src, const int N, const int R,
                                         MemoryTraverser<float, AddressingModeFunctor>* mt)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {

        float average = 0.f;

        for (int k = -R; k <= R; k++) {
            average = average + mt->get1D(src, (float)(tid - k + 0.5f)/(float)N, N);
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

    //This defines the behaviour
    using Traverser = MemoryTraverser<float, Clamp<true> >;
    Traverser* gmt;


    //First approach is to allocate the data using shared unified memory
    //HIP does not support that
/*
    {
        gmt = new MemoryTraverser<float>;
        gmt->addressMode = cuAddressModeWrap;
        gmt->filterMode = cuFilterModePoint;
        gmt->normalized = 1;

        //with unified memory I can access the variables from host
        if (gmt->addressMode == cuAddressModeWrap)
            moving_average_tr_kernel<<<iDivUp(N, 256), 256>>>(d_dst, d_src, N, R, gmt, Wrap());
        else
            moving_average_tr_kernel<<<iDivUp(N, 256), 256>>>(d_dst, d_src, N, R, gmt, Clamp());

        delete gmt;
    }
*/

    // old plain way is to use host device copy;
    {
        // just explicit helpers
        cuMemoryAddresMode addresMode = cuAddressModeWrap;
        cuMemoryFilterMode filterMode = cuFilterModePoint;
        int normalized = false;

        // base on addres mode pick clamp or wrap
        // if normalized = true address mode takes the same value;

        Traverser mt;
        //this is not necessary
        mt.addressMode = addresMode;
        mt.filterMode  = filterMode;
        mt.normalized = normalized;


        cudaMalloc((void**)&gmt, sizeof(Traverser));
        cudaMemcpy(gmt, &mt, sizeof(Traverser), cudaMemcpyHostToDevice);

        moving_average_tr_kernel<<<iDivUp(N, 256), 256>>>(d_dst, d_src, N, R, gmt);

        cudaFree(gmt);
    }

    cudaMemcpy(dst, d_dst, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_dst);
    cudaFree(d_src);

}


int main(int argc, char *argv[])
{

    typedef float T;

    const int N = 1000;
    T* host_memory = new T[N];

    T* h_cuda_output = new T[N];
    T* h_hip_output  = new T[N];

    for (int i = 0; i < N; i++) host_memory[i] = i;


    const int shift = -10;

    //original cuda texture version
    moving_average_gpu(h_cuda_output, host_memory, N, shift);

    //alternative implementation
    moving_average_tr(h_hip_output, host_memory, N, shift);

    bool implicit_comp = true;
    if (implicit_comp)
    {
        for(int i = 0; i < N; i++)
        {
            float eps = std::fabs(h_hip_output[i] - h_cuda_output[i]);
            if ( eps > 0)
                std::cout << i << " " << h_hip_output[i] << " eps: " << eps << std::endl;
        }
    }
    else
    {
            for(int i = 0; i < N; i++)
                    std::cout << i << " "
                              << "my : " << h_hip_output[i] << " "
                              << "rf : " << h_cuda_output[i] << std::endl;
    }


    delete [] host_memory;
    delete [] h_cuda_output;
    delete [] h_hip_output;

}
