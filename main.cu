#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>

#include "moving_average_cuda.hpp"
#include "moving_average_custom.hpp"

#include "rotate_image_cuda.hpp"

template <typename T>
bool test_moving_average(const int N, const int R,
                         cudaTextureFilterMode filterMode,
                         cudaTextureAddressMode addressMode,
                         int normalization)
{
    T* host_memory = new T[N];

    T* h_cuda_output = new T[N];
    T* h_hip_output  = new T[N];

    for (int i = 0; i < N; i++) host_memory[i] = i;

    //original cuda texture version
    moving_average_gpu(h_cuda_output, host_memory, N, R,
                       filterMode, addressMode, normalization);

    //alternative implementation
    moving_average_tr(h_hip_output, host_memory, N, R,
                      filterMode, addressMode, normalization);

    bool implicit_comp = true;

    bool result = true;

    if (implicit_comp)
    {
        for(int i = 0; i < N; i++)
        {
            float eps = std::fabs(h_hip_output[i] - h_cuda_output[i]);
            if ( eps > 0)
            {
                std::cout << i << " " << h_hip_output[i] << " eps: " << eps << std::endl;
                result = false;
            }
        }
    }
    else
    {
            for(int i = 0; i < N; i++)
            {
                float eps = std::fabs(h_hip_output[i] - h_cuda_output[i]);
                if ( eps > 0)
                    result = false;

                std::cout << i << " "
                          << "my : " << h_hip_output[i] << " "
                          << "rf : " << h_cuda_output[i] << std::endl;
            }
    }


    delete [] host_memory;
    delete [] h_cuda_output;
    delete [] h_hip_output;

    return result;

}


template<typename T>
bool test_rotate(const float angle,
                 cudaTextureFilterMode filterMode,
                 cudaTextureAddressMode addressMode,
                 int normalization)
{

    std::string lena_path = "data/lena_bw.pgm";
    cuda_rotate::rotate_cuda<T>(lena_path, angle, filterMode, addressMode, normalization);

    return true;

}

int main(int argc, char *argv[])
{

    bool result_avg =
            test_moving_average<float>(10, 3, cudaFilterModePoint,
                                       cudaAddressModeClamp, 1);
    if (!result_avg)
    {
        std::cout << "\n*** Result average FAILED! ***\n" << std::endl;
    }

    bool result_rot =
            test_rotate<float>(0.5, cudaFilterModePoint,
                               cudaAddressModeClamp, 1);

    if (!result_rot)
    {
        std::cout << "\n*** Result rotation FAILED! ***\n" << std::endl;
    }

}
