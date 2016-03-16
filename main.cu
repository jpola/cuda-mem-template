#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>

#include "moving_average_cuda.hpp"
#include "moving_average_custom.hpp"

#include "rotate_image_cuda.hpp"
#include "rotate_image_custom.hpp"

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
    cimg_library::CImg<T> ci =
            cuda_rotate::rotate_cuda<T>(lena_path, angle, filterMode, addressMode, normalization);
    cimg_library::CImg<T> mi =
            custom_rotate::rotate_custom<T>(lena_path, angle, filterMode, addressMode, normalization);

    cimg_library::CImg<T> diff =  ci - mi;
    //diff.abs();

    //bool result = false;
    bool result = true;
    int index = 0;
    for (auto p : diff)
    {
        if (p > 0)
        {
            std::cout << "Pixel difference ["<< index << "] = " << p << std::endl;
            result = false;
        }
        index++;
    }

    diff.save("data/diff.pgm");


    return result;

}

int main(int argc, char *argv[])
{

    cudaTextureFilterMode filterMode = cudaFilterModePoint;
    cudaTextureAddressMode addressMode = cudaAddressModeClamp;
    int normalization = 1;

    bool result_avg =
            test_moving_average<float>(100, -13, filterMode,
                                       addressMode, normalization);
    if (!result_avg)
    {
        std::cout << "\n*** Result average FAILED! ***\n" << std::endl;
    }

    bool result_rot =
            test_rotate<float>(0.5, filterMode,
                               addressMode, normalization);

    if (!result_rot)
    {
        std::cout << "\n*** Result rotation FAILED! ***\n" << std::endl;
    }

}
