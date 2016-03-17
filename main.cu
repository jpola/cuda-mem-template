#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>

#include "moving_average_cuda.hpp"
#include "moving_average_custom.hpp"

#include "rotate_image_cuda.hpp"
#include "rotate_image_custom.hpp"

template<typename T, typename S>
inline bool
compareData(cimg_library::CImg<T>&diff_image, const S epsilon, const float threshold)
{
    assert(epsilon >= 0);

    bool result = true;
    unsigned int error_count = 0;

    auto len = diff_image.size();

    for (auto p : diff_image)
    {
        float diff = p;
        bool comp = (diff <= epsilon) && (diff >= -epsilon);
        result &= comp;
        error_count += !comp;
    }

    if (threshold == 0.0f)
    {
        return (result) ? true : false;
    }
    else
    {
        if (error_count)
        {
            printf("%4.2f(%%) of bytes mismatched (count=%d)\n", (float)error_count*100/(float)len, error_count);
        }

        return (len*threshold > error_count) ? true : false;
    }
}



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

    std::string lena_path = "data/img0008.pgm";
    cimg_library::CImg<T> ci =
            cuda_rotate::rotate_cuda<T>(lena_path, angle, filterMode, addressMode, normalization);
    cimg_library::CImg<T> mi =
            custom_rotate::rotate_custom<T>(lena_path, angle, filterMode, addressMode, normalization);

    cimg_library::CImg<T> diff =  ci - mi;
    //diff.abs();

    //bool result = false;
    bool result = true;
    //int index = 0;

    //five percent;
    float treshold = 0.05f;
    result = compareData(diff, 5e-2f, treshold);

    diff.normalize(0, 255);
    diff.save("data/diff.pgm");


    return result;

}

int main(int argc, char *argv[])
{

    cudaTextureFilterMode filterMode = cudaFilterModeLinear;
    cudaTextureAddressMode addressMode = cudaAddressModeWrap;
    int normalization = 1;

    bool result_avg =
            test_moving_average<float>(100, -13, filterMode,
                                       addressMode, normalization);
    if (!result_avg)
    {
        std::cout << "\n*** Result average FAILED! ***\n" << std::endl;
    }

    bool result_rot =
            test_rotate<float>(-0.5, filterMode,
                               addressMode, normalization);

    if (!result_rot)
    {
        std::cout << "\n*** Result rotation FAILED! ***\n" << std::endl;
    }

}
