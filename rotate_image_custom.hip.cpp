#include "hip_runtime.h"
#include "rotate_image_custom.hpp"
#include "memorytraverser.hpp"
#include "hip_errors.hpp"
#include "cuda_errors.hpp"

using namespace cimg_library;

inline int iDivUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

template<typename T, typename TraverserType>
__global__ void transformKernel(hipLaunchParm lp,
                                T* outputData,
                                const T* __restrict__ sourceData,
                                int width,
                                int height,
                                T theta,
                                TraverserType* mt)
{
    // calculate normalized texture coordinates
    unsigned int x = hipBlockIdx_x*hipBlockDim_x + hipThreadIdx_x;
    unsigned int y = hipBlockIdx_y*hipBlockDim_y + hipThreadIdx_y;

    T u = (T)x - (T)width/2;
    T v = (T)y - (T)height/2;
    T tu = u*cosf(theta) - v*sinf(theta);
    T tv = v*cosf(theta) + u*sinf(theta);

    tu /= (T)width;
    tv /= (T)height;

    // read from texture and write to global memory
    T val = mt->get2D(sourceData, tu + 0.5f, tv + 0.5f);

    outputData[y*width + x] = val;
}

template<typename T, typename TraverserType>
CImg<T> rotate_custom_impl(const std::string& filename, const float angle)
{
    CImg<T> image(filename.c_str());
    T* d = image.data();

    unsigned int width = image.width();
    unsigned int height = image.height();

    //prepare input and output on device
    T* d_input;
    T* d_output;
    size_t line_bytes = width * sizeof(T);
    size_t h_elems_per_line = width;
    size_t pitch_bytes = line_bytes; //put proper here
    size_t d_elems_per_line = pitch_bytes / sizeof(T);

    size_t host_memory_size = line_bytes * height;
    size_t devcie_memory_size = pitch_bytes * height;

    hipSafeCall(hipMalloc((void**)&d_input, devcie_memory_size));
    hipSafeCall(hipMalloc((void**)&d_output, devcie_memory_size));

    for (int i = 0; i < height; i++)
    {
       // std::cout << image.at(i*width, -1) <<  " orig : " << image[i*width] << " " << d[i*width] << std::endl;

        cudaError err = cudaMemcpy(&d_input[i*d_elems_per_line],
                    &d[i*h_elems_per_line], line_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            std::cerr << "Failed at i = " << i << " "<< d[i*h_elems_per_line] << std::endl;
        }
    }
    cudaDeviceSynchronize();
    //prepare memory traverser;
    TraverserType* d_mt = nullptr;
    {
        TraverserType host_traverser;
        host_traverser.width = width;
        host_traverser.height = height;

        hipSafeCall(hipMalloc((void**)&d_mt, sizeof(TraverserType)));
        hipSafeCall(hipMemcpy(d_mt, &host_traverser, sizeof(TraverserType), hipMemcpyHostToDevice));
    }

    //call kernel
    dim3 dimBlock(8, 8, 1);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

    hipLaunchKernel(HIP_KERNEL_NAME(transformKernel<T>), dim3(dimGrid), dim3(dimBlock), 0, 0,
                    d_output,
                    d_input,
                    width, height, angle,
                    d_mt);
    hipCheckError();
    hipSafeCall(hipDeviceSynchronize());

    //TIMING
    hipEvent_t start, stop;
    hipSafeCall(hipEventCreate(&start));
    hipSafeCall(hipEventCreate(&stop));

    const int NTimes = 100;
    hipEventRecord(start);
    for (int i = 0; i < NTimes; i++)
    {
        hipLaunchKernel(HIP_KERNEL_NAME(transformKernel<T>), dim3(dimGrid), dim3(dimBlock), 0, 0, d_output,
                        d_input,
                        width, height, angle,
                        d_mt);
    }
    hipEventRecord(stop);

    hipEventSynchronize(stop);
    float miliseconds = 0;
    hipEventElapsedTime(&miliseconds, start, stop);
    std::cout << "CUSTOM TIME: " << miliseconds / (float)NTimes << " ms" <<std::endl;

    hipEventDestroy(start);
    hipEventDestroy(stop);


    //copy back the data to host
    //hipSafeCall(hipMemcpy(d, d_output, size, hipMemcpyDeviceToHost));
    for (int i = 0; i < height; i++)
    {
  //       std::cout << " i " << i <<std::endl;
        hipSafeCall(hipMemcpy(&d[i*h_elems_per_line],
                    &d_output[i*d_elems_per_line], line_bytes, hipMemcpyDeviceToHost));
    }

    //cleanup
    hipSafeCall(hipFree(d_input));
    hipSafeCall(hipFree(d_output));
    hipSafeCall(hipFree(d_mt));


    //image.normalize(0, 255);
    image.save("data/custom_result.pgm");
    return image;
}

//create a library function in memorytraverser
//which will use perfect forwading to launch implementation function
CImg<float> rotate_custom(const std::string& filename,
                          const float angle,
                          cudaTextureFilterMode filterMode,
                          cudaTextureAddressMode addressMode,
                          int normalization)
{
    typedef float T;
    //This defines the behaviour
    using TraverserClampNormPixel = MemoryTraverser<float, Clamp<NORMALIZED, float>,  PixelFilter<NEAREST, float>>;
    using TraverserClampUNormPixel = MemoryTraverser<float, Clamp<NON_NORMALIZED, float>, PixelFilter<NEAREST, float>>;

    using TraverserClampNormLinear = MemoryTraverser<float, Clamp<NORMALIZED, float>, PixelFilter<LINEAR, float>>;
    using TraverserClampUNormLinear = MemoryTraverser<float, Clamp<NON_NORMALIZED, float>, PixelFilter<LINEAR, float>>;

    using TraverserWrapNormPixel = MemoryTraverser<float, Wrap<NORMALIZED, float>, PixelFilter<NEAREST, float>>;
    using TraverserWrapUNormPixel = MemoryTraverser<float, Wrap<NON_NORMALIZED, float>, PixelFilter<NEAREST, float>>;

    using TraverserWrapNormLinear = MemoryTraverser<float, Wrap<NORMALIZED, float>, PixelFilter<LINEAR, float>>;
    using TraverserWrapUNormLinear = MemoryTraverser<float, Wrap<NON_NORMALIZED, float>, PixelFilter<LINEAR, float>>;

    if(filterMode == hipFilterModePoint)
    {
        if (addressMode == cudaAddressModeWrap)
        {
            if(normalization)
            {
                return rotate_custom_impl<T, TraverserWrapNormPixel>(filename, angle);
            }
            else
            {
                return rotate_custom_impl<T, TraverserWrapUNormPixel>(filename, angle);
            }
        }
        else //clamp
        {
            if(normalization)
            {
                return rotate_custom_impl<T, TraverserClampNormPixel>(filename, angle);
            }
            else
            {
                return rotate_custom_impl<T, TraverserClampUNormPixel>(filename, angle);
            }

        }
    }
    else //Linear interpolation
    {
        if (addressMode == cudaAddressModeWrap)
        {
            if(normalization)
            {
                return rotate_custom_impl<T, TraverserWrapNormLinear>(filename, angle);
            }
            else
            {
                return rotate_custom_impl<T, TraverserWrapUNormLinear>(filename, angle);
            }
        }
        else //clamp
        {
            if(normalization)
            {
                return rotate_custom_impl<T, TraverserClampNormLinear>(filename, angle);
            }
            else
            {
                return rotate_custom_impl<T, TraverserClampUNormLinear>(filename, angle);
            }

        }

    }

}




