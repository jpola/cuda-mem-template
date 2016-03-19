#ifndef ROTATE_IMAGE_CUSTOM_HPP
#define ROTATE_IMAGE_CUSTOM_HPP
#include <CImg.h>
#include <cuda_runtime.h>
#include <string>

#include "cuda_errors.hpp"
#include "memorytraverser.hpp"

namespace custom_rotate
{

using namespace cimg_library;

template<typename T, typename TraverserType>
__global__ void transformKernel(T* outputData,
                                T* sourceData,
                                int width,
                                int height,
                                T theta,
                                TraverserType* mt)
{
    // calculate normalized texture coordinates
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

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
    size_t size = width * height * sizeof(T);

    //prepare input and output on device
    T* d_input;
    T* d_output;
    cudaSafeCall(cudaMalloc((void**)&d_input, size));
    cudaSafeCall(cudaMalloc((void**)&d_output, size));
    cudaSafeCall(cudaMemcpy(d_input, d, size, cudaMemcpyHostToDevice));

    //prepare memory traverser;
    TraverserType* d_mt = nullptr;
    {
        TraverserType host_traverser;
        host_traverser.width = width;
        host_traverser.height = height;


        cudaSafeCall(cudaMalloc((void**)&d_mt, sizeof(TraverserType)));
        cudaSafeCall(cudaMemcpy(d_mt, &host_traverser, sizeof(TraverserType), cudaMemcpyHostToDevice));
    }

    //call kernel
    dim3 dimBlock(8, 8, 1);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

    transformKernel<T><<<dimGrid, dimBlock, 0>>>(d_output,
                                                 d_input,
                                                 width, height, angle,
                                                 d_mt);
    cudaCheckError();
    cudaSafeCall(cudaDeviceSynchronize());



    //TIMING
    cudaEvent_t start, stop;
    cudaSafeCall(cudaEventCreate(&start));
    cudaSafeCall(cudaEventCreate(&stop));

    const int NTimes = 250;
    cudaEventRecord(start);
    for (int i = 0; i < NTimes; i++)
    {
        transformKernel<T><<<dimGrid, dimBlock, 0>>>(d_output,
                                                     d_input,
                                                     width, height, angle,
                                                     d_mt);
    }
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float miliseconds = 0;
    cudaEventElapsedTime(&miliseconds, start, stop);
    std::cout << "CUSTOM TIME: " << miliseconds / (float)NTimes << " ms" <<std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    //copy back the data to host
    cudaSafeCall(cudaMemcpy(d, d_output, size, cudaMemcpyDeviceToHost));

    //cleanup
    cudaSafeCall(cudaFree(d_input));
    cudaSafeCall(cudaFree(d_output));
    cudaSafeCall(cudaFree(d_mt));


    //image.normalize(0, 255);
    image.save("data/custom_result.pgm");
    return image;
}

//create a library function in memorytraverser
//which will use perfect forwading to launch implementation function
template<typename T>
CImg<T> rotate_custom(const std::string& filename,
                                    const float angle,
                                    cudaTextureFilterMode filterMode,
                                    cudaTextureAddressMode addressMode,
                                    int normalization)
{
    //This defines the behaviour
    using TraverserClampNormPixel = MemoryTraverser<float, Clamp<NORMALIZED>,  PixelFilter<NEAREST>>;
    using TraverserClampUNormPixel = MemoryTraverser<float, Clamp<NON_NORMALIZED>, PixelFilter<NEAREST>>;

    using TraverserClampNormLinear = MemoryTraverser<float, Clamp<NORMALIZED>, PixelFilter<LINEAR>>;
    using TraverserClampUNormLinear = MemoryTraverser<float, Clamp<NON_NORMALIZED>, PixelFilter<LINEAR>>;

    using TraverserWrapNormPixel = MemoryTraverser<float, Wrap<NORMALIZED>, PixelFilter<NEAREST>>;
    using TraverserWrapUNormPixel = MemoryTraverser<float, Wrap<NON_NORMALIZED>, PixelFilter<NEAREST>>;

    using TraverserWrapNormLinear = MemoryTraverser<float, Wrap<NORMALIZED>, PixelFilter<LINEAR>>;
    using TraverserWrapUNormLinear = MemoryTraverser<float, Wrap<NON_NORMALIZED>, PixelFilter<LINEAR>>;

    if(filterMode == cudaFilterModePoint)
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


} //namespace custom_rotate


#endif // ROTATE_IMAGE_CUSTOM_HPP
