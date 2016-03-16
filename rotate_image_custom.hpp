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

template<typename T, typename AddressingModeFunctor>
__global__ void transformKernel(T* outputData,
                                T* inputData,
                                int width,
                                int height,
                                T theta,
                                MemoryTraverser<T, AddressingModeFunctor>* mt)
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

    tu += 0.5f;
    tv += 0.5f;

    // read from texture and write to global memory
    //T val = tex1D(tex, tu+0.5f, tv+0.5f);
    T val = mt->get2D(inputData, tu, tv, width, height);
    T valm = mt->get2D(inputData, tu-1, tv, width, height);
    T valp = mt->get2D(inputData, tu+1, tv, width, height);

    if (tu == 0.5f)
        printf("ssc(%d, %d), u = %f, v = %f, tu = %f, tv = %f, val = %f, valm %f, valp %f\n", x, y, u, v, tu, tv, val, valm, valp);

        if(x == 0 && y == 0)
            printf("c(%d, %d), u = %f, v = %f, tu = %f, tv = %f, val = %f, valm %f, valp %f\n", x, y, u, v, tu, tv, val, valm, valp);
        if(x == 1 && y == 1)
            printf("c(%d, %d), u = %f, v = %f, tu = %f, tv = %f, val = %f, valm %f, valp %f\n", x, y, u, v, tu, tv, val, valm, valp);
        if(x == width/2 && y == height/2)
            printf("c(%d, %d), u = %f, v = %f, tu = %f, tv = %f, val = %f, valm %f, valp %f\n", x, y, u, v, tu, tv, val, valm, valp);


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
    TraverserType* d_mt;
    TraverserType h_mt;

    cudaSafeCall(cudaMalloc((void**)&d_mt, sizeof(TraverserType)));
    cudaSafeCall(cudaMemcpy(d_mt, &h_mt, sizeof(TraverserType), cudaMemcpyHostToDevice));


    //call kernel
    dim3 dimBlock(8, 8, 1);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

    transformKernel<T><<<dimGrid, dimBlock, 0>>>(d_output, d_input,
                                                 width, height, angle,
                                                 d_mt);
    cudaCheckError();
    cudaSafeCall(cudaDeviceSynchronize());

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
    using TraverserClampNorm = MemoryTraverser<float, Clamp<true> >;
    using TraverserClampUNorm = MemoryTraverser<float, Clamp<false> >;
    using TraverserWrapNorm = MemoryTraverser<float, Wrap<true> >;
    using TraverserWrapUNorm = MemoryTraverser<float, Wrap<false> >;

    //if(filterMode == cudaFilterModePoint)
    //else //Linear

    if (addressMode == cudaAddressModeWrap)
    {
        if(normalization)
        {
            return rotate_custom_impl<T, TraverserWrapNorm>(filename, angle);
        }
        else
        {
            return rotate_custom_impl<T, TraverserWrapUNorm>(filename, angle);
        }
    }
    else //Clamp
    {
        if(normalization)
        {
            return rotate_custom_impl<T, TraverserClampNorm>(filename, angle);
        }
        else
        {
            return rotate_custom_impl<T, TraverserClampUNorm>(filename, angle);
        }
    }

}


} //namespace custom_rotate


#endif // ROTATE_IMAGE_CUSTOM_HPP
