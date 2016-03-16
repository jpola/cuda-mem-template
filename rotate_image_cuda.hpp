#ifndef ROTATE_IMAGE_CUDA_HPP
#define ROTATE_IMAGE_CUDA_HPP

#include <CImg.h>
#include <cuda_runtime.h>
#include <string>

#include "cuda_errors.hpp"

//to hide global texture objects
namespace cuda_rotate
{

using namespace cimg_library;
// Texture reference for 2D float texture
texture<float, 2, cudaReadModeElementType> tex;
texture<float, 1, cudaReadModeElementType> tx1D;

template<typename T>
__global__ void transformKernel(T* outputData,
                                int width,
                                int height,
                                T theta)
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
    //T val = tex1D(tex, tu+0.5f, tv+0.5f);
    T val = tex1D(tx1D, tu+0.5f);
//        if(x == 0 && y == 0)
//            printf("u = %f, v = %f, tu = %f, tv = %f, val = %f\n", u, v, tu, tv, val);
//        if(x == 1 && y == 0)
//            printf("u = %f, v = %f, tu = %f, tv = %f, val = %f\n", u, v, tu, tv, val);
//        if(x == 2 && y == 0)
//            printf("u = %f, v = %f, tu = %f, tv = %f, val = %f\n", u, v, tu, tv, val);
    outputData[y*width + x] = val;
}



template<typename T>
CImg<T> rotate_cuda(const std::string& filename,
                 const float angle,
                 cudaTextureFilterMode filterMode,
                 cudaTextureAddressMode addressMode,
                 int normalization)
{
    CImg<T> image(filename.c_str());

    T* d = image.data();
    unsigned int width = image.width();
    unsigned int height = image.height();

    size_t size = width * height * sizeof(T);

    //prepare texture and allocate image on device;
    cudaChannelFormatDesc channelDesc =
            cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray *cuArray;
    cudaSafeCall(cudaMallocArray(&cuArray, &channelDesc, width, height));
    cudaSafeCall(cudaMemcpyToArray(cuArray, 0, 0, d, size, cudaMemcpyHostToDevice));

//    tex.addressMode[0] = addressMode;
//    tex.addressMode[1] = addressMode;
//    tex.filterMode = filterMode;
//    tex.normalized = normalization;
    tx1D.addressMode[0] = addressMode;
    tx1D.addressMode[1] = addressMode;
    tx1D.filterMode = filterMode;
    tx1D.normalized = normalization;

    // Bind the array to the texture
    //cudaSafeCall(cudaBindTextureToArray(tex, cuArray, channelDesc));
    cudaSafeCall(cudaBindTextureToArray(tx1D, cuArray, channelDesc));

    // result data
    T* d_data;
    cudaSafeCall(cudaMalloc((void**)&d_data, size));

    dim3 dimBlock(8, 8, 1);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

    transformKernel<float><<<dimGrid, dimBlock, 0>>>(d_data, width, height, angle);
    cudaCheckError();
    cudaSafeCall(cudaDeviceSynchronize());

    //get back the results on host.
    cudaSafeCall(cudaMemcpy(d, d_data, size, cudaMemcpyDeviceToHost));

    //cudaSafeCall(cudaUnbindTexture(tex));
    cudaSafeCall(cudaUnbindTexture(tx1D));

    cudaSafeCall(cudaFree(d_data));
    cudaSafeCall(cudaFreeArray(cuArray));

    image.save("data/cuda_result.pgm");

    return image;
}


}

#endif // ROTATE_IMAGE_CUDA_HPP
