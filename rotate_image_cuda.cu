#include "rotate_image_cuda.hpp"
#include <string>

#include "cuda_errors.hpp"
#define DIV_UP(x, y) ( (y) * ( ((x)+(y)-1) / (y) ) )
//to hide global texture objects

using namespace cimg_library;
// Texture reference for 2D float texture
texture<float, 2, cudaReadModeElementType> tex;

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
    T val = tex2D(tex, tu + 0.5f, tv + 0.5f);

    outputData[y*width + x] = val;
}



CImg<float> rotate_cuda(const std::string& filename,
                        const float angle,
                        cudaTextureFilterMode filterMode,
                        cudaTextureAddressMode addressMode,
                        int normalization)
{
    typedef float T;
    CImg<T> image(filename.c_str());

    T* d = image.data();
    unsigned int width = image.width();
    unsigned int height = image.height();

    size_t size = width * height * sizeof(T);
    T* dd;
    size_t pitch;
    cudaMallocPitch((void**)&dd, &pitch, width*sizeof(T), height);
    cudaMemcpy2D(dd, pitch, d, width*sizeof(T), width*sizeof(T), height, cudaMemcpyHostToDevice);

//    {
//        int nDevices;
//        cudaGetDeviceCount(&nDevices);

//        cudaDeviceProp prop;
//        cudaGetDeviceProperties(&prop, nDevices);


//        size_t pitch_size =
//                DIV_UP(width*sizeof(T), prop.textureAlignment)
//                * prop.textureAlignment;

//        std::cout << "PITCH " << pitch
//                  << " MY PITCH: " << pitch_size << std::endl;

//    }


    //prepare texture and allocate image on device;
    cudaChannelFormatDesc channelDesc =
            cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray *cuArray;
    cudaSafeCall(cudaMallocArray(&cuArray, &channelDesc, width, height));
    cudaSafeCall(cudaMemcpy2DToArray(cuArray, 0, 0, dd, pitch, width*sizeof(T), height, cudaMemcpyDeviceToDevice));

    tex.addressMode[0] = addressMode;
    tex.addressMode[1] = addressMode;
    tex.filterMode = filterMode;
    tex.normalized = normalization;

    // Bind the array to the texture
    cudaSafeCall(cudaBindTextureToArray(tex, cuArray, channelDesc));

    // result data
    T* d_data;
    cudaSafeCall(cudaMalloc((void**)&d_data, size));

    dim3 dimBlock(8, 8, 1);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

    transformKernel<float><<<dimGrid, dimBlock, 0>>>(d_data, width, height, angle);
    cudaCheckError();
    cudaSafeCall(cudaDeviceSynchronize());

    //TIMING
    cudaEvent_t start, stop;
    cudaSafeCall(cudaEventCreate(&start));
    cudaSafeCall(cudaEventCreate(&stop));

    const int NTimes = 100;
    cudaEventRecord(start);
    for (int i = 0; i < NTimes; i++)
    {
        transformKernel<float><<<dimGrid, dimBlock, 0>>>(d_data, width, height, angle);
    }
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float miliseconds = 0;
    cudaEventElapsedTime(&miliseconds, start, stop);
    std::cout << "CUDA TIME: " << miliseconds / (float)NTimes << " ms" <<std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    //get back the results on host.
    cudaSafeCall(cudaMemcpy(d, d_data, size, cudaMemcpyDeviceToHost));

    cudaSafeCall(cudaUnbindTexture(tex));

    cudaSafeCall(cudaFree(d_data));
    cudaSafeCall(cudaFreeArray(cuArray));

    image.save("data/cuda_result.pgm");

    return image;
}

