#ifndef MOVING_AVERAGE_CUDA_HPP
#define MOVING_AVERAGE_CUDA_HPP

inline int iDivUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

texture<float, 1, cudaReadModeElementType> tex;
__global__ void moving_average_kernel(float* __restrict__ dst, const int N, const int R)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {

        float average = 0.f;

        for (int k = -R; k <= R; k++) {
            average = average + tex1D(tex, (float)(tid - k + 0.5f)/(float)N);
        }

        dst[tid] = average / (2.f * (float)R + 1.f);
    }
}

void moving_average_gpu(float* dst, float* src, const int N, const int R)
{
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    //cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

    cudaArray* cuArray;
    cudaMallocArray(&cuArray, &channelDesc, N, 1);
    cudaMemcpyToArray(cuArray, 0, 0, src, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaBindTextureToArray(tex, cuArray);

    tex.normalized=true;
    //only with normalized!
    tex.addressMode[0] = cudaAddressModeWrap;

    float* device_result;
    cudaMalloc((void**)&device_result, N * sizeof(float));

    moving_average_kernel<<<iDivUp(N, 256), 256>>>(device_result, N, R);

    cudaError err = cudaDeviceSynchronize();
    //std::cout << "Kernel execution : " << cudaGetErrorString(err) << std::endl;

    err = cudaMemcpy(dst, device_result, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaUnbindTexture(tex);
    cudaFreeArray(cuArray);
    cudaFree(&device_result);
    //std::cout << "To device : " << cudaGetErrorString(err) << std::endl;
}


#endif // MOVING_AVERAGE_CUDA_HPP
