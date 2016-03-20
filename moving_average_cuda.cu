#include "moving_average_cuda.hpp"
#include "cuda_errors.hpp"

inline int iDivUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

texture<float, 1, cudaReadModeElementType> tex1d;
__global__ void moving_average_kernel(float* __restrict__ dst, const int N, const int R)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {

        float average = 0.f;

        for (int k = -R; k <= R; k++) {
            average = average + tex1D(tex1d, (float)(tid - k + 0.5f)/(float)N);
        }

        dst[tid] = average / (2.f * (float)R + 1.f);
    }
}

void moving_average_gpu(float* dst, float* src, const int N, const int R,
                        cudaTextureFilterMode filterMode,
                        cudaTextureAddressMode addressMode,
                        int normalization)
{
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    //cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

    cudaArray* cuArray;
    cudaSafeCall(cudaMallocArray(&cuArray, &channelDesc, N, 1));
    cudaSafeCall(cudaMemcpyToArray(cuArray, 0, 0, src, N * sizeof(float), cudaMemcpyHostToDevice));

    cudaSafeCall(cudaBindTextureToArray(tex1d, cuArray));

    tex1d.filterMode = filterMode;
    tex1d.normalized = normalization;
    //only with normalized!
    tex1d.addressMode[0] = addressMode;

    float* device_result;
    cudaSafeCall(cudaMalloc((void**)&device_result, N * sizeof(float)));

    moving_average_kernel<<<iDivUp(N, 256), 256>>>(device_result, N, R);
    cudaCheckError();
    cudaSafeCall(cudaDeviceSynchronize());

    cudaSafeCall(cudaMemcpy(dst, device_result, N * sizeof(float), cudaMemcpyDeviceToHost));

    cudaSafeCall(cudaUnbindTexture(tex1d));
    cudaSafeCall(cudaFreeArray(cuArray));
    cudaSafeCall(cudaFree(device_result));

}

