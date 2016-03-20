#ifndef MOVING_AVERAGE_CUDA_HPP
#define MOVING_AVERAGE_CUDA_HPP
#include <cuda_runtime.h>

void moving_average_gpu(float* dst, float* src, const int N, const int R,
                        cudaTextureFilterMode filterMode,
                        cudaTextureAddressMode addressMode,
                        int normalization);

#endif // MOVING_AVERAGE_CUDA_HPP
