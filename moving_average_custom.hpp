#ifndef MOVING_AVERAGE_CUSTOM_HPP
#define MOVING_AVERAGE_CUSTOM_HPP
#include <cuda_runtime.h>

void moving_average_tr(float *dst, float *src, const int N, const int R,
                       cudaTextureFilterMode filterMode,
                       cudaTextureAddressMode addressMode,
                       int normalization);

#endif //MOVING_AVERAGE_CUSTOM_HPP

