#ifndef ROTATE_IMAGE_CUDA_HPP
#define ROTATE_IMAGE_CUDA_HPP
#include <CImg.h>
#include <cuda_runtime.h>
#include <string>


cimg_library::CImg<float> rotate_cuda(const std::string& filename,
                 const float angle,
                 cudaTextureFilterMode filterMode,
                 cudaTextureAddressMode addressMode,
                 int normalization);


#endif // ROTATE_IMAGE_CUDA_HPP
