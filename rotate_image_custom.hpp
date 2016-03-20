#ifndef ROTATE_IMAGE_CUSTOM_HPP
#define ROTATE_IMAGE_CUSTOM_HPP

#include <CImg.h>
#include <string>
#include <cuda_runtime.h>


using namespace cimg_library;

CImg<float> rotate_custom(const std::string& filename,
                                    const float angle,
                                    cudaTextureFilterMode filterMode,
                                    cudaTextureAddressMode addressMode,
                                    int normalization);

#endif // ROTATE_IMAGE_CUSTOM_HPP
