#ifndef ROTATE_IMAGE_CUSTOM_HPP
#define ROTATE_IMAGE_CUSTOM_HPP

#include <CImg.h>
#include <string>

using namespace cimg_library;

CImg<float> rotate_custom(const std::string& filename,
                                    const float angle);

#endif // ROTATE_IMAGE_CUSTOM_HPP
