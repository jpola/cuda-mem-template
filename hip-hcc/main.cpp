#include <iostream>
#include <algorithm>
#include <cassert>
#include <numeric>
#include <cmath>

#include "rotate_image_custom.hpp"



template<typename T>
bool test_rotate(const float angle)
{

    std::string lena_path = "../data/lena_bw.pgm";//"data/img0008.pgm";

//    cimg_library::CImg<T> ci =
//            rotate_cuda(lena_path, angle, filterMode, addressMode, normalization);

    cimg_library::CImg<T> mi =
            rotate_custom(lena_path, angle);

//    cimg_library::CImg<T> diff =  ci - mi;
//    //diff.abs();

//    //bool result = false;
//    bool result = true;
//    //int index = 0;

//    //five percent;
//    float treshold = 0.05f;
//    result = compareData(diff, 5e-2f, treshold);

//    diff.normalize(0, 255);
//    diff.save("data/diff.pgm");

    return true;

}

int main(int argc, char *argv[])
{

    bool result_rot =
            test_rotate<float>(-0.5);

    if (!result_rot)
    {
        std::cout << "\n*** Result rotation FAILED! ***\n" << std::endl;
    }

}
