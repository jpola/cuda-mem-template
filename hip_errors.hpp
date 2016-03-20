#ifndef HIP_ERRORS_HPP
#define HIP_ERRORS_HPP

#include <hip_runtime.h>
#include <iostream>

#define hipSafeCall(err) _hipSafeCall(err, __FILE__, __LINE__)
#define hipCheckError()  _hipCheckError(__FILE__, __LINE__)

/**
 * hip version of error handling functions
 */
inline void _hipSafeCall(hipError_t err, const char *file, const int line)
{
    if (err != hipSuccess)
    {
        std::cerr << "HIP failed at [" << file
                  << "] line: " << line
                  << " error: " << hipGetErrorString(err)
                  << std::endl;
        exit(-1);
    }
}

inline void _hipCheckError(const char *file, const int line)
{
    hipError_t err = hipGetLastError();

    _hipSafeCall(err, file, line);
}
#endif // HIP_ERRORS_HPP
