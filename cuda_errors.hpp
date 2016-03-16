#ifndef CUDA_ERRORS_HPP
#define CUDA_ERRORS_HPP
#include <cuda_runtime.h>
#include <iostream>

#define cudaSafeCall(err) _cudaSafeCall(err, __FILE__, __LINE__)
#define cudaCheckError()  _cudaCheckError(__FILE__, __LINE__)

/**
 * cuda version of error handling functions
 */
inline void _cudaSafeCall(cudaError err, const char *file, const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA failed at [" << file
                  << "] line: " << line
                  << " error: " << cudaGetErrorString(err)
                  << std::endl;
        exit(-1);
    }
}

inline void _cudaCheckError(const char *file, const int line)
{
    cudaError err = cudaGetLastError();

    _cudaSafeCall(err, file, line);
}
#endif // CUDA_ERRORS_HPP
