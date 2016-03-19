#ifndef MEMORYTRAVERSER_HPP
#define MEMORYTRAVERSER_HPP
#include <assert.h>
enum cuMemoryAddresMode
{
    cuAddressModeWrap = 0,
    cuAddressModeClamp = 1
};

enum cuMemoryFilterMode
{
    cuFilterModePoint = 0,
    cuFilterModeLinear = 1
};

enum COORDS_TYPE
{
    NON_NORMALIZED = 0,
    NORMALIZED = 1,
};

//AddresModeFunctors
template<enum COORDS_TYPE>
class Wrap
{
public:
    //This work only for forward wrapping if we move backward it fails.
    __device__ __host__
    float operator() (float x, float upper, float lower, float range) const
    {
        float r = upper - lower;
        return x - r * floorf(x/r);
    }
};

template<>
__device__ __host__
float Wrap<NON_NORMALIZED>::operator ()(float x, float upper, float lower, float range) const
{
    //assert(false); //is it wise to use wrapping on non normalized coords?;
    float  r = upper - lower;
    return x - r * floorf(x/r);

}

template<>
__device__ __host__
float Wrap<NORMALIZED>::operator ()(float x, float upper, float lower, float range) const
{
    //optimized with 1.f as r see general
    return range* (x - floorf(x));
}

template<enum COORDS_TYPE>
class Clamp
{
public:
    __device__ __host__ float operator() (float x, float upper, float lower, float range) const
    {
        //in case when we clamp to upper limit i have to return previous value which is upper - delta;
        float d = (upper - lower) / range;
        return fminf(upper-d, fmaxf(lower, x));
    }
};

template<>
__device__ __host__
float Clamp<NORMALIZED>::operator ()(float x, float upper, float lower, float range) const
{
    //TODO:: optimization to one line would be great!
    float d = 1.f / range;
    if (x < 0.f)
    {
        //obvious
        return 0.f;
    }
    //if larger than upper bound
    if (x > 1.f)
    {
        //return previous which is 1.f - delta;
        return (1.f-d)*range;
    }

    return x*range;
}

template<>
__device__ __host__
float Clamp<NON_NORMALIZED>::operator ()(float x, float upper, float lower, float range) const
{
    return fminf(range-1, fmaxf(lower, x));
}

__host__ __device__ inline float frac(float x)
{
    float frac, tmp = x - (float)(int)(x);
    float frac256 = (float)(int)( tmp*256.0f + 0.5f );
    frac = frac256 / 256.0f;
    return frac;
}

// MODE = true  - nearest pixel
// MODE = false - linear
enum FILTER_MODE
{
    NEAREST = 0,
    LINEAR = 1
};

template<enum FILTER_MODE>
class PixelFilter
{
public:
    __device__ __host__
    float operator() (float* data, float x, const int width) const
    {
        return data[(int)x];
    }

    __device__ __host__
    float operator() (float* data, float x, float y, const int width, const int height) const
    {
        return data[(int)(x + y*width)];
    }
};

//PixelMode specialization
template<>
__device__ __host__
float PixelFilter<NEAREST>::operator ()(float* data, float x, const int width) const
{
    return data [(int)floorf(x)];
}

template<>
__device__ __host__
float PixelFilter<NEAREST>::operator ()(float* data, float x, float y, const int width, const int height) const
{
    return data [(int)(floorf(x) + floorf(y)*width)];
}

//Linear interpolation
template<>
__device__ __host__
float PixelFilter<LINEAR>::operator() (float* data, float x, const int width) const
{
    float xb = fmaxf(0.f, x - 0.5f);

    float alpha = frac(xb);

    int i = floorf(xb);
    int ip = fminf(i+1, width-1);

    return (1.f - alpha)*data[i] + alpha * data[ip];
}

template<>
__device__ __host__
float PixelFilter<LINEAR>::operator ()(float* data, float x, float y, const int width, const int height) const
{
    float xb = x - 0.5f;
    float yb = y - 0.5f;

    xb =  fmaxf(0.f, xb);
    yb =  fmaxf(0.f, yb);

    float alpha = frac(xb);
    float beta  = frac(yb);

    int i = floorf(xb);
    int j = floorf(yb);

    int ip = fminf(i+1, width-1);
    int jp = fminf(j+1, height-1);


    return (1.f - alpha) * (1.f - beta)  * data[i  + width * j ] +
            alpha * (1.f - beta)          * data[ip + width * j ] +
            (1.f - alpha) * beta         * data[i  + width * jp] +
            alpha * beta                * data[ip + width * jp];
}


//This allows me to use new operator to use class directly on gpu;
//Otherwise do host device copy flow to put the class on GPU
//class Managed
//{
//public:
//    void *operator new(size_t len) {
//        void *ptr;
//        cudaMallocManaged(&ptr, len);
//        cudaDeviceSynchronize();
//        return ptr;
//    }
//
//    void operator delete(void *ptr) {
//        cudaDeviceSynchronize();
//        cudaFree(ptr);
//    }
//};

template <typename T, typename AddressingModeFunctor, typename FilteringFunctor>
class MemoryTraverser// : public Managed
{
public:

    __host__ __device__ T get1D(T *src, float x)
    {
        T i = addressingFunctor(x, width, 0, width);
        return filteringFunctor(src, i, width);
    }

    //TODO:: put the sizes as members
    __host__ __device__ T get2D(T *src, float x, float y)
    {
        T i = addressingFunctor(x, width, 0, width);
        T j = addressingFunctor(y, height, 0, height);

        return filteringFunctor(src, i, j, width, height);
    }


public:
    AddressingModeFunctor addressingFunctor;
    FilteringFunctor filteringFunctor;

    int width;
    int height;

    //not required
    cuMemoryAddresMode addressMode;
    cuMemoryFilterMode filterMode;
    int normalized;
};


//THIS IS SOMEHOW NOT WORKING during kernel call it throws illegal device address;
//template<typename TraverserType>
//inline
//TraverserType* create_memory_traverser(const int width, const int height = 0)
//{
//    TraverserType host_traverser;
//    host_traverser.width = width;
//    host_traverser.height = height;

//    TraverserType* device_traverser;

//    cudaSafeCall(cudaMalloc((void**)&device_traverser, sizeof(TraverserType)));
//    cudaSafeCall(cudaMemcpy(device_traverser, &host_traverser, sizeof(TraverserType), cudaMemcpyHostToDevice));
//    return device_traverser;
//}

// something like that could be used for kernel calls
// It may determine the proper type of the memoryTraverser
//template<typename Function, typename... Args>
//void Launcher(Args&&... args)
//{
//    return Function(std::forward<Args>(args)...);
//}

#endif // MEMORYTRAVERSER_HPP
