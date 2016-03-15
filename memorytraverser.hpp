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

//AddresModeFunctors
template<bool NORMALIZED>
class Wrap
{
public:
    //This work only for forward wrapping if we move backward it fails.
    __device__ __host__
    float operator() (float x, float upper, float lower, float range) const
    {
        //printf("General WRAP");
        float r = upper - lower;
        return x - r * floorf(x/r);
    }
};

template<>
__device__ __host__
float Wrap<false>::operator ()(float x, float upper, float lower, float range) const
{
    assert(false); //is it wise to use wrapping on non normalized coords;
    float  r = upper - lower;
    return x - r * floorf(x/r);

}

template<>
__device__ __host__
float Wrap<true>::operator ()(float x, float upper, float lower, float range) const
{
    float  r = 1.f;
    return range* (x - r * floorf(x/r));
}

template<bool NORMALIZED>
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
float Clamp<true>::operator ()(float x, float upper, float lower, float range) const
{
    float d = 1.f / range;
    return range * fminf(1.f - d, fmaxf(0.f, x));
}

template<>
__device__ __host__
float Clamp<false>::operator ()(float x, float upper, float lower, float range) const
{
    float d = (upper - lower) / range;
    return fminf(upper-d, fmaxf(lower, x));
}


//This allows me to use new operator to use class directly on gpu;
//Otherwise do host device copy flow to put the class on GPU
class Managed
{
public:
    void *operator new(size_t len) {
        void *ptr;
        cudaMallocManaged(&ptr, len);
        cudaDeviceSynchronize();
        return ptr;
    }

    void operator delete(void *ptr) {
        cudaDeviceSynchronize();
        cudaFree(ptr);
    }
};

template <typename T, typename AddressingModeFunctor>
class MemoryTraverser : public Managed
{
public:

    __host__ __device__ T get1D(T *src, float x, const int size)
    {
        T i = addressingFunctor(x, size, 0, size);
        T v = src[(int)i];
        return v;
    }

public:
    AddressingModeFunctor addressingFunctor;
    cuMemoryAddresMode addressMode;
    cuMemoryFilterMode filterMode;
    int normalized;
};
#endif // MEMORYTRAVERSER_HPP
