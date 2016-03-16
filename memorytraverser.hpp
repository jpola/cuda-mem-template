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
    //optimized with 1.f as r see general
    return range* (x - floorf(x));
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
float Clamp<false>::operator ()(float x, float upper, float lower, float range) const
{
    return fminf(range-1, fmaxf(lower, x));
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

        //point filtering
        i = floorf(i);

        T v = src[(int)i];
        return v;
    }

    //TODO:: put the sizes as members
    __host__ __device__ T get2D(T *src, float x, float y, const int x_size, const int y_size)
    {
        T i = addressingFunctor(x, x_size, 0, x_size);
        T j = addressingFunctor(y, y_size, 0, y_size);

        //This is point filtering;
        i = floorf(i);
        j = floorf(j);

        T v = src[(int)(i + x_size*j)];
        return v;
    }


public:
    AddressingModeFunctor addressingFunctor;
    cuMemoryAddresMode addressMode;
    cuMemoryFilterMode filterMode;
    int normalized;
};

// something like that could be used for kernel calls
// It may determine the proper type of the memoryTraverser
template<typename Function, typename... Args>
void Launcher(Args&&... args)
{
    return Function(std::forward<Args>(args)...);
}

#endif // MEMORYTRAVERSER_HPP
