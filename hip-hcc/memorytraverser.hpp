#ifndef MEMORYTRAVERSER_HPP
#define MEMORYTRAVERSER_HPP
#include <hip_runtime.h>
#include <assert.h>

enum COORDS_TYPE
{
    NON_NORMALIZED = 0,
    NORMALIZED = 1,
};

//TODO:: We will probably need specialize that with integer types;
// There is no partial member specialization. We'll have to
// create partially specialized structs and specialize the members

/*
template<enum COORDS_TYPE, typename T>
struct Warp
{
    __device__ host__
    T operator() (T x, T upper, T lower, T range) const;
};

template<enum COORDS_TYPE>
struct Warp<COORDS_TYPE, int>
{
    __device__ host__
    int operator() (int x, int upper, int lower, int range) const;
}
//instantiation here and the definiton in hip.cpp file;
template<>
__device__ __host__
int Wrap<NON_NORMALIZED>::operator ()(int x, int upper, int lower, int range) const;
*/


//AddresModeFunctors
template<enum COORDS_TYPE>
struct Wrap
{
public:
    //This work only for forward wrapping if we move backward it fails.
    __device__ __host__
    float operator() (float x, float upper, float lower, float range) const;
};

//Fully specialized templates have to be instantiated;
template<>
__device__ __host__
float Wrap<NON_NORMALIZED>::operator ()(float x, float upper, float lower, float range) const;

template<>
__device__ __host__
float Wrap<NORMALIZED>::operator ()(float x, float upper, float lower, float range) const;


template<enum COORDS_TYPE>
struct Clamp
{
public:
    __device__ __host__ float operator() (float x, float upper, float lower, float range) const;
};

template<>
__device__ __host__
float Clamp<NORMALIZED>::operator ()(float x, float upper, float lower, float range) const;

template<>
__device__ __host__
float Clamp<NON_NORMALIZED>::operator ()(float x, float upper, float lower, float range) const;



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
    float operator() (float* data, float x, const int width) const;

    __device__ __host__
    float operator() (float* data, float x, float y, const int width, const int height) const;
};

//PixelMode specialization
template<>
__device__ __host__
float PixelFilter<NEAREST>::operator ()(float* data, float x, const int width) const;


template<>
__device__ __host__
float PixelFilter<NEAREST>::operator ()(float* data, float x, float y, const int width, const int height) const;

//Linear interpolation
template<>
__device__ __host__
float PixelFilter<LINEAR>::operator() (float* data, float x, const int width) const;


template<>
__device__ __host__
float PixelFilter<LINEAR>::operator ()(float* data, float x, float y, const int width, const int height) const;



template <typename T, typename AddressingModeFunctor, typename FilteringFunctor>
struct MemoryTraverser// : public Managed
{
    __host__ __device__ T get1D(T *src, float x)
    {
        T i = addressingFunctor(x, width, 0, width);
        return filteringFunctor(src, i, width);
    }

    __host__ __device__ T get2D(T *src, float x, float y)
    {
        T i = addressingFunctor(x, width, 0, width);
        T j = addressingFunctor(y, height, 0, height);

        return filteringFunctor(src, i, j, width, height);
    }

    AddressingModeFunctor addressingFunctor;
    FilteringFunctor filteringFunctor;

    int width;
    int height;

};

#endif // MEMORYTRAVERSER_HPP
