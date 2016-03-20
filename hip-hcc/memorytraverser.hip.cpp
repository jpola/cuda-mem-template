#include <hip_runtime.h>
#include "memorytraverser.hpp"

//Wrap functor specialization
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


//Clamp functor specialization
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

//PixelMode specialization
//returns fractional part of x according to 8 bits of fractional value
__host__ __device__ inline float frac(float x)
{
    float frac, tmp = x - (float)(int)(x);
    float frac256 = (float)(int)( tmp*256.0f + 0.5f );
    frac = frac256 / 256.0f;
    return frac;
}

//Nearest neighbour 1D
template<>
__device__ __host__
float PixelFilter<NEAREST>::operator ()(float* data, float x, const int width) const
{
    return data [(int)floorf(x)];
}

//Nearest neighbour 2D
template<>
__device__ __host__
float PixelFilter<NEAREST>::operator ()(float* data, float x, float y, const int width, const int height) const
{
    return data [(int)(floorf(x) + floorf(y)*width)];
}

//Linear interpolation 1D
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

//Linear interpolation 2D
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

