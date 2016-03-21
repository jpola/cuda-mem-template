#ifndef MEMORYTRAVERSER_HPP
#define MEMORYTRAVERSER_HPP
#include <assert.h>

enum COORDS_TYPE
{
    NON_NORMALIZED = 0,
    NORMALIZED = 1,
};

//PixelMode specialization
//returns fractional part of x according to 8 bits of fractional value
template<typename T>
__host__ __device__ inline T frac(T x)
{
    float frac, tmp = x - (float)(int)(x);
    float frac256 = (float)(int)( tmp*256.0f + 0.5f );
    frac = frac256 / 256.0f;
    return frac;
}

//TODO:: We will probably need specialize that with integer types;
// There is no partial member specialization. We'll have to
// create partially specialized structs and specialize the members

// Integer specialization could be done by SFINAE and overloading
// It have to be template partially specialized. Otherwise the
// fully specialized template is an function which need to be
// defined in cpp file. This require to use -dc flag during compilation
// and destroys the other cuda code.

//Wrap functor for addressing mode
template<enum COORDS_TYPE, typename T>
struct Wrap
{
    __device__ __host__
    T operator() (T x, T upper, T lower, T range) const;
};

template<typename T>
struct Wrap<NON_NORMALIZED, T>
{
    __device__ __host__
    T operator() (T x, T upper, T lower, T range) const
    {
        //assert(false); //is it wise to use wrapping on non normalized coords?;
        T  r = upper - lower;
        return x - r * floorf(x/r);
    }
};

template<typename T>
struct Wrap<NORMALIZED, T>
{
    __device__ __host__
    T operator() (T x, T upper, T lower, T range) const
    {
        //optimized with 1.f as r see general
        return range* (x - floorf(x));
    }
};

//Calmp functor for addressing mode
template<enum COORDS_TYPE, typename T>
struct Clamp
{
public:
    __device__ __host__ T operator() (T x, T upper, T lower, T range) const;
};

template<typename T>
struct Clamp<NORMALIZED, T>
{
public:
    __device__ __host__ T operator() (T x, T upper, T lower, T range) const
    {
        //TODO:: optimization to one line would be great!
        T d = (T)1 / range;
        if (x < (T)0)
        {
            //obvious
            return (T)0;
        }
        //if larger than upper bound
        if (x > (T)1)
        {
            //return previous which is 1.f - delta;
            return ((T)1 - d)*range;
        }

        return x*range;
    }
};

template<typename T>
struct Clamp<NON_NORMALIZED, T>
{
public:
    __device__ __host__ T operator() (T x, T upper, T lower, T range) const
    {
        return fminf(range-(T)1, fmaxf(lower, x));
    }
};


//Pixel filtering mode
enum FILTER_MODE
{
    NEAREST = 0,
    LINEAR = 1
};

//Pixel filtering functor
template<enum FILTER_MODE, typename T>
class PixelFilter
{
public:
    __device__ __host__
    T operator() (T* data, T x, const int width) const;

    __device__ __host__
    T operator() (T* data, T x, T y, const int width, const int height) const;
};

template<typename T>
class PixelFilter<NEAREST, T>
{
public:
    __device__ __host__
    T operator() (T* data, T x, const int width) const
    {
        return data [(int)floorf(x)];
    }

    __device__ __host__
    T operator() (T* data, T x, T y, const int width, const int height) const
    {
        return data [(int)(floorf(x) + floorf(y)*width)];
    }
};

template<typename T>
class PixelFilter<LINEAR, T>
{
public:
    __device__ __host__
    T operator() (T* data, T x, const int width) const
    {
        float xb = fmaxf(0.f, x - 0.5f);

        float alpha = frac(xb);

        int i = floorf(xb);
        int ip = fminf(i+1, width-1);

        return (1.f - alpha)*data[i] + alpha * data[ip];
    }

    __device__ __host__
    T operator() (T* data, T x, T y, const int width, const int height) const
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
                alpha * (1.f - beta)         * data[ip + width * j ] +
                (1.f - alpha) * beta         * data[i  + width * jp] +
                alpha * beta                 * data[ip + width * jp];
    }
};

// Memory Traverser, substitutes cuda texture capabilities
// requires appropriate filter and addressing mode to avoid if statements
// during execution of getxD function
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
