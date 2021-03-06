#ifndef MEMORYTRAVERSER_HPP
#define MEMORYTRAVERSER_HPP
#include <assert.h>


template<typename T>
__device__ __forceinline__ T ldg(const T* ptr) {
#if __CUDA_ARCH__ >= 350
    return __ldg(ptr);
#else
    return *ptr;
#endif
}

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
        return x*range;
    }
};

template<typename T>
struct Clamp<NON_NORMALIZED, T>
{
public:
    __device__ __host__ T operator() (T x, T upper, T lower, T range) const
    {
        return fmaxf(fminf(x, upper - 1), lower);
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
    T operator() (const T* __restrict__ data, T x, const int width) const;

    __device__ __host__
    T operator() (const T* __restrict__ data, T x, T y, const int width, const int height) const;
};

template<typename T>
class PixelFilter<NEAREST, T>
{
public:
    __device__ __host__
    T operator() ( const T* __restrict__ data, T x, const int width) const
    {
        return ldg(&data [(int)floorf(x)]);
    }

    __device__ __host__
    T operator() (const T* __restrict__ data, T x, T y, const int width, const int height) const
    {
        return ldg(&data[(int)(floorf(x) + floorf(y)*width)]);
    }
};

template<typename T>
class PixelFilter<LINEAR, T>
{
public:
    __device__ __host__
    T operator() (const T* __restrict__ data, T x, const int width) const
    {
        assert(false);
        return -1;
    }

    __device__ __host__
    T operator() (const T* __restrict__ data, T x, T y, const int width, const int height) const
    {
        assert(false);
        return -1;
    }
};

// Memory Traverser, substitutes cuda texture capabilities
// requires appropriate filter and addressing mode to avoid if statements
// during execution of getxD function
template <typename T, typename AddressingModeFunctor, typename FilteringFunctor>
struct MemoryTraverser// : public Managed
{
    __host__ __device__ T get1D(const T* __restrict__ src, float x)
    {
        x = addressingFunctor(x, width, 0, width);
        return filteringFunctor(src, x, width);
    }

    __host__ __device__ T get2D(const T* __restrict__ src, float x, float y)
    {
        x = addressingFunctor(x, width, 0, width);
        y = addressingFunctor(y, height, 0, height);

        return filteringFunctor(src, x, y, width, height);
    }

    AddressingModeFunctor addressingFunctor;
    FilteringFunctor filteringFunctor;

    int width;
    int height;
};

template <typename T>
struct MemoryTraverser<T, Clamp<NORMALIZED, T>, PixelFilter<NEAREST, T>>
{
    __host__ __device__ T get1D(const T* __restrict__ src, float x)
    {
        x *=width;

        int xi = (int) floorf(x);

        xi = max(min(xi, width-1), 0);

        return src[xi];
    }

    __host__ __device__ T get2D(const T* __restrict__ src, float x, float y)
    {
        x *= width;
        y *= height;

        int xi = (int) floorf(x);
        int yi = (int) floorf(y);

        xi = max(min(xi, width-1), 0);
        yi = max(min(yi, height-1), 0);

        return ldg(&src[xi + yi*width]);
    }

    Clamp<NORMALIZED, T> addressingFunctor;
    PixelFilter<NEAREST, T> filteringFunctor;

    int width;
    int height;
};

template <typename T>
struct MemoryTraverser<T, Wrap<NORMALIZED, T>, PixelFilter<LINEAR, T>>
{
    __host__ __device__ T get1D(const T* __restrict__ src, float x)
    {
        x = addressingFunctor(x, width, 0, width);

        x -= 0.5f;

        float alpha = frac(x);

        int i = floorf(x);

        if (i < 0)  { i += width; alpha += 1.f; }

        int ip = (i + 1) > width -  1 ? (i + 1) -  width : i + 1;

        return (1.f - alpha) * ldg(&src[i]) + alpha * ldg(&src[ip]);
    }

    __host__ __device__ T get2D(const T* __restrict__ src, float x, float y)
    {
        x = addressingFunctor(x, width, 0, width);
        y = addressingFunctor(y, height, 0, height);

        x -= 0.5f;
        y -= 0.5f;

        float alpha = frac(x);
        float beta  = frac(y);

        int i = floorf(x);
        int j = floorf(y);

        //wrap only
        if (i < 0)  { i += width; alpha += 1.f; }
        if (j < 0) { j += height; beta += 1.f; }

        //wrap;
        int ip = (i + 1) > width -  1 ? (i + 1) -  width : i + 1;
        int jp = (j + 1) > height - 1 ? (j + 1) - height : j + 1;


        return (1.f - alpha) * (1.f - beta)  * ldg(&src[i  + width * j ]) +
                alpha * (1.f - beta)         * ldg(&src[ip + width * j ]) +
                (1.f - alpha) * beta         * ldg(&src[i  + width * jp]) +
                alpha * beta                 * ldg(&src[ip + width * jp]);
    }

    Wrap<NORMALIZED, T> addressingFunctor;
    PixelFilter<LINEAR, T> filteringFunctor;

    int width;
    int height;
};


template <typename T>
struct MemoryTraverser<T, Clamp<NORMALIZED, T>, PixelFilter<LINEAR, T>>
{
    __host__ __device__ T get1D(const T* __restrict__ src, float x)
    {
        x = addressingFunctor(x, width, 0, width);

        x -= 0.5f;

        x = fmaxf(fminf(x, width  - 1), 0);

        float alpha = frac(x);

        int i = floorf(x);

        int ip = max(min(i + 1, width -1), 0);

        return (1.f - alpha) * ldg(&src[i]) + alpha * ldg(&src[ip]);
    }

    __host__ __device__ T get2D(const T* __restrict__ src, float x, float y)
    {
        x = addressingFunctor(x, width, 0, width);
        y = addressingFunctor(y, height, 0, height);
        x -= 0.5f;
        y -= 0.5f;

        //clamp only
        x = fmaxf(fminf(x, width  - 1), 0);
        y = fmaxf(fminf(y, height - 1), 0);

        float alpha = frac(x);
        float beta  = frac(y);

        int i = floorf(x);
        int j = floorf(y);

        //clamp only
        int ip = max(min(i + 1, width -1), 0);
        int jp = max(min(j + 1, height - 1),0);

        return (1.f - alpha) * (1.f - beta)  * ldg(&src[i  + width * j ]) +
                alpha * (1.f - beta)         * ldg(&src[ip + width * j ]) +
                (1.f - alpha) * beta         * ldg(&src[i  + width * jp]) +
                alpha * beta                 * ldg(&src[ip + width * jp]);
    }

    Clamp<NORMALIZED, T> addressingFunctor;
    PixelFilter<LINEAR, T> filteringFunctor;

    int width;
    int height;
};

template <typename T>
struct MemoryTraverser<T, Clamp<NON_NORMALIZED, T>, PixelFilter<LINEAR, T>>
{
    __host__ __device__ T get1D(const T* __restrict__ src, float x)
    {
        x -= 0.5f;

        x = fmaxf(fminf(x, width  - 1), 0);

        float alpha = frac(x);

        int i = floorf(x);

        int ip = max(min(i + 1, width -1), 0);

        return (1.f - alpha) * ldg(&src[i]) + alpha * ldg(&src[ip]);
    }

    __host__ __device__ T get2D(const T* __restrict__ src, float x, float y)
    {
        x -= 0.5f;
        y -= 0.5f;

        //clamp only
        x = fmaxf(fminf(x, width  - 1), 0);
        y = fmaxf(fminf(y, height - 1), 0);

        float alpha = frac(x);
        float beta  = frac(y);

        int i = floorf(x);
        int j = floorf(y);

        //clamp only
        int ip = max(min(i + 1, width -1), 0);
        int jp = max(min(j + 1, height - 1),0);

        return (1.f - alpha) * (1.f - beta)  * ldg(&src[i  + width * j ]) +
                alpha * (1.f - beta)         * ldg(&src[ip + width * j ]) +
                (1.f - alpha) * beta         * ldg(&src[i  + width * jp]) +
                alpha * beta                 * ldg(&src[ip + width * jp]);
    }

    Clamp<NON_NORMALIZED, T> addressingFunctor;
    PixelFilter<LINEAR, T> filteringFunctor;

    int width;
    int height;
};

#endif // MEMORYTRAVERSER_HPP
