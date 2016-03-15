#ifndef MEMORYTRAVERSER_HPP
#define MEMORYTRAVERSER_HPP
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
class Wrap
{
public:
    //This work only for forward wrapping if we move backward it fails.
    __device__ float operator() (float x, float upper, float lower) const
    {
        float r = upper - lower;
        return x - r * floorf(x/r);
    }
};

class Clamp
{
public:
    __device__ float operator() (float x, float upper, float lower) const
    {
        return fmaxf(lower, fminf(x, upper));
    }
};



//POD - class
template <typename T>
class MemoryTraverser
{

public:
    template<class AddresModeFunctor>
    __host__ __device__ T get1D(T* src, float x, const int size, AddresModeFunctor AM)
    {
        //TODO: How to hide branching related to normalization

        T i;
       // if (normalized) printf("norm");
//        {
//            i = (T)AM(x, 1.f, 0.f)*size;
//        }
//        else
//        {
//            i = (T)AM(x, size, 0.f);
//        }
        i = (T)AM(x, 1.f, 0.f)*size;

        //printf("i %f\n", i);
        //Tutaj jeszcze filterMode;
        T value = src[(int)i];

        return value;
    }

public:
    cuMemoryAddresMode addressMode;
    cuMemoryFilterMode filterMode;
    int normalized;
};
#endif // MEMORYTRAVERSER_HPP
