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
    __device__ float operator() (float x, float upper, float lower, float range) const
    {
        float r = upper - lower;
        return x - r * floorf(x/r);
    }
};

class Clamp
{
public:
    __device__ float operator() (float x, float upper, float lower, float range) const
    {
        //in case when we clamp to upper limit i have to return previous value which is upper - delta;
        float d = (upper - lower) / range;

        return fminf(upper-d, fmaxf(lower, x));
    }
};

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

//POD - class
template <typename T, class AddresModeFunctor>
class MemoryTraverser : public Managed
{

public:
    template<bool norm>
    __host__ __device__ T get1D(T* src, float x, const int size, AddresModeFunctor AM)
    {
        //TODO: How to hide branching related to normalization
        T i;
        if (normalized)
        {

            i = (T)AM(x, 1.f, 0.f, size)*size;

        }
        else
        {
            i = (T)AM(x, size, 0.f, size);
        }

        //Deal with the filtering mode here
        T value = src[(int)i];
        //printf("x %f, i %f v %f \n", x, i, value);
        return value;
    }

public:
    cuMemoryAddresMode addressMode;
    cuMemoryFilterMode filterMode;
    int normalized;
};
#endif // MEMORYTRAVERSER_HPP
