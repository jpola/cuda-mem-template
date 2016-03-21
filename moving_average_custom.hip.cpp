#include "hip_runtime.h"
#include "memorytraverser.hpp"
#include "hip_errors.hpp"


inline int iDivUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

//Each new kernel have to be template to pass MemoryTraverser
//instead of using tex, we have to pass src as an additional param
template<typename TraverserType>
__global__ void moving_average_tr_kernel(hipLaunchParm lp, float* dst, float* src, const int N, const int R,
                                         TraverserType* mt)
{
    const int tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;

    if (tid < N) {

        float average = 0.f;

        for (int k = -R; k <= R; k++) {
            average = average + mt->get1D(src, (float)(tid - k + 0.5f)/(float)N);
        }

        dst[tid] = average / (2.f * (float)R + 1.f);
    }
}

template<typename TraverserType>
void moving_average_tr_impl(float *dst, float *src, const int N, const int R)
{
    //prepare data on device
    float* d_dst;
    float* d_src;

    hipSafeCall(hipMalloc((void**)&d_dst, N * sizeof(float)));
    hipSafeCall(hipMalloc((void**)&d_src, N * sizeof(float)));

    hipSafeCall(hipMemcpy(d_src, src, N * sizeof(float), hipMemcpyHostToDevice));

    TraverserType* gmt;
    // old plain way is to use host device copy;
    {
        TraverserType mt;
        mt.width = N;

        hipSafeCall(hipMalloc((void**)&gmt, sizeof(TraverserType)));
        hipSafeCall(hipMemcpy(gmt, &mt, sizeof(TraverserType), hipMemcpyHostToDevice));

        hipLaunchKernel(HIP_KERNEL_NAME(moving_average_tr_kernel), dim3(iDivUp(N,256)), dim3(256), 0, 0,
                        d_dst, d_src, N, R, gmt);
        hipCheckError();
        hipSafeCall(hipDeviceSynchronize());

        hipSafeCall(hipFree(gmt));
    }

    hipSafeCall(hipMemcpy(dst, d_dst, N * sizeof(float), hipMemcpyDeviceToHost));

    hipSafeCall(hipFree(d_dst));
    hipSafeCall(hipFree(d_src));

}

void moving_average_tr(float *dst, float *src, const int N, const int R,
                       cudaTextureFilterMode filterMode,
                       cudaTextureAddressMode addressMode,
                       int normalization)
{
    //This defines the behaviour
    using TraverserClampNormPixel = MemoryTraverser<float, Clamp<NORMALIZED, float>,  PixelFilter<NEAREST, float>>;
    using TraverserClampUNormPixel = MemoryTraverser<float, Clamp<NON_NORMALIZED, float>, PixelFilter<NEAREST, float>>;

    using TraverserClampNormLinear = MemoryTraverser<float, Clamp<NORMALIZED, float>, PixelFilter<LINEAR, float>>;
    using TraverserClampUNormLinear = MemoryTraverser<float, Clamp<NON_NORMALIZED, float>, PixelFilter<LINEAR, float>>;

    using TraverserWrapNormPixel = MemoryTraverser<float, Wrap<NORMALIZED, float>, PixelFilter<NEAREST, float>>;
    using TraverserWrapUNormPixel = MemoryTraverser<float, Wrap<NON_NORMALIZED, float>, PixelFilter<NEAREST, float>>;

    using TraverserWrapNormLinear = MemoryTraverser<float, Wrap<NORMALIZED, float>, PixelFilter<LINEAR, float>>;
    using TraverserWrapUNormLinear = MemoryTraverser<float, Wrap<NON_NORMALIZED, float>, PixelFilter<LINEAR, float>>;

    if(filterMode == hipFilterModePoint)
    {
        if (addressMode == cudaAddressModeWrap)
        {
            if(normalization)
            {
                moving_average_tr_impl<TraverserWrapNormPixel>(dst, src, N, R);
            }
            else
            {
                moving_average_tr_impl<TraverserWrapUNormPixel>(dst, src, N, R);
            }
        }
        else //clamp
        {
            if(normalization)
            {
                moving_average_tr_impl<TraverserClampNormPixel>(dst, src, N, R);
            }
            else
            {
                moving_average_tr_impl<TraverserClampUNormPixel>(dst, src, N, R);
            }
        }
    }
    else //Linear interpolation
    {
        if (addressMode == cudaAddressModeWrap)
        {
            if(normalization)
            {
                moving_average_tr_impl<TraverserWrapNormLinear>(dst, src, N, R);
            }
            else
            {
                moving_average_tr_impl<TraverserWrapUNormLinear>(dst, src, N, R);
            }
        }
        else //clamp
        {
            if(normalization)
            {
                moving_average_tr_impl<TraverserClampNormLinear>(dst, src, N, R);
            }
            else
            {
                moving_average_tr_impl<TraverserClampUNormLinear>(dst, src, N, R);
            }
        }
    }
}
