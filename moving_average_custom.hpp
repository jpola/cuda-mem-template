#ifndef MOVING_AVERAGE_CUSTOM_HPP
#define MOVING_AVERAGE_CUSTOM_HPP
#include "memorytraverser.hpp"
#include "cuda_errors.hpp"

//Each new kernel have to be template to pass MemoryTraverser
//instead of using tex, we have to pass src as an additional param
template<typename TraverserType>
__global__ void moving_average_tr_kernel(float* dst, float* src, const int N, const int R,
                                         TraverserType* mt)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {

        float average = 0.f;

        for (int k = -R; k <= R; k++) {
            average = average + mt->get1D(src, (float)(tid - k + 0.5f)/(float)N, N);
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

    cudaSafeCall(cudaMalloc((void**)&d_dst, N * sizeof(float)));
    cudaSafeCall(cudaMalloc((void**)&d_src, N * sizeof(float)));

    cudaSafeCall(cudaMemcpy(d_src, src, N * sizeof(float), cudaMemcpyHostToDevice));

    TraverserType* gmt;

    //First approach is to allocate the data using shared unified memory
    //HIP does not support that
/*
    {
        gmt = new MemoryTraverser<float>;
        gmt->addressMode = cuAddressModeWrap;
        gmt->filterMode = cuFilterModePoint;
        gmt->normalized = 1;

        //with unified memory I can access the variables from host
        if (gmt->addressMode == cuAddressModeWrap)
            moving_average_tr_kernel<<<iDivUp(N, 256), 256>>>(d_dst, d_src, N, R, gmt, Wrap());
        else
            moving_average_tr_kernel<<<iDivUp(N, 256), 256>>>(d_dst, d_src, N, R, gmt, Clamp());

        delete gmt;
    }
*/

    // old plain way is to use host device copy;
    {
        // just explicit helpers
        cuMemoryAddresMode addresMode = cuAddressModeWrap;
        cuMemoryFilterMode filterMode = cuFilterModePoint;
        int normalized = false;

        // base on addres mode pick clamp or wrap
        // if normalized = true address mode takes the same value;

        TraverserType mt;
        //this is not necessary
        mt.addressMode = addresMode;
        mt.filterMode  = filterMode;
        mt.normalized = normalized;


        cudaSafeCall(cudaMalloc((void**)&gmt, sizeof(TraverserType)));
        cudaSafeCall(cudaMemcpy(gmt, &mt, sizeof(TraverserType), cudaMemcpyHostToDevice));

        moving_average_tr_kernel<<<iDivUp(N, 256), 256>>>(d_dst, d_src, N, R, gmt);
        cudaCheckError();
        cudaSafeCall(cudaDeviceSynchronize());

        cudaSafeCall(cudaFree(gmt));
    }

    cudaSafeCall(cudaMemcpy(dst, d_dst, N * sizeof(float), cudaMemcpyDeviceToHost));

    cudaSafeCall(cudaFree(d_dst));
    cudaSafeCall(cudaFree(d_src));

}

void moving_average_tr(float *dst, float *src, const int N, const int R,
                       cudaTextureFilterMode filterMode,
                       cudaTextureAddressMode addressMode,
                       int normalization)
{
    //This defines the behaviour
    using TraverserClampNormPixel = MemoryTraverser<float, Clamp<true>,  PixelFilter<true>>;
    using TraverserClampUNormPixel = MemoryTraverser<float, Clamp<false>, PixelFilter<true>>;

    using TraverserClampNormLinear = MemoryTraverser<float, Clamp<true>, PixelFilter<false>>;
    using TraverserClampUNormLinear = MemoryTraverser<float, Clamp<false>, PixelFilter<false>>;

    using TraverserWrapNormPixel = MemoryTraverser<float, Wrap<true>, PixelFilter<true>>;
    using TraverserWrapUNormPixel = MemoryTraverser<float, Wrap<false>, PixelFilter<true>>;

    using TraverserWrapNormLinear = MemoryTraverser<float, Wrap<true>, PixelFilter<false>>;
    using TraverserWrapUNormLinear = MemoryTraverser<float, Wrap<false>, PixelFilter<false>>;

    if(filterMode == cudaFilterModePoint)
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
        std::cerr << "LINEAR IS NOT SUPPORTED" << std::endl;
        exit(-1);

    }
}
#endif // MOVING_AVERAGE_CUSTOM_HPP
