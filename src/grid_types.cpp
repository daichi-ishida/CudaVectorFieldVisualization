#include "grid_types.h"

#include <helper_cuda_drvapi.h>


void CUarrayGrid::setup(const unsigned int _xRes, const unsigned int _yRes)
{
    xRes = _xRes;
    yRes = _yRes;

    CUDA_ARRAY3D_DESCRIPTOR desc = { 0 };
    desc.Format = CU_AD_FORMAT_FLOAT;
    desc.NumChannels = 1;
    desc.Width = xRes;
    desc.Height = yRes;
    desc.Depth = 0;
    checkCudaErrors(cuArray3DCreate(&cu_array, &desc));

    CUDA_RESOURCE_DESC ResDesc;
    memset(&ResDesc, 0, sizeof(CUDA_RESOURCE_DESC));
    ResDesc.resType = CU_RESOURCE_TYPE_ARRAY;
    ResDesc.res.array.hArray = cu_array;
    checkCudaErrors(cuSurfObjectCreate(&surfData, &ResDesc));
}

void CUarrayGrid::clear()
{
    checkCudaErrors(cuSurfObjectDestroy(surfData));
    checkCudaErrors(cuArrayDestroy(cu_array));
}


void CUdeviceVector::setup(const unsigned int _xRes, const unsigned int _yRes)
{
    xRes = _xRes;
    yRes = _yRes;

    checkCudaErrors(cuMemAlloc(&cu_ptr, sizeof(float) * xRes * yRes));
}

void CUdeviceVector::clear()
{
    checkCudaErrors(cuMemFree(cu_ptr));
}

void CUdeviceVector::memsetZero()
{
    checkCudaErrors(cuMemsetD32(cu_ptr, 0, xRes*yRes));
}

void CUdeviceVector::memcpyDtoH(std::vector<float>& data)
{
    checkCudaErrors(cuMemcpyDtoH(data.data(), cu_ptr, sizeof(float)*data.size() ));
}

float CUdeviceVector::getValue(const unsigned int index)
{
    float value;
    checkCudaErrors(cuMemcpyDtoH(&value, cu_ptr+sizeof(float)*index, sizeof(float)));
    return value;
}

float* CUdeviceVector::getPtr()
{
    float* ptr = (float *)(uintptr_t)cu_ptr;
    return ptr;
}

void CUarrayGridVector::setup(const unsigned int _xRes, const unsigned int _yRes)
{
    xRes = _xRes;
    yRes = _yRes;

    x.setup(xRes, yRes);
    y.setup(xRes, yRes);
}

void CUarrayGridVector::clear()
{
    x.clear();
    y.clear();
}


void CUarrayStaggeredGrid::setup(const unsigned int _xRes, const unsigned int _yRes)
{
    xRes = _xRes;
    yRes = _yRes;

    x.setup(xRes+1, yRes);
    y.setup(xRes, yRes+1);
}

void CUarrayStaggeredGrid::clear()
{
    x.clear();
    y.clear();
}


void CUarrayCharGrid::setup(const unsigned int _xRes, const unsigned int _yRes)
{
    xRes = _xRes;
    yRes = _yRes;

    CUDA_ARRAY3D_DESCRIPTOR desc = { 0 };
    desc.Format = CU_AD_FORMAT_SIGNED_INT8;
    desc.NumChannels = 1;
    desc.Width = xRes;
    desc.Height = yRes;
    desc.Depth = 0;
    checkCudaErrors(cuArray3DCreate(&cu_array, &desc));

    CUDA_RESOURCE_DESC ResDesc;
    memset(&ResDesc, 0, sizeof(CUDA_RESOURCE_DESC));
    ResDesc.resType = CU_RESOURCE_TYPE_ARRAY;
    ResDesc.res.array.hArray = cu_array;
    checkCudaErrors(cuSurfObjectCreate(&surfData, &ResDesc));
}

void CUarrayCharGrid::clear()
{
    checkCudaErrors(cuSurfObjectDestroy(surfData));
    checkCudaErrors(cuArrayDestroy(cu_array));
}