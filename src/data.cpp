#include "data.h"

#include <utility>
#include <sstream>
#include <iomanip>

#include "constants.h"
#include <helper_cuda_drvapi.h>


void gridData::configure()
{
    xRes = xRES;
    yRes = yRES;
    dx = DX;
    dt = DT;

    density0.setup(xRes, yRes);
    density1.setup(xRes, yRes);
}

void gridData::clearAll()
{
    density0.clear();
    density1.clear();
}


void gridData::swapScalarGrid()
{
    swapGrid(density0, density1);
}


void gridData::swapRenderGrid()
{
    renderGrid = !renderGrid;
}

// --------------------------------------------------------- memcpy Host <-> CUarray ---------------------------------------------------------

void gridData::memcpyHostToCUarray(const std::vector<float>& vector, CUarray& cu_array, const unsigned int _width, const unsigned int _height, const unsigned int _depth)
{
    CUDA_MEMCPY3D copyParam = { 0 };
    copyParam.srcMemoryType = CU_MEMORYTYPE_HOST;
    copyParam.srcHost = vector.data();
    copyParam.srcPitch = _width * sizeof(float);
    copyParam.srcHeight = _height;
    copyParam.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copyParam.dstArray = cu_array;
    copyParam.WidthInBytes = _width * sizeof(float);
    copyParam.Height = _height;
    copyParam.Depth = _depth;
    checkCudaErrors(cuMemcpy3D(&copyParam));
}

void gridData::memcpyCUarrayToHost(const CUarray& cu_array, std::vector<float>& vector, const unsigned int _width, const unsigned int _height, const unsigned int _depth)
{
    CUDA_MEMCPY3D copyParam = { 0 };
    copyParam.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    copyParam.srcArray = cu_array;
    copyParam.dstMemoryType = CU_MEMORYTYPE_HOST;
    copyParam.dstHost = vector.data();
    copyParam.dstPitch = _width * sizeof(float);
    copyParam.dstHeight = _height;
    copyParam.WidthInBytes = _width * sizeof(float);
    copyParam.Height = _height;
    copyParam.Depth = _depth;
    checkCudaErrors(cuMemcpy3D(&copyParam));
}

void gridData::swapGrid(CUarrayGrid& grid1, CUarrayGrid& grid2)
{
    using std::swap;
    swap(grid1, grid2);
}
