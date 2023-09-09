#include "tinycolormap.cuh"

#include "helper_math.h"
#include <cuda.h>


extern "C"
{
    __global__ void addKernel(int* c, const int* a, const int* b)
    {
        int i = threadIdx.x;
        c[i] = a[i] + b[i];
    }

    __global__ void render(uchar3 *outcolor, const unsigned int imageW, const unsigned int imageH,       // image
                        CUtexObject densityTexture, const unsigned int xRes, const unsigned int yRes) // 2D texture
    {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;
        int offset = x + y * blockDim.x * gridDim.x;

        if ((x >= imageW) || (y >= imageH))
            return;
        float u = ((float)x + 0.5f) / (float)imageW;
        float v = ((float)y + 0.5f) / (float)imageH;

        const tinycolormap::Color color = tinycolormap::GetColor(u, tinycolormap::ColormapType::Magma);

        outcolor[offset].x = color.ri();
        outcolor[offset].y = color.gi();
        outcolor[offset].z = color.bi();
    }
}
