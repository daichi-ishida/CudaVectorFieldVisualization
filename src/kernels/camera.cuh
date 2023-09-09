#pragma once
#include <cuda.h>

#include "helper_math.h"

struct Ray
{
    float3 o;   // origin
    float3 d;   // direction
};

struct Camera
{
    __device__ Camera(const float r, float horizontalAngle, float verticalAngle, const float FoV)
    {
        // pi * Angle / 180.0f
        horizontalAngle = horizontalAngle / 180.0f;
        verticalAngle = verticalAngle / 180.0f;

        pos = make_float3(r * sinpif(verticalAngle) * cospif(horizontalAngle),
            r * sinpif(verticalAngle) * sinpif(horizontalAngle),
            r * cospif(verticalAngle));

        target = make_float3(0.0f, 0.0f, 0.0f);
        front = normalize(target - pos);
        right = make_float3(-sinpif(horizontalAngle), cospif(horizontalAngle), 0.0f);
        up = cross(right, front);
        invhalffov = 1.0f / tanf(FoV / 2.0f);
    }

    float3 pos, front, right, up, target;
    float invhalffov;

    __device__ Ray generateRay(int x, int y, unsigned int imageW, unsigned int imageH) const
    {
        float nx = ((float)x / (float)imageW) * 2.0f - 1.0f;
        float ny = ((float)y / (float)imageH) * 2.0f - 1.0f;

        // upper left corner
        Ray ray = { pos, normalize(invhalffov * front + nx * right - ny * up) };
        return ray;
    }
};