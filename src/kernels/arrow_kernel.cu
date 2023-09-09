#include "camera.cuh"
#include "tinycolormap.cuh"

#include "helper_math.h"
#include <cuda.h>


extern "C"
{
    __global__ void compute(const float planeY, const float arrowLength,
    const unsigned int imageW, const unsigned int imageH, 
    float3* outVertices, float3* outVertexNormals, float3* outVertexColors, uint3* outFaces)
    {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;
        int offset = x + y * blockDim.x * gridDim.x;

        if ((x >= imageW) || (y >= imageH)) return;

        float u = ((float)x + 0.5f) / (float)imageW;
        float v = ((float)y + 0.5f) / (float)imageH;

        float3 position = make_float3(u, v, 0.0f);
        float3 origin = make_float3(0.5f, 0.5f, 0.0f);

        float3 vector = position - origin;
        float magnitude = length(vector) / 0.707f;

        float3 forward = normalize(vector);
        float3 xAxis = normalize(make_float3(-forward.y, forward.x, 0.0f));
        float3 yAxis = normalize(cross(forward, xAxis));

        unsigned int face_id = offset * 6;
        unsigned int vertex_id = offset * 9;

        // pyramid - side
        outFaces[face_id] = make_uint3(vertex_id, vertex_id + 1, vertex_id + 2);
        outFaces[face_id + 1] = make_uint3(vertex_id, vertex_id + 2, vertex_id + 3);
        outFaces[face_id + 2] = make_uint3(vertex_id, vertex_id + 3, vertex_id + 4);
        outFaces[face_id + 3] = make_uint3(vertex_id, vertex_id + 4, vertex_id + 1);

        // pyramid - bottom
        outFaces[face_id + 4] = make_uint3(vertex_id + 5, vertex_id + 6, vertex_id +7);
        outFaces[face_id + 5] = make_uint3(vertex_id + 5, vertex_id + 7, vertex_id + 8);

        outVertexNormals[vertex_id] = forward;
        outVertexNormals[vertex_id + 1] = xAxis;
        outVertexNormals[vertex_id + 2] = yAxis;
        outVertexNormals[vertex_id + 3] = -xAxis;
        outVertexNormals[vertex_id + 4] = -yAxis;

        outVertexNormals[vertex_id + 5] = -forward;
        outVertexNormals[vertex_id + 6] = -forward;
        outVertexNormals[vertex_id + 7] = -forward;
        outVertexNormals[vertex_id + 8] = -forward;

        outVertices[vertex_id] = position + forward * arrowLength;
        outVertices[vertex_id + 1] = position + xAxis * arrowLength * 0.05f;
        outVertices[vertex_id + 2] = position + yAxis * arrowLength * 0.05f;
        outVertices[vertex_id + 3] = position - xAxis * arrowLength * 0.05f;
        outVertices[vertex_id + 4] = position - yAxis * arrowLength * 0.05f;
        outVertices[vertex_id + 5] = position + xAxis * arrowLength * 0.05f;
        outVertices[vertex_id + 6] = position + yAxis * arrowLength * 0.05f;
        outVertices[vertex_id + 7] = position - xAxis * arrowLength * 0.05f;
        outVertices[vertex_id + 8] = position - yAxis * arrowLength * 0.05f;

        const tinycolormap::Color color = tinycolormap::GetColor(magnitude, tinycolormap::ColormapType::Magma);
        float3 color3 = 2.0f * make_float3(color.r(), color.g(), color.b());
        outVertexColors[vertex_id] = color3;
        outVertexColors[vertex_id + 1] = color3;
        outVertexColors[vertex_id + 2] = color3;
        outVertexColors[vertex_id + 3] = color3;
        outVertexColors[vertex_id + 4] = color3;
        outVertexColors[vertex_id + 5] = color3;
        outVertexColors[vertex_id + 6] = color3;
        outVertexColors[vertex_id + 7] = color3;
        outVertexColors[vertex_id + 8] = color3;
    }
}