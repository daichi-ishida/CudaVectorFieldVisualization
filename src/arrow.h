#pragma once

#include <cuda.h>


class Arrow
{
public:
    Arrow();
    ~Arrow() {}

    void initialize();
    void clearAll();
    void update();
    void render();

private:
    // data
    // std::shared_ptr<gridData> data_ptr;

    void createCudaVBO(unsigned int& vbo, unsigned int target, unsigned int size, CUgraphicsResource& cudaResource);
    void deleteCudaVBO(unsigned int& vbo, CUgraphicsResource& cudaResource);

    // kernel function
    CUfunction computeArrowFunc;

    float planeY;
    float arrow_length;
    unsigned int resolution;

    static const unsigned int vertices_per_arrow = 9;
    static const unsigned int faces_per_arrow = 6;

    CUdeviceptr vertices_ptr;
    unsigned int verticesVBO;
    CUgraphicsResource cuda_vertices_vbo_resource;

    CUdeviceptr normals_ptr;
    unsigned int normalsVBO;
    CUgraphicsResource cuda_normals_vbo_resource;

    CUdeviceptr colors_ptr;
    unsigned int colorsVBO;
    CUgraphicsResource cuda_colors_vbo_resource;

    CUdeviceptr indices_ptr;
    unsigned int indicesVBO;
    CUgraphicsResource cuda_indices_vbo_resource;
};