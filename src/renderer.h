#pragma once

#include "data.h"

#include <memory>
#include <vector>

#include <cuda.h>


class Renderer
{
public:
    Renderer(std::shared_ptr<gridData> _data_ptr);
    ~Renderer(){}

    void initialize();
    void clearAll();
    void render();
    void saveImage();

private:
    void setupCUtexObject();
    void setupGLtexObject();

    // color data
    CUdeviceptr outcolor_ptr;
    std::vector<unsigned char> h_image;

    // render kernel function
    CUfunction renderFunc;

    // for display
    unsigned int texture_object;            // OpenGL
    unsigned int pbo;                       // OpenGL
    CUgraphicsResource cuda_pbo_resource;   // CUDA

    // density texture for rendering
    CUtexObject cuda_densityTexture0;        // CUDA
    CUtexObject cuda_densityTexture1;        // CUDA

    // data
    std::shared_ptr<gridData> data_ptr;
};