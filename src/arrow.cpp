#include "arrow.h"
#include "cuFunctionManager.h"

#include <glad/gl.h>
#include <cudaGL.h>
#include <vector_types.h>

#include <helper_cuda_drvapi.h>


Arrow::Arrow()
{
}

void Arrow::clearAll()
{
    deleteCudaVBO(verticesVBO, cuda_vertices_vbo_resource);
    verticesVBO = NULL;
    cuda_vertices_vbo_resource = nullptr;

    deleteCudaVBO(normalsVBO, cuda_normals_vbo_resource);
    normalsVBO = NULL;
    cuda_normals_vbo_resource = nullptr;

    deleteCudaVBO(colorsVBO, cuda_colors_vbo_resource);
    colorsVBO = NULL;
    cuda_colors_vbo_resource = nullptr;

    deleteCudaVBO(indicesVBO, cuda_indices_vbo_resource);
    indicesVBO = NULL;
    cuda_indices_vbo_resource = nullptr;
}


void Arrow::initialize()
{
    // resolution of plane Y
    resolution = 1024 * 1024;
    unsigned int totalGlyphsCount = resolution;
    unsigned int vertices_count = vertices_per_arrow * totalGlyphsCount;
    unsigned int faces_count = faces_per_arrow * totalGlyphsCount;

    createCudaVBO(verticesVBO, GL_ARRAY_BUFFER, vertices_count * sizeof(float3), cuda_vertices_vbo_resource);
    createCudaVBO(normalsVBO, GL_ARRAY_BUFFER, vertices_count * sizeof(float3), cuda_normals_vbo_resource);
    createCudaVBO(colorsVBO, GL_ARRAY_BUFFER, vertices_count * sizeof(float3), cuda_colors_vbo_resource);

    createCudaVBO(indicesVBO, GL_ELEMENT_ARRAY_BUFFER, 3 * faces_count * sizeof(uint3), cuda_indices_vbo_resource);

    // load kernel
    CuFunctionsManager::getInstance().loadModule("arrow_kernel");
    computeArrowFunc = CuFunctionsManager::getInstance().getCuFunction("arrow_kernel", "compute");

    arrow_length = 0.1f;
    planeY = 0.0f;
}

void Arrow::createCudaVBO(unsigned int& vbo, unsigned int target, unsigned int size, CUgraphicsResource& cudaResource)
{
    // create buffer object
    glGenBuffers(1, &vbo);
    glBindBuffer(target, vbo);

    // initialize buffer
    glBufferData(target, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(target, 0);

    checkCudaErrors(cuGraphicsGLRegisterBuffer(&cudaResource, vbo, CU_GRAPHICS_REGISTER_FLAGS_NONE));
}

void Arrow::deleteCudaVBO(unsigned int& vbo, CUgraphicsResource& cudaResource)
{
    if (vbo == 0) return;

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glDeleteBuffers(1, &vbo);

    if(cudaResource) return;
    checkCudaErrors(cuGraphicsUnregisterResource(cudaResource));

    vbo = 0;
}


void Arrow::update()
{
    size_t num_bytes;
    checkCudaErrors(cuGraphicsMapResources(1, &cuda_vertices_vbo_resource, 0));
    checkCudaErrors(cuGraphicsResourceGetMappedPointer(&vertices_ptr, &num_bytes, cuda_vertices_vbo_resource));

    checkCudaErrors(cuGraphicsMapResources(1, &cuda_normals_vbo_resource, 0));
    checkCudaErrors(cuGraphicsResourceGetMappedPointer(&normals_ptr, &num_bytes, cuda_normals_vbo_resource));

    checkCudaErrors(cuGraphicsMapResources(1, &cuda_colors_vbo_resource, 0));
    checkCudaErrors(cuGraphicsResourceGetMappedPointer(&colors_ptr, &num_bytes, cuda_colors_vbo_resource));

    checkCudaErrors(cuGraphicsMapResources(1, &cuda_indices_vbo_resource, 0));
    checkCudaErrors(cuGraphicsResourceGetMappedPointer(&indices_ptr, &num_bytes, cuda_indices_vbo_resource));

    unsigned int imageW = 32;
    unsigned int imageH = 32;

    // call CUDA kernel, writing results to PBO
    const dim3 blockDims(16, 16, 1);
    const dim3 gridDims( (imageW+blockDims.x-1) / blockDims.x, (imageH+blockDims.y-1) / blockDims.y);

    // run kernel
    void* kernelArgs[] = {
    &planeY, &arrow_length, &imageW, &imageH,  // input
    &vertices_ptr, &normals_ptr, &colors_ptr, &indices_ptr // output
    }; 

    checkCudaErrors(cuLaunchKernel(computeArrowFunc, gridDims.x, gridDims.y, gridDims.z, blockDims.x, blockDims.y, blockDims.z, 0, NULL, kernelArgs, 0));

    checkCudaErrors(cuGraphicsUnmapResources(1, &cuda_indices_vbo_resource, 0));
    checkCudaErrors(cuGraphicsUnmapResources(1, &cuda_colors_vbo_resource, 0));
    checkCudaErrors(cuGraphicsUnmapResources(1, &cuda_normals_vbo_resource, 0));
    checkCudaErrors(cuGraphicsUnmapResources(1, &cuda_vertices_vbo_resource, 0));
}

void Arrow::render()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // setup matrices for OpenGL/CUDA interop
    glViewport(0, 0, 1024, 1024);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    // glRotatef(-60.0f, 1.0, 0.0, 0.0);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    // glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glPolygonMode(GL_FRONT, GL_FILL);
    glEnable(GL_DEPTH_TEST);

    glBindBuffer(GL_ARRAY_BUFFER, verticesVBO);
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, NULL);

    glBindBuffer(GL_ARRAY_BUFFER, normalsVBO);
    glEnableClientState(GL_NORMAL_ARRAY);
    glNormalPointer(GL_FLOAT, 0, NULL);

    glBindBuffer(GL_ARRAY_BUFFER, colorsVBO);
    glEnableClientState(GL_COLOR_ARRAY);
    glColorPointer(3, GL_FLOAT, 0, NULL);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indicesVBO);

    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_LIGHTING);

    glDrawElements(GL_TRIANGLES, 3 * faces_per_arrow * resolution, GL_UNSIGNED_INT, NULL);

    glDisable(GL_COLOR_MATERIAL);
    glDisable(GL_LIGHTING);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);

    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
}