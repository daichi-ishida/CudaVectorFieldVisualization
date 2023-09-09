#include "renderer.h"
#include "constants.h"
#include "cuFunctionManager.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <glad/gl.h>
#include <cudaGL.h>
#include <vector_types.h>

#include <helper_cuda_drvapi.h>


Renderer::Renderer(std::shared_ptr<gridData> _data_ptr) : data_ptr(_data_ptr)
{
    // create pixel buffer object
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, WIN_WIDTH * WIN_HEIGHT * 3 * sizeof(GLubyte), 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // register this buffer object with CUDA
    checkCudaErrors(cuGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD));

    // load kernel
    CuFunctionsManager::getInstance().loadModule("render_kernel");
    renderFunc = CuFunctionsManager::getInstance().getCuFunction("render_kernel", "render");

    // setup OpenGL texture object for display
    setupGLtexObject();

    // setup CUDA texture object for kernel
    setupCUtexObject();
}

void Renderer::clearAll()
{
    // destroy CUDA texture object for kernel
    if(cuda_densityTexture0)
    {
        checkCudaErrors(cuTexObjectDestroy(cuda_densityTexture0));
    }
    if(cuda_densityTexture1)
    {
        checkCudaErrors(cuTexObjectDestroy(cuda_densityTexture1));
    }

    // destroy OpenGL texture object for display
    if(texture_object)
    {
        glDeleteTextures(1, &texture_object);
    }

    // unregister pixel buffer object from CUDA
    if(cuda_pbo_resource)
    {
        cuGraphicsUnregisterResource(cuda_pbo_resource);
    }
    if(pbo)
    {
        glDeleteBuffers(1, &pbo);
    }
}

void Renderer::render()
{
    unsigned int width = WIN_WIDTH;
    unsigned int height = WIN_HEIGHT;

    unsigned int xRes = data_ptr->xRes;
    unsigned int yRes = data_ptr->yRes;

    // map PBO to get CUDA device pointer
    checkCudaErrors(cuGraphicsMapResources(1, &cuda_pbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cuGraphicsResourceGetMappedPointer(&outcolor_ptr, &num_bytes, cuda_pbo_resource));


    // call CUDA kernel, writing results to PBO
    const dim3 blockDims(16, 16, 1);
    const dim3 gridDims( (WIN_WIDTH+blockDims.x-1) / blockDims.x, (WIN_HEIGHT+blockDims.y-1) / blockDims.y);

    CUtexObject* tex_ptr;
    if(data_ptr->renderGrid)
    {
        tex_ptr = &cuda_densityTexture1;
    }
    else
    {
        tex_ptr = &cuda_densityTexture0;
    }
    
    void* kernelArgs[] = { 
        &outcolor_ptr, &width, &height,             // image
        tex_ptr, &xRes, &yRes}; // 2D texture
    checkCudaErrors(cuLaunchKernel(renderFunc, gridDims.x, gridDims.y, gridDims.z, blockDims.x, blockDims.y, blockDims.z, 0, NULL, kernelArgs, 0));

    checkCudaErrors(cuGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

    // display results
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_2D);

    // setup matrices for OpenGL/CUDA interop
    glViewport(0, 0, WIN_WIDTH, WIN_HEIGHT);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    // draw image from PBO
    glBindTexture(GL_TEXTURE_2D, texture_object);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIN_WIDTH, WIN_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // draw textured quad
    glBegin(GL_QUADS);
    {
        glTexCoord2f(0.0f, 0.0f); glVertex2f(0.0f, 0.0f);
        glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, 0.0f);
        glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, 1.0f);
        glTexCoord2f(0.0f, 1.0f); glVertex2f(0.0f, 1.0f);
    }
    glEnd();
    glDisable(GL_TEXTURE_2D);

    glBindTexture(GL_TEXTURE_2D, 0);
}

void Renderer::saveImage()
{
    h_image.resize(WIN_WIDTH * WIN_HEIGHT * 3);
    checkCudaErrors(cuMemcpyDtoH(h_image.data(), outcolor_ptr, WIN_WIDTH * WIN_HEIGHT * 3 * sizeof(GLubyte)));
    static int count = 0;
    char filename[1024];

    snprintf(filename, sizeof(filename), "img/img%04d.png", count++);
    unsigned char* data = reinterpret_cast<unsigned char*>(h_image.data());

    int saved = stbi_write_png(filename, WIN_WIDTH, WIN_HEIGHT, 3, data, 0);  // 3 components (R, G, B)
}

void Renderer::setupCUtexObject()
{
    CUDA_RESOURCE_DESC ResDesc;
    memset(&ResDesc, 0, sizeof(CUDA_RESOURCE_DESC));
    ResDesc.resType = CU_RESOURCE_TYPE_ARRAY;
    ResDesc.res.array.hArray = data_ptr->density0.cu_array;
    
    CUDA_TEXTURE_DESC TexDesc;
    memset(&TexDesc, 0, sizeof(CUDA_TEXTURE_DESC));
    TexDesc.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
    TexDesc.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
    TexDesc.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
    TexDesc.filterMode = CU_TR_FILTER_MODE_LINEAR;

    checkCudaErrors(cuTexObjectCreate(&cuda_densityTexture0, &ResDesc, &TexDesc, NULL));

    ResDesc.res.array.hArray = data_ptr->density1.cu_array;
    checkCudaErrors(cuTexObjectCreate(&cuda_densityTexture1, &ResDesc, &TexDesc, NULL));

}

void Renderer::setupGLtexObject()
{
    // create texture for display
    glGenTextures(1, &texture_object);
    glBindTexture(GL_TEXTURE_2D, texture_object);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, WIN_WIDTH, WIN_HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);
}