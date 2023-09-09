#define GLAD_GL_IMPLEMENTATION
#include <glad/gl.h>

#include "window.h"
#include "constants.h"

void Window::create()
{
    if (glfwInit() == GLFW_FALSE)
    {
        fprintf(stderr, "Initialization failed!\n");
    }

    m_window.reset(glfwCreateWindow(WIN_WIDTH, WIN_HEIGHT, WIN_TITLE, NULL, NULL));

    if (m_window == nullptr)
    {
        fprintf(stderr, "Window creation failed!");
        glfwTerminate();
    }
    glfwMakeContextCurrent(m_window.get());

    // load OpenGL 3.x/4.x
    const int version = gladLoadGL(glfwGetProcAddress);
    if (version == 0) 
    {
        fprintf(stderr, "Failed to load OpenGL 3.x/4.x libraries!\n");
    }

    printf("Load OpenGL %d.%d\n", GLAD_VERSION_MAJOR(version), GLAD_VERSION_MINOR(version));
}

void Window::update()
{
    glfwSwapBuffers(m_window.get());
    glfwPollEvents();
}


bool Window::isClosing() const
{
    return glfwWindowShouldClose(m_window.get()) || glfwGetKey(m_window.get(), GLFW_KEY_ESCAPE);
}

void Window::close() const
{
    glfwTerminate();
}