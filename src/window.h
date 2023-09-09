#pragma once

#include <GLFW/glfw3.h>

#include <memory>

struct DestroyglfwWin
{
    void operator()(GLFWwindow* window)
    {
         glfwDestroyWindow(window);
    }
};

class Window
{
public:
    Window(){}
    ~Window(){}

    void create();
    void update();

    bool isClosing() const;
    void close() const;

private:
    std::unique_ptr<GLFWwindow, DestroyglfwWin> m_window;
};
