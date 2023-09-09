#include "window.h"
#include "cuFunctionManager.h"
#include "data.h"
// #include "renderer.h"
#include "arrow.h"

#include <memory>

#include <cuda.h>

#include "helper_cuda_drvapi.h"


int main()
{
    CUdevice device = findCudaDeviceDRV(0, nullptr);

    Window window;
    window.create();

    CuFunctionsManager::create(device);

    // init & configure

    std::shared_ptr<gridData> data_ptr = std::make_shared<gridData>();
    data_ptr->configure();

    // Renderer renderer(data_ptr);
    Arrow arrow;

    arrow.initialize();
    arrow.update();
    arrow.render();
    window.update();

    // renderer.saveImage();

    int step = 0;

    while(!window.isClosing())
    {
        printf("\n=== STEP %d ===\n", step++);

        // update
        arrow.render();
        // renderer.saveImage();
        window.update();
    }

    // free all gpu memory
    arrow.clearAll();
    data_ptr->clearAll();

    CuFunctionsManager::destroy();

    window.close();

    return 0;
}