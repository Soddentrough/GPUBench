#include "VulkanContext.h"
#include "GuiApp.h"
#include <implot.h>
#include <SDL3/SDL.h>

#include <iostream>
#include <string>
#include <sstream>

int main(int argc, char** argv) {
    int maxFrames = -1;
    std::vector<int> preselectedDevices;
    std::string preselectedBackend = "";
    std::string preselectedTest = "";
    bool autoRun = false;
    bool exitOnComplete = false;
    float uiScaleOverride = 0.0f;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--version" || arg == "-v") {
            std::cout << "GPUBench GUI v1.0.0" << std::endl;
            return 0;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: gpubench-gui [options]\n\n"
                      << "Options:\n"
                      << "  -d, --device <index>    Pre-select GPU index (e.g. 0, 1, 0,1, or dual/all)\n"
                      << "  -b, --backend <name>    Pre-select backend (vulkan, rocm, opencl)\n"
                      << "  -t, --test <name>       Pre-select a specific benchmark (e.g. FP32)\n"
                      << "      --auto-run          Automatically start benchmarks upon launch\n"
                      << "      --exit-on-complete  Exit automatically when benchmark run finishes\n"
                      << "      --frames <count>    Run for specified number of frames and exit\n"
                      << "      --ui-scale <scale>  UI scaling factor (default: auto-detected from DPI, e.g. 1.0, 1.5, 2.0, 2.5)\n"
                      << "  -h, --help              Display this help message\n"
                      << "  -v, --version           Display version information\n\n";
            return 0;
        } else if ((arg == "-d" || arg == "--device") && i + 1 < argc) {
            std::string dStr = argv[++i];
            if (dStr == "all" || dStr == "dual" || dStr == "0,1") {
                preselectedDevices = {0, 1};
            } else {
                try {
                    preselectedDevices.push_back(static_cast<uint32_t>(std::stoi(dStr)));
                } catch (...) {}
            }
        } else if ((arg == "-b" || arg == "--backend") && i + 1 < argc) {
            preselectedBackend = argv[++i];
        } else if ((arg == "-t" || arg == "--test") && i + 1 < argc) {
            preselectedTest = argv[++i];
        } else if (arg == "--auto-run") {
            autoRun = true;
        } else if (arg == "--exit-on-complete") {
            exitOnComplete = true;
        } else if (arg == "--frames" && i + 1 < argc) {
            maxFrames = std::stoi(argv[++i]);
        } else if ((arg == "--ui-scale" || arg == "--scale") && i + 1 < argc) {
            try {
                uiScaleOverride = std::stof(argv[++i]);
            } catch (...) {}
        }
    }

    gpubench::gui::VulkanContext vulkanContext;
    if (!vulkanContext.init("GPUBench v1.0.0 - Workstation GPU Profiler", 1480, 1180, uiScaleOverride)) {
        std::cerr << "Failed to initialize Vulkan GUI context!" << std::endl;
        return 1;
    }
    float effectiveScale = vulkanContext.getDisplayScale();
    std::cout << "[GPUBench GUI] Initialized. Video driver: " << (SDL_GetCurrentVideoDriver() ? SDL_GetCurrentVideoDriver() : "null")
              << " | Display scale: " << effectiveScale << "x" << std::endl;

    // Initialize ImPlot Context
    ImPlot::CreateContext();

    gpubench::gui::GuiApp app;
    app.init(effectiveScale);
    if (!preselectedDevices.empty()) {
        app.setSelectedDevices(preselectedDevices);
    }
    if (!preselectedBackend.empty()) {
        app.setSelectedBackend(preselectedBackend);
    }
    if (!preselectedTest.empty()) {
        app.selectOnlyBenchmark(preselectedTest);
    }
    if (autoRun) {
        app.startBenchmarks();
    }

    int frameCount = 0;
    bool running = true;
    while (running && !app.shouldQuit()) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            ImGui_ImplSDL3_ProcessEvent(&event);
            if (event.type == SDL_EVENT_QUIT) {
                running = false;
            }
            if (event.type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED) {
                vulkanContext.resize(event.window.data1, event.window.data2);
            }
        }

        vulkanContext.beginFrame();
        app.updateAndRender();
        vulkanContext.endFrame();

        if (exitOnComplete && app.getExecutionState() == gpubench::gui::ExecutionState::Completed) {
            running = false;
        }

        frameCount++;
        if (maxFrames > 0 && frameCount >= maxFrames) {
            running = false;
        }
    }

    ImPlot::DestroyContext();
    vulkanContext.shutdown();

    return 0;
}
