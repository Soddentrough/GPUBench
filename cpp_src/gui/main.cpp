#include "VulkanContext.h"
#include "GuiApp.h"
#include <implot.h>
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#endif

#include <iostream>
#include <string>
#include <sstream>
#include <algorithm>
#include <cctype>

int main(int argc, char** argv) {
#ifdef _WIN32
    HANDLE hStdOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hStdOut != NULL && hStdOut != INVALID_HANDLE_VALUE && GetFileType(hStdOut) != FILE_TYPE_UNKNOWN) {
        int fd = _open_osfhandle(reinterpret_cast<intptr_t>(hStdOut), 0);
        if (fd != -1) {
            _dup2(fd, 1);
            _dup2(fd, 2);
        }
    } else if (AttachConsole(ATTACH_PARENT_PROCESS)) {
        HANDLE hOut = CreateFileA("CONOUT$", GENERIC_WRITE, FILE_SHARE_WRITE, NULL, OPEN_EXISTING, 0, NULL);
        if (hOut != INVALID_HANDLE_VALUE) {
            SetStdHandle(STD_OUTPUT_HANDLE, hOut);
            SetStdHandle(STD_ERROR_HANDLE, hOut);
        }
        FILE* fp = nullptr;
        freopen_s(&fp, "CONOUT$", "w", stdout);
        freopen_s(&fp, "CONOUT$", "w", stderr);
    }
    std::cout.clear();
    std::cerr.clear();
#endif
    int maxFrames = -1;
    std::vector<int> preselectedDevices;
    std::string preselectedBackend = "";
    std::string preselectedTest = "";
    bool autoRun = false;
    bool exitOnComplete = false;
    float uiScaleOverride = 0.0f;
    uint32_t preselectedWidth = 0;
    uint32_t preselectedHeight = 0;
    bool dumpRendersFlagSet = false;
    bool dumpRendersVal = false;

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
                      << "  -r, --res <preset|WxH>  Pre-select render resolution (720p, 1080p, 1440p, 4k)\n"
                      << "      --dump-renders      Enable saving rendered output images to disk\n"
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
        } else if ((arg == "-r" || arg == "--res" || arg == "--resolution") && i + 1 < argc) {
            std::string resStr = argv[++i];
            std::string rLower = resStr;
            std::transform(rLower.begin(), rLower.end(), rLower.begin(), ::tolower);
            if (rLower == "720p" || rLower == "hd") {
                preselectedWidth = 1280; preselectedHeight = 720;
            } else if (rLower == "1080p" || rLower == "fhd") {
                preselectedWidth = 1920; preselectedHeight = 1080;
            } else if (rLower == "1440p" || rLower == "2k" || rLower == "qhd") {
                preselectedWidth = 2560; preselectedHeight = 1440;
            } else if (rLower == "4k" || rLower == "uhd" || rLower == "2160p") {
                preselectedWidth = 3840; preselectedHeight = 2160;
            } else {
                size_t xPos = rLower.find('x');
                if (xPos != std::string::npos) {
                    try {
                        preselectedWidth = static_cast<uint32_t>(std::stoul(rLower.substr(0, xPos)));
                        preselectedHeight = static_cast<uint32_t>(std::stoul(rLower.substr(xPos + 1)));
                    } catch (...) {}
                }
            }
        } else if (arg == "--dump-renders") {
            dumpRendersFlagSet = true;
            dumpRendersVal = true;
        } else if (arg == "--no-dump-renders") {
            dumpRendersFlagSet = true;
            dumpRendersVal = false;
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
    if (preselectedWidth > 0 && preselectedHeight > 0) {
        app.setRenderResolution(preselectedWidth, preselectedHeight);
    }
    if (dumpRendersFlagSet) {
        app.setDumpRenders(dumpRendersVal);
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
