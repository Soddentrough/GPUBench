#include "VulkanContext.h"
#include "GuiApp.h"

#include <CLI11.hpp>
#include <implot.h>
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#endif

#include <algorithm>
#include <cctype>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifndef GPUBENCH_VERSION
#error "GPUBENCH_VERSION must be defined by the build system"
#endif

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

    CLI::App cli{"GPUBench GUI - Workstation GPU Microarchitecture & Roofline Profiler", "gpubench-gui"};
    cli.set_version_flag("-v,--version", std::string("GPUBench GUI v") + GPUBENCH_VERSION);

    std::string deviceOpt;
    std::string preselectedBackend;
    std::string benchmarkOpt;
    std::string testOpt;
    std::string groupOpt;
    std::string preselectedScene;
    std::string resolutionOpt;
    std::string importJsonPath;
    std::string compareJsonPath;
    std::string outputJsonPath;
    std::string screenshotPath;
    float uiScaleOverride = 0.0f;
    int maxFrames = -1;
    bool autoRun = false;
    bool exitOnComplete = false;
    bool dumpRenders = false;
    bool noDumpRenders = false;
    bool verboseMode = false;
    bool debugMode = false;

    cli.add_option("-d,--device", deviceOpt, "Pre-select GPU index (e.g. 0, 1, 0,1, dual, or all)");
    cli.add_option("-k,--backend", preselectedBackend, "Pre-select compute backend (vulkan, rocm, opencl, auto)");
    cli.add_option("-b,--benchmark,--benchmarks", benchmarkOpt, "Pre-select benchmark(s) or comma-separated list");
    cli.add_option("-t,--test", testOpt, "Alias for --benchmark");
    cli.add_option("-g,--group,--groups", groupOpt, "Pre-select benchmark category (compute, memory, raytracing, scheduling, graphics, system, all)");
    cli.add_option("-s,--scene", preselectedScene, "Pre-select Ray Tracing scene (indoor, forest, outdoor, showroom, all)");
    cli.add_option("-r,--res,--resolution", resolutionOpt, "Pre-select render resolution (720p, 1080p, 1440p, 4k, 8k, auto, or WxH)");
    cli.add_option("-i,--import", importJsonPath, "Import a unified CLI/GUI JSON session file on startup");
    cli.add_option("--compare", compareJsonPath, "Load a unified CLI/GUI JSON file as comparison baseline");
    cli.add_option("-o,--output", outputJsonPath, "Export unified v2.0 JSON results to path upon completion");
    cli.add_flag("--dump-renders,--dump", dumpRenders, "Enable saving rendered scene PNGs to disk");
    cli.add_flag("--no-dump-renders,--no-dump", noDumpRenders, "Disable saving rendered scene PNGs to disk (uses GPU SharedPreviewImages)");
    cli.add_flag("--auto-run,--auto-start,--run", autoRun, "Automatically start benchmarks upon launch");
    cli.add_flag("--exit-on-complete", exitOnComplete, "Exit automatically when benchmark run finishes");
    cli.add_option("--frames", maxFrames, "Run for specified number of frames and exit");
    cli.add_option("--screenshot", screenshotPath, "Save rendered GUI window PNG before exit");
    cli.add_option("--ui-scale,--scale", uiScaleOverride, "UI scaling factor (e.g. 1.0, 1.5, 2.0)");
    cli.add_flag("--verbose", verboseMode, "Enable verbose logging output");
    cli.add_flag("--debug", debugMode, "Enable Vulkan Khronos Validation Layers");

    CLI11_PARSE(cli, argc, argv);

    bool selectAllGpus = false;
    std::vector<int> preselectedDevices;
    if (!deviceOpt.empty()) {
        std::string dLower = deviceOpt;
        std::transform(dLower.begin(), dLower.end(), dLower.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (dLower == "all") {
            selectAllGpus = true;
        } else if (dLower == "dual") {
            preselectedDevices = {0, 1};
        } else {
            std::stringstream dss(deviceOpt);
            std::string tok;
            while (std::getline(dss, tok, ',')) {
                if (!tok.empty()) {
                    try {
                        preselectedDevices.push_back(static_cast<int>(std::stoi(tok)));
                    } catch (...) {}
                }
            }
        }
    }

    std::string preselectedTest = benchmarkOpt;
    if (!preselectedTest.empty()) {
        std::string valLower = preselectedTest;
        std::transform(valLower.begin(), valLower.end(), valLower.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (valLower == "vulkan" || valLower == "rocm" || valLower == "hip" ||
            valLower == "opencl" || valLower == "cl" || valLower == "auto") {
            preselectedBackend = preselectedTest;
            preselectedTest.clear();
        }
    }
    if (preselectedTest.empty() && !testOpt.empty()) preselectedTest = testOpt;
    if (preselectedTest.empty() && !groupOpt.empty()) preselectedTest = groupOpt;

    bool autoResolution = false;
    uint32_t preselectedWidth = 0;
    uint32_t preselectedHeight = 0;
    if (!resolutionOpt.empty()) {
        std::string rLower = resolutionOpt;
        std::transform(rLower.begin(), rLower.end(), rLower.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (rLower == "auto") {
            autoResolution = true;
        } else if (rLower == "720p" || rLower == "hd") {
            preselectedWidth = 1280; preselectedHeight = 720;
        } else if (rLower == "1080p" || rLower == "fhd") {
            preselectedWidth = 1920; preselectedHeight = 1080;
        } else if (rLower == "1440p" || rLower == "2k" || rLower == "qhd") {
            preselectedWidth = 2560; preselectedHeight = 1440;
        } else if (rLower == "4k" || rLower == "uhd" || rLower == "2160p") {
            preselectedWidth = 3840; preselectedHeight = 2160;
        } else if (rLower == "8k" || rLower == "4320p") {
            preselectedWidth = 7680; preselectedHeight = 4320;
        } else {
            size_t xPos = rLower.find('x');
            if (xPos != std::string::npos) {
                try {
                    preselectedWidth = static_cast<uint32_t>(std::stoul(rLower.substr(0, xPos)));
                    preselectedHeight = static_cast<uint32_t>(std::stoul(rLower.substr(xPos + 1)));
                } catch (...) {}
            }
        }
    }

    int preferredGuiDeviceIndex = (!preselectedDevices.empty() && preselectedDevices[0] >= 0)
        ? preselectedDevices[0]
        : -1;

    gpubench::gui::VulkanContext vulkanContext;
    std::string winTitle = std::string("GPUBench v") + GPUBENCH_VERSION + " - Workstation GPU Profiler";
    if (!vulkanContext.init(winTitle.c_str(), 1760, 1000, uiScaleOverride,
                            verboseMode, debugMode, preferredGuiDeviceIndex)) {
        std::cerr << "Failed to initialize Vulkan 1.4 GUI context!" << std::endl;
        return 1;
    }

    float effectiveScale = vulkanContext.getDisplayScale();
    int curW = 0, curH = 0;
    SDL_GetWindowSize(vulkanContext.getWindow(), &curW, &curH);
    std::cout << "[GPUBench GUI] Initialized (Shared Vulkan 1.4 Device #" << vulkanContext.getActiveDeviceIndex()
              << "). Video driver: " << (SDL_GetCurrentVideoDriver() ? SDL_GetCurrentVideoDriver() : "null")
              << " | Display scale: " << effectiveScale << "x | Window: " << curW << "x" << curH << std::endl;

    ImPlot::CreateContext();

    gpubench::gui::GuiApp app;
    app.setVerbose(verboseMode);
    app.setDebug(debugMode);
    app.init(&vulkanContext, effectiveScale);

    if (selectAllGpus) {
        app.selectAllGpus();
    } else if (!preselectedDevices.empty()) {
        app.setSelectedDevices(preselectedDevices);
    }
    if (!preselectedBackend.empty()) {
        app.setSelectedBackend(preselectedBackend);
    }
    if (!preselectedScene.empty()) {
        app.setScene(preselectedScene);
    }
    if (!preselectedTest.empty()) {
        app.selectOnlyBenchmark(preselectedTest);
    }
    if (autoResolution) {
        app.applyAutoRenderResolution();
    } else if (preselectedWidth > 0 && preselectedHeight > 0) {
        app.setRenderResolution(preselectedWidth, preselectedHeight);
    }
    if (noDumpRenders) {
        app.setDumpRenders(false);
    } else if (dumpRenders) {
        app.setDumpRenders(true);
    }
    if (!importJsonPath.empty()) {
        app.importResultsFromJson(importJsonPath);
    }
    if (!compareJsonPath.empty()) {
        app.loadComparisonFromJson(compareJsonPath);
    }
    if (autoRun) {
        app.startBenchmarks();
    }

    bool exportedOnComplete = false;
    int frameCount = 0;
    bool running = true;
    auto processEvent = [&](const SDL_Event& ev) {
        ImGui_ImplSDL3_ProcessEvent(&ev);
        if (ev.type == SDL_EVENT_QUIT) {
            running = false;
        }
        if (ev.type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED) {
            vulkanContext.resize(ev.window.data1, ev.window.data2);
        }
    };

    while (running && !app.shouldQuit()) {
        SDL_Event event;
        if (maxFrames <= 0 && app.getExecutionState() == gpubench::gui::ExecutionState::Idle) {
            if (SDL_WaitEventTimeout(&event, 50)) {
                processEvent(event);
            }
        }
        while (SDL_PollEvent(&event)) {
            processEvent(event);
        }

        vulkanContext.beginFrame();
        app.updateAndRender();
        vulkanContext.endFrame();

        if (app.getExecutionState() == gpubench::gui::ExecutionState::Running) {
            SDL_Delay(2);
        }

        if (app.getExecutionState() == gpubench::gui::ExecutionState::Completed) {
            if (!outputJsonPath.empty() && !exportedOnComplete) {
                app.exportResultsToJson(outputJsonPath);
                exportedOnComplete = true;
            }
            if (exitOnComplete) {
                // Render one more frame after completion so screenshots/UI reflect completed Scorecard
                vulkanContext.beginFrame();
                app.updateAndRender();
                vulkanContext.endFrame();
                running = false;
            }
        }

        frameCount++;
        if (maxFrames > 0 && frameCount >= maxFrames) {
            running = false;
        }
    }

    if (!screenshotPath.empty()) {
        if (vulkanContext.saveScreenshot(screenshotPath)) {
            std::cout << "[GPUBench GUI] Saved screenshot to " << screenshotPath << std::endl;
        } else {
            std::cerr << "[GPUBench GUI] Failed to save screenshot to " << screenshotPath << std::endl;
        }
    }

    ImPlot::DestroyContext();
    vulkanContext.shutdown();

    return 0;
}
