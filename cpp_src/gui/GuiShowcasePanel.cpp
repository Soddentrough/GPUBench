#include "GuiApp.h"
#include "GuiUiHelpers.h"
#include "utils/KernelPath.h"

#include <SDL3/SDL.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace gpubench::gui {

namespace {

std::string resolveShowcaseKernelDir() {
    namespace fs = std::filesystem;
    const std::string base = KernelPath::find();
    std::vector<fs::path> candidates = {
        fs::path(base) / "vulkan",
        fs::path("build/kernels/vulkan"),
        fs::path("kernels/vulkan"),
        fs::path("../kernels/vulkan"),
    };
    if (const char* exeBase = SDL_GetBasePath()) {
        candidates.push_back(fs::path(exeBase) / "kernels/vulkan");
        candidates.push_back(fs::path(exeBase) / "../kernels/vulkan");
    }
    std::error_code ec;
    for (const auto& p : candidates) {
        if (fs::exists(p / "showcase_modern.mesh.spv", ec) ||
            fs::exists(p / "showcase_naive.vert.spv", ec)) {
            return p.string();
        }
    }
    return (fs::path(base) / "vulkan").string();
}

const char* modeShortLabel(gpubench::showcase::RenderMode m) {
    using gpubench::showcase::RenderMode;
    switch (m) {
        case RenderMode::Naive:  return "1. NAIVE (ANTI-PATTERN)";
        case RenderMode::Legacy: return "2. LEGACY (CLASSIC DEFERRED)";
        case RenderMode::Modern: return "3. MODERN (VULKAN 1.4)";
    }
    return "UNKNOWN";
}

const char* overlayShortLabel(gpubench::showcase::OverlayMode o) {
    using gpubench::showcase::OverlayMode;
    switch (o) {
        case OverlayMode::Beauty:          return "BEAUTY (LIT PBR)";
        case OverlayMode::Overdraw:        return "X-RAY: OVERDRAW HEAT";
        case OverlayMode::MicroTriangles:  return "X-RAY: MICRO-TRIS";
        case OverlayMode::MeshletClusters: return "X-RAY: MESHLETS";
    }
    return "BEAUTY";
}

} // namespace

void GuiApp::shutdownShowcase() {
    if (m_vulkanContext) {
        if (m_showcaseCallbackBound) {
            m_vulkanContext->setPreRenderPassCallback(nullptr);
            m_showcaseCallbackBound = false;
        }
        m_vulkanContext->waitDeviceIdleSafe();
        if (m_showcaseTex.isValid() ||
            m_showcaseTex.descriptorSet != VK_NULL_HANDLE ||
            m_showcaseTex.imageView != VK_NULL_HANDLE) {
            m_vulkanContext->destroyTexture(m_showcaseTex);
        }
    }
    m_showcaseTexGen = 0;
    if (m_showcaseRenderer) {
        m_showcaseRenderer->shutdown();
        m_showcaseRenderer.reset();
    }
}

void GuiApp::configureShowcase(int modePreset, int overlayMode, bool freezeCull, int camPreset) {
    using namespace gpubench::showcase;
    if (modePreset == 0) m_showcaseSettings.mode = RenderMode::Naive;
    else if (modePreset == 1) m_showcaseSettings.mode = RenderMode::Legacy;
    else m_showcaseSettings.mode = RenderMode::Modern;

    if (overlayMode == 1) m_showcaseSettings.overlay = OverlayMode::Overdraw;
    else if (overlayMode == 2) m_showcaseSettings.overlay = OverlayMode::MicroTriangles;
    else if (overlayMode == 3) m_showcaseSettings.overlay = OverlayMode::MeshletClusters;
    else m_showcaseSettings.overlay = OverlayMode::Beauty;

    m_showcaseTourPlaying = false;
    m_showcasePendingFreeze = freezeCull;
    m_showcasePendingCamPreset = camPreset;

    if (m_showcaseRenderer && m_showcaseRenderer->isInitialized()) {
        const auto& presets = m_showcaseRenderer->getScene().presets;
        if (m_showcasePendingCamPreset >= 0 &&
            m_showcasePendingCamPreset < static_cast<int>(presets.size())) {
            m_showcaseCamPreset = m_showcasePendingCamPreset;
            m_showcaseSettings.activeCam = presets[static_cast<size_t>(m_showcaseCamPreset)].camera;
            m_showcasePendingCamPreset = -1;
        }
        if (m_showcasePendingFreeze) {
            if (!presets.empty()) {
                m_showcaseSettings.frozenCam = presets[0].camera;
                if (presets.size() >= 4) {
                    m_showcaseCamPreset = 3;
                    m_showcaseSettings.activeCam = presets[3].camera;
                }
            }
            m_showcaseSettings.freezeCulling = true;
            m_showcasePendingFreeze = false;
        }
        m_showcaseRenderer->updateSettings(m_showcaseSettings);
    }
}

void GuiApp::renderOptimizationShowcase() {
    using namespace gpubench::showcase;

    const BringUpColors c = getBringUpColors(m_palette);
    const ImGuiIO& io = ImGui::GetIO();

    if (!m_vulkanContext || !m_vulkanContext->isInitialized()) {
        ImGui::TextColored(c.warnColor, "Vulkan GUI context is not initialized.");
        return;
    }

    auto sharedCtx = m_vulkanContext->getSharedComputeContext();
    if (!sharedCtx || !sharedCtx->getDevice()) {
        ImGui::TextColored(c.warnColor, "Shared Vulkan 1.4 device context unavailable for F4 Showcase.");
        return;
    }

    // Pause live 3D showcase while an automated benchmark batch is executing to eliminate contention.
    if (m_execState == ExecutionState::Running) {
        if (m_showcaseCallbackBound) {
            m_vulkanContext->setPreRenderPassCallback(nullptr);
            m_showcaseCallbackBound = false;
        }
        ImGui::TextColored(c.warnColor,
                           "BENCHMARK EXECUTION ACTIVE — F4 Optimization Showcase paused to isolate GPU telemetry.");
        return;
    }

    // Lazy initialization on first visit to F4
    if (!m_showcaseInitAttempted) {
        m_showcaseInitAttempted = true;
        m_showcaseRenderer = std::make_unique<ShowcaseRenderer>();
        const std::string kernelDir = resolveShowcaseKernelDir();
        if (!m_showcaseRenderer->init(sharedCtx.get(), kernelDir, 1280u, 720u)) {
            m_showcaseInitFailed = true;
            m_showcaseInitError = "ShowcaseRenderer::init failed (kernelDir=" + kernelDir + ").";
            m_showcaseRenderer->shutdown();
            m_showcaseRenderer.reset();
        } else {
            m_showcaseInitFailed = false;
            const auto& presets = m_showcaseRenderer->getScene().presets;
            if (!presets.empty()) {
                m_showcaseSettings.activeCam = presets[0].camera;
                m_showcaseSettings.frozenCam = presets[0].camera;
            }
        }
    }

    if (m_showcaseInitFailed || !m_showcaseRenderer || !m_showcaseRenderer->isInitialized()) {
        ImGui::TextColored(c.errColor, "F4 SHOWCASE INITIALIZATION FAILED: %s", m_showcaseInitError.c_str());
        if (tacticalButton("##ShowcaseRetry", "RETRY INIT", ImVec2(s(120.0f), s(28.0f)), false, c)) {
            m_showcaseInitAttempted = false;
            m_showcaseInitFailed = false;
            m_showcaseInitError.clear();
        }
        return;
    }

    const auto& scene = m_showcaseRenderer->getScene();
    const auto& presets = scene.presets;

    // Apply any CLI-deferred preset/freeze configuration now that scene presets exist
    if (m_showcasePendingCamPreset >= 0 &&
        m_showcasePendingCamPreset < static_cast<int>(presets.size())) {
        m_showcaseCamPreset = m_showcasePendingCamPreset;
        m_showcaseSettings.activeCam = presets[static_cast<size_t>(m_showcaseCamPreset)].camera;
        m_showcasePendingCamPreset = -1;
    }
    if (m_showcasePendingFreeze) {
        if (!presets.empty()) {
            m_showcaseSettings.frozenCam = presets[0].camera;
            if (presets.size() >= 4) {
                m_showcaseCamPreset = 3;
                m_showcaseSettings.activeCam = presets[3].camera;
            }
        }
        m_showcaseSettings.freezeCulling = true;
        m_showcasePendingFreeze = false;
    }

    const float availW = ImGui::GetContentRegionAvail().x;
    const float availH = ImGui::GetContentRegionAvail().y;
    const float ctrlW = std::clamp(availW * 0.315f, s(320.0f), s(470.0f));
    const float vpColW = std::max(s(320.0f), availW - ctrlW - s(8.0f));
    const float stripH = s(36.0f);
    const float vpCanvasH = std::max(s(180.0f), availH - stripH - s(6.0f));

    // ---- LEFT COLUMN (~68.5%): Interactive 3D Viewport + Control Strip ----
    ImGui::BeginChild("ShowcaseViewportCol", ImVec2(vpColW, availH), ImGuiChildFlags_None,
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    {
        ImGui::PushStyleColor(ImGuiCol_ChildBg, c.bgVoid);
        ImGui::BeginChild("ShowcaseVpCanvas", ImVec2(vpColW, vpCanvasH), ImGuiChildFlags_Borders,
                          ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
        {
            const ImVec2 innerAvail = ImGui::GetContentRegionAvail();
            const uint32_t targetW = std::clamp(static_cast<uint32_t>(std::max(320.0f, innerAvail.x)) & ~7u, 320u, 1920u);
            const uint32_t targetH = std::clamp(static_cast<uint32_t>(std::max(180.0f, innerAvail.y)) & ~7u, 180u, 1080u);

            if (targetW != m_showcaseRenderer->getDisplayWidth() ||
                targetH != m_showcaseRenderer->getDisplayHeight()) {
                m_vulkanContext->waitDeviceIdleSafe();
                if (m_showcaseTex.isValid()) {
                    m_vulkanContext->destroyTexture(m_showcaseTex);
                }
                m_showcaseRenderer->resize(targetW, targetH);
            }

            if (!m_showcaseTex.isValid() ||
                m_showcaseTexGen != m_showcaseRenderer->getImageGeneration()) {
                m_vulkanContext->waitDeviceIdleSafe();
                if (m_showcaseTex.isValid()) {
                    m_vulkanContext->destroyTexture(m_showcaseTex);
                }
                m_showcaseTex = m_vulkanContext->createTextureFromSharedView(
                    m_showcaseRenderer->getDisplayImageView(),
                    static_cast<int>(m_showcaseRenderer->getDisplayWidth()),
                    static_cast<int>(m_showcaseRenderer->getDisplayHeight()),
                    m_showcaseRenderer->getImageGeneration());
                m_showcaseTexGen = m_showcaseRenderer->getImageGeneration();
            }

            // Advance spline tour when active
            if (m_showcaseTourPlaying) {
                m_showcaseTourTime += io.DeltaTime * 0.35f;
                m_showcaseSettings.activeCam = scene.evalTourCamera(m_showcaseTourTime);
            }

            const ImVec2 imgPos = ImGui::GetCursorScreenPos();
            const ImVec2 imgSz(std::max(1.0f, innerAvail.x), std::max(1.0f, innerAvail.y));

            if (m_showcaseTex.isValid()) {
                ImGui::Image(reinterpret_cast<ImTextureID>(m_showcaseTex.descriptorSet), imgSz);
            } else {
                ImGui::Dummy(imgSz);
            }

            // Invisible interaction catcher overlaid cleanly on the image
            ImGui::SetCursorScreenPos(imgPos);
            ImGui::InvisibleButton("##ShowcaseVpInput", imgSz,
                                   ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
            const bool vpHovered = ImGui::IsItemHovered();
            const bool vpActive = ImGui::IsItemActive();

            const float wipeScreenX = imgPos.x + imgSz.x * m_showcaseSettings.splitWipeX;

            if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && vpHovered) {
                if (m_showcaseSettings.splitWipeCompare &&
                    std::fabs(io.MousePos.x - wipeScreenX) <= s(16.0f)) {
                    m_showcaseDraggingWipe = true;
                    m_showcaseDraggingCam = false;
                } else {
                    m_showcaseDraggingCam = true;
                    m_showcaseDraggingWipe = false;
                }
            }
            if (ImGui::IsMouseClicked(ImGuiMouseButton_Right) && vpHovered) {
                m_showcaseDraggingCam = true;
                m_showcaseDraggingWipe = false;
            }
            if (!ImGui::IsMouseDown(ImGuiMouseButton_Left) && !ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
                m_showcaseDraggingCam = false;
                m_showcaseDraggingWipe = false;
            }

            if (m_showcaseDraggingWipe && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                const float nx = (io.MousePos.x - imgPos.x) / std::max(1.0f, imgSz.x);
                m_showcaseSettings.splitWipeX = std::clamp(nx, 0.05f, 0.95f);
            } else if ((vpActive || m_showcaseDraggingCam) &&
                       (ImGui::IsMouseDragging(ImGuiMouseButton_Left) ||
                        ImGui::IsMouseDragging(ImGuiMouseButton_Right))) {
                m_showcaseTourPlaying = false;
                m_showcaseSettings.activeCam.yaw += io.MouseDelta.x * 0.0045f;
                m_showcaseSettings.activeCam.pitch = std::clamp(
                    m_showcaseSettings.activeCam.pitch - io.MouseDelta.y * 0.0045f, -1.35f, 1.35f);
            }

            if (vpHovered && std::fabs(io.MouseWheel) > 0.01f) {
                m_showcaseTourPlaying = false;
                const Vec3 fwd = m_showcaseSettings.activeCam.forward();
                m_showcaseSettings.activeCam.pos += fwd * (io.MouseWheel * 90.0f);
            }

            if ((vpHovered || vpActive) && !io.WantTextInput) {
                const float dt = std::min(io.DeltaTime, 0.05f);
                const float spd = (io.KeyShift ? 1200.0f : 500.0f) * dt;
                Vec3 move{0.0f, 0.0f, 0.0f};
                if (ImGui::IsKeyDown(ImGuiKey_W)) move += m_showcaseSettings.activeCam.forward();
                if (ImGui::IsKeyDown(ImGuiKey_S)) move -= m_showcaseSettings.activeCam.forward();
                if (ImGui::IsKeyDown(ImGuiKey_D)) move += m_showcaseSettings.activeCam.right();
                if (ImGui::IsKeyDown(ImGuiKey_A)) move -= m_showcaseSettings.activeCam.right();
                if (ImGui::IsKeyDown(ImGuiKey_E)) move.y += 1.0f;
                if (ImGui::IsKeyDown(ImGuiKey_Q)) move.y -= 1.0f;
                if (length(move) > 1e-4f) {
                    m_showcaseTourPlaying = false;
                    m_showcaseSettings.activeCam.pos += normalize(move) * spd;
                }
            }

            // Push live settings & arm pre-render-pass callback for this frame
            m_showcaseRenderer->updateSettings(m_showcaseSettings);
            if (!m_showcaseCallbackBound) {
                ShowcaseRenderer* rendererPtr = m_showcaseRenderer.get();
                m_vulkanContext->setPreRenderPassCallback(
                    [rendererPtr](VkCommandBuffer cmd, uint32_t frameIdx) {
                        if (rendererPtr && rendererPtr->isInitialized()) {
                            rendererPtr->recordFrame(cmd, frameIdx);
                        }
                    });
                m_showcaseCallbackBound = true;
            }

            // ---- Viewport Instrument Overlays ----
            const FrameTelemetry& tel = m_showcaseRenderer->getTelemetry();
            ImDrawList* dl = ImGui::GetWindowDrawList();
            const ImVec2 vpMax(imgPos.x + imgSz.x, imgPos.y + imgSz.y);
            dl->PushClipRect(imgPos, vpMax, true);

            drawCornerReticles(dl, imgPos, vpMax, toU32(c.phosphorDim, 0.85f), s(10.0f));

            // Top-Left HUD Readout Pill
            const float hudPad = s(8.0f);
            char perfBuf[96];
            const double shownMs = (tel.smoothedMs > 0.01) ? tel.smoothedMs : tel.totalMs;
            const double shownFps = (shownMs > 0.001) ? (1000.0 / shownMs) : 0.0;
            std::snprintf(perfBuf, sizeof(perfBuf), "%.1f FPS  |  %.2f ms  (%ux%u)",
                          shownFps, shownMs,
                          m_showcaseRenderer->getDisplayWidth(),
                          m_showcaseRenderer->getDisplayHeight());
            const char* rawModeLbl = modeShortLabel(m_showcaseSettings.mode);
            const float neededHudW = std::max(ImGui::CalcTextSize(perfBuf).x, ImGui::CalcTextSize(rawModeLbl).x) + hudPad * 2.5f;
            const float hudW = std::clamp(neededHudW, std::min(s(250.0f), imgSz.x * 0.56f), imgSz.x * 0.56f);
            {
                const float hudH = s(68.0f);
                const ImVec2 h0(imgPos.x + s(10.0f), imgPos.y + s(10.0f));
                const ImVec2 h1(h0.x + hudW, h0.y + hudH);
                dl->AddRectFilled(h0, h1, toU32(c.bgVoid, 0.84f), 0.0f);
                dl->AddRect(h0, h1, toU32(c.borderOuter, 0.9f), 0.0f, 0, 1.0f);

                const ImVec4 modeCol = (m_showcaseSettings.mode == RenderMode::Modern)
                    ? c.okColor
                    : (m_showcaseSettings.mode == RenderMode::Legacy ? c.warnColor : c.errColor);
                dl->AddRectFilled(h0, ImVec2(h0.x + s(3.5f), h1.y), toU32(modeCol), 0.0f);

                dl->PushClipRect(h0, h1, true);
                const std::string titleTxt = fitBringUpText(rawModeLbl, hudW - hudPad * 2.0f);
                dl->AddText(ImVec2(h0.x + hudPad, h0.y + s(6.0f)), toU32(modeCol), titleTxt.c_str());

                dl->AddText(ImVec2(h0.x + hudPad, h0.y + s(25.0f)), toU32(c.textPrimary),
                            fitBringUpText(perfBuf, hudW - hudPad * 2.0f).c_str());

                // Feature capability & active pills row
                float bx = h0.x + hudPad;
                const float by = h0.y + s(45.0f);
                auto drawFeatChip = [&](const char* tag, bool on) {
                    const float tw = ImGui::CalcTextSize(tag).x + s(10.0f);
                    const ImVec2 p0(bx, by);
                    const ImVec2 p1(bx + tw, by + s(16.0f));
                    dl->AddRectFilled(p0, p1, toU32(on ? c.phosphorGlow : c.bgRecessed, 0.9f));
                    dl->AddRect(p0, p1, toU32(on ? c.phosphorHi : c.borderInner), 0.0f, 0, 1.0f);
                    dl->AddText(ImVec2(p0.x + s(5.0f), p0.y + s(1.0f)),
                                toU32(on ? c.phosphorHi : c.textMuted), tag);
                    bx += tw + s(4.0f);
                };
                drawFeatChip("MESH", tel.meshShaderActive);
                drawFeatChip("DGC", tel.dgcExecutionSetActive);
                drawFeatChip("BARY", tel.barycentricActive);
                drawFeatChip("TESS", tel.tessellationActive);
                dl->PopClipRect();
            }

            // Top-Right Frame-Budget Ruler (8.3ms / 16.6ms targets)
            const float availBudgetW = imgSz.x - hudW - s(30.0f);
            if (availBudgetW >= s(200.0f)) {
                const float barCardW = std::min(s(320.0f), availBudgetW);
                const float barCardH = s(68.0f);
                const ImVec2 b1(vpMax.x - s(10.0f), imgPos.y + s(10.0f) + barCardH);
                const ImVec2 b0(b1.x - barCardW, imgPos.y + s(10.0f));
                dl->AddRectFilled(b0, b1, toU32(c.bgVoid, 0.84f), 0.0f);
                dl->AddRect(b0, b1, toU32(c.borderOuter, 0.9f), 0.0f, 0, 1.0f);

                dl->PushClipRect(b0, b1, true);
                const double frameMs = (tel.smoothedMs > 0.01) ? tel.smoothedMs : tel.totalMs;
                const char* budgetStatus = (frameMs <= 8.33)
                    ? "120+ FPS TARGET MET"
                    : (frameMs <= 16.67 ? "60 FPS TARGET MET" : "OVER 60 FPS BUDGET");
                const ImVec4 statusCol = (frameMs <= 8.33) ? c.okColor
                                        : (frameMs <= 16.67 ? c.phosphorHi : c.errColor);
                const float trackX0 = b0.x + s(8.0f);
                const float trackX1 = b1.x - s(8.0f);
                const float trackW = std::max(40.0f, trackX1 - trackX0);
                dl->AddText(ImVec2(trackX0, b0.y + s(5.0f)), toU32(statusCol),
                            fitBringUpText(budgetStatus, trackW).c_str());

                const float trackY0 = b0.y + s(26.0f);
                const float trackY1 = trackY0 + s(14.0f);
                dl->AddRectFilled(ImVec2(trackX0, trackY0), ImVec2(trackX1, trackY1), toU32(c.bgRecessed));

                const float maxScaleMs = 33.33f;
                const float cullFrac = static_cast<float>(tel.cullHizMs) / maxScaleMs;
                const float geomFrac = static_cast<float>(tel.geometryMs) / maxScaleMs;
                const float shadeFrac = static_cast<float>(tel.shadingMs) / maxScaleMs;

                float curX = trackX0;
                auto addSeg = [&](float frac, ImVec4 col) {
                    if (frac <= 0.0f || curX >= trackX1) return;
                    const float nextX = std::min(trackX1, curX + frac * trackW);
                    dl->AddRectFilled(ImVec2(curX, trackY0 + s(1.0f)), ImVec2(nextX, trackY1 - s(1.0f)), toU32(col));
                    curX = nextX;
                };
                addSeg(cullFrac, c.cyanAccent);
                addSeg(geomFrac, c.phosphorHi);
                addSeg(shadeFrac, c.okColor);

                // If totalMs > sum of pass timers (sync/overlay), show tail bar in warn/err color
                const float totalEnd = std::min(trackX1, trackX0 + (static_cast<float>(frameMs) / maxScaleMs) * trackW);
                if (totalEnd > curX + 1.0f) {
                    dl->AddRectFilled(ImVec2(curX, trackY0 + s(1.0f)), ImVec2(totalEnd, trackY1 - s(1.0f)),
                                      toU32(frameMs > 16.67 ? c.errColor : c.warnColor, 0.85f));
                }

                // 8.3ms (120Hz) and 16.6ms (60Hz) ruler tick marks
                const float x120 = trackX0 + (8.33f / maxScaleMs) * trackW;
                const float x60 = trackX0 + (16.67f / maxScaleMs) * trackW;
                dl->AddLine(ImVec2(x120, trackY0 - s(2.0f)), ImVec2(x120, trackY1 + s(2.0f)), toU32(c.textPrimary), 1.5f);
                dl->AddLine(ImVec2(x60, trackY0 - s(2.0f)), ImVec2(x60, trackY1 + s(2.0f)), toU32(c.warnColor), 1.5f);

                char legBuf[96];
                std::snprintf(legBuf, sizeof(legBuf),
                              "Cull %.2f | Geom %.2f | Shade %.2f",
                              tel.cullHizMs, tel.geometryMs, tel.shadingMs);
                dl->AddText(ImVec2(trackX0, b0.y + s(46.0f)), toU32(c.textSecondary),
                            fitBringUpText(legBuf, trackW).c_str());
                dl->PopClipRect();
            }

            // Split-Wipe interactive handle & side labels
            if (m_showcaseSettings.splitWipeCompare) {
                const float wx = imgPos.x + imgSz.x * m_showcaseSettings.splitWipeX;
                const float midY = imgPos.y + imgSz.y * 0.5f;
                const ImU32 gold = toU32(c.phosphorHi);
                dl->AddQuadFilled(ImVec2(wx, midY - s(11.0f)), ImVec2(wx + s(9.0f), midY),
                                  ImVec2(wx, midY + s(11.0f)), ImVec2(wx - s(9.0f), midY), gold);

                const char* leftLbl = (m_showcaseSettings.splitLeftMode == RenderMode::Legacy)
                    ? "LEFT: LEGACY" : "LEFT: NAIVE";
                const char* rightLbl = "RIGHT: MODERN (VK 1.4)";
                dl->AddRectFilled(ImVec2(wx - s(118.0f), vpMax.y - s(34.0f)),
                                  ImVec2(wx - s(6.0f), vpMax.y - s(12.0f)), toU32(c.bgVoid, 0.85f));
                dl->AddText(ImVec2(wx - s(112.0f), vpMax.y - s(31.0f)), toU32(c.errColor), leftLbl);

                dl->AddRectFilled(ImVec2(wx + s(6.0f), vpMax.y - s(34.0f)),
                                  ImVec2(wx + s(172.0f), vpMax.y - s(12.0f)), toU32(c.bgVoid, 0.85f));
                dl->AddText(ImVec2(wx + s(12.0f), vpMax.y - s(31.0f)), toU32(c.okColor), rightLbl);

                if (tel.hasParitySample) {
                    char psnrBuf[64];
                    std::snprintf(psnrBuf, sizeof(psnrBuf), "PARITY PSNR: %.1f dB (%.1f%% exact)",
                                  tel.parityPsnr, tel.parityExactPct);
                    const float pw = ImGui::CalcTextSize(psnrBuf).x + s(16.0f);
                    dl->AddRectFilled(ImVec2(vpMax.x - pw - s(10.0f), vpMax.y - s(34.0f)),
                                      ImVec2(vpMax.x - s(10.0f), vpMax.y - s(12.0f)), toU32(c.bgVoid, 0.88f));
                    dl->AddText(ImVec2(vpMax.x - pw - s(2.0f), vpMax.y - s(31.0f)), toU32(c.cyanAccent), psnrBuf);
                }
            }

            // X-Ray Overlay Legend (bottom-left)
            if (m_showcaseSettings.overlay != OverlayMode::Beauty) {
                const char* legDesc = (m_showcaseSettings.overlay == OverlayMode::Overdraw)
                    ? "1x Blue -> 3x Cyan -> 5x Green/Yel -> 9x+ Red"
                    : (m_showcaseSettings.overlay == OverlayMode::MicroTriangles
                           ? "Magenta <1px2 | Red 1-3px2 | Grn 3-12px2 | Cyan >12px2"
                           : "Unique Colors = 64-Vert / 124-Tri GPU Meshlet Clusters");
                const char* legTitle = overlayShortLabel(m_showcaseSettings.overlay);
                const float lx0 = imgPos.x + s(10.0f);
                const float ly1 = vpMax.y - s(10.0f);
                const float ly0 = ly1 - s(42.0f);
                const float lw = std::min(imgSz.x - s(20.0f),
                                          std::max({s(320.0f), ImGui::CalcTextSize(legDesc).x + s(18.0f),
                                                    ImGui::CalcTextSize(legTitle).x + s(18.0f)}));
                dl->AddRectFilled(ImVec2(lx0, ly0), ImVec2(lx0 + lw, ly1), toU32(c.bgVoid, 0.88f));
                dl->AddRect(ImVec2(lx0, ly0), ImVec2(lx0 + lw, ly1), toU32(c.borderOuter), 0.0f, 0, 1.0f);

                dl->AddText(ImVec2(lx0 + s(8.0f), ly0 + s(4.0f)), toU32(c.phosphorHi),
                            fitBringUpText(legTitle, lw - s(16.0f)).c_str());
                dl->AddText(ImVec2(lx0 + s(8.0f), ly0 + s(21.0f)), toU32(c.textPrimary),
                            fitBringUpText(legDesc, lw - s(16.0f)).c_str());
            }

            // Freeze Culling Banner (bottom-center / top-center)
            if (m_showcaseSettings.freezeCulling) {
                const char* fRaw = "FREEZE CULLING X-RAY: Gold frustum = frozen camera; missing shapes were culled by GPU";
                const std::string fFit = fitBringUpText(fRaw, std::max(120.0f, imgSz.x - s(36.0f)));
                const float tw = ImGui::CalcTextSize(fFit.c_str()).x + s(20.0f);
                const float fx0 = imgPos.x + (imgSz.x - tw) * 0.5f;
                const float fy0 = vpMax.y - s(36.0f);
                dl->AddRectFilled(ImVec2(fx0, fy0), ImVec2(fx0 + tw, fy0 + s(24.0f)), toU32(c.bgVoid, 0.90f));
                dl->AddRect(ImVec2(fx0, fy0), ImVec2(fx0 + tw, fy0 + s(24.0f)), toU32(c.phosphorHi), 0.0f, 0, 1.5f);
                dl->AddText(ImVec2(fx0 + s(10.0f), fy0 + s(4.0f)), toU32(c.phosphorHi), fFit.c_str());
            }

            dl->PopClipRect();
        }
        ImGui::EndChild();
        ImGui::PopStyleColor();

        // ---- Bottom Camera & Action Strip ----
        ImGui::PushStyleColor(ImGuiCol_ChildBg, c.bgElevated);
        ImGui::BeginChild("ShowcaseVpStrip", ImVec2(vpColW, stripH), ImGuiChildFlags_Borders,
                          ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
        {
            const float bh = s(22.0f);
            const bool compactStrip = (vpColW < s(820.0f));
            static const char* kShortPresetNames[4] = {"1:Nave", "2:Cloister", "3:Tess", "4:Over"};
            for (size_t pi = 0; pi < presets.size() && pi < 4; ++pi) {
                char pId[24];
                std::snprintf(pId, sizeof(pId), "##CamP%zu", pi);
                const char* pLbl = compactStrip ? kShortPresetNames[pi] : presets[pi].name;
                if (tacticalButton(pId, pLbl, ImVec2(0.0f, bh),
                                   !m_showcaseTourPlaying && m_showcaseCamPreset == static_cast<int>(pi), c)) {
                    m_showcaseTourPlaying = false;
                    m_showcaseCamPreset = static_cast<int>(pi);
                    m_showcaseSettings.activeCam = presets[pi].camera;
                }
                if (ImGui::IsItemHovered() && presets[pi].description) {
                    ImGui::SetTooltip("%s: %s", presets[pi].name, presets[pi].description);
                }
                ImGui::SameLine(0.0f, s(4.0f));
            }

            if (tacticalButton("##ScTour", m_showcaseTourPlaying ? "TOUR*" : "TOUR",
                               ImVec2(0.0f, bh), m_showcaseTourPlaying, c)) {
                m_showcaseTourPlaying = !m_showcaseTourPlaying;
                if (m_showcaseTourPlaying) m_showcaseSettings.freezeCulling = false;
            }
            ImGui::SameLine(0.0f, s(4.0f));

            if (tacticalButton("##ScWipe", compactStrip ? "WIPE" : "SPLIT-WIPE", ImVec2(0.0f, bh),
                               m_showcaseSettings.splitWipeCompare, c)) {
                m_showcaseSettings.splitWipeCompare = !m_showcaseSettings.splitWipeCompare;
                if (m_showcaseSettings.splitWipeCompare) {
                    m_showcaseSettings.mode = RenderMode::Modern;
                }
            }
            ImGui::SameLine(0.0f, s(4.0f));

            const char* frzLbl = m_showcaseSettings.freezeCulling
                ? (compactStrip ? "UNFRZ" : "UNFREEZE")
                : (compactStrip ? "FREEZE" : "FREEZE CULL");
            if (tacticalButton("##ScFreeze", frzLbl,
                               ImVec2(0.0f, bh), m_showcaseSettings.freezeCulling, c)) {
                if (!m_showcaseSettings.freezeCulling) {
                    m_showcaseTourPlaying = false;
                    m_showcaseSettings.frozenCam = m_showcaseSettings.activeCam;
                    m_showcaseSettings.freezeCulling = true;
                    if (presets.size() >= 4) {
                        m_showcaseCamPreset = 3;
                        m_showcaseSettings.activeCam = presets[3].camera;
                    }
                } else {
                    m_showcaseSettings.freezeCulling = false;
                    m_showcaseSettings.activeCam = m_showcaseSettings.frozenCam;
                }
            }
            ImGui::SameLine(0.0f, s(4.0f));

            if (tacticalButton("##ScBench3", compactStrip ? "3-WAY" : "3-WAY BENCH", ImVec2(0.0f, bh),
                               false, c, false, true)) {
                m_vulkanContext->waitDeviceIdleSafe();
                m_showcaseThreeWay = m_showcaseRenderer->runThreeWayBenchmark(
                    m_showcaseSettings.activeCam,
                    m_showcaseRenderer->getDisplayWidth(),
                    m_showcaseRenderer->getDisplayHeight());
                m_showcaseRenderer->updateSettings(m_showcaseSettings);
            }
        }
        ImGui::EndChild();
        ImGui::PopStyleColor();
    }
    ImGui::EndChild();

    ImGui::SameLine(0.0f, s(8.0f));

    // ---- RIGHT CONTROL PANEL (~31.5%) ----
    ImGui::PushStyleColor(ImGuiCol_ChildBg, c.bgPanel);
    ImGui::BeginChild("ShowcaseControlCol", ImVec2(ctrlW, availH), ImGuiChildFlags_Borders);
    {
        const float colAvail = ImGui::GetContentRegionAvail().x;
        const FrameTelemetry& tel = m_showcaseRenderer->getTelemetry();

        drawEtchedHeader("01", "PIPELINE ERA SELECTOR", c, "SAME SCENE");
        const float mbH = s(30.0f);
        const std::string era1Lbl = fitBringUpText("1. NAIVE (64x TESS + NO CULL + OVERDRAW)", colAvail - s(16.0f));
        const std::string era2Lbl = fitBringUpText("2. LEGACY (CPU CULL + MDI + G-BUFFER)", colAvail - s(16.0f));
        const std::string era3Lbl = fitBringUpText("3. MODERN (VK 1.4 MESHLETS + HI-Z + VISBUF)", colAvail - s(16.0f));
        if (tacticalButton("##EraNaive", era1Lbl.c_str(),
                           ImVec2(colAvail, mbH), m_showcaseSettings.mode == RenderMode::Naive, c,
                           m_showcaseSettings.mode == RenderMode::Naive)) {
            m_showcaseSettings.mode = RenderMode::Naive;
            m_showcaseSettings.splitWipeCompare = false;
        }
        if (tacticalButton("##EraLegacy", era2Lbl.c_str(),
                           ImVec2(colAvail, mbH), m_showcaseSettings.mode == RenderMode::Legacy, c)) {
            m_showcaseSettings.mode = RenderMode::Legacy;
            m_showcaseSettings.splitWipeCompare = false;
        }
        if (tacticalButton("##EraModern", era3Lbl.c_str(),
                           ImVec2(colAvail, mbH), m_showcaseSettings.mode == RenderMode::Modern, c)) {
            m_showcaseSettings.mode = RenderMode::Modern;
        }

        ImGui::Spacing();
        drawEtchedHeader("02", "WHY THIS CHANGES YOUR FPS", c, "EXPLAINED");
        const float explainerH = std::max(s(126.0f), ImGui::GetTextLineHeightWithSpacing() * 7.2f + s(16.0f));
        ImGui::PushStyleColor(ImGuiCol_ChildBg, c.bgRecessed);
        ImGui::BeginChild("ShowcaseExplainer", ImVec2(colAvail, explainerH), ImGuiChildFlags_Borders);
        {
            if (m_showcaseSettings.freezeCulling) {
                ImGui::TextColored(c.phosphorHi, "Freezing the Culling Camera (X-Ray View)");
                ImGui::TextWrapped(
                    "Everything inside the gold pyramid was in view of the original camera. Notice how walls and arches blocked rooms behind them: Modern mode skipped drawing those hidden objects completely before vertices even ran.");
            } else if (m_showcaseSettings.overlay == OverlayMode::Overdraw) {
                ImGui::TextColored(c.phosphorHi, "What Overdraw Looks Like");
                ImGui::TextWrapped(
                    "Blue pixels were shaded once. Bright yellow/red pixels were shaded 5 to 12+ times because distant walls and leaves were drawn first, only to be painted over. Every redraw wastes GPU shader power.");
            } else if (m_showcaseSettings.overlay == OverlayMode::MicroTriangles) {
                ImGui::TextColored(c.phosphorHi, "Why Tiny Triangles Stall GPUs");
                ImGui::TextWrapped(
                    "GPUs shade pixels in 2x2 quad blocks. Magenta (<1px2) and red triangles force whole 2x2 quads to run for a single sub-pixel dot! Turning tessellation up to 64x turns scenes into magenta dust.");
            } else if (m_showcaseSettings.mode == RenderMode::Naive) {
                ImGui::TextColored(c.errColor, "Anti-Pattern: Brute-Force Rendering");
                ImGui::TextWrapped(
                    "Draws walls back-to-front so pillars and leaves shade every pixel multiple times, never skips walls behind you, and cranks hardware tessellation to 64x. High-bandwidth cards brute-force it poorly; well-built games do not need to.");
            } else if (m_showcaseSettings.mode == RenderMode::Legacy) {
                ImGui::TextColored(c.warnColor, "Classic Deferred (DX11 / Early DX12)");
                ImGui::TextWrapped(
                    "CPU hides objects outside the screen and uses MultiDrawIndirect + G-Buffer deferred shading so hidden pixels aren't lit twice. Still chokes when CPU/vertex shaders process foliage and props behind walls.");
            } else {
                ImGui::TextColored(c.okColor, "Modern Vulkan 1.4 GPU-Driven Pipeline");
                const char* modernDesc = (tel.meshShaderSupported && tel.dgcExecutionSetSupported)
                    ? "Splits meshes into 64-vert clusters (Mesh Shaders), culls off-screen AND wall-occluded pieces via Two-Phase Hi-Z on the GPU, swaps 64x tessellation for crisp LODs, and shades each visible pixel once via Visibility Buffer + DGC ExecutionSet."
                    : (tel.meshShaderSupported
                        ? "Splits meshes into 64-vert clusters (Mesh Shaders), culls off-screen AND wall-occluded pieces via Two-Phase Hi-Z on the GPU, swaps 64x tessellation for crisp LODs, and shades each visible pixel once via Visibility Buffer + DispatchIndirect material binning."
                        : "Splits meshes into 64-vert clusters (Compute + Indirect Draw fallback), culls off-screen AND wall-occluded pieces via Two-Phase Hi-Z on the GPU, swaps 64x tessellation for crisp LODs, and shades each visible pixel once via Visibility Buffer.");
                ImGui::TextWrapped("%s", modernDesc);
            }
        }
        ImGui::EndChild();
        ImGui::PopStyleColor();

        ImGui::Spacing();
        drawEtchedHeader("03", "X-RAY DIAGNOSTIC LENS", c, "INSPECT");
        {
            const float halfW = (colAvail - s(4.0f)) * 0.5f;
            const float xbH = s(24.0f);
            if (tacticalButton("##OvBeauty", "BEAUTY", ImVec2(halfW, xbH),
                               m_showcaseSettings.overlay == OverlayMode::Beauty, c)) {
                m_showcaseSettings.overlay = OverlayMode::Beauty;
            }
            ImGui::SameLine(0.0f, s(4.0f));
            if (tacticalButton("##OvOverdraw", "OVERDRAW HEAT", ImVec2(halfW, xbH),
                               m_showcaseSettings.overlay == OverlayMode::Overdraw, c)) {
                m_showcaseSettings.overlay = OverlayMode::Overdraw;
            }
            if (tacticalButton("##OvMicro", "MICRO-TRIANGLES", ImVec2(halfW, xbH),
                               m_showcaseSettings.overlay == OverlayMode::MicroTriangles, c)) {
                m_showcaseSettings.overlay = OverlayMode::MicroTriangles;
            }
            ImGui::SameLine(0.0f, s(4.0f));
            if (tacticalButton("##OvMeshlets", "MESHLETS", ImVec2(halfW, xbH),
                               m_showcaseSettings.overlay == OverlayMode::MeshletClusters, c)) {
                m_showcaseSettings.overlay = OverlayMode::MeshletClusters;
            }
        }

        ImGui::Spacing();
        drawEtchedHeader("04", "LIVE OPTIMIZATION STACK", c,
                         m_showcaseSettings.mode == RenderMode::Modern ? "INTERACTIVE" : "MODERN ONLY");
        {
            const bool isModern = (m_showcaseSettings.mode == RenderMode::Modern);
            ImGui::BeginDisabled(!isModern);
            auto optRow = [&](const char* id, bool* v, const char* lbl, const char* note) {
                hardwareArmToggle(id, v, c, m_uiScale);
                ImGui::SameLine(0.0f, s(6.0f));
                ImGui::TextColored((v && *v && isModern) ? c.textPrimary : c.textMuted, "%s", lbl);
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled) && note) {
                    ImGui::SetTooltip("%s", note);
                }
            };
            optRow("##OptLod",  &m_showcaseSettings.opts.adaptiveLod,
                   tel.meshShaderSupported
                       ? "Adaptive LOD Meshlets (Mesh Shader)"
                       : "Adaptive LOD Meshlets (Compute Fallback)",
                   tel.meshShaderSupported
                       ? "Distance-based meshlet LOD via VK_EXT_mesh_shader eliminates sub-pixel micro-triangles."
                       : "Distance-based meshlet LOD via compute culling + indirect draw fallback (VK_EXT_mesh_shader unavailable).");
            optRow("##OptCull", &m_showcaseSettings.opts.frustumConeCull,
                   "GPU Frustum + Backface Cone Culling",
                   "Skips meshlets outside camera frustum or facing away.");
            optRow("##OptHiz",  &m_showcaseSettings.opts.hizOcclusion,
                   "Two-Phase Hi-Z Occlusion Culling",
                   "Sponza stone occluders prime Hi-Z depth pyramid, culling hidden foliage & rooms.");
            optRow("##OptVis",  &m_showcaseSettings.opts.visBuffer,
                   "Visibility Buffer (Single-Shade)",
                   "Stores compact surface ID per pixel instead of heavy forward overdraw or fat G-Buffers.");
            optRow("##OptDgc",  &m_showcaseSettings.opts.dgcMaterialBinning,
                   tel.dgcExecutionSetSupported
                       ? "DGC Material Binning (ExecutionSet)"
                       : "Material Binning (DispatchIndirect Fallback)",
                   tel.dgcExecutionSetSupported
                       ? "Batches material resolve shaders via 1x vkCmdExecuteGeneratedCommandsEXT + VkIndirectExecutionSetEXT."
                       : "Driver lacks DGC ExecutionSet; batches material resolve shaders via vkCmdDispatchIndirect fallback.");
            ImGui::EndDisabled();
        }

        ImGui::Spacing();
        drawEtchedHeader("05", "LIVE HARDWARE TELEMETRY", c,
                         scene.sceneSourceLabel.empty() ? "SPONZA" : "LIVE");
        {
            auto statRow = [&](const char* k, const char* v, ImVec4 vCol) {
                ImGui::TextColored(c.textSecondary, "%s", k);
                const float vw = ImGui::CalcTextSize(v).x;
                ImGui::SameLine(std::max(ImGui::GetCursorPosX() + s(8.0f), colAvail - vw));
                ImGui::TextColored(vCol, "%s", v);
            };
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%.2f ms (%.0f FPS)", tel.totalMs, tel.fps);
            statRow("GPU Frame Time", buf, tel.totalMs <= 8.33 ? c.okColor : (tel.totalMs <= 16.67 ? c.phosphorHi : c.errColor));

            std::snprintf(buf, sizeof(buf), "%u", tel.drawCalls);
            statRow("Draw / Dispatch Calls", buf, c.textPrimary);

            std::snprintf(buf, sizeof(buf), "%llu", static_cast<unsigned long long>(tel.clippedPrimitives));
            statRow("Clipped Primitives", buf,
                    tel.clippedPrimitives > 2000000ULL ? c.errColor : c.textPrimary);

            std::snprintf(buf, sizeof(buf), "%.2fx", tel.avgOverdraw);
            statRow("Avg Pixel Overdraw", buf,
                    tel.avgOverdraw > 2.2 ? c.errColor : (tel.avgOverdraw > 1.35 ? c.warnColor : c.okColor));

            std::snprintf(buf, sizeof(buf), "%.1f%%", tel.subPixelTriPct);
            statRow("Sub-Pixel Tris (<1px2)", buf,
                    tel.subPixelTriPct > 20.0 ? c.errColor : c.okColor);

            if (m_showcaseSettings.mode == RenderMode::Modern) {
                std::snprintf(buf, sizeof(buf), "%.1f%% (%u/%u)",
                              tel.culledPct,
                              tel.totalMeshlets - tel.renderedMeshlets,
                              tel.totalMeshlets);
                statRow("GPU Meshlets Culled", buf, tel.culledPct > 20.0 ? c.okColor : c.warnColor);
            }

            if (m_showcaseThreeWay.valid) {
                ImGui::Separator();
                ImGui::TextColored(c.phosphorHi, "3-WAY BENCHMARK SUMMARY");
                std::snprintf(buf, sizeof(buf), "%.2f ms (%.0f FPS)",
                              m_showcaseThreeWay.naiveMs, m_showcaseThreeWay.naiveFps);
                statRow("1. Naive", buf, c.errColor);
                std::snprintf(buf, sizeof(buf), "%.2f ms (%.0f FPS)",
                              m_showcaseThreeWay.legacyMs, m_showcaseThreeWay.legacyFps);
                statRow("2. Legacy", buf, c.warnColor);
                std::snprintf(buf, sizeof(buf), "%.2f ms (%.0f FPS)",
                              m_showcaseThreeWay.modernMs, m_showcaseThreeWay.modernFps);
                statRow("3. Modern", buf, c.okColor);

                std::snprintf(buf, sizeof(buf), "%.2fx vs Naive | %.2fx vs Leg",
                              m_showcaseThreeWay.modernVsNaiveSpeedup,
                              m_showcaseThreeWay.modernVsLegacySpeedup);
                statRow("Modern Speedup", buf, c.cyanAccent);

                std::snprintf(buf, sizeof(buf), "%.1f dB (%.1f%% exact)",
                              m_showcaseThreeWay.parityPsnr, m_showcaseThreeWay.parityExactPct);
                statRow("Visual Parity", buf, c.cyanAccent);
            }
        }
    }
    ImGui::EndChild();
    ImGui::PopStyleColor();
}

} // namespace gpubench::gui
