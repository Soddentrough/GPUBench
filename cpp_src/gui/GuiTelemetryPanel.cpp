#include "GuiApp.h"
#include "GuiUiHelpers.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace gpubench::gui {

void GuiApp::renderSidebarTelemetry() {
    pushBoldFont();
    ImGui::TextColored(ImVec4(0.75f, 0.82f, 0.92f, 1.0f), "LIVE HARDWARE TELEMETRY");
    popBoldFont();

    uint32_t devCount = m_telemetryWorker.getDeviceCount();
    if (devCount == 0) {
        ImGui::TextDisabled("No hardware telemetry sensors detected (N/A)");
        return;
    }

    if (devCount > 1) {
        auto names = m_telemetryWorker.getDeviceNames();
        for (uint32_t i = 0; i < devCount; ++i) {
            if (i > 0) ImGui::SameLine(0.0f, s(5.0f));
            char lbl[32];
            std::snprintf(lbl, sizeof(lbl), "GPU %u", i);
            bool sel = (m_telemetryGpuIndex == i);
            if (sel) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.05f, 0.55f, 0.78f, 1.0f));
            if (ImGui::SmallButton(lbl)) {
                m_telemetryGpuIndex = i;
            }
            if (sel) ImGui::PopStyleColor();
            if (ImGui::IsItemHovered() && i < names.size()) {
                ImGui::SetTooltip("%s", names[i].c_str());
            }
        }
    }

    DeviceTelemetrySnapshot snap{};
    if (!m_telemetryWorker.getSnapshot(m_telemetryGpuIndex, snap) || !snap.isAvailable) {
        ImGui::TextDisabled("Telemetry polling initializing... (N/A)");
        return;
    }

    ImGui::BeginChild("SidebarTelemetryCard", ImVec2(0, s(168.0f)), ImGuiChildFlags_Borders);

    auto renderSensorRow = [&](const char* label, bool available, const char* valueStr, ImVec4 valueColor) {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextDisabled("%s", label);
        ImGui::TableSetColumnIndex(1);
        pushMonoFont();
        if (available) {
            ImGui::TextColored(valueColor, "%s", valueStr);
        } else {
            ImGui::TextDisabled("N/A");
        }
        popMonoFont();
    };

    if (ImGui::BeginTable("SidebarSensorGrid", 2, ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("Sensor", ImGuiTableColumnFlags_WidthStretch, 0.46f);
        ImGui::TableSetupColumn("Reading", ImGuiTableColumnFlags_WidthStretch, 0.54f);

        char buf[64];

        std::snprintf(buf, sizeof(buf), "%.0f%%", snap.gpuBusyPct);
        renderSensorRow("GPU Load", snap.hasGpuBusy, buf,
                        (snap.gpuBusyPct > 85.0f) ? ImVec4(0.20f, 0.92f, 0.55f, 1.0f)
                                                  : ImVec4(0.85f, 0.92f, 0.98f, 1.0f));

        std::snprintf(buf, sizeof(buf), "%.0f MHz", snap.sclkMhz);
        renderSensorRow("Shader Clock", snap.hasSclk, buf, ImVec4(0.15f, 0.82f, 0.98f, 1.0f));

        std::snprintf(buf, sizeof(buf), "%.0f MHz", snap.mclkMhz);
        renderSensorRow("Memory Clock", snap.hasMclk, buf, ImVec4(0.55f, 0.78f, 0.98f, 1.0f));

        std::snprintf(buf, sizeof(buf), "%.1f W", snap.powerWatts);
        renderSensorRow("Board Power", snap.hasPower, buf, ImVec4(0.98f, 0.76f, 0.24f, 1.0f));

        bool hasTemp = snap.hasTempJct || snap.hasTempEdge;
        if (snap.hasTempJct && snap.hasTempEdge) {
            std::snprintf(buf, sizeof(buf), "%.0f C (Jct %.0f C)", snap.tempEdgeC, snap.tempJctC);
        } else if (snap.hasTempJct) {
            std::snprintf(buf, sizeof(buf), "%.0f C (Hotspot)", snap.tempJctC);
        } else {
            std::snprintf(buf, sizeof(buf), "%.0f C", snap.tempEdgeC);
        }
        float refTemp = snap.hasTempJct ? snap.tempJctC : snap.tempEdgeC;
        ImVec4 tempCol = (refTemp >= 90.0f)
            ? ImVec4(0.96f, 0.36f, 0.32f, 1.0f)
            : ((refTemp >= 75.0f) ? ImVec4(0.98f, 0.76f, 0.24f, 1.0f)
                                  : ImVec4(0.30f, 0.92f, 0.62f, 1.0f));
        renderSensorRow("Temperature", hasTemp, buf, tempCol);

        bool hasFan = snap.hasFanRpm || snap.hasFanSpeedPct;
        if (snap.hasFanRpm && snap.hasFanSpeedPct) {
            std::snprintf(buf, sizeof(buf), "%.0f RPM (%.0f%%)", snap.fanRpm, snap.fanSpeedPct);
        } else if (snap.hasFanRpm) {
            std::snprintf(buf, sizeof(buf), "%.0f RPM", snap.fanRpm);
        } else {
            std::snprintf(buf, sizeof(buf), "%.0f%%", snap.fanSpeedPct);
        }
        renderSensorRow("Cooling Fan", hasFan, buf, ImVec4(0.80f, 0.86f, 0.94f, 1.0f));

        ImGui::EndTable();
    }

    if (snap.hasVram && snap.vramTotalBytes > 0) {
        double usedGb = static_cast<double>(snap.vramUsedBytes) / (1024.0 * 1024.0 * 1024.0);
        double totalGb = static_cast<double>(snap.vramTotalBytes) / (1024.0 * 1024.0 * 1024.0);
        float frac = std::clamp(static_cast<float>(usedGb / std::max(0.001, totalGb)), 0.0f, 1.0f);
        char vramOverlay[64];
        std::snprintf(vramOverlay, sizeof(vramOverlay), "VRAM: %.2f / %.1f GB (%.0f%%)",
                      usedGb, totalGb, frac * 100.0f);
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(0.06f, 0.62f, 0.86f, 0.90f));
        ImGui::ProgressBar(frac, ImVec2(-1.0f, s(16.0f)), vramOverlay);
        ImGui::PopStyleColor();
    } else {
        ImGui::TextDisabled("VRAM Usage: N/A");
    }

    ImGui::EndChild();
}

void GuiApp::renderLiveTelemetryDock() {
    ImGui::Spacing();

    uint32_t devCount = m_telemetryWorker.getDeviceCount();
    if (devCount == 0) {
        ImGui::TextDisabled("No hardware telemetry devices detected on this system.");
        return;
    }

    // Top Device Selector & Recording Controls
    auto names = m_telemetryWorker.getDeviceNames();
    for (uint32_t i = 0; i < devCount; ++i) {
        if (i > 0) ImGui::SameLine(0.0f, s(6.0f));
        std::string label = "GPU " + std::to_string(i) + ": " +
                            (i < names.size() ? names[i] : std::string("Device"));
        bool active = (m_telemetryGpuIndex == i);
        if (active) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.05f, 0.55f, 0.78f, 1.0f));
        if (ImGui::Button(label.c_str(), ImVec2(0.0f, s(26.0f)))) {
            m_telemetryGpuIndex = i;
            m_telemetryNeedsFit = true;
        }
        if (active) ImGui::PopStyleColor();
    }

    ImGui::SameLine();
    if (ImGui::Button("Clear History", ImVec2(s(110.0f), s(26.0f)))) {
        m_telemetryWorker.resetHistory();
        m_telemetryNeedsFit = true;
    }

    ImGui::SameLine();
    if (m_telemetryWorker.isRecording()) {
        ImGui::TextColored(ImVec4(0.20f, 0.94f, 0.56f, 1.0f),
                           "  [RECORDING BENCHMARK RUN: %.1fs]",
                           m_telemetryWorker.getRunDuration());
    } else {
        ImGui::TextDisabled("  [LIVE 60s ROLLING MONITOR — 10 Hz]");
    }

    DeviceTelemetrySnapshot snap{};
    if (!m_telemetryWorker.getSnapshot(m_telemetryGpuIndex, snap)) {
        ImGui::TextDisabled("Waiting for telemetry samples...");
        return;
    }

    ImGui::Spacing();

    // 6-Card Live Sensor Strip (strict N/A whenever OS/driver sensor is absent)
    if (ImGui::BeginTable("TelemetryKpiStrip", 6, ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_SizingStretchSame)) {
        auto renderCard = [&](const char* title, bool valid, const char* valText, ImVec4 col) {
            ImGui::TableNextColumn();
            ImGui::TextDisabled("%s", title);
            pushBoldFont();
            if (valid) {
                ImGui::TextColored(col, "%s", valText);
            } else {
                ImGui::TextDisabled("N/A");
            }
            popBoldFont();
        };

        char b0[48], b1[48], b2[48], b3[48], b4[48], b5[48];
        std::snprintf(b0, sizeof(b0), "%.0f%%", snap.gpuBusyPct);
        double vramUsedGb = static_cast<double>(snap.vramUsedBytes) / (1024.0 * 1024.0 * 1024.0);
        double vramTotGb = static_cast<double>(snap.vramTotalBytes) / (1024.0 * 1024.0 * 1024.0);
        std::snprintf(b1, sizeof(b1), "%.2f / %.1f GB", vramUsedGb, vramTotGb);
        std::snprintf(b2, sizeof(b2), "%.0f MHz", snap.sclkMhz);
        std::snprintf(b3, sizeof(b3), "%.0f MHz", snap.mclkMhz);
        std::snprintf(b4, sizeof(b4), "%.1f W", snap.powerWatts);
        float tVal = snap.hasTempJct ? snap.tempJctC : snap.tempEdgeC;
        std::snprintf(b5, sizeof(b5), "%.0f C%s", tVal, snap.hasTempJct ? " (Jct)" : "");

        ImGui::TableNextRow();
        renderCard("GPU UTILIZATION", snap.hasGpuBusy, b0, ImVec4(0.22f, 0.94f, 0.56f, 1.0f));
        renderCard("VRAM ALLOCATION", snap.hasVram, b1, ImVec4(0.12f, 0.82f, 0.98f, 1.0f));
        renderCard("SHADER CLOCK", snap.hasSclk, b2, ImVec4(0.38f, 0.86f, 1.00f, 1.0f));
        renderCard("MEMORY CLOCK", snap.hasMclk, b3, ImVec4(0.62f, 0.78f, 0.98f, 1.0f));
        renderCard("PACKAGE POWER", snap.hasPower, b4, ImVec4(0.98f, 0.76f, 0.22f, 1.0f));
        renderCard("TEMPERATURE", snap.hasTempJct || snap.hasTempEdge, b5, ImVec4(0.96f, 0.48f, 0.36f, 1.0f));

        ImGui::EndTable();
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    const size_t nPts = snap.timeHistory.size();
    if (nPts < 2) {
        ImGui::TextDisabled("Collecting initial time-series samples...");
        return;
    }

    const float* tData = snap.timeHistory.data();
    const double tMin = static_cast<double>(tData[0]);
    const double tMax = std::max(tMin + 5.0, static_cast<double>(tData[nPts - 1]));

    float availH = std::max(s(360.0f), ImGui::GetContentRegionAvail().y - s(10.0f));
    float plotH = (availH - s(18.0f)) / 3.0f;

    auto makeLineSpec = [](const ImVec4& col, float weight) {
        ImPlotSpec spec;
        spec.LineColor = col;
        spec.LineWeight = weight;
        return spec;
    };

    // Chart 1: GPU Utilization (%) & VRAM Used (MB)
    if (ImPlot::BeginPlot("GPU Engine Load & VRAM Allocation###PlotLoadVram", ImVec2(-1.0f, plotH))) {
        ImPlot::SetupAxes("Time (s)", "GPU Load (%)", ImPlotAxisFlags_None, ImPlotAxisFlags_None);
        ImPlot::SetupAxis(ImAxis_Y2, "VRAM Used (MB)", ImPlotAxisFlags_AuxDefault);
        ImPlot::SetupAxisLimits(ImAxis_X1, tMin, tMax, ImPlotCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 105.0, ImPlotCond_Once);
        double maxVramMb = (snap.vramTotalBytes > 0)
            ? static_cast<double>(snap.vramTotalBytes) / (1024.0 * 1024.0)
            : 16384.0;
        ImPlot::SetupAxisLimits(ImAxis_Y2, 0.0, maxVramMb * 1.05, ImPlotCond_Always);

        if (snap.hasGpuBusy && snap.gpuBusyHistory.size() == nPts) {
            ImPlot::SetAxes(ImAxis_X1, ImAxis_Y1);
            ImPlot::PlotLine("GPU Busy (%)", tData, snap.gpuBusyHistory.data(), static_cast<int>(nPts),
                             makeLineSpec(ImVec4(0.20f, 0.92f, 0.54f, 1.0f), s(2.0f)));
        }
        if (snap.hasVram && snap.vramUsedMbHistory.size() == nPts) {
            ImPlot::SetAxes(ImAxis_X1, ImAxis_Y2);
            ImPlot::PlotLine("VRAM Used (MB)", tData, snap.vramUsedMbHistory.data(), static_cast<int>(nPts),
                             makeLineSpec(ImVec4(0.12f, 0.80f, 0.98f, 1.0f), s(2.0f)));
        }
        ImPlot::EndPlot();
    }

    // Chart 2: Shader & Memory Clocks (MHz)
    if (ImPlot::BeginPlot("Clock Frequencies (MHz)###PlotClocks", ImVec2(-1.0f, plotH))) {
        ImPlot::SetupAxes("Time (s)", "Frequency (MHz)", ImPlotAxisFlags_None, ImPlotAxisFlags_AutoFit);
        ImPlot::SetupAxisLimits(ImAxis_X1, tMin, tMax, ImPlotCond_Always);

        if (snap.hasSclk && snap.sclkHistory.size() == nPts) {
            ImPlot::PlotLine("Shader Clock (SCLK)", tData, snap.sclkHistory.data(), static_cast<int>(nPts),
                             makeLineSpec(ImVec4(0.18f, 0.84f, 0.98f, 1.0f), s(2.0f)));
        }
        if (snap.hasMclk && snap.mclkHistory.size() == nPts) {
            ImPlot::PlotLine("Memory Clock (MCLK)", tData, snap.mclkHistory.data(), static_cast<int>(nPts),
                             makeLineSpec(ImVec4(0.65f, 0.52f, 0.98f, 1.0f), s(2.0f)));
        }
        if (!snap.hasSclk && !snap.hasMclk) {
            ImPlot::PlotText("Clock telemetry N/A on current OS/driver sensor interface",
                             (tMin + tMax) * 0.5, 0.5);
        }
        ImPlot::EndPlot();
    }

    // Chart 3: Board Power (W) & Thermal Sensors (C)
    if (ImPlot::BeginPlot("Package Power & Thermal Sensors###PlotPowerTemp", ImVec2(-1.0f, plotH))) {
        ImPlot::SetupAxes("Time (s)", "Power (W)", ImPlotAxisFlags_None, ImPlotAxisFlags_AutoFit);
        ImPlot::SetupAxis(ImAxis_Y2, "Temperature (C)", ImPlotAxisFlags_AuxDefault | ImPlotAxisFlags_AutoFit);
        ImPlot::SetupAxisLimits(ImAxis_X1, tMin, tMax, ImPlotCond_Always);

        if (snap.hasPower && snap.powerHistory.size() == nPts) {
            ImPlot::SetAxes(ImAxis_X1, ImAxis_Y1);
            ImPlot::PlotLine("Board Power (W)", tData, snap.powerHistory.data(), static_cast<int>(nPts),
                             makeLineSpec(ImVec4(0.98f, 0.76f, 0.20f, 1.0f), s(2.0f)));
        }
        if (snap.hasTempJct && snap.tempJctHistory.size() == nPts) {
            ImPlot::SetAxes(ImAxis_X1, ImAxis_Y2);
            ImPlot::PlotLine("Hotspot / Junction (C)", tData, snap.tempJctHistory.data(), static_cast<int>(nPts),
                             makeLineSpec(ImVec4(0.96f, 0.38f, 0.34f, 1.0f), s(2.0f)));
        }
        if (snap.hasTempEdge && snap.tempEdgeHistory.size() == nPts) {
            ImPlot::SetAxes(ImAxis_X1, ImAxis_Y2);
            ImPlot::PlotLine("GPU Edge (C)", tData, snap.tempEdgeHistory.data(), static_cast<int>(nPts),
                             makeLineSpec(ImVec4(0.98f, 0.58f, 0.28f, 1.0f), s(1.8f)));
        }
        if (snap.hasTempMem && snap.tempMemHistory.size() == nPts) {
            ImPlot::SetAxes(ImAxis_X1, ImAxis_Y2);
            ImPlot::PlotLine("VRAM Temp (C)", tData, snap.tempMemHistory.data(), static_cast<int>(nPts),
                             makeLineSpec(ImVec4(0.35f, 0.90f, 0.72f, 1.0f), s(1.8f)));
        }
        if (!snap.hasPower && !snap.hasTempJct && !snap.hasTempEdge) {
            ImPlot::PlotText("Power / Thermal telemetry N/A on current OS/driver sensor interface",
                             (tMin + tMax) * 0.5, 0.5);
        }
        ImPlot::EndPlot();
    }
}

} // namespace gpubench::gui
