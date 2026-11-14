#include "GuiApp.h"
#include "GuiUiHelpers.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace gpubench::gui {

void GuiApp::renderSidebarTelemetry() {
    const BringUpColors c = getBringUpColors(m_palette);
    pushBoldFont();
    ImGui::TextColored(c.phosphorHi, "LIVE HARDWARE TELEMETRY");
    popBoldFont();

    uint32_t devCount = m_telemetryWorker.getDeviceCount();
    if (devCount == 0) {
        ImGui::TextColored(c.textMuted, "No hardware telemetry sensors detected (N/A)");
        return;
    }

    if (devCount > 1) {
        auto names = m_telemetryWorker.getDeviceNames();
        for (uint32_t i = 0; i < devCount; ++i) {
            if (i > 0) ImGui::SameLine(0.0f, s(4.0f));
            char lbl[32];
            std::snprintf(lbl, sizeof(lbl), "GPU %u", i);
            bool sel = (m_telemetryGpuIndex == i);
            std::string bId = "##SideTelGpu_" + std::to_string(i);
            if (tacticalButton(bId.c_str(), lbl, ImVec2(0.0f, s(20.0f)), sel, c)) {
                m_telemetryGpuIndex = i;
            }
            if (ImGui::IsItemHovered() && i < names.size()) {
                ImGui::SetTooltip("%s", names[i].c_str());
            }
        }
    }

    DeviceTelemetrySnapshot snap{};
    if (!m_telemetryWorker.getSnapshot(m_telemetryGpuIndex, snap) || !snap.isAvailable) {
        ImGui::TextColored(c.textMuted, "Telemetry polling initializing... (N/A)");
        return;
    }

    ImGui::BeginChild("SidebarTelemetryCard", ImVec2(0, s(168.0f)), ImGuiChildFlags_Borders);

    auto renderSensorRow = [&](const char* label, bool available, const char* valueStr, ImVec4 valueColor) {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextColored(c.textMuted, "%s", label);
        ImGui::TableSetColumnIndex(1);
        pushMonoFont();
        if (available) {
            ImGui::TextColored(valueColor, "%s", valueStr);
        } else {
            ImGui::TextColored(c.textMuted, "N/A");
        }
        popMonoFont();
    };

    if (ImGui::BeginTable("SidebarSensorGrid", 2, ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("Sensor", ImGuiTableColumnFlags_WidthStretch, 0.46f);
        ImGui::TableSetupColumn("Reading", ImGuiTableColumnFlags_WidthStretch, 0.54f);

        char buf[64];

        std::snprintf(buf, sizeof(buf), "%.0f%%", snap.gpuBusyPct);
        renderSensorRow("GPU Load", snap.hasGpuBusy, buf,
                        (snap.gpuBusyPct > 85.0f) ? c.okColor : c.textPrimary);

        std::snprintf(buf, sizeof(buf), "%.0f MHz", snap.sclkMhz);
        renderSensorRow("Shader Clock", snap.hasSclk, buf, c.cyanAccent);

        std::snprintf(buf, sizeof(buf), "%.0f MHz", snap.mclkMhz);
        renderSensorRow("Memory Clock", snap.hasMclk, buf, c.violetAccent);

        std::snprintf(buf, sizeof(buf), "%.1f W", snap.powerWatts);
        renderSensorRow("Board Power", snap.hasPower, buf, c.phosphorHi);

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
            ? c.errColor
            : ((refTemp >= 75.0f) ? c.warnColor : c.okColor);
        renderSensorRow("Temperature", hasTemp, buf, tempCol);

        bool hasFan = snap.hasFanRpm || snap.hasFanSpeedPct;
        if (snap.hasFanRpm && snap.hasFanSpeedPct) {
            std::snprintf(buf, sizeof(buf), "%.0f RPM (%.0f%%)", snap.fanRpm, snap.fanSpeedPct);
        } else if (snap.hasFanRpm) {
            std::snprintf(buf, sizeof(buf), "%.0f RPM", snap.fanRpm);
        } else {
            std::snprintf(buf, sizeof(buf), "%.0f%%", snap.fanSpeedPct);
        }
        renderSensorRow("Cooling Fan", hasFan, buf, c.textSecondary);

        ImGui::EndTable();
    }

    if (snap.hasVram && snap.vramTotalBytes > 0) {
        double usedGb = static_cast<double>(snap.vramUsedBytes) / (1024.0 * 1024.0 * 1024.0);
        double totalGb = static_cast<double>(snap.vramTotalBytes) / (1024.0 * 1024.0 * 1024.0);
        float frac = std::clamp(static_cast<float>(usedGb / std::max(0.001, totalGb)), 0.0f, 1.0f);
        char vramOverlay[64];
        std::snprintf(vramOverlay, sizeof(vramOverlay), "VRAM: %.2f / %.1f GB (%.0f%%)",
                      usedGb, totalGb, frac * 100.0f);
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, c.phosphorMid);
        ImGui::ProgressBar(frac, ImVec2(-1.0f, s(16.0f)), vramOverlay);
        ImGui::PopStyleColor();
    } else {
        ImGui::TextColored(c.textMuted, "VRAM Usage: N/A");
    }

    ImGui::EndChild();
}

void GuiApp::renderLiveTelemetryDock() {
    const BringUpColors c = getBringUpColors(m_palette);
    uint32_t devCount = m_telemetryWorker.getDeviceCount();
    if (devCount == 0) {
        ImGui::TextColored(c.textMuted, "OSC-04 // No hardware telemetry devices detected on host.");
        return;
    }

    // Compact DSO Control Bus
    {
        pushBoldFont();
        ImGui::TextColored(c.phosphorHi, "OSC-04");
        popBoldFont();
        ImGui::SameLine(0.0f, s(6.0f));
        ImGui::TextColored(c.textPrimary, "4-CH DSO @ 10Hz");
        ImGui::SameLine(0.0f, s(10.0f));

        auto names = m_telemetryWorker.getDeviceNames();
        for (uint32_t i = 0; i < devCount; ++i) {
            char dlbl[32];
            std::snprintf(dlbl, sizeof(dlbl), "GPU %u", i);
            bool active = (m_telemetryGpuIndex == i);
            std::string gId = "##DsoGpu_" + std::to_string(i);
            if (tacticalButton(gId.c_str(), dlbl, ImVec2(0.0f, s(20.0f)), active, c)) {
                m_telemetryGpuIndex = i;
                m_telemetryNeedsFit = true;
            }
            if (ImGui::IsItemHovered() && i < names.size()) {
                ImGui::SetTooltip("%s", names[i].c_str());
            }
            ImGui::SameLine(0.0f, s(4.0f));
        }

        if (tacticalButton("##DsoClr", "CLR", ImVec2(s(38.0f), s(20.0f)), false, c)) {
            m_telemetryWorker.resetHistory();
            m_scopeTriggers.clear();
            m_telemetryNeedsFit = true;
        }
        ImGui::SameLine(0.0f, s(4.0f));

        if (tacticalButton("##DsoExpand", m_telemetryExpanded ? "COMPACT" : "EXPAND",
                           ImVec2(s(66.0f), s(20.0f)), m_telemetryExpanded, c)) {
            m_telemetryExpanded = !m_telemetryExpanded;
        }
        ImGui::SameLine(0.0f, s(10.0f));

        if (m_telemetryWorker.isRecording()) {
            ImGui::TextColored(c.okColor, "[REC %.1fs]", m_telemetryWorker.getRunDuration());
        } else {
            ImGui::TextColored(c.textMuted, "[ROLLING 60s]");
        }
    }

    DeviceTelemetrySnapshot snap{};
    if (!m_telemetryWorker.getSnapshot(m_telemetryGpuIndex, snap)) {
        ImGui::TextColored(c.textMuted, "Waiting for 10 Hz telemetry samples...");
        return;
    }

    const size_t nPts = snap.timeHistory.size();
    const float* tData = (nPts > 0) ? snap.timeHistory.data() : nullptr;
    const double tMin = (nPts >= 2) ? static_cast<double>(tData[0]) : 0.0;
    const double tMax = (nPts >= 2) ? std::max(tMin + 5.0, static_cast<double>(tData[nPts - 1])) : 60.0;
    const double tSpan = std::max(0.5, tMax - tMin);

    // Compute dynamic Y ranges for SCLK & PWR so trace fills vertically with headroom
    float sclkCeiling = 3000.0f;
    if (snap.hasSclk && snap.sclkHistory.size() == nPts) {
        const float* d = snap.sclkHistory.data();
        for (size_t i = 0; i < nPts; ++i) sclkCeiling = std::max(sclkCeiling, d[i] * 1.15f);
    }
    float pwrCeiling = 360.0f;
    if (snap.hasPower && snap.powerHistory.size() == nPts) {
        const float* d = snap.powerHistory.data();
        for (size_t i = 0; i < nPts; ++i) pwrCeiling = std::max(pwrCeiling, d[i] * 1.15f);
    }
    double vramUsedGb = static_cast<double>(snap.vramUsedBytes) / (1024.0 * 1024.0 * 1024.0);
    double vramTotGb = std::max(0.1, static_cast<double>(snap.vramTotalBytes) / (1024.0 * 1024.0 * 1024.0));

    // Helper to draw a crisp oscilloscope trace card via ImDrawList
    auto drawScopeChannel = [&](const char* chId,
                                const char* title,
                                const char* readout,
                                ImVec4 traceCol,
                                bool hasData,
                                const TelemetryBuffer& hist,
                                float yMin,
                                float yMax,
                                ImVec2 size,
                                bool showTriggerLabels,
                                const TelemetryBuffer* secondaryHist = nullptr,
                                float secMin = 0.0f,
                                float secMax = 1.0f,
                                ImVec4 secCol = ImVec4(0, 0, 0, 0)) {
        ImVec2 p0 = ImGui::GetCursorScreenPos();
        ImVec2 p1(p0.x + size.x, p0.y + size.y);
        ImDrawList* dl = ImGui::GetWindowDrawList();

        ImGui::InvisibleButton(chId, size);
        bool hovered = ImGui::IsItemHovered();

        dl->AddRectFilled(p0, p1, toU32(c.bgRecessed), 0.0f);
        dl->AddRect(p0, p1, toU32(hovered ? c.phosphorMid : c.borderOuter), 0.0f, 0, 1.0f);
        drawCornerReticles(dl, p0, p1, toU32(traceCol, 0.75f), s(5.0f));

        // Top channel readout bar
        float hdrH = ImGui::GetTextLineHeight() + s(4.0f);
        ImVec2 hdrP1(p1.x, p0.y + hdrH);
        dl->AddRectFilled(p0, hdrP1, toU32(c.bgElevated, 0.85f), 0.0f);
        dl->AddLine(ImVec2(p0.x, p0.y + hdrH), hdrP1, toU32(c.borderInner), 1.0f);

        dl->PushClipRect(p0, hdrP1, true);
        std::string fittedReadout = fitTextToPixelWidth(readout ? readout : "", std::max(0.0f, size.x - s(12.0f)));
        ImVec2 rSz = ImGui::CalcTextSize(fittedReadout.c_str());
        float maxTitleW = std::max(0.0f, (p1.x - rSz.x - s(12.0f)) - (p0.x + s(6.0f)));
        std::string fittedTitle = fitTextToPixelWidth(title ? title : "", maxTitleW);
        if (!fittedTitle.empty()) {
            dl->AddText(ImVec2(p0.x + s(6.0f), p0.y + s(2.0f)), toU32(traceCol), fittedTitle.c_str());
        }
        if (!fittedReadout.empty()) {
            dl->AddText(ImVec2(p1.x - rSz.x - s(6.0f), p0.y + s(2.0f)),
                        toU32(hasData ? c.textPrimary : c.textMuted), fittedReadout.c_str());
        }
        dl->PopClipRect();

        // Plot Area inside graticule
        float gx0 = p0.x + s(3.0f);
        float gx1 = p1.x - s(3.0f);
        float gy0 = p0.y + hdrH + s(3.0f);
        float gy1 = p1.y - s(3.0f);
        float gw = std::max(10.0f, gx1 - gx0);
        float gh = std::max(10.0f, gy1 - gy0);

        // 4x2 Graticule Grid
        for (int xi = 1; xi <= 3; ++xi) {
            float x = gx0 + (gw * static_cast<float>(xi)) / 4.0f;
            dl->AddLine(ImVec2(x, gy0), ImVec2(x, gy1), toU32(c.gridMinor), 1.0f);
        }
        float midY = gy0 + gh * 0.5f;
        dl->AddLine(ImVec2(gx0, midY), ImVec2(gx1, midY), toU32(c.gridMajor), 1.0f);

        dl->PushClipRect(ImVec2(gx0, gy0), ImVec2(gx1, gy1), true);

        // Draw optional secondary trace first (e.g., VRAM MB inside CH4)
        if (secondaryHist && secondaryHist->size() == nPts && nPts >= 2) {
            const float* secData = secondaryHist->data();
            std::vector<ImVec2> pts;
            pts.reserve(nPts);
            float denom = std::max(0.001f, secMax - secMin);
            for (size_t i = 0; i < nPts; ++i) {
                float tx = static_cast<float>((static_cast<double>(tData[i]) - tMin) / tSpan);
                float ty = std::clamp((secData[i] - secMin) / denom, 0.0f, 1.0f);
                pts.emplace_back(gx0 + tx * gw, gy1 - ty * gh);
            }
            dl->AddPolyline(pts.data(), static_cast<int>(pts.size()), toU32(secCol, 0.78f), ImDrawFlags_None, 1.3f);
        }

        // Primary Waveform
        if (hasData && hist.size() == nPts && nPts >= 2) {
            const float* hData = hist.data();
            std::vector<ImVec2> pts;
            pts.reserve(nPts);
            float denom = std::max(0.001f, yMax - yMin);
            for (size_t i = 0; i < nPts; ++i) {
                float tx = static_cast<float>((static_cast<double>(tData[i]) - tMin) / tSpan);
                float ty = std::clamp((hData[i] - yMin) / denom, 0.0f, 1.0f);
                ImVec2 pt(gx0 + tx * gw, gy1 - ty * gh);
                pts.push_back(pt);
            }
            for (size_t i = 0; i + 1 < pts.size(); ++i) {
                dl->AddQuadFilled(
                    pts[i], pts[i + 1],
                    ImVec2(pts[i + 1].x, gy1), ImVec2(pts[i].x, gy1),
                    toU32(traceCol, 0.13f));
            }
            dl->AddPolyline(pts.data(), static_cast<int>(pts.size()), toU32(traceCol), ImDrawFlags_None, 1.8f);
        } else if (!hasData) {
            dl->AddText(ImVec2(gx0 + s(6.0f), gy0 + gh * 0.35f), toU32(c.textMuted), "NO SENSOR");
        }

        // Vertical dashed TRIG markers from m_scopeTriggers
        for (const auto& tr : m_scopeTriggers) {
            if (tr.timestampSec < tMin || tr.timestampSec > tMax) continue;
            float x = gx0 + static_cast<float>((tr.timestampSec - tMin) / tSpan) * gw;
            for (float dy = gy0; dy < gy1; dy += s(7.0f)) {
                dl->AddLine(ImVec2(x, dy), ImVec2(x, std::min(gy1, dy + s(4.0f))), toU32(c.warnColor, 0.90f), 1.2f);
            }
            if (showTriggerLabels && !tr.label.empty()) {
                std::string shortLbl = tr.label.substr(0, std::min<size_t>(10, tr.label.size()));
                dl->AddText(ImVec2(x + s(2.0f), gy0 + s(1.0f)), toU32(c.warnColor), shortLbl.c_str());
            }
        }

        dl->PopClipRect();

        if (hovered && hasData && !hist.empty()) {
            const float* hData = hist.data();
            float minV = hData[0], maxV = hData[0];
            for (size_t i = 1; i < hist.size(); ++i) {
                minV = std::min(minV, hData[i]);
                maxV = std::max(maxV, hData[i]);
            }
            ImGui::BeginTooltip();
            ImGui::TextColored(traceCol, "%s : %s", title, readout);
            ImGui::Text("Window Min/Max: %.1f .. %.1f", minV, maxV);
            if (!m_scopeTriggers.empty()) {
                const auto& lastTrig = m_scopeTriggers.back();
                ImGui::TextColored(c.warnColor, "Last TRIG: %s (%.1fs)", lastTrig.label.c_str(), lastTrig.timestampSec);
            }
            ImGui::EndTooltip();
        }
    };

    float availW = std::max(s(320.0f), ImGui::GetContentRegionAvail().x);
    float availH = std::max(s(44.0f), ImGui::GetContentRegionAvail().y);
    float gap = s(6.0f);
    float colW = (availW - gap * 3.0f) / 4.0f;
    float stripH = m_telemetryExpanded
        ? s(54.0f)
        : std::max(s(44.0f), availH - s(2.0f));

    char ch1Val[40], ch2Val[40], ch3Val[40], ch4Val[64];
    std::snprintf(ch1Val, sizeof(ch1Val), snap.hasSclk ? "%.0f MHz" : "N/A", snap.sclkMhz);
    std::snprintf(ch2Val, sizeof(ch2Val), snap.hasPower ? "%.1f W" : "N/A", snap.powerWatts);
    std::snprintf(ch3Val, sizeof(ch3Val), snap.hasGpuBusy ? "%.0f%%" : "N/A", snap.gpuBusyPct);

    float refTemp = snap.hasTempJct ? snap.tempJctC : snap.tempEdgeC;
    bool hasTemp = snap.hasTempJct || snap.hasTempEdge;
    if (hasTemp && snap.hasVram) {
        std::snprintf(ch4Val, sizeof(ch4Val), "%.0fC | %.1f/%.0fG", refTemp, vramUsedGb, vramTotGb);
    } else if (hasTemp) {
        std::snprintf(ch4Val, sizeof(ch4Val), "%.0f C", refTemp);
    } else if (snap.hasVram) {
        std::snprintf(ch4Val, sizeof(ch4Val), "%.1f/%.0fG", vramUsedGb, vramTotGb);
    } else {
        std::snprintf(ch4Val, sizeof(ch4Val), "N/A");
    }

    drawScopeChannel("##Ch1Sclk", "CH1 SCLK", ch1Val, c.cyanAccent,
                     snap.hasSclk, snap.sclkHistory, 0.0f, sclkCeiling,
                     ImVec2(colW, stripH), true);
    ImGui::SameLine(0.0f, gap);
    drawScopeChannel("##Ch2Pwr", "CH2 PWR", ch2Val, c.phosphorHi,
                     snap.hasPower, snap.powerHistory, 0.0f, pwrCeiling,
                     ImVec2(colW, stripH), false);
    ImGui::SameLine(0.0f, gap);
    drawScopeChannel("##Ch3Busy", "CH3 BUSY", ch3Val, c.okColor,
                     snap.hasGpuBusy, snap.gpuBusyHistory, 0.0f, 105.0f,
                     ImVec2(colW, stripH), false);
    ImGui::SameLine(0.0f, gap);
    const TelemetryBuffer& tempHist = snap.hasTempJct ? snap.tempJctHistory : snap.tempEdgeHistory;
    float maxVramMb = static_cast<float>(vramTotGb * 1024.0);
    drawScopeChannel("##Ch4TempVram", "CH4 TMP/VRAM", ch4Val, c.warnColor,
                     hasTemp || snap.hasVram, tempHist, 20.0f, 105.0f,
                     ImVec2(colW, stripH), false,
                     snap.hasVram ? &snap.vramUsedMbHistory : nullptr,
                     0.0f, std::max(1024.0f, maxVramMb), c.violetAccent);

    if (!m_telemetryExpanded) {
        return;
    }

    // Expanded Detailed ImPlot Traces
    if (nPts < 2) {
        ImGui::TextColored(c.textMuted, "Collecting initial time-series samples...");
        return;
    }

    ImGui::Spacing();
    const float plotGap = s(6.0f);
    const float plotColW = std::max(s(160.0f), (availW - plotGap * 2.0f) / 3.0f);
    const float plotH = std::max(s(120.0f), ImGui::GetContentRegionAvail().y - s(4.0f));

    auto makeLineSpec = [](const ImVec4& col, float weight) {
        ImPlotSpec spec;
        spec.LineColor = col;
        spec.LineWeight = weight;
        return spec;
    };

    // Chart 1: GPU Utilization (%) & VRAM Used (MB)
    if (ImPlot::BeginPlot("CH3/CH4 // ENGINE & VRAM###PlotLoadVram", ImVec2(plotColW, plotH))) {
        ImPlot::SetupAxes("Time (s)", "GPU Load (%)", ImPlotAxisFlags_None, ImPlotAxisFlags_None);
        ImPlot::SetupAxis(ImAxis_Y2, "VRAM (MB)", ImPlotAxisFlags_AuxDefault);
        ImPlot::SetupAxisLimits(ImAxis_X1, tMin, tMax, ImPlotCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 105.0, ImPlotCond_Once);
        double maxVramMbPlot = (snap.vramTotalBytes > 0)
            ? static_cast<double>(snap.vramTotalBytes) / (1024.0 * 1024.0)
            : 16384.0;
        ImPlot::SetupAxisLimits(ImAxis_Y2, 0.0, maxVramMbPlot * 1.05, ImPlotCond_Always);

        if (snap.hasGpuBusy && snap.gpuBusyHistory.size() == nPts) {
            ImPlot::SetAxes(ImAxis_X1, ImAxis_Y1);
            ImPlot::PlotLine("GPU Busy (%)", tData, snap.gpuBusyHistory.data(), static_cast<int>(nPts),
                             makeLineSpec(c.okColor, s(2.0f)));
        }
        if (snap.hasVram && snap.vramUsedMbHistory.size() == nPts) {
            ImPlot::SetAxes(ImAxis_X1, ImAxis_Y2);
            ImPlot::PlotLine("VRAM (MB)", tData, snap.vramUsedMbHistory.data(), static_cast<int>(nPts),
                             makeLineSpec(c.violetAccent, s(2.0f)));
        }
        ImPlot::EndPlot();
    }

    ImGui::SameLine(0.0f, plotGap);

    // Chart 2: Shader & Memory Clocks (MHz)
    if (ImPlot::BeginPlot("CH1 // CLOCKS (MHz)###PlotClocks", ImVec2(plotColW, plotH))) {
        ImPlot::SetupAxes("Time (s)", "MHz", ImPlotAxisFlags_None, ImPlotAxisFlags_AutoFit);
        ImPlot::SetupAxisLimits(ImAxis_X1, tMin, tMax, ImPlotCond_Always);

        if (snap.hasSclk && snap.sclkHistory.size() == nPts) {
            ImPlot::PlotLine("SCLK", tData, snap.sclkHistory.data(), static_cast<int>(nPts),
                             makeLineSpec(c.cyanAccent, s(2.0f)));
        }
        if (snap.hasMclk && snap.mclkHistory.size() == nPts) {
            ImPlot::PlotLine("MCLK", tData, snap.mclkHistory.data(), static_cast<int>(nPts),
                             makeLineSpec(c.violetAccent, s(2.0f)));
        }
        if (!snap.hasSclk && !snap.hasMclk) {
            ImPlot::PlotText("Clock telemetry N/A", (tMin + tMax) * 0.5, 0.5);
        }
        ImPlot::EndPlot();
    }

    ImGui::SameLine(0.0f, plotGap);

    // Chart 3: Board Power (W) & Thermal Sensors (C)
    if (ImPlot::BeginPlot("CH2/CH4 // POWER & THERMALS###PlotPowerTemp", ImVec2(plotColW, plotH))) {
        ImPlot::SetupAxes("Time (s)", "Power (W)", ImPlotAxisFlags_None, ImPlotAxisFlags_AutoFit);
        ImPlot::SetupAxis(ImAxis_Y2, "Temp (C)", ImPlotAxisFlags_AuxDefault | ImPlotAxisFlags_AutoFit);
        ImPlot::SetupAxisLimits(ImAxis_X1, tMin, tMax, ImPlotCond_Always);

        if (snap.hasPower && snap.powerHistory.size() == nPts) {
            ImPlot::SetAxes(ImAxis_X1, ImAxis_Y1);
            ImPlot::PlotLine("Power (W)", tData, snap.powerHistory.data(), static_cast<int>(nPts),
                             makeLineSpec(c.phosphorHi, s(2.0f)));
        }
        if (snap.hasTempJct && snap.tempJctHistory.size() == nPts) {
            ImPlot::SetAxes(ImAxis_X1, ImAxis_Y2);
            ImPlot::PlotLine("Jct (C)", tData, snap.tempJctHistory.data(), static_cast<int>(nPts),
                             makeLineSpec(c.errColor, s(2.0f)));
        }
        if (snap.hasTempEdge && snap.tempEdgeHistory.size() == nPts) {
            ImPlot::SetAxes(ImAxis_X1, ImAxis_Y2);
            ImPlot::PlotLine("Edge (C)", tData, snap.tempEdgeHistory.data(), static_cast<int>(nPts),
                             makeLineSpec(c.warnColor, s(1.8f)));
        }
        if (snap.hasTempMem && snap.tempMemHistory.size() == nPts) {
            ImPlot::SetAxes(ImAxis_X1, ImAxis_Y2);
            ImPlot::PlotLine("Mem (C)", tData, snap.tempMemHistory.data(), static_cast<int>(nPts),
                             makeLineSpec(c.okColor, s(1.8f)));
        }
        if (!snap.hasPower && !snap.hasTempJct && !snap.hasTempEdge) {
            ImPlot::PlotText("Power / Thermal N/A", (tMin + tMax) * 0.5, 0.5);
        }
        ImPlot::EndPlot();
    }
}

} // namespace gpubench::gui
