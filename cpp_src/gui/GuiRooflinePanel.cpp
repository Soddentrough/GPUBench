#include "GuiApp.h"
#include "GuiUiHelpers.h"
#include "core/RooflineModel.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace gpubench::gui {

void GuiApp::renderRooflinePanel() {
    ImGui::Spacing();

    auto rooflines = gpubench::RooflineModel::analyze(m_allResults);
    if (rooflines.empty()) {
        ImGui::Spacing();
        ImGui::TextDisabled("Empirical Roofline Model requires at least one measured GPU compute or memory bandwidth result.");
        ImGui::TextDisabled("Tip: Click 'Quick Roofline Preset' in the 'Benchmark Suite' tab and run, or import a JSON file.");
        return;
    }

    if (m_rooflineDeviceFilter < 0 || m_rooflineDeviceFilter >= static_cast<int>(rooflines.size())) {
        m_rooflineDeviceFilter = 0;
    }

    // Device Selector Pills (if multi-GPU / multi-backend results exist)
    if (rooflines.size() > 1) {
        for (size_t i = 0; i < rooflines.size(); ++i) {
            if (i > 0) sameLineIfFits(s(180.0f), s(6.0f));
            std::string lbl = "[GPU " + std::to_string(rooflines[i].deviceIndex) + "] " +
                              rooflines[i].deviceName + " (" + rooflines[i].backendName + ")";
            bool active = (m_rooflineDeviceFilter == static_cast<int>(i));
            if (active) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.05f, 0.55f, 0.78f, 1.0f));
            if (ImGui::SmallButton(lbl.c_str())) {
                m_rooflineDeviceFilter = static_cast<int>(i);
            }
            if (active) ImGui::PopStyleColor();
        }
        ImGui::Spacing();
    }

    const auto& rf = rooflines[static_cast<size_t>(m_rooflineDeviceFilter)];
    const double vramGbs = rf.primaryVramGBps();
    const double fp32Tops = rf.primaryFp32Gops() / 1000.0;
    const double maxTops = rf.highestComputeGops() / 1000.0;
    const double fp32Ridge = (vramGbs > 0.0 && fp32Tops > 0.0) ? ((fp32Tops * 1000.0) / vramGbs) : 0.0;

    // 4-Card Roofline Summary Strip
    if (ImGui::BeginTable("RooflineKpiStrip", 4, ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_SizingStretchSame)) {
        ImGui::TableNextRow();

        ImGui::TableNextColumn();
        ImGui::TextDisabled("EMPIRICAL VRAM CEILING");
        pushBoldFont();
        if (vramGbs > 0.0) ImGui::TextColored(ImVec4(0.22f, 0.92f, 0.56f, 1.0f), "%.1f GB/s", vramGbs);
        else ImGui::TextDisabled("Not Measured");
        popBoldFont();

        ImGui::TableNextColumn();
        ImGui::TextDisabled("EMPIRICAL FP32 CEILING");
        pushBoldFont();
        if (fp32Tops > 0.0) ImGui::TextColored(ImVec4(0.12f, 0.84f, 0.98f, 1.0f), "%.2f TFLOPS", fp32Tops);
        else ImGui::TextDisabled("Not Measured");
        popBoldFont();

        ImGui::TableNextColumn();
        ImGui::TextDisabled("PEAK MATRIX / TENSOR CEILING");
        pushBoldFont();
        if (maxTops > 0.0) ImGui::TextColored(ImVec4(0.98f, 0.76f, 0.22f, 1.0f), "%.2f TOPS/TFLOPS", maxTops);
        else ImGui::TextDisabled("Not Measured");
        popBoldFont();

        ImGui::TableNextColumn();
        ImGui::TextDisabled("FP32 x VRAM RIDGE POINT");
        pushBoldFont();
        if (fp32Ridge > 0.0) ImGui::TextColored(ImVec4(0.86f, 0.58f, 0.98f, 1.0f), "%.2f FLOP/Byte", fp32Ridge);
        else ImGui::TextDisabled("—");
        popBoldFont();

        ImGui::EndTable();
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Interactive Log-Log Empirical Roofline Chart
    float plotH = std::max(s(320.0f), ImGui::GetContentRegionAvail().y * 0.58f);
    const double oiMin = 0.03;
    const double oiMax = 4096.0;
    const double gopsMin = 10.0;
    const double gopsMax = std::max(100000.0, rf.highestComputeGops() * 2.5);

    std::string plotTitle = "Empirical Log-Log Roofline — " + rf.deviceName + " (" + rf.backendName + ")###RooflineLogLogPlot";
    if (ImPlot::BeginPlot(plotTitle.c_str(), ImVec2(-1.0f, plotH))) {
        ImPlot::SetupAxes("Operational Intensity (FLOP/Byte or OP/Byte)",
                          "Achieved Performance (GFLOPS / GOPS)",
                          ImPlotAxisFlags_None, ImPlotAxisFlags_None);
        ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Log10);
        ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Log10);
        ImPlot::SetupAxesLimits(oiMin, oiMax, gopsMin, gopsMax, ImPlotCond_Once);
        ImPlot::SetupLegend(ImPlotLocation_NorthWest, ImPlotLegendFlags_None);

        const double topCapGops = (rf.highestComputeGops() > 0.0) ? rf.highestComputeGops() : (gopsMax * 0.6);

        // 1. Slanted Bandwidth Ceilings (P = BW * OI, capped at highest compute ceiling)
        static const ImVec4 bwColors[] = {
            ImVec4(0.22f, 0.94f, 0.56f, 0.95f), // L0 / VRAM
            ImVec4(0.16f, 0.82f, 0.98f, 0.90f),
            ImVec4(0.54f, 0.72f, 0.98f, 0.88f),
            ImVec4(0.84f, 0.60f, 0.98f, 0.88f),
            ImVec4(0.98f, 0.76f, 0.24f, 0.90f),
            ImVec4(0.94f, 0.48f, 0.38f, 0.85f)
        };

        for (size_t bi = 0; bi < rf.bandwidthCeilings.size(); ++bi) {
            const auto& bw = rf.bandwidthCeilings[bi];
            if (bw.bandwidthGBps <= 0.0) continue;

            double kneeOi = std::clamp(topCapGops / bw.bandwidthGBps, oiMin * 1.05, oiMax);
            double xs[2] = {oiMin, kneeOi};
            double ys[2] = {std::max(0.1, bw.bandwidthGBps * oiMin), bw.bandwidthGBps * kneeOi};

            ImPlotSpec spec;
            spec.LineColor = bwColors[bi % 6];
            spec.LineWeight = (bw.level == "VRAM") ? s(2.6f) : s(1.7f);

            char lbl[96];
            std::snprintf(lbl, sizeof(lbl), "%s BW (%.0f GB/s)", bw.level.c_str(), bw.bandwidthGBps);
            ImPlot::PlotLine(lbl, xs, ys, 2, spec);
        }

        // 2. Horizontal Compute Ceilings (starting from VRAM ridge point when VRAM BW is known)
        static const ImVec4 compColors[] = {
            ImVec4(0.10f, 0.84f, 0.98f, 0.95f),
            ImVec4(0.98f, 0.78f, 0.22f, 0.95f),
            ImVec4(0.32f, 0.94f, 0.66f, 0.95f),
            ImVec4(0.96f, 0.46f, 0.68f, 0.95f),
            ImVec4(0.68f, 0.56f, 0.98f, 0.95f)
        };

        for (size_t ci = 0; ci < rf.computeCeilings.size(); ++ci) {
            const auto& cc = rf.computeCeilings[ci];
            if (cc.peakGops <= 0.0) continue;

            double startOi = (vramGbs > 0.0)
                ? std::clamp(cc.peakGops / vramGbs, oiMin, oiMax * 0.8)
                : oiMin;
            double xs[2] = {startOi, oiMax};
            double ys[2] = {cc.peakGops, cc.peakGops};

            ImPlotSpec spec;
            spec.LineColor = compColors[ci % 5];
            spec.LineWeight = s(2.2f);

            char lbl[96];
            std::snprintf(lbl, sizeof(lbl), "%s (%.1f %s)", cc.name.c_str(), cc.peakTops, cc.metric.c_str());
            ImPlot::PlotLine(lbl, xs, ys, 2, spec);
        }

        // 3. Ridge Point Markers (Diamonds)
        if (!rf.ridgePoints.empty()) {
            std::vector<double> rxs, rys;
            for (const auto& rp : rf.ridgePoints) {
                if (rp.ridgeOpsPerByte > 0.0 && rp.peakTops > 0.0) {
                    rxs.push_back(rp.ridgeOpsPerByte);
                    rys.push_back(rp.peakTops * 1000.0);
                }
            }
            if (!rxs.empty()) {
                ImPlotSpec rSpec;
                rSpec.Marker = ImPlotMarker_Diamond;
                rSpec.MarkerSize = s(6.0f);
                rSpec.LineColor = ImVec4(1.0f, 0.92f, 0.36f, 1.0f);
                rSpec.MarkerFillColor = ImVec4(1.0f, 0.92f, 0.36f, 1.0f);
                ImPlot::PlotScatter("Ridge Points", rxs.data(), rys.data(), static_cast<int>(rxs.size()), rSpec);
            }
        }

        // 4. Measured Workload Scatter Points (Vector/Memory vs Matrix Cores)
        std::vector<double> vecX, vecY, matX, matY;
        for (const auto& wp : rf.workloadPoints) {
            if (wp.operationalIntensity <= 0.0 || wp.achievedGops <= 0.0) continue;
            if (wp.isMatrix) {
                matX.push_back(wp.operationalIntensity);
                matY.push_back(wp.achievedGops);
            } else {
                vecX.push_back(wp.operationalIntensity);
                vecY.push_back(wp.achievedGops);
            }
        }

        if (!vecX.empty()) {
            ImPlotSpec vSpec;
            vSpec.Marker = ImPlotMarker_Circle;
            vSpec.MarkerSize = s(5.5f);
            vSpec.LineColor = ImVec4(0.16f, 0.90f, 1.0f, 1.0f);
            vSpec.MarkerFillColor = ImVec4(0.16f, 0.90f, 1.0f, 0.90f);
            ImPlot::PlotScatter("Vector / Bandwidth Workloads", vecX.data(), vecY.data(), static_cast<int>(vecX.size()), vSpec);
        }
        if (!matX.empty()) {
            ImPlotSpec mSpec;
            mSpec.Marker = ImPlotMarker_Square;
            mSpec.MarkerSize = s(6.0f);
            mSpec.LineColor = ImVec4(0.98f, 0.62f, 0.22f, 1.0f);
            mSpec.MarkerFillColor = ImVec4(0.98f, 0.62f, 0.22f, 0.92f);
            ImPlot::PlotScatter("Matrix / Tensor Workloads", matX.data(), matY.data(), static_cast<int>(matX.size()), mSpec);
        }

        // Interactive Hover Tooltip over Nearest Workload Point (in log10 space)
        if (ImPlot::IsPlotHovered() && !rf.workloadPoints.empty()) {
            ImPlotPoint mouse = ImPlot::GetPlotMousePos();
            if (mouse.x > 0.0 && mouse.y > 0.0) {
                double logMx = std::log10(mouse.x);
                double logMy = std::log10(mouse.y);
                const gpubench::RooflineWorkloadPoint* bestPt = nullptr;
                double bestDist2 = 0.035; // log10 proximity radius squared
                for (const auto& wp : rf.workloadPoints) {
                    if (wp.operationalIntensity <= 0.0 || wp.achievedGops <= 0.0) continue;
                    double dx = std::log10(wp.operationalIntensity) - logMx;
                    double dy = std::log10(wp.achievedGops) - logMy;
                    double d2 = dx * dx + dy * dy;
                    if (d2 < bestDist2) {
                        bestDist2 = d2;
                        bestPt = &wp;
                    }
                }
                if (bestPt) {
                    ImGui::BeginTooltip();
                    pushBoldFont();
                    ImGui::TextColored(ImVec4(0.12f, 0.86f, 0.98f, 1.0f), "%s", bestPt->label.c_str());
                    popBoldFont();
                    ImGui::Separator();
                    ImGui::Text("Operational Intensity: %.2f OP/Byte", bestPt->operationalIntensity);
                    ImGui::Text("Achieved Performance:  %.2f %s (%.0f GOPS)",
                                bestPt->achievedGops / 1000.0, bestPt->metric.c_str(), bestPt->achievedGops);
                    if (bestPt->attainmentPct > 0.0) {
                        ImGui::Text("Roofline Attainment:   %.1f%% of %s (%s)",
                                    bestPt->attainmentPct,
                                    bestPt->targetCeilingName.c_str(),
                                    bestPt->isMemoryBound ? "Memory-Bound" : "Compute-Bound");
                    }
                    ImGui::EndTooltip();
                }
            }
        }

        ImPlot::EndPlot();
    }

    ImGui::Spacing();

    // Workload Roofline Attainment Table
    if (!rf.workloadPoints.empty()) {
        ImGuiTableFlags tblFlags = ImGuiTableFlags_Borders |
                                   ImGuiTableFlags_RowBg |
                                   ImGuiTableFlags_ScrollY |
                                   ImGuiTableFlags_SizingStretchProp;
        if (ImGui::BeginTable("RooflineAttainmentTable", 6, tblFlags, ImVec2(0.0f, 0.0f))) {
            ImGui::TableSetupScrollFreeze(0, 1);
            ImGui::TableSetupColumn("Workload", ImGuiTableColumnFlags_WidthStretch, 0.34f);
            ImGui::TableSetupColumn("Operational Intensity", ImGuiTableColumnFlags_WidthStretch, 0.14f);
            ImGui::TableSetupColumn("Achieved", ImGuiTableColumnFlags_WidthStretch, 0.16f);
            ImGui::TableSetupColumn("Target Ceiling", ImGuiTableColumnFlags_WidthStretch, 0.16f);
            ImGui::TableSetupColumn("Regime", ImGuiTableColumnFlags_WidthStretch, 0.10f);
            ImGui::TableSetupColumn("Attainment", ImGuiTableColumnFlags_WidthStretch, 0.10f);
            ImGui::TableHeadersRow();

            for (const auto& wp : rf.workloadPoints) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextUnformatted(wp.label.c_str());

                ImGui::TableSetColumnIndex(1);
                pushMonoFont();
                ImGui::Text("%.2f OP/B", wp.operationalIntensity);
                popMonoFont();

                ImGui::TableSetColumnIndex(2);
                pushMonoFont();
                if (wp.achievedGops >= 1000.0) {
                    ImGui::TextColored(ImVec4(0.90f, 0.96f, 1.0f, 1.0f), "%.2f %s",
                                       wp.achievedGops / 1000.0, wp.metric.c_str());
                } else {
                    ImGui::TextColored(ImVec4(0.90f, 0.96f, 1.0f, 1.0f), "%.1f GOPS", wp.achievedGops);
                }
                popMonoFont();

                ImGui::TableSetColumnIndex(3);
                ImGui::TextDisabled("%s", wp.targetCeilingName.c_str());

                ImGui::TableSetColumnIndex(4);
                if (wp.isMemoryBound) {
                    ImGui::TextColored(ImVec4(0.22f, 0.90f, 0.62f, 1.0f), "Memory-Bound");
                } else {
                    ImGui::TextColored(ImVec4(0.18f, 0.82f, 0.98f, 1.0f), "Compute-Bound");
                }

                ImGui::TableSetColumnIndex(5);
                pushMonoFont();
                if (wp.attainmentPct > 0.0) {
                    ImVec4 attCol = (wp.attainmentPct >= 75.0)
                        ? ImVec4(0.20f, 0.92f, 0.54f, 1.0f)
                        : ((wp.attainmentPct >= 40.0) ? ImVec4(0.98f, 0.78f, 0.24f, 1.0f)
                                                      : ImVec4(0.82f, 0.86f, 0.92f, 1.0f));
                    ImGui::TextColored(attCol, "%.1f%%", wp.attainmentPct);
                } else {
                    ImGui::TextDisabled("—");
                }
                popMonoFont();
            }
            ImGui::EndTable();
        }
    }
}

} // namespace gpubench::gui
