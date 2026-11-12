#include "GuiApp.h"
#include "GuiUiHelpers.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <set>

namespace gpubench::gui {

void GuiApp::renderResultsScorecard() {
    ImGui::Spacing();

    if (m_allResults.empty()) {
        ImGui::Spacing();
        ImGui::TextDisabled("No benchmark results recorded yet.");
        ImGui::TextDisabled("Select workloads in the 'Benchmark Suite' tab and click 'RUN SELECTED SUITE', "
                            "or click 'Import/Diff' to load a unified CLI/GUI JSON file.");
        return;
    }

    // Compute Executive KPI Peaks
    double peakComputeTops = 0.0;
    std::string peakComputeName;
    std::string peakComputeUnit = "TFLOPS";
    double peakVramGbs = 0.0;
    std::string peakVramName;
    double peakRtMrays = 0.0;
    std::string peakRtName;
    size_t validCount = 0;
    size_t unsupportedCount = 0;
    size_t invalidCount = 0;

    for (const auto& r : m_allResults) {
        if (r.isUnsupported) {
            unsupportedCount++;
            continue;
        }
        if (!r.isValid || r.time_ms <= 0.0) {
            invalidCount++;
            continue;
        }
        validCount++;

        double opsPerSec = (r.operations > 0) ? ((static_cast<double>(r.operations) / r.time_ms) * 1000.0) : 0.0;
        if (r.metric == "TFLOPS" || r.metric == "TOPS") {
            double tops = opsPerSec / 1e12;
            if (tops > peakComputeTops) {
                peakComputeTops = tops;
                peakComputeName = r.benchmarkName;
                peakComputeUnit = r.metric;
            }
        }
        if (r.component == "Memory" && r.subcategory == "Bandwidth" && r.metric == "GB/s") {
            double gbs = opsPerSec / 1e9;
            if (gbs > peakVramGbs) {
                peakVramGbs = gbs;
                peakVramName = r.benchmarkName;
            }
        }
        if (r.component == "Ray Tracing" && r.metric == "MRays/s") {
            double mrays = opsPerSec / 1e6;
            if (mrays > peakRtMrays) {
                peakRtMrays = mrays;
                peakRtName = r.benchmarkName;
            }
        }
    }

    // Executive KPI Cards
    if (ImGui::BeginTable("ScorecardKpiStrip", 4, ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_SizingStretchSame)) {
        ImGui::TableNextRow();

        ImGui::TableNextColumn();
        ImGui::TextDisabled("PEAK COMPUTE");
        pushBoldFont();
        if (peakComputeTops > 0.0) {
            ImGui::TextColored(ImVec4(0.10f, 0.84f, 0.98f, 1.0f), "%.2f %s", peakComputeTops, peakComputeUnit.c_str());
        } else {
            ImGui::TextDisabled("—");
        }
        popBoldFont();
        if (!peakComputeName.empty()) {
            ImGui::TextDisabled("%s", fitTextToPixelWidth(peakComputeName, ImGui::GetContentRegionAvail().x).c_str());
        }

        ImGui::TableNextColumn();
        ImGui::TextDisabled("PEAK VRAM BANDWIDTH");
        pushBoldFont();
        if (peakVramGbs > 0.0) {
            ImGui::TextColored(ImVec4(0.22f, 0.92f, 0.56f, 1.0f), "%.1f GB/s", peakVramGbs);
        } else {
            ImGui::TextDisabled("—");
        }
        popBoldFont();
        if (!peakVramName.empty()) {
            ImGui::TextDisabled("%s", fitTextToPixelWidth(peakVramName, ImGui::GetContentRegionAvail().x).c_str());
        }

        ImGui::TableNextColumn();
        ImGui::TextDisabled("PEAK RAY TRACING");
        pushBoldFont();
        if (peakRtMrays > 0.0) {
            ImGui::TextColored(ImVec4(0.98f, 0.76f, 0.24f, 1.0f), "%.1f MRays/s", peakRtMrays);
        } else {
            ImGui::TextDisabled("—");
        }
        popBoldFont();
        if (!peakRtName.empty()) {
            ImGui::TextDisabled("%s", fitTextToPixelWidth(peakRtName, ImGui::GetContentRegionAvail().x).c_str());
        }

        ImGui::TableNextColumn();
        ImGui::TextDisabled("VALIDATION & ROOFLINE");
        pushBoldFont();
        if (invalidCount > 0) {
            ImGui::TextColored(ImVec4(0.96f, 0.36f, 0.34f, 1.0f), "%zu Valid / %zu INVALID", validCount, invalidCount);
        } else {
            ImGui::TextColored(ImVec4(0.22f, 0.92f, 0.56f, 1.0f), "%zu Valid (%zu Unsup)", validCount, unsupportedCount);
        }
        popBoldFont();
        if (ImGui::SmallButton("View Empirical Roofline ->")) {
            m_switchToRoofline = true;
        }

        ImGui::EndTable();
    }

    // Active Comparison Banner
    if (!m_comparisonResults.empty()) {
        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.08f, 0.16f, 0.24f, 0.95f));
        ImGui::BeginChild("ComparisonBanner", ImVec2(0, s(34.0f)), ImGuiChildFlags_Borders);
        ImGui::TextColored(ImVec4(0.25f, 0.88f, 0.98f, 1.0f),
                           "Comparison Mode Active  |  Baseline: %s (%zu records)",
                           m_comparisonSourceLabel.c_str(), m_comparisonResults.size());
        ImGui::SameLine();
        if (ImGui::SmallButton("Clear Baseline")) {
            clearComparisonResults();
        }
        ImGui::EndChild();
        ImGui::PopStyleColor();
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Category Filter Pills
    const char* catFilters[] = {"All", "Compute", "Memory", "Ray Tracing", "Scheduling", "Graphics", "Host System"};
    for (int i = 0; i < 7; ++i) {
        if (i > 0) sameLineIfFits(s(95.0f), s(6.0f));
        bool active = (m_activeScorecardFilter == i);
        if (active) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.05f, 0.55f, 0.78f, 1.0f));
        if (ImGui::SmallButton(catFilters[i])) {
            m_activeScorecardFilter = i;
        }
        if (active) ImGui::PopStyleColor();
    }

    // Device Filter Pills (if multiple devices exist in results)
    std::set<uint32_t> seenDevs;
    bool hasHostCpu = false;
    for (const auto& r : m_allResults) {
        if (r.component == "System" || r.backendName == "CPU") hasHostCpu = true;
        else seenDevs.insert(r.deviceIndex);
    }
    if (seenDevs.size() > 1 || (seenDevs.size() >= 1 && hasHostCpu)) {
        ImGui::SameLine(0.0f, s(18.0f));
        ImGui::TextDisabled("Device:");
        ImGui::SameLine();
        if (ImGui::SmallButton("All Devs")) m_activeDeviceScorecardFilter = -1;
        for (uint32_t d : seenDevs) {
            ImGui::SameLine();
            char dlbl[32];
            std::snprintf(dlbl, sizeof(dlbl), "GPU %u", d);
            if (ImGui::SmallButton(dlbl)) m_activeDeviceScorecardFilter = static_cast<int>(d);
        }
        if (hasHostCpu) {
            ImGui::SameLine();
            if (ImGui::SmallButton("Host CPU")) m_activeDeviceScorecardFilter = -2;
        }
    }

    ImGui::Spacing();

    auto matchesCategoryFilter = [&](const ResultData& r) {
        switch (m_activeScorecardFilter) {
        case 0: return true;
        case 1: return r.component == "Compute" && r.subcategory.find("Indirect") == std::string::npos;
        case 2: return r.component == "Memory";
        case 3: return r.component == "Ray Tracing";
        case 4: return r.component == "Scheduling" || r.subcategory.find("Indirect") != std::string::npos;
        case 5: return r.component == "Graphics" || r.component == "Raster";
        case 6: return r.component == "System" || r.backendName == "CPU";
        default: return true;
        }
    };

    auto matchesDeviceFilter = [&](const ResultData& r) {
        if (m_activeDeviceScorecardFilter == -1) return true;
        if (m_activeDeviceScorecardFilter == -2) return r.component == "System" || r.backendName == "CPU";
        if (r.component == "System" || r.backendName == "CPU") return false;
        return static_cast<int>(r.deviceIndex) == m_activeDeviceScorecardFilter;
    };

    ImGuiTableFlags flags = ImGuiTableFlags_Borders |
                            ImGuiTableFlags_RowBg |
                            ImGuiTableFlags_ScrollY |
                            ImGuiTableFlags_Resizable |
                            ImGuiTableFlags_SizingStretchProp;

    const char* deltaColTitle = m_comparisonResults.empty() ? "vs Baseline" : "vs Comparison JSON";
    if (ImGui::BeginTable("ResultsScorecardTable", 7, flags, ImVec2(0.0f, 0.0f))) {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("Device", ImGuiTableColumnFlags_WidthStretch, 0.14f);
        ImGui::TableSetupColumn("Domain", ImGuiTableColumnFlags_WidthStretch, 0.13f);
        ImGui::TableSetupColumn("Workload", ImGuiTableColumnFlags_WidthStretch, 0.30f);
        ImGui::TableSetupColumn("Throughput / Score", ImGuiTableColumnFlags_WidthStretch, 0.14f);
        ImGui::TableSetupColumn("Median Time", ImGuiTableColumnFlags_WidthStretch, 0.10f);
        ImGui::TableSetupColumn(deltaColTitle, ImGuiTableColumnFlags_WidthStretch, 0.10f);
        ImGui::TableSetupColumn("ISA / Power", ImGuiTableColumnFlags_WidthStretch, 0.09f);
        ImGui::TableHeadersRow();

        int rowId = 0;
        for (const auto& r : m_allResults) {
            if (!matchesCategoryFilter(r) || !matchesDeviceFilter(r)) continue;

            ImGui::PushID(rowId++);
            ImGui::TableNextRow();

            // Col 0: Device
            ImGui::TableSetColumnIndex(0);
            std::string devBadge = (r.component == "System" || r.backendName == "CPU")
                ? "[CPU] " + r.deviceName
                : ("[GPU " + std::to_string(r.deviceIndex) + "] " + r.deviceName);
            ImGui::TextUnformatted(fitTextToPixelWidth(devBadge, ImGui::GetContentRegionAvail().x).c_str());

            // Col 1: Domain / Subcategory
            ImGui::TableSetColumnIndex(1);
            ImGui::TextDisabled("%s", fitTextToPixelWidth(r.subcategory.empty() ? r.component : r.subcategory,
                                                           ImGui::GetContentRegionAvail().x).c_str());

            // Col 2: Workload Name + Honest Badges
            ImGui::TableSetColumnIndex(2);
            ImGui::TextUnformatted(r.benchmarkName.c_str());
            if (r.isEmulated && r.benchmarkName.find("Emulated") == std::string::npos) {
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(0.98f, 0.76f, 0.24f, 1.0f), "(Emulated)");
            }
            if (!r.isUnsupported && !r.isValid) {
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(0.98f, 0.32f, 0.32f, 1.0f), "[INVALID]");
            }

            if (ImGui::IsItemHovered()) {
                ImGui::BeginTooltip();
                pushBoldFont();
                ImGui::TextColored(ImVec4(0.10f, 0.84f, 0.98f, 1.0f), "%s", r.benchmarkName.c_str());
                popBoldFont();
                ImGui::TextDisabled("Backend: %s  |  Device: %s", r.backendName.c_str(), r.deviceName.c_str());
                ImGui::Separator();
                if (r.isUnsupported) {
                    ImGui::TextColored(ImVec4(0.96f, 0.44f, 0.36f, 1.0f), "UNSUPPORTED (%s)",
                                       r.supportCategory.empty() ? "capability" : r.supportCategory.c_str());
                    if (!r.supportNote.empty()) {
                        ImGui::TextWrapped("%s", r.supportNote.c_str());
                    }
                } else {
                    if (r.sample_count > 0) {
                        ImGui::Text("Samples (%u): min %.3f ms | median %.3f ms | p95 %.3f ms",
                                    r.sample_count, r.min_time_ms, r.median_time_ms, r.p95_time_ms);
                    }
                    if (r.operationalIntensity > 0.0) {
                        ImGui::Text("Operational Intensity: %.2f OP/Byte", r.operationalIntensity);
                    }
                    if (r.hasRegisterTelemetry) {
                        ImGui::Text("ISA Resources: %u VGPR | %u SGPR | %u B LDS | %u max waves/SIMD",
                                    r.vgprCount, r.sgprCount, r.ldsSizeBytes, r.maxWavesPerSimd);
                    }
                    if (r.hasPowerTelemetry && r.powerWatts > 0.0f) {
                        ImGui::Text("Power: %.1f W  |  Energy: %.3f J  |  Efficiency: %.3f %s",
                                    r.powerWatts, r.energyJoules, r.unitPerWatt, r.efficiencyUnit.c_str());
                    }
                    if (!r.supportNote.empty()) {
                        ImGui::TextColored(ImVec4(0.95f, 0.80f, 0.35f, 1.0f), "Note: %s", r.supportNote.c_str());
                    }
                }
                if (r.benchmarkName.rfind("RayScheduling", 0) == 0) {
                    ImGui::Separator();
                    ImGui::TextDisabled("Right-click row to inspect scene in Ray Tracing Viewport");
                }
                ImGui::EndTooltip();
            }

            if (r.benchmarkName.rfind("RayScheduling", 0) == 0 &&
                ImGui::IsItemClicked(ImGuiMouseButton_Right)) {
                std::string sc = extractSceneName(r.benchmarkName);
                if (sc == "Showroom") m_rtSceneIndex = 0;
                else if (sc == "Indoor") m_rtSceneIndex = 1;
                else if (sc == "Forest") m_rtSceneIndex = 2;
                else if (sc == "Landscape") m_rtSceneIndex = 3;
                m_switchToRtViewport = true;
            }

            // Col 3: Score
            ImGui::TableSetColumnIndex(3);
            pushMonoFont();
            if (r.isUnsupported) {
                ImGui::TextColored(ImVec4(0.65f, 0.50f, 0.45f, 1.0f), "UNSUPPORTED");
            } else if (!r.isValid) {
                ImGui::TextColored(ImVec4(0.98f, 0.32f, 0.32f, 1.0f), "INVALID");
            } else {
                ImGui::TextColored(ImVec4(0.92f, 0.97f, 1.0f, 1.0f), "%s",
                                   formatResultScoreString(r, false).c_str());
            }
            popMonoFont();

            // Col 4: Median Time
            ImGui::TableSetColumnIndex(4);
            pushMonoFont();
            double tDisp = (r.median_time_ms > 0.0) ? r.median_time_ms : r.time_ms;
            if (!r.isUnsupported && tDisp > 0.0) {
                ImGui::TextDisabled("%.3f ms", tDisp);
            } else {
                ImGui::TextDisabled("—");
            }
            popMonoFont();

            // Col 5: Speedup / Comparison Delta
            ImGui::TableSetColumnIndex(5);
            pushMonoFont();
            const ResultData* basePtr = findComparisonMatch(r);
            if (!basePtr && m_comparisonResults.empty() && r.baselineConfigIndex >= 0 && !r.isUnsupported) {
                std::string rScene = extractSceneName(r.benchmarkName);
                for (const auto& cand : m_allResults) {
                    if (cand.deviceIndex == r.deviceIndex &&
                        cand.backendName == r.backendName &&
                        cand.subcategory == r.subcategory &&
                        static_cast<int32_t>(cand.configIndex) == r.baselineConfigIndex &&
                        extractSceneName(cand.benchmarkName) == rScene &&
                        !cand.isUnsupported && cand.time_ms > 0.0) {
                        basePtr = &cand;
                        break;
                    }
                }
            }

            if (basePtr && !r.isUnsupported && r.time_ms > 0.0) {
                double curRate = computeThroughputOrInvLatency(r);
                double baseRate = computeThroughputOrInvLatency(*basePtr);
                if (curRate > 0.0 && baseRate > 0.0) {
                    double ratio = curRate / baseRate;
                    double pct = (ratio - 1.0) * 100.0;
                    ImVec4 dCol = (ratio >= 1.01)
                        ? ImVec4(0.20f, 0.92f, 0.52f, 1.0f)
                        : ((ratio <= 0.99) ? ImVec4(0.95f, 0.42f, 0.36f, 1.0f)
                                           : ImVec4(0.75f, 0.80f, 0.88f, 1.0f));
                    ImGui::TextColored(dCol, "%.2fx (%+.1f%%)", ratio, pct);
                } else {
                    ImGui::TextDisabled("—");
                }
            } else {
                ImGui::TextDisabled("—");
            }
            popMonoFont();

            // Col 6: ISA / Power
            ImGui::TableSetColumnIndex(6);
            pushMonoFont();
            if (r.hasPowerTelemetry && r.powerWatts > 0.0f) {
                ImGui::TextColored(ImVec4(0.98f, 0.78f, 0.25f, 1.0f), "%.0f W", r.powerWatts);
            } else if (r.hasRegisterTelemetry && r.vgprCount > 0) {
                ImGui::TextDisabled("%uV/%uS", r.vgprCount, r.sgprCount);
            } else {
                ImGui::TextDisabled("—");
            }
            popMonoFont();

            ImGui::PopID();
        }

        ImGui::EndTable();
    }
}

} // namespace gpubench::gui
