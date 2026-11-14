#include "GuiApp.h"
#include "GuiUiHelpers.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <set>

namespace gpubench::gui {

namespace {

bool isHostSystemResult(const ResultData& r) {
    return r.component == "System" || r.backendName == "CPU" ||
           r.backendName == "System" || r.deviceIndex == UINT32_MAX ||
           r.deviceName == "Host CPU";
}

std::string formatScorecardWorkloadName(const ResultData& r) {
    if (r.benchmarkName == "Vector" || r.benchmarkName == "Matrix") {
        if (!r.subcategory.empty()) {
            const size_t lp = r.subcategory.find('(');
            const size_t rp = r.subcategory.find(')', lp);
            if (lp != std::string::npos && rp != std::string::npos && rp > lp + 1) {
                return r.subcategory.substr(lp + 1, rp - lp - 1) + " (" + r.benchmarkName + ")";
            }
            return r.subcategory + " (" + r.benchmarkName + ")";
        }
    }
    return r.benchmarkName;
}

} // namespace

void GuiApp::renderResultsScorecard() {
    const BringUpColors c = getBringUpColors(m_palette);

    if (m_allResults.empty()) {
        drawEtchedHeader("DAT-02", "SCORECARD & DELTA MATRIX", c, "0 RECORDS");
        ImGui::Spacing();
        ImGui::TextColored(c.textSecondary, "No benchmark results recorded yet.");
        ImGui::TextColored(c.textMuted, "Select workloads in ROM-01 Workload Matrix and click [RUN TESTS], or click [SESSION] to load JSON.");
        return;
    }

    char countTag[48];
    std::snprintf(countTag, sizeof(countTag), "%zu RECORDS", m_allResults.size());
    drawEtchedHeader("DAT-02", "SCORECARD & DELTA MATRIX", c, countTag);
    ImGui::Spacing();

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
                peakComputeName = formatScorecardWorkloadName(r);
                peakComputeUnit = r.metric;
            }
        }
        if (r.component == "Memory" && !isHostSystemResult(r) &&
            r.subcategory == "Bandwidth" && r.metric == "GB/s") {
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

    // Executive KPI Strip
    if (ImGui::BeginTable("ScorecardKpiStrip", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_SizingStretchSame)) {
        ImGui::TableNextRow();

        ImGui::TableNextColumn();
        ImGui::TextColored(c.textMuted, "PEAK COMPUTE");
        pushBoldFont();
        if (peakComputeTops > 0.0) {
            ImGui::TextColored(c.cyanAccent, "%.2f %s", peakComputeTops, peakComputeUnit.c_str());
        } else {
            ImGui::TextColored(c.textMuted, "—");
        }
        popBoldFont();
        if (!peakComputeName.empty()) {
            ImGui::TextColored(c.textMuted, "%s", fitTextToPixelWidth(peakComputeName, ImGui::GetContentRegionAvail().x).c_str());
        }

        ImGui::TableNextColumn();
        ImGui::TextColored(c.textMuted, "PEAK VRAM BANDWIDTH");
        pushBoldFont();
        if (peakVramGbs > 0.0) {
            ImGui::TextColored(c.okColor, "%.1f GB/s", peakVramGbs);
        } else {
            ImGui::TextColored(c.textMuted, "—");
        }
        popBoldFont();
        if (!peakVramName.empty()) {
            ImGui::TextColored(c.textMuted, "%s", fitTextToPixelWidth(peakVramName, ImGui::GetContentRegionAvail().x).c_str());
        }

        ImGui::TableNextColumn();
        ImGui::TextColored(c.textMuted, "PEAK RAY TRACING");
        pushBoldFont();
        if (peakRtMrays > 0.0) {
            ImGui::TextColored(c.phosphorHi, "%.1f MRays/s", peakRtMrays);
        } else {
            ImGui::TextColored(c.textMuted, "—");
        }
        popBoldFont();
        if (!peakRtName.empty()) {
            ImGui::TextColored(c.textMuted, "%s", fitTextToPixelWidth(peakRtName, ImGui::GetContentRegionAvail().x).c_str());
        }

        ImGui::TableNextColumn();
        ImGui::TextColored(c.textMuted, "VALIDATION & CHARTS");
        pushBoldFont();
        if (invalidCount > 0) {
            ImGui::TextColored(c.errColor, "%zu Valid / %zu INVALID", validCount, invalidCount);
        } else {
            ImGui::TextColored(c.okColor, "%zu Valid (%zu Unsup)", validCount, unsupportedCount);
        }
        popBoldFont();
        if (tacticalButton("##BtnScorecardToRoofline", "VIEW CHARTS ->", ImVec2(0.0f, s(21.0f)), false, c)) {
            m_f2SubMode = 1;
        }

        ImGui::EndTable();
    }

    // Active Comparison Banner
    if (!m_comparisonResults.empty()) {
        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_ChildBg, c.bgElevated);
        ImGui::PushStyleColor(ImGuiCol_Border, c.cyanAccent);
        ImGui::BeginChild("ComparisonBanner", ImVec2(0, s(32.0f)), ImGuiChildFlags_Borders);
        ImGui::TextColored(c.cyanAccent,
                           "DIFF MODE ACTIVE  |  BASELINE: %s (%zu records)",
                           m_comparisonSourceLabel.c_str(), m_comparisonResults.size());
        ImGui::SameLine(0.0f, s(10.0f));
        if (tacticalButton("##BtnClearBaseline", "CLEAR BASELINE", ImVec2(0.0f, s(20.0f)), false, c, true)) {
            clearComparisonResults();
        }
        ImGui::EndChild();
        ImGui::PopStyleColor(2);
    }

    ImGui::Spacing();

    // Category Filter Strip
    const char* catFilters[] = {"ALL", "COMPUTE", "MEMORY", "RAY TRACING", "SCHEDULING", "GRAPHICS", "HOST CPU"};
    for (int i = 0; i < 7; ++i) {
        float bw = ImGui::CalcTextSize(catFilters[i]).x + s(16.0f);
        if (i > 0) sameLineIfFits(bw, s(4.0f));
        bool active = (m_activeScorecardFilter == i);
        std::string fId = "##ScCat_" + std::to_string(i);
        if (tacticalButton(fId.c_str(), catFilters[i], ImVec2(0.0f, s(21.0f)), active, c)) {
            m_activeScorecardFilter = i;
        }
    }

    // Device Filter Strip (if multiple devices exist in results)
    std::set<uint32_t> seenDevs;
    bool hasHostCpu = false;
    for (const auto& r : m_allResults) {
        if (isHostSystemResult(r)) hasHostCpu = true;
        else seenDevs.insert(r.deviceIndex);
    }
    if (seenDevs.size() > 1 || (seenDevs.size() >= 1 && hasHostCpu)) {
        ImGui::SameLine(0.0f, s(14.0f));
        ImGui::TextColored(c.textMuted, "DUT:");
        ImGui::SameLine(0.0f, s(4.0f));
        if (tacticalButton("##ScDevAll", "ALL DUT", ImVec2(0.0f, s(21.0f)), (m_activeDeviceScorecardFilter == -1), c)) {
            m_activeDeviceScorecardFilter = -1;
        }
        for (uint32_t d : seenDevs) {
            ImGui::SameLine(0.0f, s(4.0f));
            char dlbl[32];
            std::snprintf(dlbl, sizeof(dlbl), "GPU %u", d);
            std::string dId = "##ScDevGpu_" + std::to_string(d);
            if (tacticalButton(dId.c_str(), dlbl, ImVec2(0.0f, s(21.0f)),
                               (m_activeDeviceScorecardFilter == static_cast<int>(d)), c)) {
                m_activeDeviceScorecardFilter = static_cast<int>(d);
            }
        }
        if (hasHostCpu) {
            ImGui::SameLine(0.0f, s(4.0f));
            if (tacticalButton("##ScDevCpu", "HOST CPU", ImVec2(0.0f, s(21.0f)),
                               (m_activeDeviceScorecardFilter == -2), c)) {
                m_activeDeviceScorecardFilter = -2;
            }
        }
    }

    ImGui::Spacing();

    auto matchesCategoryFilter = [&](const ResultData& r) {
        switch (m_activeScorecardFilter) {
        case 0: return true;
        case 1: return r.component == "Compute" && r.subcategory.find("Indirect") == std::string::npos;
        case 2: return r.component == "Memory" && !isHostSystemResult(r);
        case 3: return r.component == "Ray Tracing";
        case 4: return r.component == "Scheduling" || r.subcategory.find("Indirect") != std::string::npos;
        case 5: return r.component == "Graphics" || r.component == "Raster";
        case 6: return isHostSystemResult(r);
        default: return true;
        }
    };

    auto matchesDeviceFilter = [&](const ResultData& r) {
        if (m_activeDeviceScorecardFilter == -1) return true;
        if (m_activeDeviceScorecardFilter == -2) return isHostSystemResult(r);
        if (isHostSystemResult(r)) return false;
        return static_cast<int>(r.deviceIndex) == m_activeDeviceScorecardFilter;
    };

    auto resolveRowBaseline = [&](const ResultData& r) -> const ResultData* {
        if (const ResultData* cmp = findComparisonMatch(r)) {
            return cmp;
        }
        if (m_comparisonResults.empty() && r.baselineConfigIndex >= 0 && !r.isUnsupported) {
            std::string rScene = extractSceneName(r.benchmarkName);
            for (const auto& cand : m_allResults) {
                if (cand.deviceIndex == r.deviceIndex &&
                    cand.backendName == r.backendName &&
                    cand.subcategory == r.subcategory &&
                    static_cast<int32_t>(cand.configIndex) == r.baselineConfigIndex &&
                    extractSceneName(cand.benchmarkName) == rScene &&
                    !cand.isUnsupported && cand.time_ms > 0.0) {
                    return &cand;
                }
            }
        }
        return nullptr;
    };

    std::vector<const ResultData*> filteredRows;
    filteredRows.reserve(m_allResults.size());
    bool showDeltaColumn = !m_comparisonResults.empty();
    for (const auto& r : m_allResults) {
        if (!matchesCategoryFilter(r) || !matchesDeviceFilter(r)) continue;
        filteredRows.push_back(&r);
        if (!showDeltaColumn && resolveRowBaseline(r) != nullptr) {
            showDeltaColumn = true;
        }
    }

    ImGuiTableFlags flags = ImGuiTableFlags_Borders |
                            ImGuiTableFlags_RowBg |
                            ImGuiTableFlags_ScrollY |
                            ImGuiTableFlags_Resizable |
                            ImGuiTableFlags_SizingStretchProp;

    const char* tableId = showDeltaColumn ? "ResultsScorecardTableWithBase" : "ResultsScorecardTableNoBase";
    const int numCols = showDeltaColumn ? 7 : 6;
    if (ImGui::BeginTable(tableId, numCols, flags, ImVec2(0.0f, 0.0f))) {
        ImGui::TableSetupScrollFreeze(0, 1);
        if (showDeltaColumn) {
            const char* deltaColTitle = m_comparisonResults.empty() ? "Speedup" : "vs Comparison JSON";
            ImGui::TableSetupColumn("Device", ImGuiTableColumnFlags_WidthStretch, 0.14f);
            ImGui::TableSetupColumn("Domain", ImGuiTableColumnFlags_WidthStretch, 0.13f);
            ImGui::TableSetupColumn("Workload", ImGuiTableColumnFlags_WidthStretch, 0.30f);
            ImGui::TableSetupColumn("Throughput / Score", ImGuiTableColumnFlags_WidthStretch, 0.14f);
            ImGui::TableSetupColumn("Median Time", ImGuiTableColumnFlags_WidthStretch, 0.10f);
            ImGui::TableSetupColumn(deltaColTitle, ImGuiTableColumnFlags_WidthStretch, 0.10f);
            ImGui::TableSetupColumn("ISA / Power", ImGuiTableColumnFlags_WidthStretch, 0.09f);
        } else {
            ImGui::TableSetupColumn("Device", ImGuiTableColumnFlags_WidthStretch, 0.15f);
            ImGui::TableSetupColumn("Domain", ImGuiTableColumnFlags_WidthStretch, 0.14f);
            ImGui::TableSetupColumn("Workload", ImGuiTableColumnFlags_WidthStretch, 0.35f);
            ImGui::TableSetupColumn("Throughput / Score", ImGuiTableColumnFlags_WidthStretch, 0.16f);
            ImGui::TableSetupColumn("Median Time", ImGuiTableColumnFlags_WidthStretch, 0.10f);
            ImGui::TableSetupColumn("ISA / Power", ImGuiTableColumnFlags_WidthStretch, 0.10f);
        }
        ImGui::TableHeadersRow();

        int rowId = 0;
        for (const ResultData* rPtr : filteredRows) {
            const ResultData& r = *rPtr;

            ImGui::PushID(rowId++);
            ImGui::TableNextRow();

            // Col 0: Device
            ImGui::TableSetColumnIndex(0);
            std::string devBadge = isHostSystemResult(r)
                ? "[CPU] " + r.deviceName
                : ("[GPU " + std::to_string(r.deviceIndex) + "] " + r.deviceName);
            ImGui::TextColored(c.textSecondary, "%s", fitTextToPixelWidth(devBadge, ImGui::GetContentRegionAvail().x).c_str());

            // Col 1: Domain / Subcategory
            ImGui::TableSetColumnIndex(1);
            ImGui::TextColored(c.textMuted, "%s", fitTextToPixelWidth(r.subcategory.empty() ? r.component : r.subcategory,
                                                                      ImGui::GetContentRegionAvail().x).c_str());

            // Col 2: Workload Name + Honest Badges
            ImGui::TableSetColumnIndex(2);
            const std::string displayWorkload = formatScorecardWorkloadName(r);
            ImGui::TextColored(c.textPrimary, "%s", displayWorkload.c_str());
            if (r.isEmulated && displayWorkload.find("Emulated") == std::string::npos) {
                ImGui::SameLine();
                ImGui::TextColored(c.warnColor, "(Emulated)");
            }
            if (!r.isUnsupported && !r.isValid) {
                ImGui::SameLine();
                ImGui::TextColored(c.errColor, "[INVALID]");
            }

            if (ImGui::IsItemHovered()) {
                ImGui::BeginTooltip();
                pushBoldFont();
                ImGui::TextColored(c.phosphorHi, "%s", displayWorkload.c_str());
                popBoldFont();
                ImGui::TextColored(c.textMuted, "Backend: %s  |  Device: %s", r.backendName.c_str(), r.deviceName.c_str());
                ImGui::Separator();
                if (r.isUnsupported) {
                    ImGui::TextColored(c.errColor, "UNSUPPORTED (%s)",
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
                        ImGui::TextColored(c.warnColor, "Note: %s", r.supportNote.c_str());
                    }
                }
                if (r.benchmarkName.rfind("RayScheduling", 0) == 0) {
                    ImGui::Separator();
                    ImGui::TextColored(c.textMuted, "Right-click row to inspect scene in F3 RT Parity Viewport");
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
                ImGui::TextColored(c.textMuted, "UNSUPPORTED");
            } else if (!r.isValid) {
                ImGui::TextColored(c.errColor, "INVALID");
            } else {
                ImGui::TextColored(c.textPrimary, "%s",
                                   formatResultScoreString(r, false).c_str());
            }
            popMonoFont();

            // Col 4: Median Time
            ImGui::TableSetColumnIndex(4);
            pushMonoFont();
            double tDisp = (r.median_time_ms > 0.0) ? r.median_time_ms : r.time_ms;
            if (!r.isUnsupported && tDisp > 0.0) {
                ImGui::TextColored(c.textSecondary, "%.3f ms", tDisp);
            } else {
                ImGui::TextColored(c.textMuted, "—");
            }
            popMonoFont();

            int powerColIdx = 5;
            if (showDeltaColumn) {
                // Col 5: Speedup / Comparison Delta
                ImGui::TableSetColumnIndex(5);
                pushMonoFont();
                const ResultData* basePtr = resolveRowBaseline(r);
                if (basePtr && !r.isUnsupported && r.time_ms > 0.0) {
                    double curRate = computeThroughputOrInvLatency(r);
                    double baseRate = computeThroughputOrInvLatency(*basePtr);
                    if (curRate > 0.0 && baseRate > 0.0) {
                        double ratio = curRate / baseRate;
                        double pct = (ratio - 1.0) * 100.0;
                        ImVec4 dCol = (ratio >= 1.01)
                            ? c.okColor
                            : ((ratio <= 0.99) ? c.errColor : c.textSecondary);
                        ImGui::TextColored(dCol, "%.2fx (%+.1f%%)", ratio, pct);
                    } else {
                        ImGui::TextColored(c.textMuted, "—");
                    }
                } else {
                    ImGui::TextColored(c.textMuted, "—");
                }
                popMonoFont();
                powerColIdx = 6;
            }

            // ISA / Power
            ImGui::TableSetColumnIndex(powerColIdx);
            pushMonoFont();
            if (r.hasPowerTelemetry && r.powerWatts > 0.0f) {
                ImGui::TextColored(c.warnColor, "%.0f W", r.powerWatts);
            } else if (r.hasRegisterTelemetry && r.vgprCount > 0) {
                ImGui::TextColored(c.textSecondary, "%uV/%uS", r.vgprCount, r.sgprCount);
            } else {
                ImGui::TextColored(c.textMuted, "—");
            }
            popMonoFont();

            ImGui::PopID();
        }

        ImGui::EndTable();
    }
}

} // namespace gpubench::gui
