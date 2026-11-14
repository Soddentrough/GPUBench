#include "GuiApp.h"
#include "GuiUiHelpers.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace gpubench::gui {

namespace {

bool isHostSystemResult(const ResultData& r) {
    return r.component == "System" || r.backendName == "CPU" ||
           r.backendName == "System" || r.deviceIndex == UINT32_MAX ||
           r.deviceName == "Host CPU";
}

enum ChartDomain : int {
    DOMAIN_COMPUTE = 0,
    DOMAIN_MEMORY = 1,
    DOMAIN_RAY = 2,
    DOMAIN_BVH = 3,
    DOMAIN_SCHED_GFX = 4,
    DOMAIN_LATENCY = 5,
    DOMAIN_COUNT = 6
};

struct ChartDeviceGroup {
    bool isHost{false};
    uint32_t deviceIndex{0};
    std::string deviceName;
    std::string backendName;
    std::vector<const ResultData*> results;
};

struct ChartBarItem {
    const ResultData* res{nullptr};
    std::string shortLabel;
    std::string fullLabel;
    double perfVal{0.0};
    std::string perfText;
    double effVal{0.0};
    std::string effText;
    double watts{0.0};
    double medianMs{0.0};
    bool isMatrix{false};
};

std::string buildChartFullLabel(const ResultData& r) {
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

std::string abbreviateChartTickLabel(const std::string& full) {
    std::string s = full;
    if (s.rfind("ILP & Dual-Issue (", 0) == 0) {
        s = s.substr(18);
        if (!s.empty() && s.back() == ')') s.pop_back();
        if (s.find("Ping-Pong") != std::string::npos) {
            s = (s.rfind("FP32", 0) == 0) ? "FP32 Base" : "INT32 Base";
        } else if (s.find("Moderate ILP") != std::string::npos) {
            s = (s.rfind("FP32", 0) == 0) ? "FP32 Mid ILP" : "INT32 Mid ILP";
        } else if (s.find("Peak ILP") != std::string::npos) {
            s = (s.rfind("FP32", 0) == 0) ? "FP32 Peak ILP" : "INT32 Peak ILP";
        } else if (s.find("Combined FP32 + INT32") != std::string::npos) {
            s = "FP32+INT32 Dual";
        }
    } else if (s.rfind("Device Memory Bandwidth (", 0) == 0) {
        s = "VRAM: " + s.substr(25);
        if (!s.empty() && s.back() == ')') s.pop_back();
    } else if (s.rfind("RayScheduling - ", 0) == 0) {
        s = "RT " + s.substr(16);
    } else if (s.rfind("Cache Latency Curve - ", 0) == 0) {
        s = "Cache: " + s.substr(22);
    }
    if (s.size() > 16) {
        s = s.substr(0, 14) + "..";
    }
    return s;
}

int classifyChartDomain(const ResultData& r) {
    if (isLowerBetterMetric(r.metric)) {
        return DOMAIN_LATENCY;
    }
    if (r.metric.find("FLOPS") != std::string::npos || r.metric.find("OPS") != std::string::npos) {
        return DOMAIN_COMPUTE;
    }
    if (r.metric.find("GB/s") != std::string::npos || r.metric.find("TB/s") != std::string::npos) {
        return DOMAIN_MEMORY;
    }
    if (r.metric == "MRays/s" || r.metric == "MHits/s") {
        return DOMAIN_RAY;
    }
    if (r.metric == "GIS/s" || r.metric == "MBVH/s" ||
        r.metric == "MTris/s" || r.metric == "MInst/s") {
        return DOMAIN_BVH;
    }
    return DOMAIN_SCHED_GFX;
}

const char* domainTabName(int dom) {
    switch (dom) {
    case DOMAIN_COMPUTE:   return "COMPUTE";
    case DOMAIN_MEMORY:    return "MEMORY BW";
    case DOMAIN_RAY:       return "RAY TRACING";
    case DOMAIN_BVH:       return "BVH & INTERSECT";
    case DOMAIN_SCHED_GFX: return "SCHED & GFX";
    case DOMAIN_LATENCY:   return "LATENCY";
    default:               return "OTHER";
    }
}

const char* domainPerfAxisLabel(int dom) {
    switch (dom) {
    case DOMAIN_COMPUTE:   return "Throughput (TFLOPS / TOPS)";
    case DOMAIN_MEMORY:    return "Bandwidth (GB/s)";
    case DOMAIN_RAY:       return "Ray Throughput (MRays/s)";
    case DOMAIN_BVH:       return "BVH / Primitives Rate (MOp/s)";
    case DOMAIN_SCHED_GFX: return "Throughput (MItems/s or MOp/s)";
    case DOMAIN_LATENCY:   return "Latency (ns — lower is faster)";
    default:               return "Throughput";
    }
}

const char* domainEffAxisLabel(int dom) {
    switch (dom) {
    case DOMAIN_COMPUTE:   return "Efficiency (GFLOPS/W | GOPS/W)";
    case DOMAIN_MEMORY:    return "Efficiency (GB/s per Watt)";
    case DOMAIN_RAY:       return "Efficiency (MRays/s per Watt)";
    case DOMAIN_BVH:       return "Efficiency (MOp/s per Watt)";
    case DOMAIN_SCHED_GFX: return "Efficiency (MOp/s per Watt)";
    case DOMAIN_LATENCY:   return "Board Power (Watts)";
    default:               return "Perf / Watt";
    }
}

double computeLatencyNs(const ResultData& r) {
    if (r.metric == "ns") {
        return (r.operations > 0) ? ((r.time_ms * 1e6) / static_cast<double>(r.operations)) : r.time_ms;
    }
    if (r.metric == "us") {
        return (r.operations > 1) ? ((r.time_ms * 1e6) / static_cast<double>(r.operations)) : (r.time_ms * 1000.0);
    }
    return r.time_ms * 1e6;
}

} // namespace

void GuiApp::renderRooflinePanel() {
    const BringUpColors c = getBringUpColors(m_palette);

    std::vector<ChartDeviceGroup> devGroups;
    for (const auto& r : m_allResults) {
        if (r.isUnsupported || !r.isValid || r.time_ms <= 0.0) continue;
        if (r.operations == 0 && r.metric != "us" && r.metric != "ms") continue;

        const bool rHost = isHostSystemResult(r);
        ChartDeviceGroup* target = nullptr;
        for (auto& g : devGroups) {
            if (g.isHost == rHost &&
                (rHost || (g.deviceIndex == r.deviceIndex && g.backendName == r.backendName))) {
                target = &g;
                break;
            }
        }
        if (!target) {
            devGroups.push_back(ChartDeviceGroup{
                rHost, r.deviceIndex, r.deviceName, rHost ? "CPU" : r.backendName, {}
            });
            target = &devGroups.back();
        }
        target->results.push_back(&r);
    }

    if (devGroups.empty()) {
        drawEtchedHeader("INS-02", "THROUGHPUT & POWER EFFICIENCY", c, "NO DATA");
        ImGui::Spacing();
        ImGui::TextColored(c.textSecondary, "Run at least one valid GPU or CPU workload to view Throughput & Power Efficiency charts.");
        ImGui::TextColored(c.textMuted, "Tip: Click [RUN TESTS] in the top bar or load a saved JSON file via [SESSION].");
        return;
    }

    if (m_rooflineDeviceFilter < 0 || m_rooflineDeviceFilter >= static_cast<int>(devGroups.size())) {
        m_rooflineDeviceFilter = 0;
    }
    const auto& devGroup = devGroups[static_cast<size_t>(m_rooflineDeviceFilter)];

    char devTag[96];
    if (devGroup.isHost) {
        std::snprintf(devTag, sizeof(devTag), "HOST CPU");
    } else {
        std::snprintf(devTag, sizeof(devTag), "GPU %u | %s", devGroup.deviceIndex, devGroup.backendName.c_str());
    }
    drawEtchedHeader("INS-02", "THROUGHPUT & POWER EFFICIENCY", c, devTag);

    // Group device results by domain
    std::array<std::vector<ChartBarItem>, DOMAIN_COUNT> domainItems;
    for (const ResultData* rPtr : devGroup.results) {
        const ResultData& r = *rPtr;
        const int dom = classifyChartDomain(r);
        const double opsPerSec = (r.operations > 0)
            ? ((static_cast<double>(r.operations) / r.time_ms) * 1000.0)
            : 0.0;
        const double watts = (r.hasPowerTelemetry && r.powerWatts > 0.0f)
            ? static_cast<double>(r.powerWatts)
            : 0.0;

        ChartBarItem item{};
        item.res = &r;
        item.fullLabel = buildChartFullLabel(r);
        item.shortLabel = abbreviateChartTickLabel(item.fullLabel);
        item.perfText = formatResultScoreString(r, true);
        item.watts = watts;
        item.medianMs = (r.median_time_ms > 0.0) ? r.median_time_ms : r.time_ms;
        item.isMatrix = (item.fullLabel.find("Matrix") != std::string::npos ||
                         item.fullLabel.find("FP8") != std::string::npos ||
                         item.fullLabel.find("INT8") != std::string::npos ||
                         item.fullLabel.find("WMMA") != std::string::npos);

        switch (dom) {
        case DOMAIN_COMPUTE:
            item.perfVal = opsPerSec / 1e12;
            if (watts > 0.0) {
                item.effVal = (opsPerSec / 1e9) / watts;
                char eb[48];
                const char* unitStr = (r.metric.find("FLOP") != std::string::npos) ? "GFLOPS/W" : "GOPS/W";
                if (item.effVal >= 100.0) {
                    std::snprintf(eb, sizeof(eb), "%.0f %s", item.effVal, unitStr);
                } else {
                    std::snprintf(eb, sizeof(eb), "%.1f %s", item.effVal, unitStr);
                }
                item.effText = eb;
            }
            break;
        case DOMAIN_MEMORY:
            item.perfVal = opsPerSec / 1e9;
            if (watts > 0.0) {
                item.effVal = item.perfVal / watts;
                char eb[48];
                std::snprintf(eb, sizeof(eb), "%.2f GB/s/W", item.effVal);
                item.effText = eb;
            }
            break;
        case DOMAIN_RAY:
            item.perfVal = opsPerSec / 1e6;
            if (watts > 0.0) {
                item.effVal = item.perfVal / watts;
                char eb[48];
                std::snprintf(eb, sizeof(eb), "%.2f MR/s/W", item.effVal);
                item.effText = eb;
            }
            break;
        case DOMAIN_BVH:
        case DOMAIN_SCHED_GFX:
            item.perfVal = opsPerSec / 1e6;
            if (watts > 0.0) {
                item.effVal = item.perfVal / watts;
                char eb[48];
                std::snprintf(eb, sizeof(eb), "%.1f M/s/W", item.effVal);
                item.effText = eb;
            }
            break;
        case DOMAIN_LATENCY:
            item.perfVal = computeLatencyNs(r);
            if (watts > 0.0) {
                item.effVal = watts;
                char eb[48];
                std::snprintf(eb, sizeof(eb), "%.0f W", watts);
                item.effText = eb;
            }
            break;
        default:
            break;
        }

        if (item.perfVal > 0.0) {
            domainItems[dom].push_back(std::move(item));
        }
    }

    int firstPopulatedDom = -1;
    int populatedCount = 0;
    for (int d = 0; d < DOMAIN_COUNT; ++d) {
        if (!domainItems[d].empty()) {
            if (firstPopulatedDom < 0) firstPopulatedDom = d;
            populatedCount++;
        }
    }
    if (firstPopulatedDom < 0) return;

    if (m_chartDomainFilter < 0 || m_chartDomainFilter >= DOMAIN_COUNT ||
        domainItems[m_chartDomainFilter].empty()) {
        m_chartDomainFilter = firstPopulatedDom;
    }

    // Optional selector bar (Devices & Metric Domains)
    if (devGroups.size() > 1 || populatedCount > 1) {
        ImGui::Spacing();
        bool needSameLine = false;
        if (devGroups.size() > 1) {
            for (size_t i = 0; i < devGroups.size(); ++i) {
                std::string lbl = devGroups[i].isHost
                    ? "HOST CPU"
                    : ("GPU " + std::to_string(devGroups[i].deviceIndex) + " (" + devGroups[i].backendName + ")");
                float bw = ImGui::CalcTextSize(lbl.c_str()).x + s(16.0f);
                if (needSameLine) sameLineIfFits(bw, s(4.0f));
                needSameLine = true;
                std::string bid = "##ChartDev_" + std::to_string(i);
                if (tacticalButton(bid.c_str(), lbl.c_str(), ImVec2(0.0f, s(21.0f)),
                                   m_rooflineDeviceFilter == static_cast<int>(i), c)) {
                    m_rooflineDeviceFilter = static_cast<int>(i);
                }
            }
        }
        if (populatedCount > 1) {
            if (devGroups.size() > 1) {
                ImGui::SameLine(0.0f, s(12.0f));
                ImGui::TextColored(c.textMuted, "|");
                needSameLine = true;
            }
            for (int d = 0; d < DOMAIN_COUNT; ++d) {
                if (domainItems[d].empty()) continue;
                char dlbl[48];
                std::snprintf(dlbl, sizeof(dlbl), "%s (%zu)", domainTabName(d), domainItems[d].size());
                float bw = ImGui::CalcTextSize(dlbl).x + s(16.0f);
                if (needSameLine) sameLineIfFits(bw, s(4.0f));
                needSameLine = true;
                std::string bid = "##ChartDom_" + std::to_string(d);
                if (tacticalButton(bid.c_str(), dlbl, ImVec2(0.0f, s(21.0f)), m_chartDomainFilter == d, c)) {
                    m_chartDomainFilter = d;
                }
            }
        }
    }

    std::vector<ChartBarItem> bars = domainItems[m_chartDomainFilter];
    const bool isLatencyDom = (m_chartDomainFilter == DOMAIN_LATENCY);

    // Sort ascending so highest-performing entry is drawn at the top row (y = count - 1)
    std::sort(bars.begin(), bars.end(), [&](const ChartBarItem& a, const ChartBarItem& b) {
        return isLatencyDom ? (a.perfVal > b.perfVal) : (a.perfVal < b.perfVal);
    });

    const size_t totalInDomain = bars.size();
    const size_t maxBars = (m_f2SubMode == 0) ? 7u : 15u;
    if (bars.size() > maxBars) {
        // Keep the top-performing tail (end of ascending-sorted array)
        bars.erase(bars.begin(), bars.end() - static_cast<std::ptrdiff_t>(maxBars));
    }

    // Compute domain KPI stats across all entries in this domain
    const auto& allDom = domainItems[m_chartDomainFilter];
    const ChartBarItem* topPerfItem = &bars.back();
    const ChartBarItem* topEffItem = nullptr;
    double sumWatts = 0.0;
    double maxWatts = 0.0;
    int powerCount = 0;
    double fp32PerfVal = 0.0;

    for (const auto& it : allDom) {
        if (m_chartDomainFilter == DOMAIN_COMPUTE && it.fullLabel == "FP32") {
            fp32PerfVal = it.perfVal;
        }
        if (it.watts > 0.0) {
            sumWatts += it.watts;
            if (it.watts > maxWatts) maxWatts = it.watts;
            powerCount++;
            if (!isLatencyDom && (!topEffItem || it.effVal > topEffItem->effVal)) {
                topEffItem = &it;
            }
        }
    }
    const bool hasAnyPower = (powerCount > 0);

    ImGui::Spacing();

    // 4-Cell Summary Strip
    if (ImGui::BeginTable("PerfEfficiencyKpiStrip", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_SizingStretchSame)) {
        ImGui::TableNextRow();

        ImGui::TableNextColumn();
        ImGui::TextColored(c.textMuted, isLatencyDom ? "FASTEST LATENCY" : "TOP THROUGHPUT");
        pushBoldFont();
        ImGui::TextColored(c.phosphorHi, "%s", topPerfItem->perfText.c_str());
        popBoldFont();
        ImGui::TextColored(c.textMuted, "%s",
                           fitTextToPixelWidth(topPerfItem->fullLabel, ImGui::GetContentRegionAvail().x).c_str());

        ImGui::TableNextColumn();
        ImGui::TextColored(c.textMuted, isLatencyDom ? "AVG / PEAK POWER" : "PEAK POWER EFFICIENCY");
        pushBoldFont();
        if (isLatencyDom && hasAnyPower) {
            ImGui::TextColored(c.warnColor, "%.0f W avg / %.0f W pk", sumWatts / powerCount, maxWatts);
        } else if (topEffItem && !topEffItem->effText.empty()) {
            ImGui::TextColored(c.okColor, "%s", topEffItem->effText.c_str());
        } else {
            ImGui::TextColored(c.textMuted, "N/A (No Power Sensor)");
        }
        popBoldFont();
        if (topEffItem) {
            ImGui::TextColored(c.textMuted, "%s (%.0f W)",
                               fitTextToPixelWidth(topEffItem->shortLabel, std::max(s(60.0f), ImGui::GetContentRegionAvail().x - s(55.0f))).c_str(),
                               topEffItem->watts);
        }

        ImGui::TableNextColumn();
        ImGui::TextColored(c.textMuted, hasAnyPower ? "BOARD POWER RANGE" : "FASTEST KERNEL TIME");
        pushBoldFont();
        if (hasAnyPower) {
            ImGui::TextColored(c.warnColor, "%.0f W avg  |  %.0f W peak", sumWatts / powerCount, maxWatts);
        } else {
            double minMs = topPerfItem->medianMs;
            for (const auto& it : allDom) minMs = std::min(minMs, it.medianMs);
            ImGui::TextColored(c.cyanAccent, "%.2f ms median", minMs);
        }
        popBoldFont();

        ImGui::TableNextColumn();
        ImGui::TextColored(c.textMuted, "WORKLOADS COMPARED");
        pushBoldFont();
        if (fp32PerfVal > 0.0 && topPerfItem->perfVal > fp32PerfVal * 1.05) {
            ImGui::TextColored(c.cyanAccent, "%zu shown (%.1fx vs FP32)", bars.size(), topPerfItem->perfVal / fp32PerfVal);
        } else {
            ImGui::TextColored(c.textPrimary, "%zu of %zu shown", bars.size(), totalInDomain);
        }
        popBoldFont();

        ImGui::EndTable();
    }

    ImGui::Spacing();

    const int count = static_cast<int>(bars.size());
    std::vector<double> yPos(static_cast<size_t>(count));
    std::vector<std::string> leftTickStorage(static_cast<size_t>(count));
    std::vector<const char*> leftTickPtrs(static_cast<size_t>(count));
    std::vector<double> perfVals(static_cast<size_t>(count));
    std::vector<double> rightVals(static_cast<size_t>(count));
    std::vector<std::string> rightTickStorage(static_cast<size_t>(count));
    std::vector<const char*> rightTickPtrs(static_cast<size_t>(count));
    std::vector<ImU32> leftFillCols(static_cast<size_t>(count));
    std::vector<ImU32> leftLineCols(static_cast<size_t>(count));
    std::vector<ImU32> rightFillCols(static_cast<size_t>(count));
    std::vector<ImU32> rightLineCols(static_cast<size_t>(count));

    double maxPerf = 0.0;
    double maxRight = 0.0;
    for (int i = 0; i < count; ++i) {
        const auto& b = bars[static_cast<size_t>(i)];
        yPos[i] = static_cast<double>(i);
        leftTickStorage[i] = b.shortLabel;
        perfVals[i] = b.perfVal;
        if (b.perfVal > maxPerf) maxPerf = b.perfVal;

        if (b.isMatrix) {
            leftFillCols[i] = toU32(c.phosphorHi, 0.82f);
            leftLineCols[i] = toU32(c.phosphorHi, 1.0f);
        } else {
            leftFillCols[i] = toU32(c.cyanAccent, 0.82f);
            leftLineCols[i] = toU32(c.cyanAccent, 1.0f);
        }

        if (hasAnyPower) {
            rightVals[i] = b.effVal;
            if (b.watts > 0.0) {
                char wtLbl[24];
                std::snprintf(wtLbl, sizeof(wtLbl), "%.0f W", b.watts);
                rightTickStorage[i] = wtLbl;
            } else {
                rightTickStorage[i] = "N/A";
            }
            rightFillCols[i] = toU32(isLatencyDom ? c.warnColor : c.okColor, 0.82f);
            rightLineCols[i] = toU32(isLatencyDom ? c.warnColor : c.okColor, 1.0f);
        } else {
            rightVals[i] = b.medianMs;
            rightTickStorage[i] = b.shortLabel;
            rightFillCols[i] = toU32(c.violetAccent, 0.82f);
            rightLineCols[i] = toU32(c.violetAccent, 1.0f);
        }
        if (rightVals[i] > maxRight) maxRight = rightVals[i];
    }
    for (int i = 0; i < count; ++i) {
        leftTickPtrs[i] = leftTickStorage[i].c_str();
        rightTickPtrs[i] = rightTickStorage[i].c_str();
    }
    maxPerf = std::max(maxPerf, 1e-6);
    maxRight = std::max(maxRight, 1e-6);

    const float plotH = std::max(s(175.0f), ImGui::GetContentRegionAvail().y - s(4.0f));
    const ImPlotFlags chartFlags = ImPlotFlags_NoLegend | ImPlotFlags_NoMouseText | ImPlotFlags_NoBoxSelect;
    const ImPlotAxisFlags xAxisFlags = ImPlotAxisFlags_Lock | ImPlotAxisFlags_NoHighlight;
    const ImPlotAxisFlags yAxisFlags = ImPlotAxisFlags_Lock | ImPlotAxisFlags_NoGridLines | ImPlotAxisFlags_NoHighlight;

    auto renderHoverTooltipForBar = [&](int idx) {
        if (idx < 0 || idx >= count) return;
        const auto& b = bars[static_cast<size_t>(idx)];
        ImGui::BeginTooltip();
        pushBoldFont();
        ImGui::TextColored(c.phosphorHi, "%s", b.fullLabel.c_str());
        popBoldFont();
        ImGui::Separator();
        ImGui::Text("Throughput / Score: %s", b.perfText.c_str());
        if (m_chartDomainFilter == DOMAIN_COMPUTE && fp32PerfVal > 0.0 && b.fullLabel != "FP32") {
            ImGui::Text("vs FP32 Vector:     %.2fx", b.perfVal / fp32PerfVal);
        }
        ImGui::Text("Median Kernel Time: %.3f ms", b.medianMs);
        if (b.watts > 0.0) {
            ImGui::TextColored(c.warnColor, "Board Power Draw:   %.1f W", b.watts);
            if (!isLatencyDom && !b.effText.empty()) {
                ImGui::TextColored(c.okColor, "Power Efficiency:   %s", b.effText.c_str());
            }
        }
        ImGui::EndTooltip();
    };

    if (ImGui::BeginTable("DualPerfCharts", 2, ImGuiTableFlags_SizingStretchSame)) {
        // ---- LEFT CHART: RAW THROUGHPUT ----
        ImGui::TableNextColumn();
        std::string leftTitle = std::string(domainTabName(m_chartDomainFilter)) + " THROUGHPUT###RawThroughputBars";
        if (ImPlot::BeginPlot(leftTitle.c_str(), ImVec2(-1.0f, plotH), chartFlags)) {
            ImPlot::SetupAxes(domainPerfAxisLabel(m_chartDomainFilter), nullptr, xAxisFlags, yAxisFlags);
            ImPlot::SetupAxisTicks(ImAxis_Y1, yPos.data(), count, leftTickPtrs.data(), false);
            ImPlot::SetupAxesLimits(0.0, maxPerf * 1.48, -0.6, static_cast<double>(count) - 0.4, ImPlotCond_Always);

            ImPlotSpec spec;
            spec.Flags = static_cast<ImPlotItemFlags>(ImPlotBarsFlags_Horizontal) |
                         static_cast<ImPlotItemFlags>(ImPlotItemFlags_NoLegend);
            spec.FillColors = leftFillCols.data();
            spec.LineColors = leftLineCols.data();
            spec.LineWeight = 1.0f;
            ImPlot::PlotBars("##RawPerfBars", perfVals.data(), count, 0.62, 0.0, spec);

            ImPlot::PushStyleColor(ImPlotCol_InlayText, c.textPrimary);
            for (int i = 0; i < count; ++i) {
                const std::string& tip = bars[static_cast<size_t>(i)].perfText;
                const float txtW = ImGui::CalcTextSize(tip.c_str()).x;
                ImPlot::PlotText(tip.c_str(), perfVals[i], yPos[i], ImVec2(txtW * 0.5f + s(6.0f), 0.0f));
            }
            ImPlot::PopStyleColor();

            if (ImPlot::IsPlotHovered()) {
                const int hoveredIdx = static_cast<int>(std::round(ImPlot::GetPlotMousePos().y));
                renderHoverTooltipForBar(hoveredIdx);
            }
            ImPlot::EndPlot();
        }

        // ---- RIGHT CHART: POWER EFFICIENCY (OR MEDIAN TIME IF NO POWER TELEMETRY) ----
        ImGui::TableNextColumn();
        const char* rightTitle = hasAnyPower
            ? (isLatencyDom ? "BOARD POWER DRAW (W)###PowerEffBars" : "POWER EFFICIENCY (PERF / WATT)###PowerEffBars")
            : "MEDIAN KERNEL TIME (ms)###PowerEffBars";
        const char* rightXAxis = hasAnyPower
            ? domainEffAxisLabel(m_chartDomainFilter)
            : "Median Kernel Duration (ms)";

        if (ImPlot::BeginPlot(rightTitle, ImVec2(-1.0f, plotH), chartFlags)) {
            ImPlot::SetupAxes(rightXAxis, nullptr, xAxisFlags, yAxisFlags);
            ImPlot::SetupAxisTicks(ImAxis_Y1, yPos.data(), count, rightTickPtrs.data(), false);
            ImPlot::SetupAxesLimits(0.0, maxRight * 1.45, -0.6, static_cast<double>(count) - 0.4, ImPlotCond_Always);

            ImPlotSpec spec;
            spec.Flags = static_cast<ImPlotItemFlags>(ImPlotBarsFlags_Horizontal) |
                         static_cast<ImPlotItemFlags>(ImPlotItemFlags_NoLegend);
            spec.FillColors = rightFillCols.data();
            spec.LineColors = rightLineCols.data();
            spec.LineWeight = 1.0f;
            ImPlot::PlotBars("##EffBars", rightVals.data(), count, 0.62, 0.0, spec);

            ImPlot::PushStyleColor(ImPlotCol_InlayText, c.textPrimary);
            for (int i = 0; i < count; ++i) {
                char buf[48];
                if (hasAnyPower) {
                    std::snprintf(buf, sizeof(buf), "%s",
                                  bars[static_cast<size_t>(i)].effText.empty()
                                      ? "N/A"
                                      : bars[static_cast<size_t>(i)].effText.c_str());
                } else {
                    std::snprintf(buf, sizeof(buf), "%.2f ms", rightVals[i]);
                }
                const float txtW = ImGui::CalcTextSize(buf).x;
                ImPlot::PlotText(buf, rightVals[i], yPos[i], ImVec2(txtW * 0.5f + s(6.0f), 0.0f));
            }
            ImPlot::PopStyleColor();

            if (ImPlot::IsPlotHovered()) {
                const int hoveredIdx = static_cast<int>(std::round(ImPlot::GetPlotMousePos().y));
                renderHoverTooltipForBar(hoveredIdx);
            }
            ImPlot::EndPlot();
        }

        ImGui::EndTable();
    }
}

} // namespace gpubench::gui
