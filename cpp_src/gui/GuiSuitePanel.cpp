#include "GuiApp.h"
#include "GuiSuiteLogic.h"
#include "GuiUiHelpers.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <set>
#include <unordered_map>

namespace gpubench::gui {

namespace {

bool isCoreSuiteSlot(const BenchmarkItem& itm, bool isRocm) {
    const bool isIndirect = (itm.id == "GPU-Driven Compute Dispatch" ||
                             itm.id == "In-Shader Indirect Synthesis");
    const bool indirectCore = isRocm
        ? (isIndirect && (itm.configIndex == 1 || itm.configIndex == 3))
        : (isIndirect && (itm.configIndex == 1 || itm.configIndex == 3 || itm.configIndex == 5 || itm.configIndex == 7));
    return itm.id == "FP32" || itm.id == "FP16" || itm.id == "BF16" ||
           itm.id == "FP8" || itm.id == "INT8" ||
           (itm.id == "Device Memory Bandwidth" && itm.configIndex == 0) ||
           itm.id == "L0 Cache Bandwidth" || itm.id == "L1 Cache Bandwidth" ||
           itm.id == "L2 Cache Bandwidth" || itm.id == "L3 Cache Bandwidth" ||
           (itm.id == "PCIe & Host-Device Interconnect" && itm.configIndex <= 1) ||
           indirectCore ||
           (itm.id == "RayScheduling" && (itm.configIndex <= 2 || itm.configIndex == 4 || itm.configIndex == 5));
}

void syncSelectionFlags(std::vector<BenchmarkCategory>& categories) {
    for (auto& cat : categories) {
        bool catAny = false;
        for (auto& sg : cat.subgroups) {
            bool sgAny = false;
            for (const auto& itm : sg.items) {
                if (itm.selected && itm.isSupported) {
                    sgAny = true;
                    break;
                }
            }
            sg.allSelected = sgAny;
            if (sgAny) catAny = true;
        }
        cat.allSelected = catAny;
    }
}

void setCategoryArmed(BenchmarkCategory& cat, bool arm) {
    bool any = false;
    for (auto& sg : cat.subgroups) {
        bool sgAny = false;
        for (auto& itm : sg.items) {
            itm.selected = arm ? itm.isSupported : false;
            if (itm.selected) sgAny = true;
        }
        sg.allSelected = sgAny;
        if (sgAny) any = true;
    }
    cat.allSelected = any;
}

} // namespace

void GuiApp::initializeBenchmarkCategories() {
    m_categories.clear();

    const std::vector<std::pair<std::string, std::string>> categoryDefs = {
        {"Compute",      "Core shader math, AI/tensor matrix throughput, neural shading, shader ILP/dual-issue, and on-chip LDS memory."},
        {"Memory",       "VRAM streaming bandwidth, L0-L3 cache hierarchy bandwidth & latency cliffs, and PCIe/ReBAR/P2P interconnect speed."},
        {"Ray Tracing",  "BVH acceleration builds, hardware ray traversal/intersection, Opacity Micromaps, bounce divergence, and 4-scene path tracing."},
        {"Scheduling",   "GPU-driven compute dispatch compaction/DGC, Async Compute + DMA engine overlap, and shader-to-shader Work Graphs."},
        {"Raster",       "Fixed-function rasterization and Render Output Unit (ROP) pixel fill, HDR, and alpha-blending throughput."},
        {"Host System",  "CPU host system RAM streaming bandwidth and access latency—gating game logic, asset streaming, and APU graphics."}
    };

    for (const auto& [catName, catDesc] : categoryDefs) {
        BenchmarkCategory cat;
        cat.name = catName;
        cat.description = catDesc;
        cat.allSelected = true;
        m_categories.push_back(std::move(cat));
    }

    auto findCategory = [this](const std::string& name) -> BenchmarkCategory* {
        for (auto& cat : m_categories) {
            if (cat.name == name) return &cat;
        }
        return nullptr;
    };

    auto findOrAddSubgroup = [](BenchmarkCategory& cat, const std::string& sgName,
                                const std::string& comp, const std::string& engineId,
                                const std::string& desc) -> BenchmarkSubgroup& {
        for (auto& sg : cat.subgroups) {
            if (sg.name == sgName) return sg;
        }
        BenchmarkSubgroup sg;
        sg.name = sgName;
        sg.component = comp;
        sg.engineId = engineId;
        sg.description = desc;
        sg.allSelected = true;
        sg.collapsed = false;
        cat.subgroups.push_back(std::move(sg));
        return cat.subgroups.back();
    };

    const auto catalog = GetBenchmarkMetadataCatalogAPI();
    std::set<uint32_t> seenRaySchedulingConfigs;
    bool addedCacheLatencyCurve = false;

    for (const auto& meta : catalog) {
        // Collapse 15-step Cache Latency Curve into one sweep toggle with 15 subResults
        if (meta.benchmarkName == "Cache Latency Curve") {
            if (addedCacheLatencyCurve) continue;
            addedCacheLatencyCurve = true;

            BenchmarkCategory* memCat = findCategory("Memory");
            if (!memCat) continue;
            auto& sg = findOrAddSubgroup(
                *memCat,
                "GPU Cache & VRAM Access Latency",
                "Memory",
                "Cache Latency Curve",
                "Measures memory round-trip latency across 16 KB to 256 MB working sets, exposing cache size cliffs that stall irregular shader fetches in BVH traversal and indirection tables.");
            BenchmarkItem item(
                "Cache Latency Curve",
                meta.subcategory,
                "Cache Latency Curve (16 KB -> 256 MB Sweep)",
                "Memory",
                meta.metric,
                fallbackDescriptionForBenchmark(meta),
                true,
                -1);
            item.whatItTests = meta.whatItTests;
            item.pipelineImpact = meta.pipelineImpact;
            item.method = meta.method;
            item.configCaveat = meta.configCaveat;
            sg.items.push_back(std::move(item));
            continue;
        }

        // Collapse 4 scene instances of RayScheduling into unified "RayScheduling" items
        bool isRaySched = (meta.benchmarkName.rfind("RayScheduling", 0) == 0);
        if (isRaySched) {
            if (!seenRaySchedulingConfigs.insert(meta.configIndex).second) {
                continue;
            }
        }

        std::string targetCatName;
        std::string sgDesc;
        std::string sgTitle = subgroupTitleForCatalogEntry(meta, targetCatName, sgDesc);
        BenchmarkCategory* cat = findCategory(targetCatName);
        if (!cat) continue;

        std::string itemEngineId = isRaySched ? "RayScheduling" : meta.benchmarkName;
        std::string sgEngineId = isRaySched ? "RayScheduling" : meta.benchmarkName;
        std::string itemComponent = (!meta.isDeviceDependent || meta.component == "System")
            ? "System"
            : meta.component;

        auto& sg = findOrAddSubgroup(*cat, sgTitle, itemComponent, sgEngineId, sgDesc);
        if (sg.description.empty() && !sgDesc.empty()) {
            sg.description = sgDesc;
        }

        std::string friendlyName;
        if (isRaySched) {
            friendlyName = meta.configName.empty() ? meta.displayName : meta.configName;
        } else if (sgTitle.rfind("Shader & AI Matrix Compute", 0) == 0 ||
                   sgTitle.rfind("2:4 Sparse Matrix", 0) == 0 ||
                   sgTitle.rfind("GPU Cache Bandwidth", 0) == 0 ||
                   sgTitle.rfind("GPU Cache & VRAM Access Latency", 0) == 0) {
            friendlyName = meta.displayName;
        } else {
            friendlyName = meta.configName.empty() ? meta.displayName : meta.configName;
        }

        BenchmarkItem item(
            itemEngineId,
            meta.subcategory,
            friendlyName,
            itemComponent,
            meta.metric,
            fallbackDescriptionForBenchmark(meta),
            true,
            static_cast<int>(meta.configIndex));
        item.whatItTests = meta.whatItTests;
        item.pipelineImpact = meta.pipelineImpact;
        item.method = meta.method;
        item.configCaveat = meta.configCaveat;
        sg.items.push_back(std::move(item));
    }
}

void GuiApp::hydrateMetadataFromCatalog() {
    // Categories are already populated directly from GetBenchmarkMetadataCatalogAPI();
    // attach rich procedural schematics and 16:9 thumbnail descriptors.
    applyVisualizationMetadata();
}

bool GuiApp::matchesItem(const ResultData& r, const BenchmarkItem& itm, uint32_t activeDev) const {
    return matchesWorkloadItem(r, itm.id, itm.name, itm.category, itm.configIndex, activeDev);
}

GuiApp::BenchmarkDisplayInfo GuiApp::getBenchmarkDisplayInfo(
    const BenchmarkItem& item,
    uint32_t targetDeviceIndex,
    const std::string& sceneOverride) const
{
    BenchmarkDisplayInfo info;
    info.pipelineExplanation = item.description;

    // 1. Multi-step Cache Latency Curve
    if (item.id == "Cache Latency Curve") {
        std::vector<const ResultData*> curvePoints;
        for (const auto& r : m_allResults) {
            if (r.deviceIndex == targetDeviceIndex &&
                r.benchmarkName.rfind("Cache Latency Curve", 0) == 0 &&
                !r.isUnsupported && r.time_ms > 0.0) {
                curvePoints.push_back(&r);
            }
        }
        if (!curvePoints.empty()) {
            std::sort(curvePoints.begin(), curvePoints.end(),
                      [](const ResultData* a, const ResultData* b) { return a->configIndex < b->configIndex; });
            info.hasResult = true;
            info.primaryResult = *curvePoints.front();

            double minNs = 1e18, maxNs = 0.0;
            for (const auto* pt : curvePoints) {
                double ns = (pt->operations > 0) ? ((pt->time_ms * 1e6) / static_cast<double>(pt->operations)) : 0.0;
                minNs = std::min(minNs, ns);
                maxNs = std::max(maxNs, ns);
                BenchmarkDisplayInfo::SubResult sr;
                sr.label = cleanWorkloadName(pt->benchmarkName, pt->subcategory);
                sr.score = formatResultScoreString(*pt, false);
                info.subResults.push_back(std::move(sr));
            }
            char rangeBuf[64];
            std::snprintf(rangeBuf, sizeof(rangeBuf), "%.1f -> %.1f ns", minNs, maxNs);
            info.scoreText = rangeBuf;
        }
        return info;
    }

    // 2. Scene-parameterized RayScheduling
    if (item.id == "RayScheduling" && item.configIndex >= 0) {
        std::string targetScene = !sceneOverride.empty() ? sceneOverride : m_scene;
        auto matchesSceneFilter = [&](const std::string& bName) {
            if (targetScene == "all") return true;
            std::string low = bName;
            for (char& ch : low) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
            if (targetScene == "indoor") return low.find("indoor") != std::string::npos || low.find("sponza") != std::string::npos;
            if (targetScene == "forest") return low.find("forest") != std::string::npos;
            if (targetScene == "landscape" || targetScene == "outdoor") return low.find("landscape") != std::string::npos || low.find("outdoor") != std::string::npos;
            if (targetScene == "showroom") return low.find("showroom") != std::string::npos;
            return true;
        };

        std::vector<const ResultData*> sceneMatches;
        for (const auto& r : m_allResults) {
            if (r.deviceIndex == targetDeviceIndex &&
                r.benchmarkName.rfind("RayScheduling", 0) == 0 &&
                static_cast<int>(r.configIndex) == item.configIndex &&
                matchesSceneFilter(r.benchmarkName)) {
                sceneMatches.push_back(&r);
            }
        }
        if (sceneMatches.empty()) return info;

        info.hasResult = true;
        info.primaryResult = *sceneMatches.back();
        if (info.primaryResult.isUnsupported) {
            info.scoreText = "UNSUPPORTED";
            return info;
        }
        info.scoreText = formatResultScoreString(info.primaryResult, false);

        // Loaded comparison JSON takes priority if present; otherwise intra-suite baselineConfigIndex
        const ResultData* baseRes = findComparisonMatch(info.primaryResult);
        if (!baseRes && info.primaryResult.baselineConfigIndex >= 0) {
            std::string primaryScene = extractSceneName(info.primaryResult.benchmarkName);
            for (const auto& r : m_allResults) {
                if (r.deviceIndex == targetDeviceIndex &&
                    r.benchmarkName.rfind("RayScheduling", 0) == 0 &&
                    static_cast<int32_t>(r.configIndex) == info.primaryResult.baselineConfigIndex &&
                    extractSceneName(r.benchmarkName) == primaryScene &&
                    !r.isUnsupported && r.time_ms > 0.0) {
                    baseRes = &r;
                    break;
                }
            }
        } else if (!baseRes && info.primaryResult.baselineConfigIndex < 0) {
            // Check if another config uses this config as its baseline
            const BringUpColors pal = getBringUpColors(m_palette);
            for (const auto& r : m_allResults) {
                if (r.deviceIndex == targetDeviceIndex &&
                    r.benchmarkName.rfind("RayScheduling", 0) == 0 &&
                    r.baselineConfigIndex == static_cast<int32_t>(item.configIndex)) {
                    info.isBaseline = true;
                    info.deltaText = "Baseline";
                    info.deltaColor = pal.cyanAccent;
                    break;
                }
            }
        }

        if (baseRes) {
            const BringUpColors pal = getBringUpColors(m_palette);
            double curRate = computeThroughputOrInvLatency(info.primaryResult);
            double baseRate = computeThroughputOrInvLatency(*baseRes);
            if (curRate > 0.0 && baseRate > 0.0) {
                info.hasComparison = true;
                info.baselineResult = *baseRes;
                info.speedupRatio = curRate / baseRate;
                info.percentDelta = (info.speedupRatio - 1.0) * 100.0;
                char dbuf[48];
                std::snprintf(dbuf, sizeof(dbuf), "%.2fx (%+.1f%%)", info.speedupRatio, info.percentDelta);
                info.deltaText = dbuf;
                info.deltaColor = (info.speedupRatio >= 1.01)
                    ? pal.okColor
                    : ((info.speedupRatio <= 0.99) ? pal.errColor : pal.textSecondary);
            }
        }
        return info;
    }

    // 3. Standard single-config benchmarks (Compute, Memory, Scheduling, OMM, BVH, Raster, Host System)
    const ResultData* matched = nullptr;
    for (auto it = m_allResults.rbegin(); it != m_allResults.rend(); ++it) {
        if (matchesItem(*it, item, targetDeviceIndex)) {
            matched = &(*it);
            break;
        }
    }
    if (!matched) return info;

    info.hasResult = true;
    info.primaryResult = *matched;
    if (matched->isUnsupported) {
        info.scoreText = "UNSUPPORTED";
        return info;
    }
    info.scoreText = formatResultScoreString(*matched, false);

    const ResultData* baseRes = findComparisonMatch(*matched);
    if (!baseRes && matched->baselineConfigIndex >= 0) {
        for (auto it = m_allResults.rbegin(); it != m_allResults.rend(); ++it) {
            if (it->deviceIndex == matched->deviceIndex &&
                it->backendName == matched->backendName &&
                static_cast<int32_t>(it->configIndex) == matched->baselineConfigIndex &&
                it->subcategory == matched->subcategory &&
                !it->isUnsupported && it->time_ms > 0.0) {
                baseRes = &(*it);
                break;
            }
        }
    } else if (!baseRes && matched->baselineConfigIndex < 0) {
        const BringUpColors pal = getBringUpColors(m_palette);
        for (const auto& r : m_allResults) {
            if (r.deviceIndex == matched->deviceIndex &&
                r.subcategory == matched->subcategory &&
                r.baselineConfigIndex == static_cast<int32_t>(matched->configIndex)) {
                info.isBaseline = true;
                info.deltaText = "Baseline";
                info.deltaColor = pal.cyanAccent;
                break;
            }
        }
    }

    if (baseRes) {
        const BringUpColors pal = getBringUpColors(m_palette);
        double curRate = computeThroughputOrInvLatency(*matched);
        double baseRate = computeThroughputOrInvLatency(*baseRes);
        if (curRate > 0.0 && baseRate > 0.0) {
            info.hasComparison = true;
            info.baselineResult = *baseRes;
            info.speedupRatio = curRate / baseRate;
            info.percentDelta = (info.speedupRatio - 1.0) * 100.0;
            char dbuf[48];
            std::snprintf(dbuf, sizeof(dbuf), "%.2fx (%+.1f%%)", info.speedupRatio, info.percentDelta);
            info.deltaText = dbuf;
            info.deltaColor = (info.speedupRatio >= 1.01)
                ? pal.okColor
                : ((info.speedupRatio <= 0.99) ? pal.errColor : pal.textSecondary);
        }
    }

    return info;
}

void GuiApp::renderBenchmarkSuitePanel() {
    const BringUpColors c = getBringUpColors(m_palette);
    const bool isRunning = (m_execState == ExecutionState::Running);
    const uint32_t activeGpu = m_telemetryGpuIndex;
    const std::string activeScene = (m_scene == "all") ? m_suiteActiveScene : m_scene;

    const bool isRocmBackend = (resolvedBackendName() == "rocm");
    size_t totalArmed = 0;
    size_t totalSupported = 0;
    bool matchesCorePreset = true;
    for (const auto& cat : m_categories) {
        for (const auto& sg : cat.subgroups) {
            for (const auto& itm : sg.items) {
                const bool wantCore = itm.isSupported && isCoreSuiteSlot(itm, isRocmBackend);
                if (itm.selected != wantCore) {
                    matchesCorePreset = false;
                }
                if (itm.isSupported) {
                    totalSupported++;
                    if (itm.selected) totalArmed++;
                }
            }
        }
    }
    const bool isAllPresetActive = (totalSupported > 0 && totalArmed == totalSupported);
    const bool isNonePresetActive = (totalArmed == 0);
    const bool isCorePresetActive = (totalArmed > 0 && matchesCorePreset);

    char rightTag[48];
    std::snprintf(rightTag, sizeof(rightTag), "%zu/%zu ARMED", totalArmed, totalSupported);
    drawEtchedHeader("ROM-01", "WORKLOAD MATRIX", c, rightTag);

    auto allCategoriesMask = [&]() -> uint32_t {
        if (m_categories.empty()) return 0u;
        return (m_categories.size() >= 32)
            ? 0xFFFFFFFFu
            : ((1u << m_categories.size()) - 1u);
    };
    auto isCategoryInView = [&](size_t ci) -> bool {
        if (m_suiteViewCategoryMask == 0u) return true;
        if (ci >= 32) return true;
        return (m_suiteViewCategoryMask & (1u << ci)) != 0u;
    };
    auto searchMatchesAnyItemInMask = [&](uint32_t mask) -> bool {
        if (m_suiteSearchBuf[0] == '\0') return true;
        const std::string q(m_suiteSearchBuf);
        for (size_t ci = 0; ci < m_categories.size(); ++ci) {
            if (mask != 0u && ci < 32 && (mask & (1u << ci)) == 0u) continue;
            const auto& cat = m_categories[ci];
            if (containsCaseInsensitive(cat.name, q)) return true;
            for (const auto& sg : cat.subgroups) {
                if (containsCaseInsensitive(sg.name, q)) return true;
                for (const auto& itm : sg.items) {
                    if (itm.selected && itm.isSupported &&
                        (containsCaseInsensitive(itm.name, q) || containsCaseInsensitive(itm.id, q))) {
                        return true;
                    }
                }
            }
        }
        return false;
    };
    auto computeArmedCategoryMask = [&]() -> uint32_t {
        uint32_t mask = 0u;
        for (size_t ci = 0; ci < m_categories.size() && ci < 32; ++ci) {
            bool catHasArmed = false;
            for (const auto& sg : m_categories[ci].subgroups) {
                for (const auto& itm : sg.items) {
                    if (itm.selected && itm.isSupported) {
                        catHasArmed = true;
                        break;
                    }
                }
                if (catHasArmed) break;
            }
            if (catHasArmed) mask |= (1u << ci);
        }
        return mask;
    };
    auto syncViewToArmedCategories = [&](int focusCatIdx) {
        const uint32_t allMask = allCategoriesMask();
        const uint32_t armedMask = computeArmedCategoryMask();
        m_suiteViewCategoryMask = (armedMask == 0u || (armedMask & allMask) == allMask)
            ? 0u
            : armedMask;
        if (m_suiteSearchBuf[0] != '\0' && !searchMatchesAnyItemInMask(m_suiteViewCategoryMask)) {
            m_suiteSearchBuf[0] = '\0';
        }
        if (focusCatIdx >= 0) {
            m_suiteScrollToCategoryIdx = focusCatIdx;
            m_suiteScrollRetries = 2;
        }
    };

    // Preset & Bulk Selection Bar
    if (isRunning) ImGui::BeginDisabled();
    const float btnH = s(22.0f);
    if (tacticalButton("##BtnSelectAll", "ALL", ImVec2(s(46.0f), btnH), isAllPresetActive, c)) {
        for (auto& cat : m_categories) {
            setCategoryArmed(cat, true);
        }
        syncSelectionFlags(m_categories);
        syncViewToArmedCategories(0);
    }
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        ImGui::SetTooltip("Arm all %zu supported benchmarks across every group.", totalSupported);
    }
    ImGui::SameLine(0.0f, s(4.0f));
    if (tacticalButton("##BtnDeselectAll", "NONE", ImVec2(s(50.0f), btnH), isNonePresetActive, c)) {
        for (auto& cat : m_categories) {
            setCategoryArmed(cat, false);
        }
        syncSelectionFlags(m_categories);
        syncViewToArmedCategories(-1);
    }
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        ImGui::SetTooltip("Disarm all benchmarks so you can pick specific groups or slots.");
    }
    ImGui::SameLine(0.0f, s(4.0f));
    if (tacticalButton("##BtnRooflinePreset", "CORE SUITE", ImVec2(s(88.0f), btnH), isCorePresetActive, c)) {
        int firstArmedCat = -1;
        for (size_t ci = 0; ci < m_categories.size(); ++ci) {
            for (auto& sg : m_categories[ci].subgroups) {
                for (auto& itm : sg.items) {
                    itm.selected = itm.isSupported && isCoreSuiteSlot(itm, isRocmBackend);
                    if (itm.selected && firstArmedCat < 0) firstArmedCat = static_cast<int>(ci);
                }
            }
        }
        syncSelectionFlags(m_categories);
        syncViewToArmedCategories(firstArmedCat);
    }
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        ImGui::SetTooltip("Arm curated Core Suite (FP32/FP16/BF16/FP8/INT8, VRAM & L0-L3 Cache BW, PCIe DMA, Core RT Path Tracing).");
    }
    if (isRunning) ImGui::EndDisabled();

    const char* unsupLabel = m_hideUnsupported ? "UNSUP: HIDDEN" : "UNSUP: SHOWN";
    const float unsupW = ImGui::CalcTextSize(unsupLabel).x + s(16.0f);
    sameLineIfFits(unsupW, s(6.0f));
    if (tacticalButton("##BtnHideUnsup", unsupLabel, ImVec2(0.0f, btnH), m_hideUnsupported, c)) {
        m_hideUnsupported = !m_hideUnsupported;
    }

    // Test Group Arming Bar (toggle whole categories ON/OFF for execution)
    ImGui::Spacing();
    if (isRunning) ImGui::BeginDisabled();
    {
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(c.textMuted, "ARM:");
        for (size_t ci = 0; ci < m_categories.size(); ++ci) {
            auto& cat = m_categories[ci];
            size_t catArmed = 0;
            size_t catSup = 0;
            for (const auto& sg : cat.subgroups) {
                for (const auto& itm : sg.items) {
                    if (itm.isSupported) {
                        catSup++;
                        if (itm.selected) catArmed++;
                    }
                }
            }
            const bool groupOn = (catArmed > 0);
            const bool partial = (catArmed > 0 && catArmed < catSup);
            const char* tag = shortCategoryTag(cat.name);
            char lbl[32];
            std::snprintf(lbl, sizeof(lbl), "%s%s", tag, partial ? "*" : "");

            const float pillW = std::max(s(36.0f), ImGui::CalcTextSize(lbl).x + s(14.0f));
            sameLineIfFits(pillW, s(3.0f));
            std::string grpId = "##GrpArm_" + std::to_string(ci);
            if (tacticalButton(grpId.c_str(), lbl, ImVec2(pillW, s(21.0f)), groupOn, c)) {
                const bool arming = !groupOn;
                setCategoryArmed(cat, arming);
                syncSelectionFlags(m_categories);
                syncViewToArmedCategories(arming ? static_cast<int>(ci) : -1);
            }
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                ImGui::SetTooltip("%s: %zu/%zu armed (click to turn %s).",
                                  cat.name.c_str(), catArmed, catSup, groupOn ? "OFF" : "ON");
            }
        }
    }
    if (isRunning) ImGui::EndDisabled();

    // Slot Filter Input
    ImGui::Spacing();
    {
        float availSearchW = ImGui::GetContentRegionAvail().x;
        bool hasSearch = (m_suiteSearchBuf[0] != '\0');
        float clrW = hasSearch ? s(40.0f) : 0.0f;
        float inputW = std::max(s(80.0f), availSearchW - (hasSearch ? (clrW + s(4.0f)) : 0.0f));
        ImGui::SetNextItemWidth(inputW);
        pushMonoFont();
        ImGui::InputTextWithHint("##SuiteSearchInput", "FILTER SLOTS (FP16, WMMA, L2, BVH)...",
                                 m_suiteSearchBuf, sizeof(m_suiteSearchBuf));
        popMonoFont();
        if (hasSearch) {
            ImGui::SameLine(0.0f, s(4.0f));
            if (tacticalButton("##BtnClearSuiteSearch", "CLR", ImVec2(clrW, s(22.0f)), false, c)) {
                m_suiteSearchBuf[0] = '\0';
            }
        }
    }

    // Category View Filter Matrix (multi-category bitmask: 0 = ALL)
    ImGui::Spacing();
    {
        const uint32_t allMask = allCategoriesMask();
        const bool multiMod = ImGui::GetIO().KeyCtrl || ImGui::GetIO().KeyShift;
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(c.textMuted, "VIEW:");
        ImGui::SameLine(0.0f, s(4.0f));
        bool allActive = (m_suiteViewCategoryMask == 0u);
        if (tacticalButton("##CatAll", "ALL", ImVec2(0.0f, s(21.0f)), allActive, c)) {
            m_suiteViewCategoryMask = 0u;
        }
        for (size_t ci = 0; ci < m_categories.size(); ++ci) {
            const uint32_t bit = (ci < 32) ? (1u << ci) : 0u;
            std::string pillLabel = std::string(shortCategoryTag(m_categories[ci].name)) +
                                    " [" + std::to_string(m_categories[ci].totalItemCount()) + "]";
            float pillW = ImGui::CalcTextSize(pillLabel.c_str()).x + s(16.0f);
            sameLineIfFits(pillW, s(4.0f));
            bool active = (m_suiteViewCategoryMask != 0u) && ((m_suiteViewCategoryMask & bit) != 0u);
            std::string btnId = "##CatBtn_" + std::to_string(ci);
            if (tacticalButton(btnId.c_str(), pillLabel.c_str(), ImVec2(0.0f, s(21.0f)), active, c)) {
                if (bit != 0u) {
                    if (multiMod) {
                        if (m_suiteViewCategoryMask == 0u) {
                            m_suiteViewCategoryMask = bit;
                        } else {
                            m_suiteViewCategoryMask ^= bit;
                            if (m_suiteViewCategoryMask == 0u ||
                                (m_suiteViewCategoryMask & allMask) == allMask) {
                                m_suiteViewCategoryMask = 0u;
                            }
                        }
                    } else {
                        m_suiteViewCategoryMask = bit;
                    }
                }
            }
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                ImGui::SetTooltip("View %s (Ctrl/Shift+click to multi-select view groups).",
                                  m_categories[ci].name.c_str());
            }
        }
    }

    // Cache DisplayInfo once per visible item
    struct CachedSlotRow {
        size_t ci{0};
        size_t sgi{0};
        size_t ii{0};
        BenchmarkDisplayInfo disp;
    };
    std::vector<CachedSlotRow> cachedRows;
    const std::string searchQuery(m_suiteSearchBuf);

    for (size_t ci = 0; ci < m_categories.size(); ++ci) {
        if (!isCategoryInView(ci)) {
            continue;
        }
        const auto& cat = m_categories[ci];
        for (size_t sgi = 0; sgi < cat.subgroups.size(); ++sgi) {
            const auto& sg = cat.subgroups[sgi];
            for (size_t ii = 0; ii < sg.items.size(); ++ii) {
                const auto& itm = sg.items[ii];
                if (m_hideUnsupported && !itm.isSupported) continue;
                if (!searchQuery.empty() &&
                    !containsCaseInsensitive(itm.name, searchQuery) &&
                    !containsCaseInsensitive(itm.id, searchQuery) &&
                    !containsCaseInsensitive(sg.name, searchQuery) &&
                    !containsCaseInsensitive(cat.name, searchQuery)) {
                    continue;
                }
                CachedSlotRow row{};
                row.ci = ci;
                row.sgi = sgi;
                row.ii = ii;
                row.disp = getBenchmarkDisplayInfo(itm, activeGpu, activeScene);
                cachedRows.push_back(std::move(row));
            }
        }
    }

    ImGui::Spacing();
    ImGui::Separator();

    const bool useInnerScroll = (ImGui::GetContentRegionAvail().y >= s(140.0f));
    if (useInnerScroll) {
        ImGui::BeginChild("SuiteScrollRegion", ImVec2(0, 0), ImGuiChildFlags_None);
    }

    size_t cursorRow = 0;
    for (size_t ci = 0; ci < m_categories.size(); ++ci) {
        if (!isCategoryInView(ci)) {
            continue;
        }

        auto& cat = m_categories[ci];
        bool anyCatVisible = false;
        for (size_t r = cursorRow; r < cachedRows.size(); ++r) {
            if (cachedRows[r].ci == ci) {
                anyCatVisible = true;
                break;
            }
            if (cachedRows[r].ci > ci) break;
        }
        if (!anyCatVisible) continue;

        ImGui::PushID(static_cast<int>(ci));
        ImGui::Spacing();
        if (m_suiteScrollToCategoryIdx == static_cast<int>(ci) && m_suiteScrollRetries > 0) {
            ImGui::SetScrollHereY(0.05f);
            if (--m_suiteScrollRetries <= 0) {
                m_suiteScrollToCategoryIdx = -1;
            }
        }

        // Category Etched Band with group Arm toggle & +ALL / -CLR
        {
            size_t catArmed = 0;
            size_t catSupported = 0;
            for (const auto& sg : cat.subgroups) {
                for (const auto& itm : sg.items) {
                    if (itm.isSupported) {
                        catSupported++;
                        if (itm.selected) catArmed++;
                    }
                }
            }
            bool catToggle = (catArmed > 0);
            const bool catDisabled = isRunning || (catSupported == 0);
            if (catDisabled) ImGui::BeginDisabled();
            if (hardwareArmToggle("##CatArm", &catToggle, c, m_uiScale)) {
                setCategoryArmed(cat, catToggle);
                syncSelectionFlags(m_categories);
            }
            if (catDisabled) ImGui::EndDisabled();

            ImGui::SameLine(0.0f, s(6.0f));
            const float rightBtnsW = isRunning ? 0.0f : (s(32.0f) * 2.0f + s(8.0f));
            const float labelAvailW = std::max(s(60.0f), ImGui::GetContentRegionAvail().x - rightBtnsW);
            char catBuf[128];
            std::snprintf(catBuf, sizeof(catBuf), "// %s [%zu/%zu]",
                          cat.name.c_str(), catArmed, catSupported);
            pushBoldFont();
            ImGui::TextColored(c.phosphorHi, "%s", fitTextToPixelWidth(catBuf, labelAvailW).c_str());
            popBoldFont();
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled) && !cat.description.empty()) {
                ImGui::SetTooltip("%s", cat.description.c_str());
            }

            if (!isRunning) {
                const float smallBtnW = s(32.0f);
                const float btnH2 = s(18.0f);
                ImGui::SameLine();
                const float endX = ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x;
                const float btnsX = endX - (smallBtnW * 2.0f + s(4.0f));
                if (btnsX > ImGui::GetCursorPosX() + s(4.0f)) {
                    ImGui::SetCursorPosX(btnsX);
                }
                if (tacticalButton("##CatAll", "+ALL", ImVec2(smallBtnW, btnH2), false, c)) {
                    setCategoryArmed(cat, true);
                    syncSelectionFlags(m_categories);
                }
                ImGui::SameLine(0.0f, s(4.0f));
                if (tacticalButton("##CatClr", "-CLR", ImVec2(smallBtnW, btnH2), false, c)) {
                    setCategoryArmed(cat, false);
                    syncSelectionFlags(m_categories);
                }
            }
        }

        // RayTracing Scene Selector Strip
        if (cat.name == "Ray Tracing") {
            const char* sceneIds[] = {"all", "indoor", "showroom", "forest", "landscape"};
            const char* scenePills[] = {"ALL 4", "INDOOR", "SHOWROOM", "FOREST", "LANDSCAPE"};
            for (int si = 0; si < 5; ++si) {
                float pw = ImGui::CalcTextSize(scenePills[si]).x + s(14.0f);
                if (si > 0) sameLineIfFits(pw, s(4.0f));
                bool activeSceneSel = (m_scene == sceneIds[si]);
                std::string sId = "##RtScene_" + std::to_string(si);
                if (tacticalButton(sId.c_str(), scenePills[si], ImVec2(0.0f, s(20.0f)), activeSceneSel, c)) {
                    setScene(sceneIds[si]);
                }
            }
            ImGui::Spacing();
        }

        for (size_t sgi = 0; sgi < cat.subgroups.size(); ++sgi) {
            auto& sg = cat.subgroups[sgi];

            size_t startIdx = cursorRow;
            while (startIdx < cachedRows.size() &&
                   (cachedRows[startIdx].ci < ci ||
                    (cachedRows[startIdx].ci == ci && cachedRows[startIdx].sgi < sgi))) {
                startIdx++;
            }
            size_t endIdx = startIdx;
            while (endIdx < cachedRows.size() && cachedRows[endIdx].ci == ci && cachedRows[endIdx].sgi == sgi) {
                endIdx++;
            }
            cursorRow = endIdx;
            if (startIdx == endIdx) continue;

            size_t selectedCount = 0;
            size_t supportedCount = 0;
            for (const auto& itm : sg.items) {
                if (itm.isSupported) {
                    supportedCount++;
                    if (itm.selected) selectedCount++;
                }
            }

            ImGui::PushID(static_cast<int>(sgi));
            ImGui::Spacing();

            // Zero-radius Subgroup Header Bar
            {
                ImDrawList* dl = ImGui::GetWindowDrawList();
                ImVec2 hdrP0 = ImGui::GetCursorScreenPos();
                float fullW = std::max(s(120.0f), ImGui::GetContentRegionAvail().x);
                float hdrH = s(22.0f);
                float smallBtnW = isRunning ? 0.0f : s(32.0f);
                float toggleW = isRunning ? fullW : std::max(s(80.0f), fullW - (smallBtnW * 2.0f + s(8.0f)));

                ImVec2 hdrP1(hdrP0.x + fullW, hdrP0.y + hdrH);
                dl->AddRectFilled(hdrP0, hdrP1, toU32(c.bgElevated), 0.0f);
                dl->AddRect(hdrP0, hdrP1, toU32(c.borderInner), 0.0f, 0, 1.0f);

                if (ImGui::InvisibleButton("##SGHdrToggle", ImVec2(toggleW, hdrH))) {
                    sg.collapsed = !sg.collapsed;
                }
                bool hdrHovered = ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled);
                if (hdrHovered) {
                    dl->AddRectFilled(hdrP0, ImVec2(hdrP0.x + toggleW, hdrP1.y), toU32(c.bgHover, 0.55f), 0.0f);
                    if (!sg.description.empty()) {
                        ImGui::BeginTooltip();
                        ImGui::PushTextWrapPos(ImGui::GetCursorPos().x + s(380.0f));
                        pushBoldFont();
                        ImGui::TextColored(c.phosphorHi, "%s", sg.name.c_str());
                        popBoldFont();
                        ImGui::Separator();
                        ImGui::TextWrapped("%s", sg.description.c_str());
                        ImGui::PopTextWrapPos();
                        ImGui::EndTooltip();
                    }
                }

                char countBuf[32];
                std::snprintf(countBuf, sizeof(countBuf), "[%zu/%zu]", selectedCount, supportedCount);
                const float countW = ImGui::CalcTextSize(countBuf).x;
                const std::string titleText = std::string(sg.collapsed ? "[+] " : "[-] ") + sg.name;
                const float maxTitleW = std::max(s(40.0f), toggleW - s(16.0f) - countW);
                const std::string fittedTitle = fitTextToPixelWidth(titleText, maxTitleW);
                const float textY = hdrP0.y + (hdrH - ImGui::GetTextLineHeight()) * 0.5f;
                dl->AddText(ImVec2(hdrP0.x + s(6.0f), textY),
                            toU32(hdrHovered ? c.phosphorHi : c.textPrimary), fittedTitle.c_str());
                dl->AddText(ImVec2(hdrP0.x + toggleW - s(6.0f) - countW, textY),
                            toU32(c.textSecondary), countBuf);

                if (!isRunning) {
                    ImGui::SameLine(0.0f, s(4.0f));
                    if (tacticalButton("##SGAll", "+ALL", ImVec2(smallBtnW, hdrH), false, c)) {
                        for (auto& itm : sg.items) {
                            if (itm.isSupported) itm.selected = true;
                        }
                        sg.collapsed = false;
                        syncSelectionFlags(m_categories);
                    }
                    ImGui::SameLine(0.0f, s(4.0f));
                    if (tacticalButton("##SGNone", "-CLR", ImVec2(smallBtnW, hdrH), false, c)) {
                        for (auto& itm : sg.items) itm.selected = false;
                        syncSelectionFlags(m_categories);
                    }
                }
            }

            if (!sg.collapsed) {
                ImGuiTableFlags tblFlags = ImGuiTableFlags_BordersInnerH |
                                           ImGuiTableFlags_RowBg |
                                           ImGuiTableFlags_SizingStretchProp;
                if (ImGui::BeginTable("SubgroupItemsTable", 4, tblFlags)) {
                    ImGui::TableSetupColumn("Arm",   ImGuiTableColumnFlags_WidthFixed,   s(22.0f));
                    ImGui::TableSetupColumn("Slot",  ImGuiTableColumnFlags_WidthStretch, 0.67f);
                    ImGui::TableSetupColumn("Score", ImGuiTableColumnFlags_WidthStretch, 0.33f);
                    ImGui::TableSetupColumn("Run",   ImGuiTableColumnFlags_WidthFixed,   s(24.0f));

                    for (size_t r = startIdx; r < endIdx; ++r) {
                        const auto& cached = cachedRows[r];
                        const size_t ii = cached.ii;
                        auto& itm = sg.items[ii];
                        const auto& disp = cached.disp;

                        const bool isInspected =
                            (m_inspectedCategoryIdx == static_cast<int>(ci) &&
                             m_inspectedSubgroupIdx == static_cast<int>(sgi) &&
                             m_inspectedItemIdx == static_cast<int>(ii));

                        ImGui::PushID(static_cast<int>(ii));
                        ImGui::TableNextRow(ImGuiTableRowFlags_None, s(23.0f));

                        if (isInspected) {
                            ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg1, toU32(c.phosphorGlow));
                            if (m_suiteAutoScrollToInspected) {
                                ImGui::SetScrollHereY(0.35f);
                                m_suiteAutoScrollToInspected = false;
                            }
                        }

                        // Col 0: Hardware Arm Toggle + 3px Left Phosphor Indicator
                        ImGui::TableSetColumnIndex(0);
                        ImVec2 cellPos = ImGui::GetCursorScreenPos();
                        if (isInspected) {
                            ImDrawList* dl = ImGui::GetWindowDrawList();
                            dl->AddRectFilled(
                                ImVec2(cellPos.x - s(5.0f), cellPos.y - s(2.0f)),
                                ImVec2(cellPos.x - s(2.0f), cellPos.y + s(19.0f)),
                                toU32(c.phosphorHi), 0.0f);
                        }
                        bool rowDisabled = isRunning || !itm.isSupported;
                        if (rowDisabled) ImGui::BeginDisabled();
                        if (hardwareArmToggle("##ArmSlot", &itm.selected, c, m_uiScale)) {
                            syncSelectionFlags(m_categories);
                        }
                        if (rowDisabled) ImGui::EndDisabled();

                        // Col 1: Clickable Slot Label -> F1 ISA & Die Inspector
                        ImGui::TableSetColumnIndex(1);
                        {
                            float colW = std::max(s(40.0f), ImGui::GetContentRegionAvail().x);
                            std::string compactName = compactWorkloadSlotLabel(itm.name, sg.name);
                            std::string fittedName = fitTextToPixelWidth(compactName, colW - s(4.0f));
                            ImVec2 textPos = ImGui::GetCursorScreenPos();
                            float rowH = s(18.0f);
                            if (ImGui::InvisibleButton("##SelectSlot", ImVec2(colW, rowH))) {
                                selectInspectedBenchmark(static_cast<int>(ci), static_cast<int>(sgi), static_cast<int>(ii), true);
                            }
                            bool lblHovered = ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled);
                            if (lblHovered) {
                                renderBenchmarkTooltip(itm, activeScene);
                            }
                            ImVec4 txtCol = !itm.isSupported
                                ? c.textMuted
                                : ((isInspected || lblHovered) ? c.phosphorHi : c.textPrimary);
                            ImGui::GetWindowDrawList()->AddText(
                                ImVec2(textPos.x, textPos.y + (rowH - ImGui::GetTextLineHeight()) * 0.5f),
                                toU32(txtCol), fittedName.c_str());
                        }

                        // Col 2: Tabular Score + Delta Badge
                        ImGui::TableSetColumnIndex(2);
                        pushMonoFont();
                        if (m_hasCurrentlyRunningResult && matchesItem(m_currentlyRunningResult, itm, activeGpu)) {
                            ImGui::TextColored(c.warnColor, "RUNNING...");
                        } else if (!itm.isSupported) {
                            ImGui::TextColored(c.errColor, "UNSUP");
                            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled) &&
                                (!itm.supportReason.empty() || itm.hasSupportDiscrepancy)) {
                                if (itm.hasSupportDiscrepancy) {
                                    ImGui::SetTooltip("%s: %s\nHW: %s\nDRV: %s",
                                        itm.limitationCategory.empty() ? "Unsupported" : itm.limitationCategory.c_str(),
                                        itm.supportReason.c_str(),
                                        itm.hardwareSupportText.c_str(),
                                        itm.driverSupportText.c_str());
                                } else {
                                    ImGui::SetTooltip("%s: %s",
                                        itm.limitationCategory.empty() ? "Unsupported" : itm.limitationCategory.c_str(),
                                        itm.supportReason.c_str());
                                }
                            }
                        } else if (disp.hasResult) {
                            ImGui::TextColored(c.textPrimary, "%s", disp.scoreText.c_str());
                            const float remW = ImGui::GetContentRegionAvail().x;
                            if (!disp.deltaText.empty() && remW > s(34.0f)) {
                                ImGui::SameLine(0.0f, s(4.0f));
                                const float badgeAvailW = std::max(s(12.0f), ImGui::GetContentRegionAvail().x);
                                std::string badge = disp.deltaText;
                                if (ImGui::CalcTextSize(badge.c_str()).x > badgeAvailW - s(4.0f) && disp.hasComparison) {
                                    char cbuf[24];
                                    std::snprintf(cbuf, sizeof(cbuf),
                                                  (std::abs(disp.percentDelta) >= 9.95) ? "%+.0f%%" : "%+.1f%%",
                                                  disp.percentDelta);
                                    badge = cbuf;
                                }
                                ImGui::TextColored(disp.deltaColor, "%s",
                                    fitTextToPixelWidth(badge, badgeAvailW).c_str());
                            }
                        } else {
                            ImGui::TextColored(c.textMuted, "— %s", itm.metricType.c_str());
                            if (itm.hasSupportDiscrepancy && ImGui::GetContentRegionAvail().x > s(36.0f)) {
                                ImGui::SameLine(0.0f, s(4.0f));
                                ImGui::TextColored(c.warnColor, "[DRV]");
                                if (ImGui::IsItemHovered()) {
                                    ImGui::SetTooltip("Hardware vs Driver Support Differ\nHW: %s\nDRV: %s",
                                        itm.hardwareSupportText.c_str(),
                                        itm.driverSupportText.c_str());
                                }
                            }
                        }
                        popMonoFont();

                        // Col 3: Single-Slot Trigger [>]
                        ImGui::TableSetColumnIndex(3);
                        if (rowDisabled) ImGui::BeginDisabled();
                        if (tacticalButton("##RunSlot", ">", ImVec2(s(22.0f), s(18.0f)), false, c)) {
                            runSingleBenchmarkItem(itm);
                        }
                        if (rowDisabled) ImGui::EndDisabled();
                        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled) && itm.isSupported) {
                            ImGui::SetTooltip("Execute single slot: %s", itm.name.c_str());
                        }

                        ImGui::PopID();
                    }
                    ImGui::EndTable();
                }
            }

            ImGui::PopID();
        }

        ImGui::Spacing();
        ImGui::PopID();
    }

    if (useInnerScroll) {
        ImGui::EndChild();
    }
}

} // namespace gpubench::gui
