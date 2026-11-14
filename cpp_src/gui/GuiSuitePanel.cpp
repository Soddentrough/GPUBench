#include "GuiApp.h"
#include "GuiUiHelpers.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <set>
#include <unordered_map>

namespace gpubench::gui {

namespace {

std::string fallbackDescriptionForBenchmark(const BenchmarkConfigMetadata& meta) {
    if (!meta.description.empty()) {
        return meta.description;
    }
    const std::string& bn = meta.benchmarkName;
    if (bn == "FP64") return "IEEE-754 double-precision (64-bit) FMA throughput across SIMD vector lanes.";
    if (bn == "FP32") return "Single-precision (32-bit) FMA throughput (including VOPD dual-issue where supported).";
    if (bn == "FP16") return (meta.configIndex == 0)
        ? "Packed 16-bit half-precision vector FMA throughput across shader ALU lanes."
        : "Hardware Cooperative Matrix / WMMA / MFMA tensor tile throughput in FP16.";
    if (bn == "BF16") return (meta.configIndex == 0)
        ? "BFloat16 vector dot/FMA throughput (with hardware ISA emulation detection)."
        : "Hardware Cooperative Matrix / WMMA / MFMA tensor tile throughput in BF16.";
    if (bn == "FP8") return "8-bit E4M3 floating-point hardware matrix core throughput.";
    if (bn == "FP6") return "OCP MXFP6 (E3M2) microscaling hardware matrix throughput.";
    if (bn == "FP4") return "OCP MXFP4 (E2M1) microscaling hardware matrix throughput.";
    if (bn == "INT8") return (meta.configIndex == 0)
        ? "Packed 8-bit integer dot-product (SDOT/UDOT) throughput across vector ALUs."
        : "8-bit integer Cooperative Matrix / WMMA / MFMA tensor throughput.";
    if (bn == "INT4") return "Sub-byte 4-bit integer Cooperative Matrix / WMMA / MFMA throughput.";
    if (bn == "DualIssue" || bn.find("Dual-Issue") != std::string::npos) return "Instruction-level parallelism (ILP) & FP32/INT32 co-issue scaling from serialized chains to peak wave occupancy.";
    if (bn == "LDS Bank Conflicts") return "Shared memory / Local Data Share (LDS) bank serialization sweep from conflict-free stride-1 to 32-way bank conflicts.";
    if (bn == "Pixel Fill Rate") return "Fixed-function rasterizer & Render Output Unit (ROP) pixel/blend fill throughput via Vulkan dynamic rendering.";
    if (bn == "System Memory Bandwidth") return "Host CPU DRAM streaming read/write/copy throughput using runtime-dispatched AVX-512, AVX2, or ARM64 NEON non-temporal stores.";
    if (bn == "System Memory Latency") return "Host CPU DRAM pointer-chasing latency using 2 MB Large Pages / Transparent Huge Pages to isolate DRAM access from TLB walks.";
    if (bn.find("Latency") != std::string::npos) return "Dependent pointer-chasing latency isolating GPU cache hierarchy and VRAM controller round-trip time.";
    if (bn.find("Cache Bandwidth") != std::string::npos) return "Streaming coalesced bandwidth targeting a dedicated GPU cache hierarchy residency tier.";
    if (bn.rfind("RayScheduling", 0) == 0) {
        return "Hardware ray-tracing execution paradigm & lighting stage (" + meta.configName + ") evaluated across complex 3D PBR geometry.";
    }
    return meta.displayName + " (" + meta.subcategory + ") — hardware throughput & efficiency microbenchmark.";
}

std::string subgroupTitleForCatalogEntry(const BenchmarkConfigMetadata& meta, std::string& outCategory) {
    if (!meta.isDeviceDependent || meta.component == "System") {
        outCategory = "Host System";
        if (meta.benchmarkName.find("Bandwidth") != std::string::npos) {
            return "Host DRAM Streaming Bandwidth (AVX-512 / AVX2 / NEON)";
        }
        return "Host DRAM Pointer-Chasing Latency (2 MB Large Pages)";
    }

    if (meta.component == "Scheduling" || meta.benchmarkName == "In-Shader Indirect Synthesis") {
        outCategory = "Scheduling";
        if (meta.benchmarkName.find("Work Graphs") != std::string::npos) {
            return "GPU Work Graphs (VK_AMDX_shader_enqueue)";
        }
        if (meta.benchmarkName.find("Async") != std::string::npos) {
            return "Multi-Queue Async Compute (ACE) & Copy (SDMA) Overlap";
        }
        return "In-Shader Indirect Command Synthesis";
    }

    if (meta.component == "Graphics" || meta.component == "Raster") {
        outCategory = "Graphics";
        return "ROP Pixel & Blend Fill Rate";
    }

    if (meta.component == "Memory") {
        outCategory = "Memory";
        if (meta.benchmarkName.find("Interconnect") != std::string::npos) {
            return "PCIe DMA, ReBAR & Multi-GPU P2P Interconnect";
        }
        if (meta.benchmarkName == "Device Memory Bandwidth") {
            return "Device VRAM Streaming Bandwidth";
        }
        if (meta.benchmarkName.find("Cache Bandwidth") != std::string::npos) {
            return "Cache Hierarchy Bandwidth (L0 -> L3)";
        }
        return "Pointer-Chasing Latency & Working-Set Curve";
    }

    if (meta.component == "Ray Tracing") {
        outCategory = "Ray Tracing";
        if (meta.benchmarkName.rfind("RayScheduling", 0) == 0) {
            return meta.subcategory.empty() ? "Scene Ray Scheduling" : meta.subcategory;
        }
        if (meta.benchmarkName.find("Opacity Micromap") != std::string::npos) {
            return "Opacity Micromap Alpha-Cutout Traversal (OMM)";
        }
        return "Hardware BVH Build, Traversal & Divergence Stress";
    }

    outCategory = "Compute";
    if (meta.benchmarkName.find("Sparse") != std::string::npos ||
        meta.benchmarkName.find("Cooperative Vector") != std::string::npos) {
        return "Structured Sparsity & Neural Shading (2026)";
    }
    if (meta.benchmarkName == "DualIssue" || meta.benchmarkName.find("Dual-Issue") != std::string::npos) {
        return "Instruction Co-Issue & ILP (DualIssue)";
    }
    if (meta.benchmarkName == "LDS Bank Conflicts") {
        return "Shared Memory / LDS Bank Conflicts";
    }
    return "Compute Precision (Vec & Matrix)";
}

} // namespace

void GuiApp::initializeBenchmarkCategories() {
    m_categories.clear();

    const std::vector<std::pair<std::string, std::string>> categoryDefs = {
        {"Compute",      "Floating-point, quantized matrix, 2:4 sparse, neural Cooperative Vector, ILP, and LDS bank throughput."},
        {"Memory",       "VRAM streaming bandwidth, L0-L3 cache hierarchy, PCIe/ReBAR/P2P interconnect, and pointer-chasing latency."},
        {"Ray Tracing",  "Hardware BVH build/traversal, Opacity Micromaps (OMM), SER/DGC scheduling, and 4-scene PBR path tracing."},
        {"Scheduling",   "GPU Work Graphs (producer-consumer), Async Compute/Copy engine overlap, and GPU indirect command synthesis."},
        {"Graphics",     "Vulkan 1.4 dynamic rendering rasterization and Render Output Unit (ROP) pixel/blend fill throughput."},
        {"Host System",  "CPU host DRAM streaming bandwidth (AVX-512 / AVX2 / NEON) and 2 MB Large Page pointer-chasing latency."}
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
                "Pointer-Chasing Latency & Working-Set Curve",
                "Memory",
                "Cache Latency Curve",
                "Dependent pointer-chasing latency across GPU cache tiers and 16 KB -> 256 MB working sets.");
            BenchmarkItem item(
                "Cache Latency Curve",
                meta.subcategory,
                "Cache Latency Curve (16 KB -> 256 MB Sweep)",
                "Memory",
                meta.metric,
                fallbackDescriptionForBenchmark(meta),
                true,
                -1);
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
        std::string sgTitle = subgroupTitleForCatalogEntry(meta, targetCatName);
        BenchmarkCategory* cat = findCategory(targetCatName);
        if (!cat) continue;

        std::string itemEngineId = isRaySched ? "RayScheduling" : meta.benchmarkName;
        std::string sgEngineId = isRaySched ? "RayScheduling" : meta.benchmarkName;
        std::string itemComponent = (!meta.isDeviceDependent || meta.component == "System")
            ? "System"
            : meta.component;

        auto& sg = findOrAddSubgroup(*cat, sgTitle, itemComponent, sgEngineId, "");

        std::string friendlyName;
        if (isRaySched) {
            friendlyName = meta.configName.empty() ? meta.displayName : meta.configName;
        } else if (sgTitle.rfind("Compute Precision", 0) == 0 ||
                   sgTitle.rfind("Structured Sparsity", 0) == 0 ||
                   sgTitle.rfind("Hardware BVH", 0) == 0 ||
                   sgTitle.rfind("Cache Hierarchy Bandwidth", 0) == 0 ||
                   sgTitle.rfind("Pointer-Chasing Latency", 0) == 0) {
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
        sg.items.push_back(std::move(item));
    }
}

void GuiApp::hydrateMetadataFromCatalog() {
    // Categories are already populated directly from GetBenchmarkMetadataCatalogAPI();
    // attach rich procedural schematics and 16:9 thumbnail descriptors.
    applyVisualizationMetadata();
}

bool GuiApp::matchesItem(const ResultData& r, const BenchmarkItem& itm, uint32_t activeDev) const {
    const bool rIsHost = (r.component == "System" || r.backendName == "CPU" ||
                          r.backendName == "System" || r.deviceIndex == UINT32_MAX ||
                          r.deviceName == "Host CPU");
    if (itm.category == "System") {
        if (!rIsHost) return false;
    } else {
        if (rIsHost || r.deviceIndex != activeDev) return false;
    }

    if (itm.id == "Cache Latency Curve") {
        return r.benchmarkName.rfind("Cache Latency Curve", 0) == 0;
    }

    if (itm.id == "RayScheduling") {
        if (r.benchmarkName.rfind("RayScheduling", 0) != 0) return false;
        return (itm.configIndex >= 0) && (static_cast<int>(r.configIndex) == itm.configIndex);
    }

    // Normalized multi-precision Vector / Matrix records from ResultImporter
    if (r.benchmarkName == "Vector" || r.benchmarkName == "Matrix") {
        const bool subMatches = (r.subcategory == itm.id) ||
                                (r.subcategory.find("(" + itm.id + ")") != std::string::npos);
        if (!subMatches) return false;
        if (itm.name.find("(Matrix)") != std::string::npos) {
            return r.benchmarkName == "Matrix";
        }
        if (itm.name.find("(Vector)") != std::string::npos) {
            return r.benchmarkName == "Vector";
        }
        return (itm.configIndex >= 0) && (static_cast<int>(r.configIndex) == itm.configIndex);
    }

    // Normalized Device Memory Bandwidth records from ResultImporter
    if (itm.id == "Device Memory Bandwidth") {
        if (r.benchmarkName == "GPU VRAM Bandwidth" && itm.configIndex == 3) {
            return true;
        }
        if ((r.benchmarkName.rfind("Device Memory Bandwidth", 0) == 0 ||
             r.benchmarkName.rfind("VRAM Bandwidth", 0) == 0) &&
            static_cast<int>(r.configIndex) == itm.configIndex) {
            return true;
        }
    }

    // Normalized RayIntersect & Pixel Fill Rate records from ResultImporter
    if (itm.id == "RayIntersect" && static_cast<int>(r.configIndex) == itm.configIndex) {
        if ((itm.configIndex == 0 && r.benchmarkName == "Ray-Triangle Intersection") ||
            (itm.configIndex == 1 && r.benchmarkName == "Ray-Box Intersection") ||
            r.benchmarkName.rfind("RayTracing (", 0) == 0) {
            return true;
        }
    }
    if (itm.id == "Pixel Fill Rate" && r.subcategory == "Pixel Fill Rate" &&
        static_cast<int>(r.configIndex) == itm.configIndex) {
        return true;
    }

    if (itm.configIndex >= 0 && static_cast<int>(r.configIndex) == itm.configIndex) {
        if (r.benchmarkName == itm.id || r.benchmarkName.rfind(itm.id + " (", 0) == 0 ||
            r.benchmarkName.rfind(itm.id + " -", 0) == 0) {
            return true;
        }
    }

    return r.benchmarkName == itm.id;
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

    // 3. Standard single-config benchmarks (Compute, Memory, Scheduling, OMM, BVH, Graphics, Host System)
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

    size_t totalArmed = 0;
    size_t totalSupported = 0;
    for (const auto& cat : m_categories) {
        for (const auto& sg : cat.subgroups) {
            for (const auto& itm : sg.items) {
                if (itm.isSupported) {
                    totalSupported++;
                    if (itm.selected) totalArmed++;
                }
            }
        }
    }

    char rightTag[48];
    std::snprintf(rightTag, sizeof(rightTag), "SELECTED %zu/%zu", totalArmed, totalSupported);
    drawEtchedHeader("ROM-01", "WORKLOAD SLOT MATRIX", c, rightTag);

    // Preset & Bulk Selection Bar
    if (isRunning) ImGui::BeginDisabled();
    const float btnH = s(22.0f);
    if (tacticalButton("##BtnSelectAll", "ALL", ImVec2(s(46.0f), btnH), false, c)) {
        for (auto& cat : m_categories) {
            bool catSel = false;
            for (auto& sg : cat.subgroups) {
                bool sgSel = false;
                for (auto& itm : sg.items) {
                    itm.selected = itm.isSupported;
                    if (itm.selected) sgSel = true;
                }
                sg.allSelected = sgSel;
                if (sgSel) catSel = true;
            }
            cat.allSelected = catSel;
        }
    }
    ImGui::SameLine(0.0f, s(4.0f));
    if (tacticalButton("##BtnDeselectAll", "NONE", ImVec2(s(50.0f), btnH), false, c)) {
        for (auto& cat : m_categories) {
            cat.allSelected = false;
            for (auto& sg : cat.subgroups) {
                sg.allSelected = false;
                for (auto& itm : sg.items) itm.selected = false;
            }
        }
    }
    ImGui::SameLine(0.0f, s(4.0f));
    if (tacticalButton("##BtnRooflinePreset", "CORE SUITE", ImVec2(s(88.0f), btnH), false, c)) {
        for (auto& cat : m_categories) {
            bool catSel = false;
            for (auto& sg : cat.subgroups) {
                bool sgSel = false;
                for (auto& itm : sg.items) {
                    bool pick = itm.isSupported && (
                        itm.id == "FP32" || itm.id == "FP16" || itm.id == "BF16" ||
                        itm.id == "FP8" || itm.id == "INT8" ||
                        (itm.id == "Device Memory Bandwidth" && itm.configIndex == 0) ||
                        itm.id == "L0 Cache Bandwidth" || itm.id == "L1 Cache Bandwidth" ||
                        itm.id == "L2 Cache Bandwidth" || itm.id == "L3 Cache Bandwidth" ||
                        (itm.id == "PCIe & Host-Device Interconnect" && itm.configIndex <= 1) ||
                        (itm.id == "RayScheduling" && (itm.configIndex == 9 || itm.configIndex == 10 || itm.configIndex == 11)));
                    itm.selected = pick;
                    if (pick) sgSel = true;
                }
                sg.allSelected = sgSel;
                if (sgSel) catSel = true;
            }
            cat.allSelected = catSel;
        }
    }
    if (isRunning) ImGui::EndDisabled();

    ImGui::SameLine(0.0f, s(6.0f));
    if (tacticalButton("##BtnHideUnsup", m_hideUnsupported ? "UNSUP: HIDDEN" : "UNSUP: SHOWN",
                       ImVec2(0.0f, btnH), m_hideUnsupported, c)) {
        m_hideUnsupported = !m_hideUnsupported;
    }

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

    // Category Selector Matrix
    ImGui::Spacing();
    {
        bool allActive = (m_suiteCategoryFilter == 0);
        if (tacticalButton("##CatAll", "ALL", ImVec2(0.0f, s(21.0f)), allActive, c)) {
            m_suiteCategoryFilter = 0;
        }
        for (size_t ci = 0; ci < m_categories.size(); ++ci) {
            std::string shortName = m_categories[ci].name;
            if (shortName == "Ray Tracing") shortName = "RT";
            else if (shortName == "Scheduling") shortName = "SCHED";
            else if (shortName == "Graphics") shortName = "GFX";
            else if (shortName == "Host System") shortName = "HOST";
            else if (shortName == "Compute") shortName = "COMP";
            else if (shortName == "Memory") shortName = "MEM";

            std::string pillLabel = shortName + " [" + std::to_string(m_categories[ci].totalItemCount()) + "]";
            float pillW = ImGui::CalcTextSize(pillLabel.c_str()).x + s(16.0f);
            sameLineIfFits(pillW, s(4.0f));
            bool active = (m_suiteCategoryFilter == static_cast<int>(ci + 1));
            std::string btnId = "##CatBtn_" + std::to_string(ci);
            if (tacticalButton(btnId.c_str(), pillLabel.c_str(), ImVec2(0.0f, s(21.0f)), active, c)) {
                m_suiteCategoryFilter = static_cast<int>(ci + 1);
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
        if (m_suiteCategoryFilter != 0 && m_suiteCategoryFilter != static_cast<int>(ci + 1)) {
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

    ImGui::BeginChild("SuiteScrollRegion", ImVec2(0, 0), ImGuiChildFlags_None);

    size_t cursorRow = 0;
    for (size_t ci = 0; ci < m_categories.size(); ++ci) {
        if (m_suiteCategoryFilter != 0 && m_suiteCategoryFilter != static_cast<int>(ci + 1)) {
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

        // Category Etched Band
        pushBoldFont();
        ImGui::TextColored(c.phosphorHi, "// %s", cat.name.c_str());
        popBoldFont();
        ImGui::SameLine();
        ImGui::TextColored(c.textMuted, "%s", fitTextToPixelWidth(cat.description, ImGui::GetContentRegionAvail().x).c_str());

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
            for (const auto& itm : sg.items) {
                if (itm.selected && itm.isSupported) selectedCount++;
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
                bool hdrHovered = ImGui::IsItemHovered();
                if (hdrHovered) {
                    dl->AddRectFilled(hdrP0, ImVec2(hdrP0.x + toggleW, hdrP1.y), toU32(c.bgHover, 0.55f), 0.0f);
                }

                char hdrBuf[220];
                std::snprintf(hdrBuf, sizeof(hdrBuf), "%s %s [%zu/%zu]",
                              sg.collapsed ? "[+]" : "[-]", sg.name.c_str(), selectedCount, sg.items.size());
                std::string fittedHdr = fitTextToPixelWidth(hdrBuf, toggleW - s(10.0f));
                dl->AddText(ImVec2(hdrP0.x + s(6.0f), hdrP0.y + (hdrH - ImGui::GetTextLineHeight()) * 0.5f),
                            toU32(hdrHovered ? c.phosphorHi : c.textPrimary), fittedHdr.c_str());

                if (!isRunning) {
                    ImGui::SameLine(0.0f, s(4.0f));
                    if (tacticalButton("##SGAll", "+ALL", ImVec2(smallBtnW, hdrH), false, c)) {
                        for (auto& itm : sg.items) {
                            if (itm.isSupported) itm.selected = true;
                        }
                        sg.allSelected = true;
                    }
                    ImGui::SameLine(0.0f, s(4.0f));
                    if (tacticalButton("##SGNone", "-CLR", ImVec2(smallBtnW, hdrH), false, c)) {
                        for (auto& itm : sg.items) itm.selected = false;
                        sg.allSelected = false;
                    }
                }
            }

            if (!sg.collapsed) {
                ImGuiTableFlags tblFlags = ImGuiTableFlags_BordersInnerH |
                                           ImGuiTableFlags_RowBg |
                                           ImGuiTableFlags_SizingStretchProp;
                if (ImGui::BeginTable("SubgroupItemsTable", 4, tblFlags)) {
                    ImGui::TableSetupColumn("Arm",   ImGuiTableColumnFlags_WidthFixed,   s(22.0f));
                    ImGui::TableSetupColumn("Slot",  ImGuiTableColumnFlags_WidthStretch, 0.58f);
                    ImGui::TableSetupColumn("Score", ImGuiTableColumnFlags_WidthStretch, 0.42f);
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
                            bool anySg = false;
                            for (const auto& check : sg.items) {
                                if (check.selected) { anySg = true; break; }
                            }
                            sg.allSelected = anySg;
                        }
                        if (rowDisabled) ImGui::EndDisabled();

                        // Col 1: Clickable Slot Label -> F1 ISA & Die Inspector
                        ImGui::TableSetColumnIndex(1);
                        {
                            float colW = std::max(s(40.0f), ImGui::GetContentRegionAvail().x);
                            std::string fittedName = fitTextToPixelWidth(itm.name, colW - s(4.0f));
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
                            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled) && !itm.supportReason.empty()) {
                                ImGui::SetTooltip("%s: %s",
                                    itm.limitationCategory.empty() ? "Unsupported" : itm.limitationCategory.c_str(),
                                    itm.supportReason.c_str());
                            }
                        } else if (disp.hasResult) {
                            ImGui::TextColored(c.textPrimary, "%s", disp.scoreText.c_str());
                            if (!disp.deltaText.empty() && ImGui::GetContentRegionAvail().x > s(55.0f)) {
                                ImGui::SameLine(0.0f, s(4.0f));
                                ImGui::TextColored(disp.deltaColor, "%s",
                                    fitTextToPixelWidth(disp.deltaText, ImGui::GetContentRegionAvail().x).c_str());
                            }
                        } else {
                            ImGui::TextColored(c.textMuted, "— %s", itm.metricType.c_str());
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

    ImGui::EndChild();
}

} // namespace gpubench::gui
