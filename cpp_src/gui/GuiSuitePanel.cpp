#include "GuiApp.h"
#include "GuiUiHelpers.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <set>

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
    if (bn == "DualIssue") return "Instruction-level parallelism (ILP) & FP32/INT32 co-issue scaling from serialized chains to peak wave occupancy.";
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
    if (meta.benchmarkName == "DualIssue") {
        return "Instruction Co-Issue & ILP (DualIssue)";
    }
    if (meta.benchmarkName == "LDS Bank Conflicts") {
        return "Shared Memory / LDS Bank Conflicts";
    }
    return "Compute Precision (Vector & Matrix Cores)";
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
    if (itm.category != "System" && r.deviceIndex != activeDev) {
        return false;
    }

    if (itm.id == "Cache Latency Curve") {
        return r.benchmarkName.rfind("Cache Latency Curve", 0) == 0;
    }

    if (itm.id == "RayScheduling") {
        if (r.benchmarkName.rfind("RayScheduling", 0) != 0) return false;
        return (itm.configIndex >= 0) && (static_cast<int>(r.configIndex) == itm.configIndex);
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
            for (const auto& r : m_allResults) {
                if (r.deviceIndex == targetDeviceIndex &&
                    r.benchmarkName.rfind("RayScheduling", 0) == 0 &&
                    r.baselineConfigIndex == static_cast<int32_t>(item.configIndex)) {
                    info.isBaseline = true;
                    info.deltaText = "Baseline";
                    info.deltaColor = ImVec4(0.55f, 0.72f, 0.90f, 1.0f);
                    break;
                }
            }
        }

        if (baseRes) {
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
                    ? ImVec4(0.20f, 0.92f, 0.52f, 1.0f)
                    : ((info.speedupRatio <= 0.99) ? ImVec4(0.95f, 0.42f, 0.36f, 1.0f)
                                                   : ImVec4(0.75f, 0.80f, 0.88f, 1.0f));
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
        for (const auto& r : m_allResults) {
            if (r.deviceIndex == matched->deviceIndex &&
                r.subcategory == matched->subcategory &&
                r.baselineConfigIndex == static_cast<int32_t>(matched->configIndex)) {
                info.isBaseline = true;
                info.deltaText = "Baseline";
                info.deltaColor = ImVec4(0.55f, 0.72f, 0.90f, 1.0f);
                break;
            }
        }
    }

    if (baseRes) {
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
                ? ImVec4(0.20f, 0.92f, 0.52f, 1.0f)
                : ((info.speedupRatio <= 0.99) ? ImVec4(0.95f, 0.42f, 0.36f, 1.0f)
                                               : ImVec4(0.75f, 0.80f, 0.88f, 1.0f));
        }
    }

    return info;
}

void GuiApp::renderBenchmarkSuitePanel() {
    ImGui::Spacing();
    bool isRunning = (m_execState == ExecutionState::Running);

    // Preset & Bulk Selection Bar
    if (isRunning) ImGui::BeginDisabled();
    if (ImGui::Button("Select All Supported", ImVec2(s(155.0f), s(26.0f)))) {
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
    ImGui::SameLine();
    if (ImGui::Button("Deselect All", ImVec2(s(105.0f), s(26.0f)))) {
        for (auto& cat : m_categories) {
            cat.allSelected = false;
            for (auto& sg : cat.subgroups) {
                sg.allSelected = false;
                for (auto& itm : sg.items) itm.selected = false;
            }
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Quick Roofline Preset", ImVec2(s(160.0f), s(26.0f)))) {
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
                        (itm.id == "RayScheduling" && (itm.configIndex == 9 || itm.configIndex == 10)));
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

    ImGui::SameLine();
    ImGui::Checkbox("Hide Unsupported", &m_hideUnsupported);

    // Category Filter Pill Row
    ImGui::Spacing();
    {
        bool allActive = (m_suiteCategoryFilter == 0);
        if (allActive) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.05f, 0.55f, 0.78f, 1.0f));
        if (ImGui::Button("All Categories", ImVec2(0.0f, s(25.0f)))) m_suiteCategoryFilter = 0;
        if (allActive) ImGui::PopStyleColor();

        for (size_t ci = 0; ci < m_categories.size(); ++ci) {
            std::string pillLabel = m_categories[ci].name + " (" +
                                    std::to_string(m_categories[ci].totalItemCount()) + ")";
            float pillW = ImGui::CalcTextSize(pillLabel.c_str()).x + s(20.0f);
            sameLineIfFits(pillW, s(6.0f));
            bool active = (m_suiteCategoryFilter == static_cast<int>(ci + 1));
            if (active) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.05f, 0.55f, 0.78f, 1.0f));
            if (ImGui::Button(pillLabel.c_str(), ImVec2(0.0f, s(25.0f)))) {
                m_suiteCategoryFilter = static_cast<int>(ci + 1);
            }
            if (active) ImGui::PopStyleColor();
        }
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    uint32_t activeGpu = m_telemetryGpuIndex;

    ImGui::BeginChild("SuiteScrollRegion", ImVec2(0, 0), ImGuiChildFlags_None);

    for (size_t ci = 0; ci < m_categories.size(); ++ci) {
        if (m_suiteCategoryFilter != 0 && m_suiteCategoryFilter != static_cast<int>(ci + 1)) {
            continue;
        }

        auto& cat = m_categories[ci];
        ImGui::PushID(static_cast<int>(ci));

        // Category Header Banner
        pushBoldFont();
        ImGui::TextColored(ImVec4(0.06f, 0.80f, 0.98f, 1.0f), "%s", cat.name.c_str());
        popBoldFont();
        ImGui::SameLine();
        ImGui::TextDisabled("— %s", cat.description.c_str());

        // Scene Selector Bar when inside Ray Tracing category
        if (cat.name == "Ray Tracing") {
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.70f, 0.80f, 0.92f, 1.0f), "RayScheduling Active Scene:");
            const char* sceneIds[] = {"all", "indoor", "showroom", "forest", "landscape"};
            const char* scenePills[] = {
                "All 4 Scenes",
                "Indoor Atrium (Sponza)",
                "Showroom Studio",
                "Open-World Forest",
                "Outdoor Landscape"
            };
            for (int si = 0; si < 5; ++si) {
                float pw = ImGui::CalcTextSize(scenePills[si]).x + s(18.0f);
                sameLineIfFits(pw, s(6.0f));
                bool activeScene = (m_scene == sceneIds[si]);
                if (activeScene) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.06f, 0.52f, 0.72f, 1.0f));
                if (ImGui::SmallButton(scenePills[si])) {
                    setScene(sceneIds[si]);
                }
                if (activeScene) ImGui::PopStyleColor();
            }
            ImGui::Spacing();
        }

        for (size_t sgi = 0; sgi < cat.subgroups.size(); ++sgi) {
            auto& sg = cat.subgroups[sgi];

            size_t visibleCount = 0;
            size_t selectedCount = 0;
            for (const auto& itm : sg.items) {
                if (!m_hideUnsupported || itm.isSupported) visibleCount++;
                if (itm.selected && itm.isSupported) selectedCount++;
            }
            if (visibleCount == 0) continue;

            ImGui::PushID(static_cast<int>(sgi));
            ImGui::Spacing();

            // Subgroup Header Card
            ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0.11f, 0.14f, 0.20f, 0.95f));
            std::string sgHeader = sg.name + "  [" + std::to_string(selectedCount) + "/" +
                                   std::to_string(sg.items.size()) + "]###SGHeader";
            bool open = ImGui::CollapsingHeader(sgHeader.c_str(), ImGuiTreeNodeFlags_DefaultOpen);
            ImGui::PopStyleColor();

            if (open) {
                if (!isRunning) {
                    if (ImGui::SmallButton("All")) {
                        for (auto& itm : sg.items) {
                            if (itm.isSupported) itm.selected = true;
                        }
                        sg.allSelected = true;
                    }
                    ImGui::SameLine();
                    if (ImGui::SmallButton("None")) {
                        for (auto& itm : sg.items) itm.selected = false;
                        sg.allSelected = false;
                    }
                }

                ImGuiTableFlags tblFlags = ImGuiTableFlags_BordersInnerH |
                                           ImGuiTableFlags_RowBg |
                                           ImGuiTableFlags_SizingStretchProp;
                if (ImGui::BeginTable("SubgroupItemsTable", 4, tblFlags)) {
                    ImGui::TableSetupColumn("Workload", ImGuiTableColumnFlags_WidthStretch, 0.50f);
                    ImGui::TableSetupColumn("Metric", ImGuiTableColumnFlags_WidthStretch, 0.13f);
                    ImGui::TableSetupColumn("Latest Result", ImGuiTableColumnFlags_WidthStretch, 0.21f);
                    ImGui::TableSetupColumn("vs Baseline", ImGuiTableColumnFlags_WidthStretch, 0.16f);

                    for (size_t ii = 0; ii < sg.items.size(); ++ii) {
                        auto& itm = sg.items[ii];
                        if (m_hideUnsupported && !itm.isSupported) continue;

                        ImGui::PushID(static_cast<int>(ii));
                        ImGui::TableNextRow();

                        // Column 0: Checkbox + Workload Name
                        ImGui::TableSetColumnIndex(0);
                        bool rowDisabled = isRunning || !itm.isSupported;
                        if (rowDisabled) ImGui::BeginDisabled();
                        if (ImGui::Checkbox(itm.name.c_str(), &itm.selected)) {
                            bool anySg = false;
                            for (const auto& check : sg.items) {
                                if (check.selected) { anySg = true; break; }
                            }
                            sg.allSelected = anySg;
                        }
                        if (rowDisabled) ImGui::EndDisabled();

                        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                            renderBenchmarkTooltip(itm, (m_scene == "all") ? m_suiteActiveScene : m_scene);
                        }

                        if (!itm.isSupported) {
                            ImGui::SameLine();
                            ImGui::TextColored(ImVec4(0.92f, 0.45f, 0.35f, 1.0f), "[UNSUPPORTED]");
                            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled) && !itm.supportReason.empty()) {
                                ImGui::SetTooltip("%s: %s",
                                    itm.limitationCategory.empty() ? "Unsupported" : itm.limitationCategory.c_str(),
                                    itm.supportReason.c_str());
                            }
                        }

                        // Column 1: Metric Unit
                        ImGui::TableSetColumnIndex(1);
                        ImGui::TextDisabled("%s", itm.metricType.c_str());

                        // Column 2 & 3: Live Score + Speedup Delta
                        BenchmarkDisplayInfo disp = getBenchmarkDisplayInfo(
                            itm, activeGpu, (m_scene == "all") ? m_suiteActiveScene : m_scene);

                        ImGui::TableSetColumnIndex(2);
                        if (m_hasCurrentlyRunningResult && matchesItem(m_currentlyRunningResult, itm, activeGpu)) {
                            ImGui::TextColored(ImVec4(0.98f, 0.78f, 0.22f, 1.0f), "Running...");
                        } else if (disp.hasResult) {
                            pushMonoFont();
                            ImGui::TextColored(ImVec4(0.90f, 0.96f, 1.0f, 1.0f), "%s", disp.scoreText.c_str());
                            popMonoFont();
                        } else if (!itm.isSupported) {
                            ImGui::TextDisabled("N/A");
                        } else {
                            ImGui::TextDisabled("—");
                        }

                        ImGui::TableSetColumnIndex(3);
                        if (!disp.deltaText.empty()) {
                            pushMonoFont();
                            ImGui::TextColored(disp.deltaColor, "%s", disp.deltaText.c_str());
                            popMonoFont();
                        } else {
                            ImGui::TextDisabled("—");
                        }

                        ImGui::PopID();
                    }
                    ImGui::EndTable();
                }
            }

            ImGui::PopID();
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        ImGui::PopID();
    }

    ImGui::EndChild();
}

} // namespace gpubench::gui
