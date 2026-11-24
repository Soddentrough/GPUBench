#include "GuiApp.h"
#include "GuiBringUpTheme.h"
#include "GuiUiHelpers.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace gpubench::gui {

std::string GuiApp::currentSceneTag() const {
    if (m_scene == "showroom") return "showroom";
    if (m_scene == "forest") return "forest";
    if (m_scene == "landscape" || m_scene == "outdoor") return "outdoor";
    if (m_scene == "indoor") return "indoor";
    if (m_suiteActiveScene == "landscape") return "outdoor";
    if (!m_suiteActiveScene.empty() && m_suiteActiveScene != "all") return m_suiteActiveScene;
    return "indoor";
}

void GuiApp::applyVisualizationMetadata() {
    const bool isRocm = (resolvedBackendName() == "rocm");

    for (auto& cat : m_categories) {
        for (auto& sg : cat.subgroups) {
            for (auto& item : sg.items) {
                BenchmarkVisualization& viz = item.viz;
                viz = BenchmarkVisualization{};

                // 1. RayScheduling 3D Scene Actual Renders (falling back to shipped scene thumbnails / procedural schematics)
                if (item.id == "RayScheduling") {
                    const std::string& nm = item.name;
                    const bool isDgcCfg = containsCaseInsensitive(nm, "DGC");
                    const bool isLegacyInd = containsCaseInsensitive(nm, "Legacy Indirect");
                    const bool isWavefront = isDgcCfg || isLegacyInd || containsCaseInsensitive(nm, "Wavefront");
                    const bool isMega = containsCaseInsensitive(nm, "Megakernel");

                    if (containsCaseInsensitive(nm, "Queue Compaction")) {
                        viz.diagramId = "queue_compaction";
                        viz.caption = "Wave Ballot Stream Compaction & Compact Active Ray Worklist";
                        viz.linkedViewportMode = 0;
                        viz.linkedViewportSubIndex = 1;
                    } else if (containsCaseInsensitive(nm, "BVH Traversal")) {
                        viz.assetPath = "renders/render_{scene}_stage1_bvh.png";
                        viz.fallbackAssetPath = "assets/thumbnails/thumb_{scene}_stage1_bvh.png";
                        viz.diagramId = "bvh_raw_traversal";
                        viz.caption = "Screen-Space Ray Dispatch Ordering & BVH Traversal Cache Locality";
                        viz.linkedViewportMode = 1;
                        viz.linkedViewportSubIndex = 0;
                    } else if (containsCaseInsensitive(nm, "128 Lights")) {
                        viz.assetPath = isMega
                            ? "renders/render_{scene}_multilight_128_traditional.png"
                            : "renders/render_{scene}_multilight_128_dgc.png";
                        viz.fallbackAssetPath = "assets/thumbnails/thumb_{scene}_multilight_128_dgc.png";
                        viz.diagramId = isWavefront ? "dgc_wavefront_dispatch" : "bvh_rayquery";
                        viz.caption = isWavefront
                            ? "128-Light Direct Evaluation via Spatial Light-Binned DGC Dispatch"
                            : "128-Light Direct Evaluation via Monolithic Compute Megakernel";
                        viz.linkedViewportMode = 0;
                        viz.linkedViewportSubIndex = 4;
                    } else if (containsCaseInsensitive(nm, "Alpha Cutout")) {
                        viz.assetPath = "renders/render_{scene}_alpha_cutout.png";
                        viz.fallbackAssetPath = "assets/thumbnails/thumb_{scene}_alpha_cutout.png";
                        viz.diagramId = "bvh_cutout_layers";
                        viz.caption = "Alpha-Cutout Foliage Traversal Divergence & Mid-Traversal Alpha Testing";
                        viz.linkedViewportMode = 3;
                        viz.linkedViewportSubIndex = 1;
                    } else if (containsCaseInsensitive(nm, "Path Tracing") && containsCaseInsensitive(nm, "16 SPP")) {
                        viz.assetPath = isMega
                            ? "renders/render_{scene}_pathtracing_16spp_traditional.png"
                            : "renders/render_{scene}_pathtracing_16spp_worklist.png";
                        viz.fallbackAssetPath = "assets/thumbnails/thumb_{scene}_pathtracing_16spp.png";
                        viz.diagramId = isWavefront ? "dgc_wavefront_dispatch" : "cone_divergence";
                        viz.caption = isWavefront
                            ? "Unbiased Monte Carlo Path Tracing (16 SPP — Octant-Sorted Wavefront DGC)"
                            : "Unbiased Monte Carlo Path Tracing (16 SPP — Compute Megakernel)";
                        viz.linkedViewportMode = 2;
                        viz.linkedViewportSubIndex = 0;
                    } else if (containsCaseInsensitive(nm, "Path Tracing")) {
                        viz.assetPath = isMega
                            ? "renders/render_{scene}_pathtracing_1spp_traditional.png"
                            : "renders/render_{scene}_pathtracing_1spp_worklist.png";
                        viz.fallbackAssetPath = "assets/thumbnails/thumb_{scene}_pathtracing_1spp.png";
                        viz.diagramId = isWavefront ? "dgc_wavefront_dispatch" : "cone_divergence";
                        viz.caption = isWavefront
                            ? "Multi-Bounce Monte Carlo Path Tracing (Octant-Sorted Wavefront DGC)"
                            : "Multi-Bounce Monte Carlo Path Tracing (Megakernel / RTP+SER)";
                        viz.linkedViewportMode = 2;
                        viz.linkedViewportSubIndex = 1;
                    } else if (containsCaseInsensitive(nm, "Shadow")) {
                        viz.assetPath = "renders/render_{scene}_stage3_shadow.png";
                        viz.fallbackAssetPath = "assets/thumbnails/thumb_{scene}_stage3_shadow.png";
                        viz.diagramId = isWavefront ? "dgc_wavefront_dispatch" : "bvh_rayquery";
                        viz.caption = isWavefront
                            ? "Directional Shadow Ray Visibility Pass (Wavefront DGC)"
                            : "Directional Shadow Ray Visibility Pass (RayQuery Megakernel)";
                        viz.linkedViewportMode = 1;
                        viz.linkedViewportSubIndex = 2;
                    } else if (containsCaseInsensitive(nm, "Ambient Occlusion") || containsCaseInsensitive(nm, "RTAO")) {
                        viz.assetPath = "renders/render_{scene}_stage4_rtao.png";
                        viz.fallbackAssetPath = "assets/thumbnails/thumb_{scene}_stage4_rtao.png";
                        viz.diagramId = "cone_divergence";
                        viz.caption = "Cosine-Weighted Hemisphere RTAO Pass";
                        viz.linkedViewportMode = 1;
                        viz.linkedViewportSubIndex = 3;
                    } else if (containsCaseInsensitive(nm, "Indirect") || containsCaseInsensitive(nm, "GI")) {
                        viz.assetPath = "renders/render_{scene}_stage6_indirect.png";
                        viz.fallbackAssetPath = "assets/thumbnails/thumb_{scene}_stage6_indirect.png";
                        viz.diagramId = isWavefront ? "dgc_wavefront_dispatch" : "cone_divergence";
                        viz.caption = isWavefront
                            ? "Incoherent Diffuse GI Bounce (Octant-Sorted Wavefront DGC)"
                            : "Incoherent Diffuse GI Bounce (Hemisphere Ray Divergence)";
                        viz.linkedViewportMode = 1;
                        viz.linkedViewportSubIndex = 5;
                    } else if (containsCaseInsensitive(nm, "Primary")) {
                        viz.assetPath = "renders/render_{scene}_stage2_primary.png";
                        viz.fallbackAssetPath = "assets/thumbnails/thumb_{scene}_stage2_primary.png";
                        viz.diagramId = isWavefront ? "dgc_wavefront_dispatch" : "bvh_rayquery";
                        viz.caption = isWavefront
                            ? "Primary G-Buffer Ray Generation & Wavefront Material Compaction"
                            : "Coherent Primary G-Buffer Ray Cast & BVH Traversal";
                        viz.linkedViewportMode = 1;
                        viz.linkedViewportSubIndex = 1;
                    } else if (containsCaseInsensitive(nm, "Material")) {
                        viz.assetPath = isMega
                            ? "renders/render_{scene}_traditional_megakernel.png"
                            : "renders/render_{scene}_worklist_dgc.png";
                        viz.fallbackAssetPath = "assets/thumbnails/thumb_material_lineup.png";
                        viz.diagramId = "dgc_wavefront_dispatch";
                        if (isDgcCfg) {
                            viz.caption = "8-Archetype BSDF Material Shading via Single-Call DGC ExecutionSet";
                        } else if (isLegacyInd) {
                            viz.caption = "8-Archetype BSDF Material Shading via 8x Legacy vkCmdDispatchIndirect";
                        } else if (containsCaseInsensitive(nm, "SER")) {
                            viz.caption = "8-Archetype BSDF Material Shading via RTP + Shader Execution Reordering";
                        } else {
                            viz.caption = "8-Archetype BSDF Material Shading via Monolithic Compute Megakernel";
                        }
                        viz.linkedViewportMode = 0;
                        viz.linkedViewportSubIndex = 1;
                    } else if (containsCaseInsensitive(nm, "Full Frame") || containsCaseInsensitive(nm, "Deferred")) {
                        viz.assetPath = isMega
                            ? "renders/render_{scene}_traditional_megakernel.png"
                            : "renders/render_{scene}_stage7_final.png";
                        viz.fallbackAssetPath = "assets/thumbnails/thumb_{scene}_stage7_final.png";
                        viz.diagramId = isWavefront ? "dgc_wavefront_dispatch" : "bvh_rayquery";
                        viz.caption = isDgcCfg
                            ? "End-to-End Hybrid RT Frame (Wavefront Classify + DGC ExecutionSet)"
                            : (isLegacyInd
                                ? "End-to-End Hybrid RT Frame (Wavefront Classify + 8x Legacy Indirect)"
                                : "End-to-End Hybrid Ray-Traced Beauty Frame");
                        viz.linkedViewportMode = 0;
                        viz.linkedViewportSubIndex = 2;
                    } else {
                        viz.assetPath = "renders/render_{scene}_stage5_direct.png";
                        viz.fallbackAssetPath = "assets/thumbnails/thumb_{scene}_stage5_direct.png";
                        viz.diagramId = "bvh_rayquery";
                        viz.caption = "Direct PBR Illumination & Reflection Evaluation";
                        viz.linkedViewportMode = 1;
                        viz.linkedViewportSubIndex = 4;
                    }
                    continue;
                }

                // 2. Hardware BVH & OMM Captures / Schematics
                if (containsCaseInsensitive(item.id, "Opacity Micromap")) {
                    viz.diagramId = "omm_micromap";
                    viz.caption = "VK_EXT_opacity_micromap 2-State & 4-State Subdivision Triangle Grid";
                    continue;
                }
                if (containsCaseInsensitive(item.id, "Acceleration Structure") ||
                    containsCaseInsensitive(item.id, "RayASBuild")) {
                    if (containsCaseInsensitive(item.name, "Indoor Corridor")) {
                        viz.assetPath = "assets/thumbnails/thumb_indoor_stage1_bvh.png";
                        viz.caption = "Top-Level AS (TLAS): 20K Indoor Corridor Instances across 5K BLAS Meshes";
                    } else if (containsCaseInsensitive(item.name, "Dense Jungle")) {
                        viz.assetPath = "assets/thumbnails/thumb_forest_stage1_bvh.png";
                        viz.caption = "Top-Level AS (TLAS): 50K Dense Jungle Instances across 500 BLAS Meshes";
                    } else if (containsCaseInsensitive(item.name, "Massive Open World")) {
                        viz.assetPath = "assets/thumbnails/thumb_outdoor_stage1_bvh.png";
                        viz.caption = "Top-Level AS (TLAS): 200K Open-World Instances across 5K BLAS Meshes";
                    } else {
                        viz.assetPath = "assets/thumbnails/thumb_blas_wireframe.png";
                        viz.caption = "Bottom-Level Acceleration Structure (BLAS) Triangle Mesh BVH";
                    }
                    viz.diagramId = "bvh_rayquery";
                    viz.linkedViewportMode = 3;
                    viz.linkedViewportSubIndex = 0;
                    continue;
                }
                if (containsCaseInsensitive(item.id, "Any-Hit") || containsCaseInsensitive(item.id, "RayAnyHit")) {
                    viz.assetPath = "assets/thumbnails/thumb_alpha_layers.png";
                    viz.diagramId = "bvh_cutout_layers";
                    viz.caption = "16-Layer Stacked Alpha-Cutout Planes (Opaque Fast-Path vs Any-Hit Shader)";
                    viz.linkedViewportMode = 3;
                    viz.linkedViewportSubIndex = 1;
                    continue;
                }
                if (containsCaseInsensitive(item.id, "RayProcedural")) {
                    viz.diagramId = "bvh_procedural";
                    viz.caption = "Vulkan Ray Tracing Pipeline Procedural AABB Intersection Shader (.rint)";
                    continue;
                }
                if (containsCaseInsensitive(item.id, "RayIntersect")) {
                    viz.diagramId = "bvh_rayquery";
                    viz.caption = "Hardware RayQuery Inline BVH Box & Triangle Intersection";
                    continue;
                }
                if (containsCaseInsensitive(item.id, "Raw") && containsCaseInsensitive(item.id, "Traversal")) {
                    viz.diagramId = "bvh_raw_traversal";
                    viz.caption = "Raw Fixed-Function BVH Box/Triangle Intersection across Stratified Layers";
                    viz.linkedViewportMode = 1;
                    viz.linkedViewportSubIndex = 0;
                    continue;
                }
                if (containsCaseInsensitive(item.id, "Divergence")) {
                    viz.diagramId = "cone_divergence";
                    viz.caption = "Secondary Bounce Cone Spread (90 deg -> 0 deg) & Wavefront Ray Divergence";
                    continue;
                }
                if (containsCaseInsensitive(item.id, "Payload")) {
                    viz.diagramId = "payload_registers";
                    viz.caption = "Ray Payload VGPR Footprint (16B / 128B / 256B) vs Active Wave Occupancy";
                    continue;
                }

                // 3. Host System DRAM (must precede GPU Cache / Memory Bandwidth matching!)
                if (item.category == "System" ||
                    containsCaseInsensitive(item.id, "System Memory")) {
                    viz.diagramId = "host_sysmem_dram";
                    viz.caption = containsCaseInsensitive(item.id, "Latency")
                        ? "Host CPU Pointer-Chasing Chain across L1/L2/L3 Cache & System DRAM (THP / Large Pages)"
                        : "Host CPU SIMD Cores (AVX-512 / AVX2 / NEON) <-> L1/L2/L3 <-> System DDR/LPDDR DRAM";
                    continue;
                }

                // 4. Raster / ROP
                if (item.id == "Pixel Fill Rate" || item.category == "Raster") {
                    viz.diagramId = "raster_rop_fill";
                    viz.caption = "Fixed-Function Rasterizer -> Fragment Shader -> ROP / Alpha Blend -> Framebuffer";
                    continue;
                }

                // 5. Compute, Interconnect & Scheduling Schematics
                if (containsCaseInsensitive(item.id, "In-Shader Indirect")) {
                    if (isRocm && item.configIndex >= 2 && item.configIndex <= 4) {
                        viz.diagramId = "rocm_persistent_queue";
                        viz.caption = "ROCm/HIP Persistent Work-Queue Compaction (Classify + Fixed-Grid Consumer)";
                    } else {
                        viz.diagramId = "indirect_dispatch";
                        viz.caption = (item.configIndex == 6)
                            ? "GPU Sparse-Bucket Compaction -> 1x vkCmdExecuteGeneratedCommandsEXT (DGC ExecutionSet)"
                            : "GPU-Side Indirect Dispatch Argument Synthesis (vkCmdDispatchIndirect) vs CPU Direct";
                    }
                    continue;
                }
                if (containsCaseInsensitive(item.id, "Sparse")) {
                    viz.diagramId = "sparse_2_4";
                    viz.caption = "Hardware 2:4 Structured Sparsity (50% Non-Zero Weight Compression + 2-Bit Metadata)";
                    continue;
                }
                if (containsCaseInsensitive(item.id, "Cooperative Vector")) {
                    viz.diagramId = "coop_vector_mlp";
                    viz.caption = "VK_NV_cooperative_vector In-Shader Neural BSDF / Radiance MLP Layer";
                    continue;
                }
                if (containsCaseInsensitive(item.id, "Interconnect")) {
                    if (item.configIndex == 4 || containsCaseInsensitive(item.name, "Unified")) {
                        viz.diagramId = "apu_unified_memory";
                        viz.caption = "APU / Unified Memory Host CPU <-> Coherent Shared Fabric <-> GPU Latency";
                    } else {
                        viz.diagramId = "pcie_interconnect";
                        viz.caption = "Host Pinned DMA <-> PCIe Root Complex <-> ReBAR / Multi-GPU P2P Crossbar";
                    }
                    continue;
                }
#ifdef GPUBENCH_ENABLE_EXPERIMENTAL_WORKGRAPHS
                if (containsCaseInsensitive(item.id, "Work Graphs")) {
                    viz.diagramId = "workgraphs_dag";
                    viz.caption = "GPU-Autonomous Producer -> Consumer Node Enqueue (Zero Host Roundtrip)";
                    continue;
                }
#endif
                if (containsCaseInsensitive(item.id, "Async Compute")) {
                    if (item.configIndex >= 2) {
                        viz.diagramId = "async_compute_dma";
                        viz.caption = isRocm
                            ? "Concurrent HIP Compute Stream + Asynchronous DMA Copy Stream"
                            : "Concurrent Async Compute Queue (ACE) + Dedicated Transfer Engine (SDMA)";
                    } else {
                        viz.diagramId = "async_ace_overlap";
                        viz.caption = "Concurrent Vulkan Graphics Queue (3D) + Async Compute Engine (ACE)";
                    }
                    continue;
                }
                if (containsCaseInsensitive(item.name, "Matrix") &&
                    (item.id == "FP16" || item.id == "BF16" || item.id == "FP8" ||
                     item.id == "FP6" || item.id == "FP4" || item.id == "INT8" || item.id == "INT4")) {
                    viz.diagramId = "coop_matrix_tile";
                    viz.caption = "Hardware Matrix Core (Cooperative Matrix / WMMA / MFMA) Tile D = A * B + C";
                    continue;
                }
                if (item.id == "FP64" || item.id == "FP32" || item.id == "FP16" ||
                    item.id == "BF16" || item.id == "INT8" ||
                    item.id == "ILP & Dual-Issue" || containsCaseInsensitive(item.id, "Dual-Issue")) {
                    viz.diagramId = "simd_vector_alu";
                    viz.caption = (item.id == "ILP & Dual-Issue" || containsCaseInsensitive(item.id, "Dual-Issue"))
                        ? "SIMD Vector ALU Instruction-Level Parallelism (ILP) & FP32/INT32 Co-Issue"
                        : "SIMD Vector ALU Execution Lanes -> Vector Register File (VGPR)";
                    continue;
                }
                if (item.id == "LDS Bank Conflicts") {
                    viz.diagramId = "lds_banks";
                    viz.caption = "32-Bank Shared Memory / LDS Lane-to-Bank Serialization";
                    continue;
                }
                if (containsCaseInsensitive(item.id, "Cache") || containsCaseInsensitive(item.id, "Memory Bandwidth")) {
                    viz.diagramId = "cache_hierarchy";
                    viz.caption = "GPU L0 Vector -> L1 -> L2 -> L3/MALL -> GDDR/HBM VRAM Hierarchy";
                    continue;
                }
            }
        }
    }
}

void GuiApp::renderBenchmarkTooltip(const BenchmarkItem& item, const std::string& sceneOverride) {
    const BringUpColors c = activeBringUpColors();
    ImGui::BeginTooltip();
    float wrapW = s(360.0f);
    ImGui::PushTextWrapPos(ImGui::GetCursorPos().x + wrapW);

    pushBoldFont();
    ImGui::TextColored(c.phosphorHi, "%s", item.name.c_str());
    popBoldFont();
    ImGui::TextColored(c.textMuted, "%s  |  %s  |  UNIT: %s",
                       item.category.c_str(), item.subcategory.c_str(), item.metricType.c_str());
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::TextWrapped("%s", item.description.c_str());

    if (item.viz.hasVisualization()) {
        ImGui::Spacing();
        float canvasW = s(340.0f);
        float canvasH = std::round(canvasW / std::max(1.2f, item.viz.aspectRatio));

        bool drewImage = false;
        const std::string scTag = [&]() {
            std::string tag = !sceneOverride.empty() ? sceneOverride : currentSceneTag();
            if (tag == "landscape") tag = "outdoor";
            return tag;
        }();
        auto tryDrawAsset = [&](const std::string& tmpl) -> bool {
            if (tmpl.empty()) return false;
            std::string path = tmpl;
            size_t pos = path.find("{scene}");
            if (pos != std::string::npos) {
                path.replace(pos, 7, scTag);
            }
            VulkanContext::TextureResource tex = getOrLoadTexture(path);
            if (tex.isValid()) {
                ImGui::Image(reinterpret_cast<ImTextureID>(tex.descriptorSet), ImVec2(canvasW, canvasH));
                return true;
            }
            return false;
        };
        drewImage = tryDrawAsset(item.viz.assetPath) || tryDrawAsset(item.viz.fallbackAssetPath);
        if (!drewImage && !item.viz.diagramId.empty()) {
            ImVec2 p0 = ImGui::GetCursorScreenPos();
            ImVec2 p1(p0.x + canvasW, p0.y + canvasH);
            ImGui::Dummy(ImVec2(canvasW, canvasH));
            renderProceduralDiagram(item.viz.diagramId, p0, p1);
        }

        if (!item.viz.caption.empty()) {
            ImGui::TextColored(c.textSecondary, "%s", item.viz.caption.c_str());
        }
    }

    if (!item.configCaveat.empty()) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::TextColored(c.warnColor, "NOTE / CAVEAT: %s", item.configCaveat.c_str());
    }

    if (item.hasSupportDiscrepancy) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::TextColored(c.warnColor, "HARDWARE VS DRIVER SUPPORT");
        if (!item.hardwareSupportText.empty()) {
            ImGui::TextColored(c.textSecondary, "HW:  %s", item.hardwareSupportText.c_str());
        }
        if (!item.driverSupportText.empty()) {
            ImGui::TextColored(c.textSecondary, "DRV: %s", item.driverSupportText.c_str());
        }
        if (!item.supportDiscrepancySummary.empty()) {
            ImGui::TextColored(c.warnColor, "%s", item.supportDiscrepancySummary.c_str());
        }
    }

    if (!item.isSupported && !item.supportReason.empty()) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::TextColored(c.errColor, "%s: %s",
                           item.limitationCategory.empty() ? "UNSUPPORTED" : item.limitationCategory.c_str(),
                           item.supportReason.c_str());
    }

    ImGui::PopTextWrapPos();
    ImGui::EndTooltip();
}

void GuiApp::renderProceduralDiagram(const std::string& diagramId, ImVec2 p0, ImVec2 p1) const {
    const BringUpColors pal = activeBringUpColors();
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float w = p1.x - p0.x;
    const float h = p1.y - p0.y;
    const ImU32 bgCol      = toU32(pal.bgRecessed);
    const ImU32 elevCol    = toU32(pal.bgElevated);
    const ImU32 borderCol  = toU32(pal.borderOuter);
    const ImU32 gridCol    = toU32(pal.gridMinor);
    const ImU32 cyanCol    = toU32(pal.cyanAccent);
    const ImU32 greenCol   = toU32(pal.okColor);
    const ImU32 amberCol   = toU32(pal.phosphorHi);
    const ImU32 midCol     = toU32(pal.phosphorMid);
    const ImU32 redCol     = toU32(pal.errColor);
    const ImU32 textCol    = toU32(pal.textPrimary);
    const ImU32 mutedCol   = toU32(pal.textMuted);
    const ImU32 invCol     = toU32(pal.textInverse);

    dl->AddRectFilled(p0, p1, bgCol, 0.0f);
    // Oscilloscope graticule background grid
    for (int gx = 1; gx < 6; ++gx) {
        float x = p0.x + (w * gx) / 6.0f;
        dl->AddLine(ImVec2(x, p0.y), ImVec2(x, p1.y), gridCol, 1.0f);
    }
    for (int gy = 1; gy < 4; ++gy) {
        float y = p0.y + (h * gy) / 4.0f;
        dl->AddLine(ImVec2(p0.x, y), ImVec2(p1.x, y), gridCol, 1.0f);
    }
    dl->AddRect(p0, p1, borderCol, 0.0f, 0, 1.0f);
    drawCornerReticles(dl, p0, p1, amberCol, 6.0f);
    dl->PushClipRect(p0, p1, true);

    if (diagramId == "sparse_2_4") {
        float cell = std::min(w * 0.078f, h * 0.145f);
        ImVec2 orig(p0.x + w * 0.07f, p0.y + h * 0.25f);
        static const int mask[4][4] = {
            {1, 0, 1, 0},
            {0, 1, 0, 1},
            {1, 1, 0, 0},
            {0, 0, 1, 1}
        };
        dl->AddText(ImVec2(orig.x, p0.y + h * 0.07f), mutedCol, "DENSE 4x4 TILE A");
        for (int r = 0; r < 4; ++r) {
            for (int col = 0; col < 4; ++col) {
                ImVec2 c0(orig.x + col * (cell + 2.0f), orig.y + r * (cell + 2.0f));
                ImVec2 c1(c0.x + cell, c0.y + cell);
                dl->AddRectFilled(c0, c1, mask[r][col] ? cyanCol : toU32(pal.ledOff), 0.0f);
            }
        }
        ImVec2 arrowA(orig.x + 4 * (cell + 2.0f) + w * 0.04f, p0.y + h * 0.50f);
        ImVec2 arrowB(arrowA.x + w * 0.11f, arrowA.y);
        dl->AddLine(arrowA, arrowB, amberCol, 2.0f);
        dl->AddText(ImVec2(arrowA.x, arrowA.y - h * 0.17f), amberCol, "2:4 PACK");

        ImVec2 comp(arrowB.x + w * 0.05f, orig.y);
        dl->AddText(ImVec2(comp.x, p0.y + h * 0.07f), greenCol, "2:4 + META (2X)");
        for (int r = 0; r < 4; ++r) {
            for (int col = 0; col < 2; ++col) {
                ImVec2 c0(comp.x + col * (cell + 2.0f), comp.y + r * (cell + 2.0f));
                ImVec2 c1(c0.x + cell, c0.y + cell);
                dl->AddRectFilled(c0, c1, greenCol, 0.0f);
            }
            ImVec2 m0(comp.x + 2 * (cell + 2.0f) + w * 0.03f, comp.y + r * (cell + 2.0f));
            dl->AddRectFilled(m0, ImVec2(m0.x + cell * 1.3f, m0.y + cell), amberCol, 0.0f);
        }
    } else if (diagramId == "coop_vector_mlp") {
        const int nodes[4] = {4, 6, 6, 4};
        const float xs[4] = {0.14f, 0.38f, 0.62f, 0.86f};
        ImVec2 pts[4][6];
        for (int l = 0; l < 4; ++l) {
            for (int n = 0; n < nodes[l]; ++n) {
                float fy = 0.22f + 0.64f * (static_cast<float>(n) / static_cast<float>(nodes[l] - 1));
                pts[l][n] = ImVec2(p0.x + w * xs[l], p0.y + h * fy);
            }
        }
        const ImU32 wireCol = toU32(pal.phosphorMid, 0.45f);
        for (int l = 0; l < 3; ++l) {
            for (int a = 0; a < nodes[l]; ++a) {
                for (int b = 0; b < nodes[l + 1]; ++b) {
                    dl->AddLine(pts[l][a], pts[l + 1][b], wireCol, 1.0f);
                }
            }
        }
        for (int l = 0; l < 4; ++l) {
            ImU32 col = (l == 0) ? cyanCol : ((l == 3) ? greenCol : amberCol);
            for (int n = 0; n < nodes[l]; ++n) {
                float r = s(3.5f);
                dl->AddRectFilled(ImVec2(pts[l][n].x - r, pts[l][n].y - r),
                                  ImVec2(pts[l][n].x + r, pts[l][n].y + r), col, 0.0f);
            }
        }
        dl->AddText(ImVec2(p0.x + w * 0.06f, p0.y + h * 0.06f), cyanCol, "IN[16]");
        dl->AddText(ImVec2(p0.x + w * 0.33f, p0.y + h * 0.06f), amberCol, "MATVEC + RELU");
        dl->AddText(ImVec2(p0.x + w * 0.78f, p0.y + h * 0.06f), greenCol, "OUT[16]");
    } else if (diagramId == "pcie_interconnect" || diagramId == "interconnect_fabric") {
        ImVec2 boxHost0(p0.x + w * 0.05f, p0.y + h * 0.26f);
        ImVec2 boxHost1(p0.x + w * 0.34f, p0.y + h * 0.74f);
        ImVec2 boxGpuA0(p0.x + w * 0.47f, p0.y + h * 0.14f);
        ImVec2 boxGpuA1(p0.x + w * 0.94f, p0.y + h * 0.48f);
        ImVec2 boxGpuB0(p0.x + w * 0.47f, p0.y + h * 0.56f);
        ImVec2 boxGpuB1(p0.x + w * 0.94f, p0.y + h * 0.88f);

        dl->AddRectFilled(boxHost0, boxHost1, elevCol, 0.0f);
        dl->AddRect(boxHost0, boxHost1, cyanCol, 0.0f);
        dl->AddText(ImVec2(boxHost0.x + s(6.0f), boxHost0.y + s(8.0f)), cyanCol, "HOST PINNED");
        dl->AddText(ImVec2(boxHost0.x + s(6.0f), boxHost0.y + s(24.0f)), mutedCol, "DMA / REBAR");

        dl->AddRectFilled(boxGpuA0, boxGpuA1, elevCol, 0.0f);
        dl->AddRect(boxGpuA0, boxGpuA1, greenCol, 0.0f);
        dl->AddText(ImVec2(boxGpuA0.x + s(8.0f), boxGpuA0.y + s(8.0f)), greenCol, "GPU 0 VRAM (H2D / D2H)");

        dl->AddRectFilled(boxGpuB0, boxGpuB1, elevCol, 0.0f);
        dl->AddRect(boxGpuB0, boxGpuB1, amberCol, 0.0f);
        dl->AddText(ImVec2(boxGpuB0.x + s(8.0f), boxGpuB0.y + s(8.0f)), amberCol, "GPU 1 PEER P2P FABRIC");

        dl->AddLine(ImVec2(boxHost1.x, (boxHost0.y + boxHost1.y) * 0.5f),
                    ImVec2(boxGpuA0.x, (boxGpuA0.y + boxGpuA1.y) * 0.5f), cyanCol, 2.0f);
        dl->AddLine(ImVec2((boxGpuA0.x + boxGpuA1.x) * 0.5f, boxGpuA1.y),
                    ImVec2((boxGpuB0.x + boxGpuB1.x) * 0.5f, boxGpuB0.y), amberCol, 2.0f);
    } else if (diagramId == "omm_micromap") {
        ImVec2 vTop(p0.x + w * 0.34f, p0.y + h * 0.12f);
        ImVec2 vL(p0.x + w * 0.08f, p0.y + h * 0.86f);
        ImVec2 vR(p0.x + w * 0.60f, p0.y + h * 0.86f);
        const int N = 4;
        auto bary = [&](float u, float v) {
            float w0 = 1.0f - u - v;
            return ImVec2(w0 * vTop.x + u * vL.x + v * vR.x,
                          w0 * vTop.y + u * vL.y + v * vR.y);
        };
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N - i; ++j) {
                ImVec2 a = bary(float(i) / N, float(j) / N);
                ImVec2 b = bary(float(i + 1) / N, float(j) / N);
                ImVec2 c = bary(float(i) / N, float(j + 1) / N);
                ImU32 col = ((i + j) == 1 || (i + j) == 2) ? amberCol : (((i + j) > 2) ? greenCol : toU32(pal.ledOff));
                dl->AddTriangleFilled(a, b, c, col);
                dl->AddTriangle(a, b, c, bgCol, 1.2f);
            }
        }
        dl->AddText(ImVec2(p0.x + w * 0.64f, p0.y + h * 0.22f), greenCol, "OPAQUE (HW HIT)");
        dl->AddText(ImVec2(p0.x + w * 0.64f, p0.y + h * 0.44f), mutedCol, "EMPTY (HW SKIP)");
        dl->AddText(ImVec2(p0.x + w * 0.64f, p0.y + h * 0.66f), amberCol, "EDGE (AHS)");
#ifdef GPUBENCH_ENABLE_EXPERIMENTAL_WORKGRAPHS
    } else if (diagramId == "workgraphs_dag") {
        ImVec2 prod0(p0.x + w * 0.06f, p0.y + h * 0.32f);
        ImVec2 prod1(p0.x + w * 0.34f, p0.y + h * 0.68f);
        dl->AddRectFilled(prod0, prod1, elevCol, 0.0f);
        dl->AddRect(prod0, prod1, cyanCol, 0.0f);
        dl->AddText(ImVec2(prod0.x + s(6.0f), prod0.y + s(8.0f)), cyanCol, "PRODUCER");
        dl->AddText(ImVec2(prod0.x + s(6.0f), prod0.y + s(24.0f)), mutedCol, "ENQUEUE");

        const char* tiers[3] = {"CONSUMER HEAVY", "CONSUMER MED", "CONSUMER LIGHT"};
        const ImU32 cols[3] = {redCol, amberCol, greenCol};
        for (int t = 0; t < 3; ++t) {
            float y0 = p0.y + h * (0.12f + t * 0.28f);
            ImVec2 c0(p0.x + w * 0.54f, y0);
            ImVec2 c1(p0.x + w * 0.94f, y0 + h * 0.22f);
            dl->AddRectFilled(c0, c1, elevCol, 0.0f);
            dl->AddRect(c0, c1, cols[t], 0.0f);
            dl->AddText(ImVec2(c0.x + s(6.0f), c0.y + s(4.0f)), cols[t], tiers[t]);
            dl->AddLine(ImVec2(prod1.x, (prod0.y + prod1.y) * 0.5f),
                        ImVec2(c0.x, (c0.y + c1.y) * 0.5f), cols[t], 1.8f);
        }
#endif
    } else if (diagramId == "async_compute_dma") {
        dl->AddText(ImVec2(p0.x + w * 0.05f, p0.y + h * 0.14f), mutedCol, "SERIAL:");
        dl->AddRectFilled(ImVec2(p0.x + w * 0.28f, p0.y + h * 0.12f),
                          ImVec2(p0.x + w * 0.60f, p0.y + h * 0.30f), amberCol, 0.0f);
        dl->AddRectFilled(ImVec2(p0.x + w * 0.62f, p0.y + h * 0.12f),
                          ImVec2(p0.x + w * 0.94f, p0.y + h * 0.30f), greenCol, 0.0f);

        dl->AddText(ImVec2(p0.x + w * 0.05f, p0.y + h * 0.52f), greenCol, "OVERLAP:");
        dl->AddRectFilled(ImVec2(p0.x + w * 0.28f, p0.y + h * 0.44f),
                          ImVec2(p0.x + w * 0.72f, p0.y + h * 0.64f), amberCol, 0.0f);
        dl->AddText(ImVec2(p0.x + w * 0.30f, p0.y + h * 0.46f), invCol, "COMPUTE STREAM");

        dl->AddRectFilled(ImVec2(p0.x + w * 0.28f, p0.y + h * 0.69f),
                          ImVec2(p0.x + w * 0.66f, p0.y + h * 0.89f), greenCol, 0.0f);
        dl->AddText(ImVec2(p0.x + w * 0.30f, p0.y + h * 0.71f), invCol, "DMA COPY ENGINE");
    } else if (diagramId == "async_ace_overlap") {
        dl->AddText(ImVec2(p0.x + w * 0.05f, p0.y + h * 0.14f), mutedCol, "SERIAL:");
        dl->AddRectFilled(ImVec2(p0.x + w * 0.28f, p0.y + h * 0.12f),
                          ImVec2(p0.x + w * 0.60f, p0.y + h * 0.30f), cyanCol, 0.0f);
        dl->AddRectFilled(ImVec2(p0.x + w * 0.62f, p0.y + h * 0.12f),
                          ImVec2(p0.x + w * 0.94f, p0.y + h * 0.30f), amberCol, 0.0f);

        dl->AddText(ImVec2(p0.x + w * 0.05f, p0.y + h * 0.52f), greenCol, "OVERLAP:");
        dl->AddRectFilled(ImVec2(p0.x + w * 0.28f, p0.y + h * 0.44f),
                          ImVec2(p0.x + w * 0.68f, p0.y + h * 0.64f), cyanCol, 0.0f);
        dl->AddText(ImVec2(p0.x + w * 0.30f, p0.y + h * 0.46f), invCol, "GFX QUEUE (3D)");

        dl->AddRectFilled(ImVec2(p0.x + w * 0.28f, p0.y + h * 0.69f),
                          ImVec2(p0.x + w * 0.72f, p0.y + h * 0.89f), amberCol, 0.0f);
        dl->AddText(ImVec2(p0.x + w * 0.30f, p0.y + h * 0.71f), invCol, "ASYNC COMPUTE (ACE)");
    } else if (diagramId == "host_sysmem_dram") {
        const char* stages[4] = {"CPU SIMD", "L1/L2/L3", "MEM CTRL", "SYS DRAM"};
        const ImU32 cols[4] = {cyanCol, midCol, amberCol, greenCol};
        float boxW = w * 0.19f;
        float gap = w * 0.045f;
        float y0 = p0.y + h * 0.30f;
        float y1 = p0.y + h * 0.74f;
        dl->AddText(ImVec2(p0.x + w * 0.05f, p0.y + h * 0.08f), cyanCol, "HOST CPU MEMORY SUBSYSTEM (NO GPU)");
        for (int i = 0; i < 4; ++i) {
            float x0 = p0.x + w * 0.05f + i * (boxW + gap);
            dl->AddRectFilled(ImVec2(x0, y0), ImVec2(x0 + boxW, y1), elevCol, 0.0f);
            dl->AddRect(ImVec2(x0, y0), ImVec2(x0 + boxW, y1), cols[i], 0.0f);
            dl->AddText(ImVec2(x0 + s(5.0f), y0 + s(10.0f)), cols[i], stages[i]);
            if (i < 3) {
                dl->AddLine(ImVec2(x0 + boxW, (y0 + y1) * 0.5f),
                            ImVec2(x0 + boxW + gap, (y0 + y1) * 0.5f), cols[i + 1], 1.8f);
            }
        }
    } else if (diagramId == "apu_unified_memory") {
        dl->AddText(ImVec2(p0.x + w * 0.05f, p0.y + h * 0.08f), cyanCol, "UNIFIED MEMORY COHERENCY FABRIC");
        ImVec2 c0(p0.x + w * 0.05f, p0.y + h * 0.30f), c1(p0.x + w * 0.30f, p0.y + h * 0.78f);
        ImVec2 f0(p0.x + w * 0.36f, p0.y + h * 0.30f), f1(p0.x + w * 0.64f, p0.y + h * 0.78f);
        ImVec2 g0(p0.x + w * 0.70f, p0.y + h * 0.30f), g1(p0.x + w * 0.95f, p0.y + h * 0.78f);
        dl->AddRectFilled(c0, c1, elevCol, 0.0f); dl->AddRect(c0, c1, cyanCol, 0.0f);
        dl->AddText(ImVec2(c0.x + s(6.0f), c0.y + s(10.0f)), cyanCol, "HOST CPU");
        dl->AddRectFilled(f0, f1, elevCol, 0.0f); dl->AddRect(f0, f1, amberCol, 0.0f);
        dl->AddText(ImVec2(f0.x + s(6.0f), f0.y + s(10.0f)), amberCol, "COHERENT");
        dl->AddText(ImVec2(f0.x + s(6.0f), f0.y + s(26.0f)), mutedCol, "SHARED RAM");
        dl->AddRectFilled(g0, g1, elevCol, 0.0f); dl->AddRect(g0, g1, greenCol, 0.0f);
        dl->AddText(ImVec2(g0.x + s(6.0f), g0.y + s(10.0f)), greenCol, "GPU CU");
        dl->AddLine(ImVec2(c1.x, (c0.y + c1.y) * 0.5f), ImVec2(f0.x, (f0.y + f1.y) * 0.5f), cyanCol, 2.0f);
        dl->AddLine(ImVec2(f1.x, (f0.y + f1.y) * 0.5f), ImVec2(g0.x, (g0.y + g1.y) * 0.5f), greenCol, 2.0f);
    } else if (diagramId == "raster_rop_fill") {
        const char* stages[4] = {"RASTER", "FRAG", "ROP/BLD", "FB OUT"};
        const ImU32 cols[4] = {cyanCol, midCol, amberCol, greenCol};
        float boxW = w * 0.19f;
        float gap = w * 0.045f;
        float y0 = p0.y + h * 0.30f;
        float y1 = p0.y + h * 0.74f;
        dl->AddText(ImVec2(p0.x + w * 0.05f, p0.y + h * 0.08f), amberCol, "FIXED-FUNCTION RASTER & ROP PIPELINE");
        for (int i = 0; i < 4; ++i) {
            float x0 = p0.x + w * 0.05f + i * (boxW + gap);
            dl->AddRectFilled(ImVec2(x0, y0), ImVec2(x0 + boxW, y1), elevCol, 0.0f);
            dl->AddRect(ImVec2(x0, y0), ImVec2(x0 + boxW, y1), cols[i], 0.0f);
            dl->AddText(ImVec2(x0 + s(5.0f), y0 + s(10.0f)), cols[i], stages[i]);
            if (i < 3) {
                dl->AddLine(ImVec2(x0 + boxW, (y0 + y1) * 0.5f),
                            ImVec2(x0 + boxW + gap, (y0 + y1) * 0.5f), cols[i + 1], 1.8f);
            }
        }
    } else if (diagramId == "indirect_dispatch" || diagramId == "dgc_wavefront_dispatch") {
        const bool isRt = (diagramId == "dgc_wavefront_dispatch");
        const char* stages[4] = {
            isRt ? "PRIMARY" : "CLASSIFY",
            isRt ? "BIN/SORT" : "COMPACT",
            "IND / DGC",
            isRt ? "SHADERS" : "WORKER"
        };
        const ImU32 cols[4] = {cyanCol, midCol, amberCol, greenCol};
        float boxW = w * 0.19f;
        float gap = w * 0.045f;
        float y0 = p0.y + h * 0.30f;
        float y1 = p0.y + h * 0.74f;
        dl->AddText(ImVec2(p0.x + w * 0.05f, p0.y + h * 0.08f), amberCol,
                    isRt ? "GPU-DRIVEN WAVEFRONT / DGC SCHEDULING"
                         : "GPU INDIRECT COMMAND SYNTHESIS / DGC");
        for (int i = 0; i < 4; ++i) {
            float x0 = p0.x + w * 0.05f + i * (boxW + gap);
            dl->AddRectFilled(ImVec2(x0, y0), ImVec2(x0 + boxW, y1), elevCol, 0.0f);
            dl->AddRect(ImVec2(x0, y0), ImVec2(x0 + boxW, y1), cols[i], 0.0f);
            dl->AddText(ImVec2(x0 + s(5.0f), y0 + s(10.0f)), cols[i], stages[i]);
            if (i < 3) {
                dl->AddLine(ImVec2(x0 + boxW, (y0 + y1) * 0.5f),
                            ImVec2(x0 + boxW + gap, (y0 + y1) * 0.5f), cols[i + 1], 1.8f);
            }
        }
    } else if (diagramId == "rocm_persistent_queue") {
        const char* stages[3] = {"CLASSIFY", "WORK QUEUE", "PERSIST WGs"};
        const ImU32 cols[3] = {cyanCol, amberCol, greenCol};
        float boxW = w * 0.26f;
        float gap = w * 0.06f;
        float y0 = p0.y + h * 0.30f;
        float y1 = p0.y + h * 0.74f;
        dl->AddText(ImVec2(p0.x + w * 0.05f, p0.y + h * 0.08f), cyanCol, "ROCm / HIP PERSISTENT WORK-QUEUE COMPACTION");
        for (int i = 0; i < 3; ++i) {
            float x0 = p0.x + w * 0.05f + i * (boxW + gap);
            dl->AddRectFilled(ImVec2(x0, y0), ImVec2(x0 + boxW, y1), elevCol, 0.0f);
            dl->AddRect(ImVec2(x0, y0), ImVec2(x0 + boxW, y1), cols[i], 0.0f);
            dl->AddText(ImVec2(x0 + s(6.0f), y0 + s(10.0f)), cols[i], stages[i]);
            if (i < 2) {
                dl->AddLine(ImVec2(x0 + boxW, (y0 + y1) * 0.5f),
                            ImVec2(x0 + boxW + gap, (y0 + y1) * 0.5f), cols[i + 1], 1.8f);
            }
        }
    } else if (diagramId == "bvh_rayquery" || diagramId == "bvh_procedural") {
        const bool isProc = (diagramId == "bvh_procedural");
        const char* stages[4] = {
            isProc ? "RAYGEN" : "RAYQUERY",
            "TLAS INST",
            isProc ? "AABB BLAS" : "BLAS BVH",
            isProc ? ".RINT HIT" : "TRI/BOX HW"
        };
        const ImU32 cols[4] = {cyanCol, midCol, amberCol, greenCol};
        float boxW = w * 0.19f;
        float gap = w * 0.045f;
        float y0 = p0.y + h * 0.30f;
        float y1 = p0.y + h * 0.74f;
        dl->AddText(ImVec2(p0.x + w * 0.05f, p0.y + h * 0.08f), cyanCol,
                    isProc ? "PROCEDURAL AABB RAY TRACING PIPELINE"
                           : "HARDWARE INLINE RAYQUERY BVH TRAVERSAL");
        for (int i = 0; i < 4; ++i) {
            float x0 = p0.x + w * 0.05f + i * (boxW + gap);
            dl->AddRectFilled(ImVec2(x0, y0), ImVec2(x0 + boxW, y1), elevCol, 0.0f);
            dl->AddRect(ImVec2(x0, y0), ImVec2(x0 + boxW, y1), cols[i], 0.0f);
            dl->AddText(ImVec2(x0 + s(5.0f), y0 + s(10.0f)), cols[i], stages[i]);
            if (i < 3) {
                dl->AddLine(ImVec2(x0 + boxW, (y0 + y1) * 0.5f),
                            ImVec2(x0 + boxW + gap, (y0 + y1) * 0.5f), cols[i + 1], 1.8f);
            }
        }
    } else if (diagramId == "cone_divergence") {
        ImVec2 hitPt(p0.x + w * 0.50f, p0.y + h * 0.82f);
        dl->AddLine(ImVec2(p0.x + w * 0.14f, p0.y + h * 0.82f),
                    ImVec2(p0.x + w * 0.86f, p0.y + h * 0.82f), mutedCol, 2.0f);
        dl->AddLine(ImVec2(p0.x + w * 0.18f, p0.y + h * 0.22f), hitPt, cyanCol, 2.2f);
        for (int i = -3; i <= 3; ++i) {
            float ang = -1.10f + i * 0.18f;
            ImVec2 rayEnd(hitPt.x + std::cos(ang) * w * 0.36f,
                          hitPt.y + std::sin(ang) * h * 0.65f);
            dl->AddLine(hitPt, rayEnd, (i == 0) ? greenCol : amberCol, 1.6f);
        }
        dl->AddText(ImVec2(p0.x + w * 0.06f, p0.y + h * 0.06f), cyanCol, "PRIMARY -> GGX DIVERGENCE CONE");
    } else if (diagramId == "lds_banks") {
        const int banks = 8;
        float bw = (w * 0.84f) / banks;
        for (int b = 0; b < banks; ++b) {
            ImVec2 b0(p0.x + w * 0.08f + b * bw, p0.y + h * 0.58f);
            ImVec2 b1(b0.x + bw - 4.0f, p0.y + h * 0.86f);
            dl->AddRectFilled(b0, b1, (b == 0) ? redCol : elevCol, 0.0f);
            dl->AddRect(b0, b1, (b == 0) ? redCol : borderCol, 0.0f);
            ImVec2 lanePt(p0.x + w * 0.08f + b * bw + bw * 0.4f, p0.y + h * 0.20f);
            dl->AddRectFilled(ImVec2(lanePt.x - 3.0f, lanePt.y - 3.0f),
                              ImVec2(lanePt.x + 3.0f, lanePt.y + 3.0f), cyanCol, 0.0f);
            dl->AddLine(lanePt, ImVec2((b0.x + b1.x) * 0.5f, b0.y), (b % 2 == 0) ? greenCol : amberCol, 1.5f);
        }
        dl->AddText(ImVec2(p0.x + w * 0.08f, p0.y + h * 0.37f), textCol, "WAVE LANES -> 32 LDS BANKS");
    } else if (diagramId == "payload_registers") {
        const char* labels[3] = {"16B PAYLOAD (MAX WAVES)", "128B PAYLOAD (GI)", "256B PAYLOAD (SPILL RISK)"};
        const float frac[3] = {0.28f, 0.58f, 0.90f};
        const ImU32 cols[3] = {greenCol, amberCol, redCol};
        for (int i = 0; i < 3; ++i) {
            float y = p0.y + h * (0.14f + i * 0.28f);
            dl->AddRectFilled(ImVec2(p0.x + w * 0.06f, y),
                              ImVec2(p0.x + w * (0.06f + 0.88f * frac[i]), y + h * 0.20f), cols[i], 0.0f);
            dl->AddText(ImVec2(p0.x + w * 0.08f, y + s(2.0f)), invCol, labels[i]);
        }
    } else if (diagramId == "coop_matrix_tile" || diagramId == "wmma_matrix_core") {
        float boxH = h * 0.52f;
        float boxW = w * 0.22f;
        float y = p0.y + h * 0.24f;
        ImVec2 a0(p0.x + w * 0.05f, y);
        ImVec2 b0(p0.x + w * 0.34f, y);
        ImVec2 w0(p0.x + w * 0.64f, y);
        dl->AddRectFilled(a0, ImVec2(a0.x + boxW, y + boxH), elevCol, 0.0f);
        dl->AddRect(a0, ImVec2(a0.x + boxW, y + boxH), cyanCol, 0.0f);
        dl->AddText(ImVec2(a0.x + s(6.0f), y + s(8.0f)), cyanCol, "TILE A");
        dl->AddText(ImVec2(a0.x + s(6.0f), y + s(24.0f)), mutedCol, "M x K");

        dl->AddRectFilled(b0, ImVec2(b0.x + boxW, y + boxH), elevCol, 0.0f);
        dl->AddRect(b0, ImVec2(b0.x + boxW, y + boxH), amberCol, 0.0f);
        dl->AddText(ImVec2(b0.x + s(6.0f), y + s(8.0f)), amberCol, "TILE B");
        dl->AddText(ImVec2(b0.x + s(6.0f), y + s(24.0f)), mutedCol, "K x N");

        dl->AddRectFilled(w0, ImVec2(w0.x + w * 0.30f, y + boxH), elevCol, 0.0f);
        dl->AddRect(w0, ImVec2(w0.x + w * 0.30f, y + boxH), greenCol, 0.0f, 0, 1.5f);
        dl->AddText(ImVec2(w0.x + s(6.0f), y + s(8.0f)), greenCol, "MATRIX CORE");
        dl->AddText(ImVec2(w0.x + s(6.0f), y + s(24.0f)), textCol, "D=A*B+C");

        dl->AddLine(ImVec2(a0.x + boxW, y + boxH * 0.5f), ImVec2(b0.x, y + boxH * 0.5f), amberCol, 1.5f);
        dl->AddLine(ImVec2(b0.x + boxW, y + boxH * 0.5f), ImVec2(w0.x, y + boxH * 0.5f), greenCol, 1.5f);
    } else if (diagramId == "simd_vector_alu") {
        // SIMD vector issue lanes -> Vector ALU -> VGPR bank
        dl->AddText(ImVec2(p0.x + w * 0.05f, p0.y + h * 0.08f), amberCol, "SIMD VECTOR ALU // LANE EXECUTION");
        for (int lane = 0; lane < 4; ++lane) {
            float y0 = p0.y + h * (0.28f + lane * 0.16f);
            float y1 = y0 + h * 0.11f;
            ImVec2 l0(p0.x + w * 0.06f, y0), l1(p0.x + w * 0.36f, y1);
            ImVec2 u0(p0.x + w * 0.42f, y0), u1(p0.x + w * 0.68f, y1);
            ImVec2 r0(p0.x + w * 0.74f, y0), r1(p0.x + w * 0.94f, y1);
            dl->AddRectFilled(l0, l1, elevCol, 0.0f);
            dl->AddRect(l0, l1, cyanCol, 0.0f);
            dl->AddRectFilled(u0, u1, elevCol, 0.0f);
            dl->AddRect(u0, u1, amberCol, 0.0f);
            dl->AddRectFilled(r0, r1, elevCol, 0.0f);
            dl->AddRect(r0, r1, greenCol, 0.0f);
            dl->AddLine(ImVec2(l1.x, (y0 + y1) * 0.5f), ImVec2(u0.x, (y0 + y1) * 0.5f), midCol, 1.5f);
            dl->AddLine(ImVec2(u1.x, (y0 + y1) * 0.5f), ImVec2(r0.x, (y0 + y1) * 0.5f), greenCol, 1.5f);
        }
    } else if (diagramId == "queue_compaction") {
        dl->AddText(ImVec2(p0.x + w * 0.05f, p0.y + h * 0.08f), cyanCol, "SPARSE WAVEFRONT (BALLOT MASK)");
        const int active[8] = {1, 0, 1, 0, 0, 1, 1, 0};
        const float cellW = (w * 0.88f) / 8.0f;
        for (int i = 0; i < 8; ++i) {
            ImVec2 c0(p0.x + w * 0.06f + i * cellW, p0.y + h * 0.24f);
            ImVec2 c1(c0.x + cellW - s(4.0f), p0.y + h * 0.42f);
            dl->AddRectFilled(c0, c1, active[i] ? amberCol : toU32(pal.ledOff), 0.0f);
            dl->AddRect(c0, c1, active[i] ? amberCol : borderCol, 0.0f);
        }
        dl->AddText(ImVec2(p0.x + w * 0.05f, p0.y + h * 0.50f), greenCol, "COMPACTED WORKLIST (100% ACTIVE)");
        int outIdx = 0;
        for (int i = 0; i < 8; ++i) {
            if (!active[i]) continue;
            ImVec2 srcMid(p0.x + w * 0.06f + (i + 0.45f) * cellW, p0.y + h * 0.42f);
            ImVec2 d0(p0.x + w * 0.06f + outIdx * cellW, p0.y + h * 0.67f);
            ImVec2 d1(d0.x + cellW - s(4.0f), p0.y + h * 0.86f);
            dl->AddLine(srcMid, ImVec2((d0.x + d1.x) * 0.5f, d0.y), greenCol, 1.5f);
            dl->AddRectFilled(d0, d1, greenCol, 0.0f);
            ++outIdx;
        }
    } else if (diagramId == "bvh_raw_traversal") {
        dl->AddText(ImVec2(p0.x + w * 0.05f, p0.y + h * 0.07f), cyanCol, "COHERENT RAY WARP -> STRATIFIED BVH LAYERS");
        for (int layer = 0; layer < 5; ++layer) {
            float x0 = p0.x + w * (0.28f + layer * 0.13f);
            float x1 = x0 + w * 0.08f;
            float y0 = p0.y + h * (0.22f + (layer % 2) * 0.06f);
            float y1 = p0.y + h * (0.86f - (layer % 2) * 0.06f);
            ImU32 col = (layer < 2) ? midCol : ((layer < 4) ? amberCol : greenCol);
            dl->AddRectFilled(ImVec2(x0, y0), ImVec2(x1, y1), elevCol, 0.0f);
            dl->AddRect(ImVec2(x0, y0), ImVec2(x1, y1), col, 0.0f);
        }
        for (int r = 0; r < 4; ++r) {
            float ry = p0.y + h * (0.31f + r * 0.16f);
            dl->AddLine(ImVec2(p0.x + w * 0.06f, ry), ImVec2(p0.x + w * 0.90f, ry), cyanCol, 1.8f);
        }
    } else if (diagramId == "bvh_cutout_layers") {
        const char* stages[4] = {"TLAS INST", "BLAS BVH", "TRI / BOX HW", "AHS / HIT"};
        const ImU32 cols[4] = {cyanCol, midCol, amberCol, greenCol};
        float boxW = w * 0.19f;
        float gap = w * 0.045f;
        float y0 = p0.y + h * 0.30f;
        float y1 = p0.y + h * 0.70f;
        for (int i = 0; i < 4; ++i) {
            float x0 = p0.x + w * 0.05f + i * (boxW + gap);
            dl->AddRectFilled(ImVec2(x0, y0), ImVec2(x0 + boxW, y1), elevCol, 0.0f);
            dl->AddRect(ImVec2(x0, y0), ImVec2(x0 + boxW, y1), cols[i], 0.0f);
            dl->AddText(ImVec2(x0 + s(5.0f), y0 + s(10.0f)), cols[i], stages[i]);
            if (i < 3) {
                dl->AddLine(ImVec2(x0 + boxW, (y0 + y1) * 0.5f),
                            ImVec2(x0 + boxW + gap, (y0 + y1) * 0.5f), cols[i + 1], 1.8f);
            }
        }
    } else {
        // Memory & Cache hierarchy pyramid (L0 -> L1 -> L2 -> L3/MALL -> VRAM)
        const char* tiers[5] = {"L0 VECTOR CACHE", "L1 SHADER ARRAY", "L2 SHARED CACHE", "L3 / INFINITY MALL", "GDDR6 / HBM VRAM"};
        for (int i = 0; i < 5; ++i) {
            float frac = 0.38f + i * 0.13f;
            float x0 = p0.x + w * (0.5f - frac * 0.5f);
            float x1 = p0.x + w * (0.5f + frac * 0.5f);
            float y0 = p0.y + h * (0.09f + i * 0.172f);
            float y1 = y0 + h * 0.140f;
            ImU32 col = (i < 2) ? cyanCol : ((i < 4) ? amberCol : greenCol);
            dl->AddRectFilled(ImVec2(x0, y0), ImVec2(x1, y1), elevCol, 0.0f);
            dl->AddRect(ImVec2(x0, y0), ImVec2(x1, y1), col, 0.0f);
            dl->AddText(ImVec2(x0 + s(8.0f), y0 + s(2.0f)), col, tiers[i]);
        }
    }
    dl->PopClipRect();
}

} // namespace gpubench::gui
