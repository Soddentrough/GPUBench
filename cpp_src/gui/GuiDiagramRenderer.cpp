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
    for (auto& cat : m_categories) {
        for (auto& sg : cat.subgroups) {
            for (auto& item : sg.items) {
                BenchmarkVisualization& viz = item.viz;
                viz = BenchmarkVisualization{};

                // 1. RayScheduling 3D Scene Actual Renders (with procedural schematic fallback until captured)
                if (item.id == "RayScheduling") {
                    const std::string& nm = item.name;
                    if (containsCaseInsensitive(nm, "128 Lights")) {
                        viz.assetPath = "renders/render_{scene}_multilight_128_dgc.png";
                        viz.diagramId = "bvh_cutout_layers";
                        viz.caption = "128-Light Direct Evaluation (Megakernel vs DGC Light Binning)";
                        viz.linkedViewportMode = 0;
                        viz.linkedViewportSubIndex = 4;
                    } else if (containsCaseInsensitive(nm, "Alpha Cutout")) {
                        viz.assetPath = "renders/render_{scene}_alpha_cutout.png";
                        viz.diagramId = "bvh_cutout_layers";
                        viz.caption = "Alpha-Cutout Any-Hit Foliage & Grate Traversal";
                        viz.linkedViewportMode = 3;
                        viz.linkedViewportSubIndex = 1;
                    } else if (containsCaseInsensitive(nm, "Path Tracing") && containsCaseInsensitive(nm, "16 SPP")) {
                        viz.assetPath = "renders/render_{scene}_pathtracing_16spp_worklist.png";
                        viz.diagramId = "cone_divergence";
                        viz.caption = "Unbiased Monte Carlo Path Tracing (16 SPP Converged)";
                        viz.linkedViewportMode = 2;
                        viz.linkedViewportSubIndex = 0;
                    } else if (containsCaseInsensitive(nm, "Path Tracing") || containsCaseInsensitive(nm, "Persistent")) {
                        viz.assetPath = "renders/render_{scene}_pathtracing_1spp_worklist.png";
                        viz.diagramId = "cone_divergence";
                        viz.caption = "Unbiased Path Tracing & Persistent Ray Queue";
                        viz.linkedViewportMode = 2;
                        viz.linkedViewportSubIndex = 1;
                    } else if (containsCaseInsensitive(nm, "Shadow")) {
                        viz.assetPath = "renders/render_{scene}_stage3_shadow.png";
                        viz.diagramId = "bvh_cutout_layers";
                        viz.caption = "Directional Shadow Ray Visibility Pass";
                        viz.linkedViewportMode = 1;
                        viz.linkedViewportSubIndex = 2;
                    } else if (containsCaseInsensitive(nm, "Ambient Occlusion") || containsCaseInsensitive(nm, "RTAO")) {
                        viz.assetPath = "renders/render_{scene}_stage4_rtao.png";
                        viz.diagramId = "cone_divergence";
                        viz.caption = "Cosine-Weighted Hemisphere RTAO Pass";
                        viz.linkedViewportMode = 1;
                        viz.linkedViewportSubIndex = 3;
                    } else if (containsCaseInsensitive(nm, "Indirect") || containsCaseInsensitive(nm, "GI")) {
                        viz.assetPath = "renders/render_{scene}_stage6_indirect.png";
                        viz.diagramId = "cone_divergence";
                        viz.caption = "Secondary Diffuse Global Illumination Bounce";
                        viz.linkedViewportMode = 1;
                        viz.linkedViewportSubIndex = 5;
                    } else if (containsCaseInsensitive(nm, "Primary")) {
                        viz.assetPath = "renders/render_{scene}_stage2_primary.png";
                        viz.diagramId = "bvh_cutout_layers";
                        viz.caption = "Coherent Primary G-Buffer Ray Generation";
                        viz.linkedViewportMode = 1;
                        viz.linkedViewportSubIndex = 1;
                    } else if (containsCaseInsensitive(nm, "Material")) {
                        viz.assetPath = "renders/render_{scene}_worklist_dgc.png";
                        viz.diagramId = "cone_divergence";
                        viz.caption = "Heterogeneous Uber-Material BSDF Shading";
                        viz.linkedViewportMode = 0;
                        viz.linkedViewportSubIndex = 1;
                    } else if (containsCaseInsensitive(nm, "Full Frame") || containsCaseInsensitive(nm, "Deferred")) {
                        viz.assetPath = "renders/render_{scene}_stage7_final.png";
                        viz.diagramId = "bvh_cutout_layers";
                        viz.caption = "Full Multi-Stage Deferred Ray-Traced Frame";
                        viz.linkedViewportMode = 0;
                        viz.linkedViewportSubIndex = 2;
                    } else {
                        viz.assetPath = "renders/render_{scene}_stage5_direct.png";
                        viz.diagramId = "bvh_cutout_layers";
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
                    viz.assetPath = "renders/render_{scene}_stage2_primary.png";
                    viz.diagramId = "bvh_cutout_layers";
                    viz.caption = "Hardware Bottom-Level (BLAS) & Multi-Instance Top-Level (TLAS) BVH";
                    viz.linkedViewportMode = 3;
                    viz.linkedViewportSubIndex = 0;
                    continue;
                }
                if (containsCaseInsensitive(item.id, "Any-Hit") || containsCaseInsensitive(item.id, "RayAnyHit")) {
                    viz.assetPath = "renders/render_{scene}_alpha_cutout.png";
                    viz.diagramId = "bvh_cutout_layers";
                    viz.caption = "Multi-Layer Alpha-Cutout Any-Hit Shader Invocation";
                    viz.linkedViewportMode = 3;
                    viz.linkedViewportSubIndex = 1;
                    continue;
                }
                if (containsCaseInsensitive(item.id, "Raw") && containsCaseInsensitive(item.id, "Traversal")) {
                    viz.assetPath = "renders/render_{scene}_stage1_bvh.png";
                    viz.diagramId = "bvh_cutout_layers";
                    viz.caption = "Raw Fixed-Function BVH Box/Triangle Intersection Units";
                    viz.linkedViewportMode = 1;
                    viz.linkedViewportSubIndex = 0;
                    continue;
                }
                if (containsCaseInsensitive(item.id, "Divergence")) {
                    viz.diagramId = "cone_divergence";
                    viz.caption = "GGX Roughness Cone Spread & SIMD Wavefront Ray Divergence";
                    continue;
                }
                if (containsCaseInsensitive(item.id, "Payload")) {
                    viz.diagramId = "payload_registers";
                    viz.caption = "Ray Payload VGPR Footprint vs Active Wave Occupancy";
                    continue;
                }

                // 3. 2026 Compute, Interconnect & Scheduling Schematics
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
                    viz.diagramId = "pcie_interconnect";
                    viz.caption = "Host Pinned DMA <-> PCIe Root Complex <-> ReBAR / Multi-GPU P2P Crossbar";
                    continue;
                }
                if (containsCaseInsensitive(item.id, "Work Graphs")) {
                    viz.diagramId = "workgraphs_dag";
                    viz.caption = "GPU-Autonomous Producer -> Consumer Node Enqueue (Zero Host Roundtrip)";
                    continue;
                }
                if (containsCaseInsensitive(item.id, "Async Compute")) {
                    viz.diagramId = "async_ace_overlap";
                    viz.caption = "Concurrent Graphics Queue + Async Compute Engine (ACE) + Copy Engine (SDMA)";
                    continue;
                }
                if (containsCaseInsensitive(item.name, "Matrix") &&
                    (item.id == "FP16" || item.id == "BF16" || item.id == "FP8" ||
                     item.id == "FP6" || item.id == "FP4" || item.id == "INT8" || item.id == "INT4")) {
                    viz.diagramId = "coop_matrix_tile";
                    viz.caption = "Hardware Cooperative Matrix / WMMA / MFMA Sub-Tile D = A * B + C";
                    continue;
                }
                if (item.id == "LDS Bank Conflicts") {
                    viz.diagramId = "lds_banks";
                    viz.caption = "32-Bank Shared Memory / LDS Lane-to-Bank Serialization";
                    continue;
                }
                if (containsCaseInsensitive(item.id, "Cache") || containsCaseInsensitive(item.id, "Memory Bandwidth")) {
                    viz.diagramId = "cache_hierarchy";
                    viz.caption = "L0 Vector -> L1 -> L2 -> L3 Infinity/MALL -> GDDR/HBM VRAM Hierarchy";
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
        if (!item.viz.assetPath.empty()) {
            std::string path = item.viz.assetPath;
            std::string scTag = !sceneOverride.empty() ? sceneOverride : currentSceneTag();
            if (scTag == "landscape") scTag = "outdoor";
            size_t pos = path.find("{scene}");
            if (pos != std::string::npos) {
                path.replace(pos, 7, scTag);
            }
            VulkanContext::TextureResource tex = getOrLoadTexture(path);
            if (tex.isValid()) {
                ImGui::Image(reinterpret_cast<ImTextureID>(tex.descriptorSet), ImVec2(canvasW, canvasH));
                drewImage = true;
            }
        }
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
    } else if (diagramId == "async_ace_overlap") {
        dl->AddText(ImVec2(p0.x + w * 0.05f, p0.y + h * 0.10f), mutedCol, "SERIAL:");
        dl->AddRectFilled(ImVec2(p0.x + w * 0.30f, p0.y + h * 0.09f),
                          ImVec2(p0.x + w * 0.62f, p0.y + h * 0.24f), cyanCol, 0.0f);
        dl->AddRectFilled(ImVec2(p0.x + w * 0.63f, p0.y + h * 0.09f),
                          ImVec2(p0.x + w * 0.95f, p0.y + h * 0.24f), amberCol, 0.0f);

        dl->AddText(ImVec2(p0.x + w * 0.05f, p0.y + h * 0.44f), greenCol, "OVERLAP:");
        dl->AddRectFilled(ImVec2(p0.x + w * 0.30f, p0.y + h * 0.38f),
                          ImVec2(p0.x + w * 0.68f, p0.y + h * 0.54f), cyanCol, 0.0f);
        dl->AddText(ImVec2(p0.x + w * 0.32f, p0.y + h * 0.39f), invCol, "GFX QUEUE");

        dl->AddRectFilled(ImVec2(p0.x + w * 0.30f, p0.y + h * 0.58f),
                          ImVec2(p0.x + w * 0.72f, p0.y + h * 0.74f), amberCol, 0.0f);
        dl->AddText(ImVec2(p0.x + w * 0.32f, p0.y + h * 0.59f), invCol, "ACE COMPUTE");

        dl->AddRectFilled(ImVec2(p0.x + w * 0.30f, p0.y + h * 0.78f),
                          ImVec2(p0.x + w * 0.64f, p0.y + h * 0.92f), greenCol, 0.0f);
        dl->AddText(ImVec2(p0.x + w * 0.32f, p0.y + h * 0.78f), invCol, "SDMA COPY");
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
        dl->AddText(ImVec2(a0.x + s(6.0f), y + s(24.0f)), mutedCol, "16x16");

        dl->AddRectFilled(b0, ImVec2(b0.x + boxW, y + boxH), elevCol, 0.0f);
        dl->AddRect(b0, ImVec2(b0.x + boxW, y + boxH), amberCol, 0.0f);
        dl->AddText(ImVec2(b0.x + s(6.0f), y + s(8.0f)), amberCol, "TILE B");
        dl->AddText(ImVec2(b0.x + s(6.0f), y + s(24.0f)), mutedCol, "16x16");

        dl->AddRectFilled(w0, ImVec2(w0.x + w * 0.30f, y + boxH), elevCol, 0.0f);
        dl->AddRect(w0, ImVec2(w0.x + w * 0.30f, y + boxH), greenCol, 0.0f, 0, 1.5f);
        dl->AddText(ImVec2(w0.x + s(6.0f), y + s(8.0f)), greenCol, "WMMA CORE");
        dl->AddText(ImVec2(w0.x + s(6.0f), y + s(24.0f)), textCol, "D=A*B+C");

        dl->AddLine(ImVec2(a0.x + boxW, y + boxH * 0.5f), ImVec2(b0.x, y + boxH * 0.5f), amberCol, 1.5f);
        dl->AddLine(ImVec2(b0.x + boxW, y + boxH * 0.5f), ImVec2(w0.x, y + boxH * 0.5f), greenCol, 1.5f);
    } else if (diagramId == "simd_vector_alu") {
        // Wave32 SIMD vector issue lanes -> dual-issue VALU -> VGPR bank
        dl->AddText(ImVec2(p0.x + w * 0.05f, p0.y + h * 0.08f), amberCol, "WAVE32 SIMD // DUAL-ISSUE VALU");
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
    } else if (diagramId == "bvh_cutout_layers") {
        const char* stages[4] = {"TLAS INST", "BLAS BVH4", "TRI / BOX HW", "AHS / HIT"};
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
