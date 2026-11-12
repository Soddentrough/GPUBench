#include "GuiApp.h"
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

                // 1. RayScheduling 3D Scene Stage Thumbnails
                if (item.id == "RayScheduling") {
                    const std::string& nm = item.name;
                    if (containsCaseInsensitive(nm, "128 Lights")) {
                        viz.assetPath = "assets/thumbnails/thumb_{scene}_multilight_128_dgc.png";
                        viz.caption = "128-Light Direct Evaluation (Megakernel vs DGC Light Binning)";
                        viz.linkedViewportMode = 0;
                        viz.linkedViewportSubIndex = 4;
                    } else if (containsCaseInsensitive(nm, "Alpha Cutout")) {
                        viz.assetPath = "assets/thumbnails/thumb_{scene}_alpha_cutout.png";
                        viz.caption = "Alpha-Cutout Any-Hit Foliage & Grate Traversal";
                        viz.linkedViewportMode = 3;
                        viz.linkedViewportSubIndex = 1;
                    } else if (containsCaseInsensitive(nm, "Path Tracing") && containsCaseInsensitive(nm, "16 SPP")) {
                        viz.assetPath = "assets/thumbnails/thumb_{scene}_pathtracing_16spp.png";
                        viz.caption = "Unbiased Monte Carlo Path Tracing (16 SPP Converged)";
                        viz.linkedViewportMode = 0;
                        viz.linkedViewportSubIndex = 3;
                    } else if (containsCaseInsensitive(nm, "Path Tracing") || containsCaseInsensitive(nm, "Persistent")) {
                        viz.assetPath = "assets/thumbnails/thumb_{scene}_pathtracing_1spp.png";
                        viz.caption = "Unbiased Path Tracing & Persistent Ray Queue";
                        viz.linkedViewportMode = 0;
                        viz.linkedViewportSubIndex = 3;
                    } else if (containsCaseInsensitive(nm, "Shadow")) {
                        viz.assetPath = "assets/thumbnails/thumb_{scene}_stage3_shadow.png";
                        viz.caption = "Directional Shadow Ray Visibility Pass";
                        viz.linkedViewportMode = 1;
                        viz.linkedViewportSubIndex = 2;
                    } else if (containsCaseInsensitive(nm, "Ambient Occlusion") || containsCaseInsensitive(nm, "RTAO")) {
                        viz.assetPath = "assets/thumbnails/thumb_{scene}_stage4_rtao.png";
                        viz.caption = "Cosine-Weighted Hemisphere RTAO Pass";
                        viz.linkedViewportMode = 1;
                        viz.linkedViewportSubIndex = 3;
                    } else if (containsCaseInsensitive(nm, "Indirect") || containsCaseInsensitive(nm, "GI")) {
                        viz.assetPath = "assets/thumbnails/thumb_{scene}_stage6_indirect.png";
                        viz.caption = "Secondary Diffuse Global Illumination Bounce";
                        viz.linkedViewportMode = 1;
                        viz.linkedViewportSubIndex = 5;
                    } else if (containsCaseInsensitive(nm, "Primary")) {
                        viz.assetPath = "assets/thumbnails/thumb_{scene}_stage2_primary.png";
                        viz.caption = "Coherent Primary G-Buffer Ray Generation";
                        viz.linkedViewportMode = 1;
                        viz.linkedViewportSubIndex = 1;
                    } else if (containsCaseInsensitive(nm, "Material")) {
                        viz.assetPath = "assets/thumbnails/thumb_material_lineup.png";
                        viz.caption = "Heterogeneous Uber-Material BSDF Shading";
                        viz.linkedViewportMode = 2;
                        viz.linkedViewportSubIndex = 0;
                    } else if (containsCaseInsensitive(nm, "Full Frame") || containsCaseInsensitive(nm, "Deferred")) {
                        viz.assetPath = "assets/thumbnails/thumb_{scene}_stage7_final.png";
                        viz.caption = "Full Multi-Stage Deferred Ray-Traced Frame";
                        viz.linkedViewportMode = 0;
                        viz.linkedViewportSubIndex = 2;
                    } else {
                        viz.assetPath = "assets/thumbnails/thumb_{scene}_stage5_direct.png";
                        viz.caption = "Direct PBR Illumination & Reflection Evaluation";
                        viz.linkedViewportMode = 1;
                        viz.linkedViewportSubIndex = 4;
                    }
                    continue;
                }

                // 2. Hardware BVH & OMM Thumbnails / Schematics
                if (containsCaseInsensitive(item.id, "Opacity Micromap")) {
                    viz.diagramId = "omm_micromap";
                    viz.caption = "VK_EXT_opacity_micromap 2-State & 4-State Subdivision Triangle Grid";
                    continue;
                }
                if (containsCaseInsensitive(item.id, "Acceleration Structure") ||
                    containsCaseInsensitive(item.id, "RayASBuild")) {
                    viz.assetPath = "assets/thumbnails/thumb_blas_wireframe.png";
                    viz.caption = "Hardware Bottom-Level (BLAS) & Multi-Instance Top-Level (TLAS) BVH";
                    viz.linkedViewportMode = 3;
                    viz.linkedViewportSubIndex = 0;
                    continue;
                }
                if (containsCaseInsensitive(item.id, "Any-Hit") || containsCaseInsensitive(item.id, "RayAnyHit")) {
                    viz.assetPath = "assets/thumbnails/thumb_alpha_layers.png";
                    viz.caption = "Multi-Layer Alpha-Cutout Any-Hit Shader Invocation";
                    viz.linkedViewportMode = 3;
                    viz.linkedViewportSubIndex = 1;
                    continue;
                }
                if (containsCaseInsensitive(item.id, "Raw") && containsCaseInsensitive(item.id, "Traversal")) {
                    viz.assetPath = "assets/thumbnails/thumb_{scene}_stage1_bvh.png";
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
    ImGui::BeginTooltip();
    float wrapW = s(360.0f);
    ImGui::PushTextWrapPos(ImGui::GetCursorPos().x + wrapW);

    pushBoldFont();
    ImGui::TextColored(ImVec4(0.08f, 0.84f, 0.98f, 1.0f), "%s", item.name.c_str());
    popBoldFont();
    ImGui::TextDisabled("%s  |  %s  |  Unit: %s",
                        item.category.c_str(), item.subcategory.c_str(), item.metricType.c_str());
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::TextWrapped("%s", item.description.c_str());

    if (item.viz.hasVisualization()) {
        ImGui::Spacing();
        float canvasW = s(340.0f);
        float canvasH = std::round(canvasW / std::max(1.2f, item.viz.aspectRatio));

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
            }
        } else if (!item.viz.diagramId.empty()) {
            ImVec2 p0 = ImGui::GetCursorScreenPos();
            ImVec2 p1(p0.x + canvasW, p0.y + canvasH);
            ImGui::Dummy(ImVec2(canvasW, canvasH));
            renderProceduralDiagram(item.viz.diagramId, p0, p1);
        }

        if (!item.viz.caption.empty()) {
            ImGui::TextColored(ImVec4(0.65f, 0.76f, 0.90f, 1.0f), "%s", item.viz.caption.c_str());
        }
    }

    if (!item.isSupported && !item.supportReason.empty()) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.95f, 0.42f, 0.36f, 1.0f), "%s: %s",
                           item.limitationCategory.empty() ? "Unsupported" : item.limitationCategory.c_str(),
                           item.supportReason.c_str());
    }

    ImGui::PopTextWrapPos();
    ImGui::EndTooltip();
}

void GuiApp::renderProceduralDiagram(const std::string& diagramId, ImVec2 p0, ImVec2 p1) const {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float w = p1.x - p0.x;
    const float h = p1.y - p0.y;
    const ImU32 bgCol = IM_COL32(14, 18, 26, 255);
    const ImU32 borderCol = IM_COL32(48, 62, 86, 255);
    const ImU32 cyanCol = IM_COL32(18, 205, 248, 255);
    const ImU32 greenCol = IM_COL32(48, 224, 132, 255);
    const ImU32 amberCol = IM_COL32(245, 186, 54, 255);
    const ImU32 redCol = IM_COL32(238, 88, 82, 255);
    const ImU32 mutedCol = IM_COL32(120, 136, 162, 255);

    dl->AddRectFilled(p0, p1, bgCol, s(6.0f));
    dl->AddRect(p0, p1, borderCol, s(6.0f), 0, 1.0f);

    if (diagramId == "sparse_2_4") {
        // 4x4 block showing 2:4 Structured Sparsity: exactly 2 non-zero weights per 4 elements -> compressed 4x2 + metadata
        float cell = std::min(w * 0.085f, h * 0.16f);
        ImVec2 orig(p0.x + w * 0.07f, p0.y + h * 0.24f);
        static const int mask[4][4] = {
            {1, 0, 1, 0},
            {0, 1, 0, 1},
            {1, 1, 0, 0},
            {0, 0, 1, 1}
        };
        dl->AddText(ImVec2(orig.x, p0.y + h * 0.07f), mutedCol, "Dense 4x4 A");
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                ImVec2 c0(orig.x + c * (cell + 2.0f), orig.y + r * (cell + 2.0f));
                ImVec2 c1(c0.x + cell, c0.y + cell);
                dl->AddRectFilled(c0, c1, mask[r][c] ? cyanCol : IM_COL32(30, 36, 48, 255), 2.0f);
            }
        }
        ImVec2 arrowA(orig.x + 4 * (cell + 2.0f) + w * 0.04f, p0.y + h * 0.50f);
        ImVec2 arrowB(arrowA.x + w * 0.12f, arrowA.y);
        dl->AddLine(arrowA, arrowB, amberCol, 2.0f);
        dl->AddText(ImVec2(arrowA.x, arrowA.y - h * 0.16f), amberCol, "2:4 Pack");

        ImVec2 comp(arrowB.x + w * 0.05f, orig.y);
        dl->AddText(ImVec2(comp.x, p0.y + h * 0.07f), greenCol, "Compressed 4x2 + Meta (2x TFLOPS)");
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 2; ++c) {
                ImVec2 c0(comp.x + c * (cell + 2.0f), comp.y + r * (cell + 2.0f));
                ImVec2 c1(c0.x + cell, c0.y + cell);
                dl->AddRectFilled(c0, c1, greenCol, 2.0f);
            }
            ImVec2 m0(comp.x + 2 * (cell + 2.0f) + w * 0.03f, comp.y + r * (cell + 2.0f));
            dl->AddRectFilled(m0, ImVec2(m0.x + cell * 1.3f, m0.y + cell), amberCol, 2.0f);
        }
    } else if (diagramId == "coop_vector_mlp") {
        // 3-layer Neural BSDF MLP diagram (16 -> 64 -> 64 -> 16)
        const int nodes[4] = {4, 6, 6, 4};
        const float xs[4] = {0.14f, 0.38f, 0.62f, 0.86f};
        ImVec2 pts[4][6];
        for (int l = 0; l < 4; ++l) {
            for (int n = 0; n < nodes[l]; ++n) {
                float fy = 0.20f + 0.64f * (static_cast<float>(n) / static_cast<float>(nodes[l] - 1));
                pts[l][n] = ImVec2(p0.x + w * xs[l], p0.y + h * fy);
            }
        }
        for (int l = 0; l < 3; ++l) {
            for (int a = 0; a < nodes[l]; ++a) {
                for (int b = 0; b < nodes[l + 1]; ++b) {
                    dl->AddLine(pts[l][a], pts[l + 1][b], IM_COL32(36, 118, 168, 120), 1.0f);
                }
            }
        }
        for (int l = 0; l < 4; ++l) {
            ImU32 col = (l == 0) ? cyanCol : ((l == 3) ? greenCol : amberCol);
            for (int n = 0; n < nodes[l]; ++n) {
                dl->AddCircleFilled(pts[l][n], s(4.5f), col);
            }
        }
        dl->AddText(ImVec2(p0.x + w * 0.06f, p0.y + h * 0.05f), cyanCol, "In[16]");
        dl->AddText(ImVec2(p0.x + w * 0.34f, p0.y + h * 0.05f), amberCol, "MatVec + ReLU (64x64)");
        dl->AddText(ImVec2(p0.x + w * 0.79f, p0.y + h * 0.05f), greenCol, "Out[16]");
    } else if (diagramId == "pcie_interconnect") {
        // Host RAM <-> PCIe Root <-> GPU 0 VRAM (<-> GPU 1 P2P)
        ImVec2 boxHost0(p0.x + w * 0.05f, p0.y + h * 0.28f);
        ImVec2 boxHost1(p0.x + w * 0.31f, p0.y + h * 0.74f);
        ImVec2 boxGpuA0(p0.x + w * 0.46f, p0.y + h * 0.16f);
        ImVec2 boxGpuA1(p0.x + w * 0.94f, p0.y + h * 0.50f);
        ImVec2 boxGpuB0(p0.x + w * 0.46f, p0.y + h * 0.58f);
        ImVec2 boxGpuB1(p0.x + w * 0.94f, p0.y + h * 0.88f);

        dl->AddRectFilled(boxHost0, boxHost1, IM_COL32(26, 42, 66, 255), 4.0f);
        dl->AddRect(boxHost0, boxHost1, cyanCol, 4.0f);
        dl->AddText(ImVec2(boxHost0.x + s(8.0f), boxHost0.y + s(12.0f)), cyanCol, "Pinned Host");
        dl->AddText(ImVec2(boxHost0.x + s(8.0f), boxHost0.y + s(28.0f)), mutedCol, "DMA / ReBAR");

        dl->AddRectFilled(boxGpuA0, boxGpuA1, IM_COL32(22, 54, 46, 255), 4.0f);
        dl->AddRect(boxGpuA0, boxGpuA1, greenCol, 4.0f);
        dl->AddText(ImVec2(boxGpuA0.x + s(8.0f), boxGpuA0.y + s(10.0f)), greenCol, "GPU 0 VRAM (H2D / D2H)");

        dl->AddRectFilled(boxGpuB0, boxGpuB1, IM_COL32(52, 42, 22, 255), 4.0f);
        dl->AddRect(boxGpuB0, boxGpuB1, amberCol, 4.0f);
        dl->AddText(ImVec2(boxGpuB0.x + s(8.0f), boxGpuB0.y + s(8.0f)), amberCol, "GPU 1 Peer P2P / Coherency");

        dl->AddLine(ImVec2(boxHost1.x, (boxHost0.y + boxHost1.y) * 0.5f),
                    ImVec2(boxGpuA0.x, (boxGpuA0.y + boxGpuA1.y) * 0.5f), cyanCol, 2.5f);
        dl->AddLine(ImVec2((boxGpuA0.x + boxGpuA1.x) * 0.5f, boxGpuA1.y),
                    ImVec2((boxGpuB0.x + boxGpuB1.x) * 0.5f, boxGpuB0.y), amberCol, 2.5f);
    } else if (diagramId == "omm_micromap") {
        // Subdivided triangle showing Opaque (green), Transparent (dark), Unknown->AHS (amber) micro-triangles
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
                ImU32 col = ((i + j) == 1 || (i + j) == 2) ? amberCol : (((i + j) > 2) ? greenCol : IM_COL32(34, 42, 58, 255));
                dl->AddTriangleFilled(a, b, c, col);
                dl->AddTriangle(a, b, c, IM_COL32(14, 18, 26, 255), 1.2f);
            }
        }
        dl->AddText(ImVec2(p0.x + w * 0.64f, p0.y + h * 0.22f), greenCol, "Opaque (HW Hit)");
        dl->AddText(ImVec2(p0.x + w * 0.64f, p0.y + h * 0.44f), mutedCol, "Empty (HW Skip)");
        dl->AddText(ImVec2(p0.x + w * 0.64f, p0.y + h * 0.66f), amberCol, "Edge (AHS Only)");
    } else if (diagramId == "workgraphs_dag") {
        // Producer Node -> GPU Ring / Enqueue -> 3 Consumer Nodes
        ImVec2 prod0(p0.x + w * 0.06f, p0.y + h * 0.34f);
        ImVec2 prod1(p0.x + w * 0.32f, p0.y + h * 0.66f);
        dl->AddRectFilled(prod0, prod1, IM_COL32(20, 52, 74, 255), 4.0f);
        dl->AddRect(prod0, prod1, cyanCol, 4.0f);
        dl->AddText(ImVec2(prod0.x + s(6.0f), prod0.y + s(10.0f)), cyanCol, "Producer");
        dl->AddText(ImVec2(prod0.x + s(6.0f), prod0.y + s(26.0f)), mutedCol, "Classify Tile");

        const char* tiers[3] = {"Consumer Heavy", "Consumer Med", "Consumer Light"};
        const ImU32 cols[3] = {redCol, amberCol, greenCol};
        for (int t = 0; t < 3; ++t) {
            float y0 = p0.y + h * (0.12f + t * 0.28f);
            ImVec2 c0(p0.x + w * 0.56f, y0);
            ImVec2 c1(p0.x + w * 0.94f, y0 + h * 0.22f);
            dl->AddRectFilled(c0, c1, IM_COL32(28, 34, 46, 255), 4.0f);
            dl->AddRect(c0, c1, cols[t], 4.0f);
            dl->AddText(ImVec2(c0.x + s(8.0f), c0.y + s(6.0f)), cols[t], tiers[t]);
            dl->AddLine(ImVec2(prod1.x, (prod0.y + prod1.y) * 0.5f),
                        ImVec2(c0.x, (c0.y + c1.y) * 0.5f), cols[t], 2.0f);
        }
    } else if (diagramId == "async_ace_overlap") {
        // Timeline bars comparing Serial vs Overlapped Graphics + ACE + SDMA
        dl->AddText(ImVec2(p0.x + w * 0.05f, p0.y + h * 0.08f), mutedCol, "Serial Queue:");
        dl->AddRectFilled(ImVec2(p0.x + w * 0.30f, p0.y + h * 0.08f),
                          ImVec2(p0.x + w * 0.62f, p0.y + h * 0.24f), cyanCol, 3.0f);
        dl->AddRectFilled(ImVec2(p0.x + w * 0.63f, p0.y + h * 0.08f),
                          ImVec2(p0.x + w * 0.95f, p0.y + h * 0.24f), amberCol, 3.0f);

        dl->AddText(ImVec2(p0.x + w * 0.05f, p0.y + h * 0.42f), greenCol, "Concurrent:");
        dl->AddRectFilled(ImVec2(p0.x + w * 0.30f, p0.y + h * 0.38f),
                          ImVec2(p0.x + w * 0.66f, p0.y + h * 0.54f), cyanCol, 3.0f);
        dl->AddText(ImVec2(p0.x + w * 0.32f, p0.y + h * 0.39f), IM_COL32(10, 14, 20, 255), "GFX Queue");

        dl->AddRectFilled(ImVec2(p0.x + w * 0.30f, p0.y + h * 0.58f),
                          ImVec2(p0.x + w * 0.68f, p0.y + h * 0.74f), amberCol, 3.0f);
        dl->AddText(ImVec2(p0.x + w * 0.32f, p0.y + h * 0.59f), IM_COL32(10, 14, 20, 255), "ACE Compute");

        dl->AddRectFilled(ImVec2(p0.x + w * 0.30f, p0.y + h * 0.78f),
                          ImVec2(p0.x + w * 0.64f, p0.y + h * 0.92f), greenCol, 3.0f);
        dl->AddText(ImVec2(p0.x + w * 0.32f, p0.y + h * 0.78f), IM_COL32(10, 14, 20, 255), "SDMA Copy");
    } else if (diagramId == "cone_divergence") {
        ImVec2 hitPt(p0.x + w * 0.50f, p0.y + h * 0.82f);
        dl->AddLine(ImVec2(p0.x + w * 0.15f, p0.y + h * 0.82f),
                    ImVec2(p0.x + w * 0.85f, p0.y + h * 0.82f), mutedCol, 2.0f);
        dl->AddLine(ImVec2(p0.x + w * 0.18f, p0.y + h * 0.20f), hitPt, cyanCol, 2.2f);
        for (int i = -3; i <= 3; ++i) {
            float ang = -1.10f + i * 0.18f;
            ImVec2 rayEnd(hitPt.x + std::cos(ang) * w * 0.36f,
                          hitPt.y + std::sin(ang) * h * 0.68f);
            dl->AddLine(hitPt, rayEnd, (i == 0) ? greenCol : amberCol, 1.6f);
        }
        dl->AddText(ImVec2(p0.x + w * 0.06f, p0.y + h * 0.07f), cyanCol, "Primary Hit -> GGX Bounce Cone");
    } else if (diagramId == "lds_banks") {
        const int banks = 8;
        float bw = (w * 0.84f) / banks;
        for (int b = 0; b < banks; ++b) {
            ImVec2 b0(p0.x + w * 0.08f + b * bw, p0.y + h * 0.58f);
            ImVec2 b1(b0.x + bw - 4.0f, p0.y + h * 0.86f);
            dl->AddRectFilled(b0, b1, (b == 0) ? redCol : IM_COL32(34, 48, 70, 255), 3.0f);
            ImVec2 lanePt(p0.x + w * 0.08f + b * bw + bw * 0.4f, p0.y + h * 0.18f);
            dl->AddCircleFilled(lanePt, s(4.0f), cyanCol);
            dl->AddLine(lanePt, ImVec2((b0.x + b1.x) * 0.5f, b0.y), (b % 2 == 0) ? greenCol : amberCol, 1.5f);
        }
        dl->AddText(ImVec2(p0.x + w * 0.08f, p0.y + h * 0.36f), mutedCol, "Wavefront Lanes -> 32 Shared LDS Banks");
    } else if (diagramId == "payload_registers") {
        const char* labels[3] = {"16B Payload (High Waves)", "128B Payload (GI)", "256B Payload (VGPR Spill Risk)"};
        const float frac[3] = {0.22f, 0.56f, 0.90f};
        const ImU32 cols[3] = {greenCol, amberCol, redCol};
        for (int i = 0; i < 3; ++i) {
            float y = p0.y + h * (0.14f + i * 0.28f);
            dl->AddRectFilled(ImVec2(p0.x + w * 0.06f, y),
                              ImVec2(p0.x + w * (0.06f + 0.88f * frac[i]), y + h * 0.20f), cols[i], 3.0f);
            dl->AddText(ImVec2(p0.x + w * 0.08f, y + s(2.0f)), IM_COL32(12, 16, 22, 255), labels[i]);
        }
    } else if (diagramId == "coop_matrix_tile") {
        float sz = std::min(w * 0.18f, h * 0.48f);
        ImVec2 a0(p0.x + w * 0.08f, p0.y + h * 0.28f);
        ImVec2 b0(a0.x + sz + w * 0.08f, a0.y);
        ImVec2 d0(b0.x + sz + w * 0.14f, a0.y);
        dl->AddRectFilled(a0, ImVec2(a0.x + sz, a0.y + sz), cyanCol, 3.0f);
        dl->AddRectFilled(b0, ImVec2(b0.x + sz, b0.y + sz), amberCol, 3.0f);
        dl->AddRectFilled(d0, ImVec2(d0.x + sz, d0.y + sz), greenCol, 3.0f);
        dl->AddText(ImVec2(a0.x + s(6.0f), a0.y + sz + s(4.0f)), cyanCol, "Tile A (MxK)");
        dl->AddText(ImVec2(b0.x + s(6.0f), b0.y + sz + s(4.0f)), amberCol, "Tile B (KxN)");
        dl->AddText(ImVec2(d0.x + s(6.0f), d0.y + sz + s(4.0f)), greenCol, "Accum D (MxN)");
    } else {
        // Cache hierarchy pyramid (L0 -> L1 -> L2 -> L3 -> VRAM)
        const char* tiers[5] = {"L0 Vector Cache", "L1 Shader Array", "L2 Shared Cache", "L3 / MALL Infinity", "GDDR / HBM VRAM"};
        for (int i = 0; i < 5; ++i) {
            float frac = 0.34f + i * 0.14f;
            float x0 = p0.x + w * (0.5f - frac * 0.5f);
            float x1 = p0.x + w * (0.5f + frac * 0.5f);
            float y0 = p0.y + h * (0.08f + i * 0.175f);
            float y1 = y0 + h * 0.145f;
            ImU32 col = (i < 2) ? cyanCol : ((i < 4) ? amberCol : greenCol);
            dl->AddRectFilled(ImVec2(x0, y0), ImVec2(x1, y1), IM_COL32(26, 36, 52, 255), 3.0f);
            dl->AddRect(ImVec2(x0, y0), ImVec2(x1, y1), col, 3.0f);
            dl->AddText(ImVec2(x0 + s(8.0f), y0 + s(1.0f)), col, tiers[i]);
        }
    }
}

} // namespace gpubench::gui
