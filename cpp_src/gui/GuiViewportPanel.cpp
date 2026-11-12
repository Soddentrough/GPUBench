#include "GuiApp.h"
#include "GuiUiHelpers.h"
#include "core/VulkanContext.h"
#include "yyjson.h"

#include <SDL3/SDL.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace gpubench::gui {

namespace {

struct SceneMeta {
    const char* title;
    const char* tag;      // render_<tag>_*
    const char* thumbTag; // thumb_<thumbTag>_*
    const char* geometryNote;
};

constexpr SceneMeta kScenes[4] = {
    {"Showroom Studio",    "showroom", "showroom", "High-poly automotive clear-coat studio with area lights & glossy reflections"},
    {"Indoor Atrium",      "indoor",   "indoor",   "First-party MIT/CC0 Renaissance Colonnade Atrium (426K triangles, multi-bounce GI)"},
    {"Open-World Forest",  "forest",   "forest",   "Dense instanced alpha-cutout foliage canopy & bark geometry"},
    {"Outdoor Landscape",  "outdoor",  "outdoor",  "Procedural eroded heightfield terrain with directional sun & sky atmosphere"}
};

bool loadViewportJsonContent(VulkanContext* guiVk, const std::string& bareJsonName, std::string& outJson) {
    if (guiVk && guiVk->isInitialized()) {
        if (auto sharedCtx = guiVk->getSharedComputeContext()) {
            if (sharedCtx->getInMemoryJson(bareJsonName, outJson) ||
                sharedCtx->getInMemoryJson("renders/" + bareJsonName, outJson)) {
                return !outJson.empty();
            }
        }
    }

    namespace fs = std::filesystem;
    std::vector<std::string> roots = {"renders/", "../renders/", "../../renders/"};
    if (const char* base = SDL_GetBasePath()) {
        roots.push_back(std::string(base) + "renders/");
        roots.push_back(std::string(base) + "../renders/");
    }

    for (const auto& root : roots) {
        std::string fullPath = root + bareJsonName;
        std::error_code ec;
        if (fs::exists(fullPath, ec) && fs::is_regular_file(fullPath, ec)) {
            std::ifstream in(fullPath, std::ios::binary);
            if (in.is_open()) {
                std::ostringstream ss;
                ss << in.rdbuf();
                outJson = ss.str();
                return !outJson.empty();
            }
        }
    }
    return false;
}

struct ParsedParityProfile {
    bool valid{false};
    std::string gpu;
    std::string resolution;
    double tradFps{0.0};
    double tradMrays{0.0};
    double tradMs{0.0};
    double workFps{0.0};
    double workMrays{0.0};
    double workMs{0.0};
    double psnr{0.0};
    double exactPct{0.0};
    uint64_t diffPixels{0};
    std::string status;
};

ParsedParityProfile parseParityProfileJson(const std::string& jsonStr) {
    ParsedParityProfile p{};
    if (jsonStr.empty()) return p;

    yyjson_doc* doc = yyjson_read(jsonStr.data(), jsonStr.size(), 0);
    if (!doc) return p;

    yyjson_val* root = yyjson_doc_get_root(doc);
    if (yyjson_is_obj(root)) {
        p.valid = true;
        if (yyjson_val* v = yyjson_obj_get(root, "gpu"); yyjson_is_str(v)) p.gpu = yyjson_get_str(v);
        if (yyjson_val* v = yyjson_obj_get(root, "resolution"); yyjson_is_str(v)) p.resolution = yyjson_get_str(v);

        if (yyjson_val* trad = yyjson_obj_get(root, "traditional"); yyjson_is_obj(trad)) {
            if (yyjson_val* v = yyjson_obj_get(trad, "fps"); yyjson_is_num(v)) p.tradFps = yyjson_get_num(v);
            if (yyjson_val* v = yyjson_obj_get(trad, "mrays"); yyjson_is_num(v)) p.tradMrays = yyjson_get_num(v);
            if (yyjson_val* v = yyjson_obj_get(trad, "frame_ms"); yyjson_is_num(v)) p.tradMs = yyjson_get_num(v);
        }
        if (yyjson_val* work = yyjson_obj_get(root, "worklist"); yyjson_is_obj(work)) {
            if (yyjson_val* v = yyjson_obj_get(work, "fps"); yyjson_is_num(v)) p.workFps = yyjson_get_num(v);
            if (yyjson_val* v = yyjson_obj_get(work, "mrays"); yyjson_is_num(v)) p.workMrays = yyjson_get_num(v);
            if (yyjson_val* v = yyjson_obj_get(work, "frame_ms"); yyjson_is_num(v)) p.workMs = yyjson_get_num(v);
        }
        if (yyjson_val* par = yyjson_obj_get(root, "parity"); yyjson_is_obj(par)) {
            if (yyjson_val* v = yyjson_obj_get(par, "psnr"); yyjson_is_num(v)) p.psnr = yyjson_get_num(v);
            if (yyjson_val* v = yyjson_obj_get(par, "exact_pct"); yyjson_is_num(v)) p.exactPct = yyjson_get_num(v);
            if (yyjson_val* v = yyjson_obj_get(par, "diff_pixels"); yyjson_is_uint(v)) p.diffPixels = yyjson_get_uint(v);
            if (yyjson_val* v = yyjson_obj_get(par, "status"); yyjson_is_str(v)) p.status = yyjson_get_str(v);
        }
    }

    yyjson_doc_free(doc);
    return p;
}

struct ParsedPipelineStage {
    std::string id;
    std::string title;
    std::string passType;
    double timeMs{0.0};
    double mrays{0.0};
    double fps{0.0};
};

struct ParsedPipelineBreakdown {
    bool valid{false};
    uint64_t triangles{0};
    double bvhBuildMs{0.0};
    std::vector<ParsedPipelineStage> stages;
};

ParsedPipelineBreakdown parsePipelineBreakdownJson(const std::string& jsonStr) {
    ParsedPipelineBreakdown bd{};
    if (jsonStr.empty()) return bd;

    yyjson_doc* doc = yyjson_read(jsonStr.data(), jsonStr.size(), 0);
    if (!doc) return bd;

    yyjson_val* root = yyjson_doc_get_root(doc);
    if (yyjson_is_obj(root)) {
        bd.valid = true;
        if (yyjson_val* v = yyjson_obj_get(root, "triangles"); yyjson_is_uint(v)) bd.triangles = yyjson_get_uint(v);
        if (yyjson_val* v = yyjson_obj_get(root, "bvh_build_time_ms"); yyjson_is_num(v)) bd.bvhBuildMs = yyjson_get_num(v);

        if (yyjson_val* arr = yyjson_obj_get(root, "stages"); yyjson_is_arr(arr)) {
            size_t idx = 0, maxIdx = 0;
            yyjson_val* st = nullptr;
            yyjson_arr_foreach(arr, idx, maxIdx, st) {
                if (!yyjson_is_obj(st)) continue;
                ParsedPipelineStage s{};
                if (yyjson_val* v = yyjson_obj_get(st, "id"); yyjson_is_str(v)) s.id = yyjson_get_str(v);
                if (yyjson_val* v = yyjson_obj_get(st, "title"); yyjson_is_str(v)) s.title = yyjson_get_str(v);
                if (yyjson_val* v = yyjson_obj_get(st, "pass_type"); yyjson_is_str(v)) s.passType = yyjson_get_str(v);
                if (yyjson_val* v = yyjson_obj_get(st, "time_ms"); yyjson_is_num(v)) s.timeMs = yyjson_get_num(v);
                if (yyjson_val* v = yyjson_obj_get(st, "mrays"); yyjson_is_num(v)) s.mrays = yyjson_get_num(v);
                if (yyjson_val* v = yyjson_obj_get(st, "fps"); yyjson_is_num(v)) s.fps = yyjson_get_num(v);
                bd.stages.push_back(std::move(s));
            }
        }
    }

    yyjson_doc_free(doc);
    return bd;
}

} // namespace

void GuiApp::renderRayTracingViewport() {
    ImGui::Spacing();

    // Top Mode Selector
    const char* modeLabels[] = {
        "Scenes & Visual Parity",
        "7-Stage Pipeline Passes",
        "PBR Material Gallery",
        "Geometry & BVH Inspection"
    };
    for (int m = 0; m < 4; ++m) {
        if (m > 0) sameLineIfFits(s(180.0f), s(6.0f));
        bool active = (m_rtViewportMode == m);
        if (active) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.05f, 0.55f, 0.78f, 1.0f));
        if (ImGui::Button(modeLabels[m], ImVec2(0.0f, s(27.0f)))) {
            m_rtViewportMode = m;
        }
        if (active) ImGui::PopStyleColor();
    }

    ImGui::Spacing();

    // Scene Selector Pills (applicable to modes 0, 1, 3)
    m_rtSceneIndex = std::clamp(m_rtSceneIndex, 0, 3);
    if (m_rtViewportMode != 2) {
        ImGui::TextDisabled("Active Scene:");
        for (int si = 0; si < 4; ++si) {
            sameLineIfFits(s(150.0f), s(6.0f));
            bool sel = (m_rtSceneIndex == si);
            if (sel) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.08f, 0.48f, 0.68f, 1.0f));
            if (ImGui::SmallButton(kScenes[si].title)) {
                m_rtSceneIndex = si;
            }
            if (sel) ImGui::PopStyleColor();
        }
        ImGui::SameLine();
        ImGui::TextDisabled("— %s", kScenes[m_rtSceneIndex].geometryNote);
    }

    ImGui::Separator();
    ImGui::Spacing();

    const SceneMeta& curScene = kScenes[m_rtSceneIndex];
    const std::string tag = curScene.tag;
    const std::string thumbTag = curScene.thumbTag;

    auto drawImageFramed = [&](const std::string& primaryRelPath,
                               const std::string& fallbackThumbRelPath,
                               const char* caption,
                               float targetW) {
        VulkanContext::TextureResource tex = getOrLoadTexture(primaryRelPath);
        bool isLiveFrame = tex.isValid();
        if (!isLiveFrame && !fallbackThumbRelPath.empty()) {
            tex = getOrLoadTexture(fallbackThumbRelPath);
        }

        float imgH = std::round(targetW * (9.0f / 16.0f));
        if (tex.isValid() && tex.width > 0 && tex.height > 0) {
            imgH = std::round(targetW * (static_cast<float>(tex.height) / static_cast<float>(tex.width)));
        }

        ImGui::BeginGroup();
        if (tex.isValid()) {
            ImGui::Image(reinterpret_cast<ImTextureID>(tex.descriptorSet), ImVec2(targetW, imgH));
            if (isLiveFrame) {
                ImGui::TextColored(ImVec4(0.20f, 0.92f, 0.56f, 1.0f), "[LIVE %dx%d%s] %s",
                                   tex.width, tex.height,
                                   tex.borrowsExternalImage ? " GPU VRAM" : " PNG",
                                   caption);
            } else {
                ImGui::TextColored(ImVec4(0.70f, 0.78f, 0.90f, 1.0f), "[REFERENCE PREVIEW] %s", caption);
            }
        } else {
            ImGui::Dummy(ImVec2(targetW, imgH));
            ImGui::TextDisabled("%s (Run RayScheduling to populate live render)", caption);
        }
        ImGui::EndGroup();
    };

    if (m_rtViewportMode == 0) {
        // Scenes & Visual Parity Mode
        const char* parityViews[] = {
            "DGC Wavefront Render",
            "Megakernel Baseline",
            "Side-by-Side + 10x Difference Heatmap",
            "Path Tracing (1 SPP vs 16 SPP)",
            "128-Light Direct Evaluation"
        };
        for (int pv = 0; pv < 5; ++pv) {
            if (pv > 0) sameLineIfFits(s(170.0f), s(6.0f));
            bool active = (m_rtParityViewMode == pv);
            if (active) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.05f, 0.55f, 0.78f, 1.0f));
            if (ImGui::SmallButton(parityViews[pv])) m_rtParityViewMode = pv;
            if (active) ImGui::PopStyleColor();
        }
        ImGui::Spacing();

        std::string profileJsonStr;
        std::string jsonKey = (m_rtParityViewMode == 3)
            ? ("render_" + tag + "_pt16_profile.json")
            : ("render_" + tag + "_profile.json");
        ParsedParityProfile prof{};
        if (loadViewportJsonContent(m_vulkanContext, jsonKey, profileJsonStr)) {
            prof = parseParityProfileJson(profileJsonStr);
        }

        if (prof.valid) {
            double speedup = (prof.tradMs > 0.0 && prof.workMs > 0.0) ? (prof.tradMs / prof.workMs) : 1.0;
            ImVec4 statusCol = (prof.psnr >= 45.0)
                ? ImVec4(0.20f, 0.94f, 0.54f, 1.0f)
                : ImVec4(0.96f, 0.38f, 0.34f, 1.0f);
            ImGui::TextColored(statusCol,
                               "%s  |  PSNR: %.2f dB  |  Exact Match: %.2f%%  |  Megakernel: %.2f ms (%.0f MRays/s)  |  DGC: %.2f ms (%.0f MRays/s, %.2fx)",
                               prof.status.empty() ? "PARITY TELEMETRY" : prof.status.c_str(),
                               prof.psnr, prof.exactPct,
                               prof.tradMs, prof.tradMrays,
                               prof.workMs, prof.workMrays, speedup);
            ImGui::Spacing();
        }

        float availW = std::max(s(320.0f), ImGui::GetContentRegionAvail().x - s(8.0f));

        if (m_rtParityViewMode == 0) {
            drawImageFramed("renders/render_" + tag + "_worklist_dgc.png",
                            "assets/thumbnails/thumb_" + thumbTag + "_stage7_final.png",
                            "DGC Wavefront Material-Sorted Render",
                            std::min(availW, s(980.0f)));
        } else if (m_rtParityViewMode == 1) {
            drawImageFramed("renders/render_" + tag + "_traditional_megakernel.png",
                            "assets/thumbnails/thumb_" + thumbTag + "_stage7_final.png",
                            "Traditional Monolithic Megakernel Render",
                            std::min(availW, s(980.0f)));
        } else if (m_rtParityViewMode == 2) {
            float colW = (availW - s(20.0f)) / 3.0f;
            drawImageFramed("renders/render_" + tag + "_traditional_megakernel.png",
                            "assets/thumbnails/thumb_" + thumbTag + "_stage7_final.png",
                            "Megakernel", colW);
            ImGui::SameLine(0.0f, s(10.0f));
            drawImageFramed("renders/render_" + tag + "_worklist_dgc.png",
                            "assets/thumbnails/thumb_" + thumbTag + "_stage7_final.png",
                            "DGC Wavefront", colW);
            ImGui::SameLine(0.0f, s(10.0f));
            drawImageFramed("renders/render_" + tag + "_difference_heatmap.png",
                            "assets/thumbnails/thumb_" + thumbTag + "_stage1_bvh.png",
                            "10x Difference Heatmap", colW);
        } else if (m_rtParityViewMode == 3) {
            float colW = (availW - s(12.0f)) / 2.0f;
            drawImageFramed("renders/render_" + tag + "_pathtracing_1spp_worklist.png",
                            "assets/thumbnails/thumb_" + thumbTag + "_pathtracing_1spp.png",
                            "Unbiased Path Tracing (1 SPP)", colW);
            ImGui::SameLine(0.0f, s(12.0f));
            drawImageFramed("renders/render_" + tag + "_pathtracing_16spp_worklist.png",
                            "assets/thumbnails/thumb_" + thumbTag + "_pathtracing_16spp.png",
                            "Unbiased Path Tracing (16 SPP Converged)", colW);
        } else {
            float colW = (availW - s(12.0f)) / 2.0f;
            drawImageFramed("renders/render_" + tag + "_multilight_128_traditional.png",
                            "assets/thumbnails/thumb_" + thumbTag + "_multilight_128_dgc.png",
                            "128-Light Megakernel", colW);
            ImGui::SameLine(0.0f, s(12.0f));
            drawImageFramed("renders/render_" + tag + "_multilight_128_dgc.png",
                            "assets/thumbnails/thumb_" + thumbTag + "_multilight_128_dgc.png",
                            "128-Light DGC Spatial Binning", colW);
        }
    } else if (m_rtViewportMode == 1) {
        // 7-Stage Pipeline Passes
        static const char* stageIds[7] = {
            "stage1_bvh", "stage2_primary", "stage3_shadow", "stage4_rtao",
            "stage5_direct", "stage6_indirect", "stage7_final"
        };
        static const char* stageLabels[7] = {
            "1. BVH Heatmap", "2. G-Buffer Normals", "3. Sun Shadows", "4. RTAO",
            "5. Direct PBR", "6. Indirect GI", "7. Final Beauty"
        };

        m_rtPassIndex = std::clamp(m_rtPassIndex, 0, 6);
        for (int si = 0; si < 7; ++si) {
            if (si > 0) sameLineIfFits(s(130.0f), s(6.0f));
            bool active = (m_rtPassIndex == si);
            if (active) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.05f, 0.55f, 0.78f, 1.0f));
            if (ImGui::SmallButton(stageLabels[si])) m_rtPassIndex = si;
            if (active) ImGui::PopStyleColor();
        }
        ImGui::Spacing();

        std::string bdJson;
        ParsedPipelineBreakdown bd{};
        if (loadViewportJsonContent(m_vulkanContext, "render_" + tag + "_pipeline_breakdown.json", bdJson)) {
            bd = parsePipelineBreakdownJson(bdJson);
        }

        float availW = std::max(s(320.0f), ImGui::GetContentRegionAvail().x - s(8.0f));
        std::string sid = stageIds[m_rtPassIndex];
        drawImageFramed("renders/render_" + tag + "_" + sid + ".png",
                        "assets/thumbnails/thumb_" + thumbTag + "_" + sid + ".png",
                        stageLabels[m_rtPassIndex],
                        std::min(availW, s(860.0f)));

        if (bd.valid && !bd.stages.empty()) {
            ImGui::Spacing();
            ImGui::TextDisabled("Scene Triangles: %llu  |  BVH Build: %.2f ms",
                                static_cast<unsigned long long>(bd.triangles), bd.bvhBuildMs);
            if (ImGui::BeginTable("PipelineStageBreakdownTbl", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
                ImGui::TableSetupColumn("Stage", ImGuiTableColumnFlags_WidthStretch, 0.36f);
                ImGui::TableSetupColumn("Pass Description", ImGuiTableColumnFlags_WidthStretch, 0.34f);
                ImGui::TableSetupColumn("Duration", ImGuiTableColumnFlags_WidthStretch, 0.15f);
                ImGui::TableSetupColumn("Throughput", ImGuiTableColumnFlags_WidthStretch, 0.15f);
                ImGui::TableHeadersRow();
                for (const auto& st : bd.stages) {
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0); ImGui::TextUnformatted(st.title.c_str());
                    ImGui::TableSetColumnIndex(1); ImGui::TextDisabled("%s", st.passType.c_str());
                    ImGui::TableSetColumnIndex(2); ImGui::Text("%.2f ms", st.timeMs);
                    ImGui::TableSetColumnIndex(3); ImGui::TextColored(ImVec4(0.15f, 0.86f, 0.98f, 1.0f), "%.1f MRays/s", st.mrays);
                }
                ImGui::EndTable();
            }
        }
    } else if (m_rtViewportMode == 2) {
        // PBR Material Gallery
        float availW = std::max(s(320.0f), ImGui::GetContentRegionAvail().x - s(16.0f));
        float colW = (availW - s(12.0f)) / 2.0f;
        drawImageFramed("", "assets/thumbnails/thumb_material_lineup.png",
                        "8-Archetype PBR Material BSDF Lineup", colW);
        ImGui::SameLine(0.0f, s(12.0f));
        drawImageFramed("", "assets/thumbnails/thumb_material_range.png",
                        "Roughness, Clear-Coat, Transmission & Subsurface Sweep", colW);
    } else {
        // Geometry & BVH Inspection
        float availW = std::max(s(320.0f), ImGui::GetContentRegionAvail().x - s(20.0f));
        float colW = (availW - s(20.0f)) / 3.0f;
        drawImageFramed("", "assets/thumbnails/thumb_blas_wireframe.png",
                        "BLAS / TLAS Wireframe Hierarchy", colW);
        ImGui::SameLine(0.0f, s(10.0f));
        drawImageFramed("renders/render_" + tag + "_alpha_cutout.png",
                        "assets/thumbnails/thumb_" + thumbTag + "_alpha_cutout.png",
                        "Alpha-Cutout Any-Hit / OMM Traversal", colW);
        ImGui::SameLine(0.0f, s(10.0f));
        drawImageFramed("renders/render_" + tag + "_stage1_bvh.png",
                        "assets/thumbnails/thumb_" + thumbTag + "_stage1_bvh.png",
                        "BVH Traversal Step Heatmap", colW);
    }
}

} // namespace gpubench::gui
