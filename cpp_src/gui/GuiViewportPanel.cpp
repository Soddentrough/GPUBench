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
    const BringUpColors c = getBringUpColors(m_palette);
    m_rtSceneIndex = std::clamp(m_rtSceneIndex, 0, 3);
    const SceneMeta& curScene = kScenes[m_rtSceneIndex];
    const std::string tag = curScene.tag;
    const std::string thumbTag = curScene.thumbTag;

    drawEtchedHeader("OPT-03", "RAY TRACING PARITY & STAGE INSPECTOR", c, curScene.title);
    ImGui::Spacing();

    // Top Mode Selector
    const char* modeLabels[] = {
        "SPLIT-WIPE PARITY",
        "7-STAGE PIPELINE",
        "PBR MATERIALS",
        "BVH & GEOMETRY"
    };
    for (int m = 0; m < 4; ++m) {
        float bw = ImGui::CalcTextSize(modeLabels[m]).x + s(18.0f);
        if (m > 0) sameLineIfFits(bw, s(5.0f));
        bool active = (m_rtViewportMode == m);
        std::string mId = "##RtMode_" + std::to_string(m);
        if (tacticalButton(mId.c_str(), modeLabels[m], ImVec2(0.0f, s(23.0f)), active, c)) {
            m_rtViewportMode = m;
        }
    }

    // Scene Selector Strip (applicable to modes 0, 1, 3)
    if (m_rtViewportMode != 2) {
        ImGui::Spacing();
        ImGui::TextColored(c.textMuted, "SCENE:");
        for (int si = 0; si < 4; ++si) {
            float bw = ImGui::CalcTextSize(kScenes[si].title).x + s(14.0f);
            sameLineIfFits(bw, s(4.0f));
            bool sel = (m_rtSceneIndex == si);
            std::string sId = "##RtSceneSel_" + std::to_string(si);
            if (tacticalButton(sId.c_str(), kScenes[si].title, ImVec2(0.0f, s(21.0f)), sel, c)) {
                m_rtSceneIndex = si;
            }
        }
        const float lineRemain = (ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x) -
                               ImGui::GetItemRectMax().x - s(14.0f);
        if (lineRemain > s(120.0f)) {
            ImGui::SameLine(0.0f, s(8.0f));
            ImGui::TextColored(c.textMuted, "%s", fitTextToPixelWidth(curScene.geometryNote, lineRemain).c_str());
        }
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    auto resolveTex = [&](const std::string& primaryRelPath,
                          const std::string& fallbackThumbRelPath,
                          bool& outIsLive) -> VulkanContext::TextureResource {
        VulkanContext::TextureResource tex = primaryRelPath.empty()
            ? VulkanContext::TextureResource{}
            : getOrLoadTexture(primaryRelPath);
        outIsLive = tex.isValid();
        if (!outIsLive && !fallbackThumbRelPath.empty()) {
            tex = getOrLoadTexture(fallbackThumbRelPath);
        }
        return tex;
    };

    auto drawImageFramed = [&](const std::string& primaryRelPath,
                               const std::string& fallbackThumbRelPath,
                               const char* caption,
                               float targetW) {
        bool isLiveFrame = false;
        VulkanContext::TextureResource tex = resolveTex(primaryRelPath, fallbackThumbRelPath, isLiveFrame);

        float imgH = std::round(targetW * (9.0f / 16.0f));
        if (tex.isValid() && tex.width > 0 && tex.height > 0) {
            imgH = std::round(targetW * (static_cast<float>(tex.height) / static_cast<float>(tex.width)));
        }

        ImGui::BeginGroup();
        ImDrawList* dl = ImGui::GetWindowDrawList();
        ImVec2 p0 = ImGui::GetCursorScreenPos();
        ImVec2 p1(p0.x + targetW, p0.y + imgH);
        dl->AddRectFilled(p0, p1, toU32(c.bgRecessed), 0.0f);

        if (tex.isValid()) {
            ImGui::Image(reinterpret_cast<ImTextureID>(tex.descriptorSet), ImVec2(targetW, imgH));
            dl->AddRect(p0, p1, toU32(c.borderOuter), 0.0f, 0, 1.0f);
            drawCornerReticles(dl, p0, p1, toU32(isLiveFrame ? c.okColor : c.phosphorMid), s(8.0f));
            if (isLiveFrame) {
                ImGui::TextColored(c.okColor, "[LIVE %dx%d%s] %s",
                                   tex.width, tex.height,
                                   tex.borrowsExternalImage ? " GPU VRAM" : " PNG",
                                   caption);
            } else {
                ImGui::TextColored(c.cyanAccent, "[REF] %s", caption);
            }
        } else {
            ImGui::Dummy(ImVec2(targetW, imgH));
            dl->AddRect(p0, p1, toU32(c.borderOuter), 0.0f, 0, 1.0f);
            drawCornerReticles(dl, p0, p1, toU32(c.textMuted), s(8.0f));
            ImGui::TextColored(c.textMuted, "%s (Run RayScheduling to capture)", caption);
        }
        ImGui::EndGroup();
    };

    if (m_rtViewportMode == 0) {
        // Scenes & Visual Parity Mode
        const char* parityViews[] = {
            "SPLIT-WIPE COMPARATOR",
            "DGC WAVEFRONT",
            "MEGAKERNEL",
            "3-WAY + 10x HEATMAP",
            "128-LIGHT SPATIAL BINNING"
        };
        for (int pv = 0; pv < 5; ++pv) {
            float bw = ImGui::CalcTextSize(parityViews[pv]).x + s(16.0f);
            if (pv > 0) sameLineIfFits(bw, s(4.0f));
            bool active = (m_rtParityViewMode == pv);
            std::string pId = "##RtPv_" + std::to_string(pv);
            if (tacticalButton(pId.c_str(), parityViews[pv], ImVec2(0.0f, s(21.0f)), active, c)) {
                m_rtParityViewMode = pv;
            }
        }
        ImGui::Spacing();

        std::string profileJsonStr;
        const bool isPtPair = (m_rtParityViewMode == 0 && m_rtComparatorPair == 2);
        std::string jsonKey = isPtPair
            ? ("render_" + tag + "_pt16_profile.json")
            : ("render_" + tag + "_profile.json");
        ParsedParityProfile prof{};
        if (loadViewportJsonContent(m_vulkanContext, jsonKey, profileJsonStr)) {
            prof = parseParityProfileJson(profileJsonStr);
        }

        if (prof.valid) {
            double speedup = (prof.tradMs > 0.0 && prof.workMs > 0.0) ? (prof.tradMs / prof.workMs) : 1.0;
            ImVec4 statusCol = (prof.psnr >= 45.0) ? c.okColor : c.errColor;
            pushMonoFont();
            ImGui::TextColored(statusCol,
                               "%s  |  PSNR: %.2f dB  |  EXACT: %.2f%%  |  MEGAKERNEL: %.2f ms (%.0f MR/s)  |  DGC: %.2f ms (%.0f MR/s, %.2fx)",
                               prof.status.empty() ? "PARITY TELEMETRY" : prof.status.c_str(),
                               prof.psnr, prof.exactPct,
                               prof.tradMs, prof.tradMrays,
                               prof.workMs, prof.workMrays, speedup);
            popMonoFont();
            ImGui::Spacing();
        }

        float availW = std::max(s(320.0f), ImGui::GetContentRegionAvail().x - s(6.0f));
        float availH = std::max(s(220.0f), ImGui::GetContentRegionAvail().y - s(6.0f));

        if (m_rtParityViewMode == 0) {
            // Interactive Split-Wipe Reticle Comparator
            const char* pairLabels[] = {
                "DGC vs MEGAKERNEL",
                "DGC vs STAGE-1 BVH",
                "16 SPP vs 1 SPP",
                "FINAL PBR vs NORMALS"
            };
            m_rtComparatorPair = std::clamp(m_rtComparatorPair, 0, 3);
            for (int pr = 0; pr < 4; ++pr) {
                float bw = ImGui::CalcTextSize(pairLabels[pr]).x + s(14.0f);
                if (pr > 0) sameLineIfFits(bw, s(4.0f));
                std::string prId = "##RtCmpPair_" + std::to_string(pr);
                if (tacticalButton(prId.c_str(), pairLabels[pr], ImVec2(0.0f, s(20.0f)),
                                   m_rtComparatorPair == pr, c)) {
                    m_rtComparatorPair = pr;
                }
            }
            sameLineIfFits(s(95.0f), s(10.0f));
            if (tacticalButton("##RtResetWipe", "CENTER 50%", ImVec2(s(88.0f), s(20.0f)), false, c)) {
                m_rtSplitRatio = 0.5f;
            }

            std::string leftRel, leftThumb, leftBadge;
            std::string rightRel, rightThumb, rightBadge;
            switch (m_rtComparatorPair) {
            case 1:
                leftRel    = "renders/render_" + tag + "_worklist_dgc.png";
                leftThumb  = "assets/thumbnails/thumb_" + thumbTag + "_stage7_final.png";
                leftBadge  = "LEFT: DGC FINAL";
                rightRel   = "renders/render_" + tag + "_stage1_bvh.png";
                rightThumb = "assets/thumbnails/thumb_" + thumbTag + "_stage1_bvh.png";
                rightBadge = "RIGHT: STAGE-1 BVH";
                break;
            case 2:
                leftRel    = "renders/render_" + tag + "_pathtracing_16spp_worklist.png";
                leftThumb  = "assets/thumbnails/thumb_" + thumbTag + "_pathtracing_16spp.png";
                leftBadge  = "LEFT: 16 SPP CONVERGED";
                rightRel   = "renders/render_" + tag + "_pathtracing_1spp_worklist.png";
                rightThumb = "assets/thumbnails/thumb_" + thumbTag + "_pathtracing_1spp.png";
                rightBadge = "RIGHT: 1 SPP RAW";
                break;
            case 3:
                leftRel    = "renders/render_" + tag + "_stage7_final.png";
                leftThumb  = "assets/thumbnails/thumb_" + thumbTag + "_stage7_final.png";
                leftBadge  = "LEFT: 7. FINAL PBR";
                rightRel   = "renders/render_" + tag + "_stage2_primary.png";
                rightThumb = "assets/thumbnails/thumb_" + thumbTag + "_stage2_primary.png";
                rightBadge = "RIGHT: 2. G-BUF NORMALS";
                break;
            case 0:
            default:
                leftRel    = "renders/render_" + tag + "_worklist_dgc.png";
                leftThumb  = "assets/thumbnails/thumb_" + thumbTag + "_stage7_final.png";
                leftBadge  = "LEFT: DGC WAVEFRONT";
                rightRel   = "renders/render_" + tag + "_traditional_megakernel.png";
                rightThumb = "assets/thumbnails/thumb_" + thumbTag + "_stage7_final.png";
                rightBadge = "RIGHT: MEGAKERNEL";
                break;
            }

            bool leftLive = false, rightLive = false;
            VulkanContext::TextureResource texL = resolveTex(leftRel, leftThumb, leftLive);
            VulkanContext::TextureResource texR = resolveTex(rightRel, rightThumb, rightLive);

            float aspect = 16.0f / 9.0f;
            if (texL.isValid() && texL.width > 0 && texL.height > 0) {
                aspect = static_cast<float>(texL.width) / static_cast<float>(texL.height);
            } else if (texR.isValid() && texR.width > 0 && texR.height > 0) {
                aspect = static_cast<float>(texR.width) / static_cast<float>(texR.height);
            }

            ImGui::Spacing();
            availW = std::max(s(320.0f), ImGui::GetContentRegionAvail().x - s(4.0f));
            availH = std::max(s(180.0f), ImGui::GetContentRegionAvail().y - s(28.0f));
            float canvasW = std::min({availW, availH * aspect, s(1060.0f)});
            float canvasH = std::round(canvasW / std::max(0.5f, aspect));

            ImVec2 p0 = ImGui::GetCursorScreenPos();
            ImVec2 p1(p0.x + canvasW, p0.y + canvasH);
            ImGui::InvisibleButton("##RtSplitWipeCanvas", ImVec2(canvasW, canvasH));
            bool hovered = ImGui::IsItemHovered();
            if (ImGui::IsItemActive() && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
                m_rtSplitRatio = std::clamp((ImGui::GetIO().MousePos.x - p0.x) / std::max(1.0f, canvasW), 0.02f, 0.98f);
            }
            if (hovered && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                m_rtSplitRatio = 0.5f;
            }

            m_rtSplitRatio = std::clamp(m_rtSplitRatio, 0.02f, 0.98f);
            float splitX = std::round(p0.x + canvasW * m_rtSplitRatio);

            ImDrawList* dl = ImGui::GetWindowDrawList();
            dl->AddRectFilled(p0, p1, toU32(c.bgRecessed), 0.0f);

            // Left Halves
            dl->PushClipRect(p0, ImVec2(splitX, p1.y), true);
            if (texL.isValid()) {
                dl->AddImage(reinterpret_cast<ImTextureID>(texL.descriptorSet), p0, p1);
            } else {
                dl->AddText(ImVec2(p0.x + s(16.0f), p0.y + canvasH * 0.5f), toU32(c.textMuted), "LEFT CHANNEL: NO IMAGE");
            }
            {
                std::string lFull = leftBadge + (leftLive ? " [LIVE]" : " [REF]");
                ImVec2 bSz = ImGui::CalcTextSize(lFull.c_str());
                ImVec2 bp0(p0.x + s(10.0f), p0.y + s(10.0f));
                ImVec2 bp1(bp0.x + bSz.x + s(12.0f), bp0.y + bSz.y + s(6.0f));
                dl->AddRectFilled(bp0, bp1, toU32(c.bgVoid, 0.86f), 0.0f);
                dl->AddRect(bp0, bp1, toU32(leftLive ? c.okColor : c.phosphorHi), 0.0f, 0, 1.0f);
                dl->AddText(ImVec2(bp0.x + s(6.0f), bp0.y + s(3.0f)), toU32(leftLive ? c.okColor : c.phosphorHi), lFull.c_str());
            }
            dl->PopClipRect();

            // Right Halves
            dl->PushClipRect(ImVec2(splitX, p0.y), p1, true);
            if (texR.isValid()) {
                dl->AddImage(reinterpret_cast<ImTextureID>(texR.descriptorSet), p0, p1);
            } else {
                dl->AddText(ImVec2(splitX + s(16.0f), p0.y + canvasH * 0.5f), toU32(c.textMuted), "RIGHT CHANNEL: NO IMAGE");
            }
            {
                std::string rFull = rightBadge + (rightLive ? " [LIVE]" : " [REF]");
                ImVec2 bSz = ImGui::CalcTextSize(rFull.c_str());
                ImVec2 bp1(p1.x - s(10.0f), p0.y + s(10.0f) + bSz.y + s(6.0f));
                ImVec2 bp0(bp1.x - (bSz.x + s(12.0f)), p0.y + s(10.0f));
                dl->AddRectFilled(bp0, bp1, toU32(c.bgVoid, 0.86f), 0.0f);
                dl->AddRect(bp0, bp1, toU32(c.cyanAccent), 0.0f, 0, 1.0f);
                dl->AddText(ImVec2(bp0.x + s(6.0f), bp0.y + s(3.0f)), toU32(c.cyanAccent), rFull.c_str());
            }
            dl->PopClipRect();

            // Outer frame + Corner reticles + Vertical wipe hairline & center reticle handle
            dl->AddRect(p0, p1, toU32(c.borderOuter), 0.0f, 0, 1.0f);
            drawCornerReticles(dl, p0, p1, toU32(c.phosphorHi), s(10.0f));

            dl->AddLine(ImVec2(splitX, p0.y), ImVec2(splitX, p1.y), toU32(c.phosphorHi), 2.0f);
            float tickLen = s(8.0f);
            dl->AddTriangleFilled(
                ImVec2(splitX - s(5.0f), p0.y),
                ImVec2(splitX + s(5.0f), p0.y),
                ImVec2(splitX, p0.y + tickLen),
                toU32(c.phosphorHi));
            dl->AddTriangleFilled(
                ImVec2(splitX - s(5.0f), p1.y),
                ImVec2(splitX + s(5.0f), p1.y),
                ImVec2(splitX, p1.y - tickLen),
                toU32(c.phosphorHi));

            float midY = p0.y + canvasH * 0.5f;
            float knobW = s(22.0f);
            float knobH = s(14.0f);
            ImVec2 kp0(splitX - knobW, midY - knobH);
            ImVec2 kp1(splitX + knobW, midY + knobH);
            dl->AddRectFilled(kp0, kp1, toU32(hovered ? c.bgActive : c.bgVoid, 0.92f), 0.0f);
            dl->AddRect(kp0, kp1, toU32(c.phosphorHi), 0.0f, 0, 1.5f);
            const char* handleLbl = "< || >";
            ImVec2 hSz = ImGui::CalcTextSize(handleLbl);
            dl->AddText(ImVec2(splitX - hSz.x * 0.5f, midY - hSz.y * 0.5f), toU32(c.phosphorHi), handleLbl);

            // Live Wipe Telemetry Strip below Canvas
            pushMonoFont();
            char footerBuf[256];
            if (prof.valid && prof.tradMs > 0.0 && prof.workMs > 0.0) {
                std::snprintf(footerBuf, sizeof(footerBuf),
                              "WIPE %.0f%%  |  DELTA: %.2fx FASTER (DGC %.2fms vs MEGA %.2fms)  |  PSNR %.2f dB  |  DRAG HANDLE OR DOUBLE-CLICK CENTER",
                              m_rtSplitRatio * 100.0f, prof.tradMs / prof.workMs, prof.workMs, prof.tradMs, prof.psnr);
            } else {
                std::snprintf(footerBuf, sizeof(footerBuf),
                              "WIPE %.0f%%  |  %s (%dx%d)  <->  %s (%dx%d)  |  DRAG RETICLE TO COMPARE",
                              m_rtSplitRatio * 100.0f,
                              leftBadge.c_str(), texL.width, texL.height,
                              rightBadge.c_str(), texR.width, texR.height);
            }
            ImGui::TextColored(c.textSecondary, "%s", fitTextToPixelWidth(footerBuf, availW).c_str());
            popMonoFont();
        } else if (m_rtParityViewMode == 1) {
            drawImageFramed("renders/render_" + tag + "_worklist_dgc.png",
                            "assets/thumbnails/thumb_" + thumbTag + "_stage7_final.png",
                            "DGC Wavefront Material-Sorted Render",
                            std::min(availW, s(980.0f)));
        } else if (m_rtParityViewMode == 2) {
            drawImageFramed("renders/render_" + tag + "_traditional_megakernel.png",
                            "assets/thumbnails/thumb_" + thumbTag + "_stage7_final.png",
                            "Traditional Monolithic Megakernel Render",
                            std::min(availW, s(980.0f)));
        } else if (m_rtParityViewMode == 3) {
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
            "1. BVH HEATMAP", "2. G-BUF NORMALS", "3. SUN SHADOWS", "4. RTAO",
            "5. DIRECT PBR", "6. INDIRECT GI", "7. FINAL BEAUTY"
        };

        m_rtPassIndex = std::clamp(m_rtPassIndex, 0, 6);
        for (int si = 0; si < 7; ++si) {
            float bw = ImGui::CalcTextSize(stageLabels[si]).x + s(14.0f);
            if (si > 0) sameLineIfFits(bw, s(4.0f));
            bool active = (m_rtPassIndex == si);
            std::string stId = "##RtStage_" + std::to_string(si);
            if (tacticalButton(stId.c_str(), stageLabels[si], ImVec2(0.0f, s(21.0f)), active, c)) {
                m_rtPassIndex = si;
            }
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
                        std::min(availW, s(760.0f)));

        if (bd.valid && !bd.stages.empty()) {
            ImGui::Spacing();
            ImGui::TextColored(c.textMuted, "Scene Triangles: %llu  |  BVH Build: %.2f ms",
                               static_cast<unsigned long long>(bd.triangles), bd.bvhBuildMs);
            if (ImGui::BeginTable("PipelineStageBreakdownTbl", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
                ImGui::TableSetupColumn("Stage", ImGuiTableColumnFlags_WidthStretch, 0.36f);
                ImGui::TableSetupColumn("Pass Description", ImGuiTableColumnFlags_WidthStretch, 0.34f);
                ImGui::TableSetupColumn("Duration", ImGuiTableColumnFlags_WidthStretch, 0.15f);
                ImGui::TableSetupColumn("Throughput", ImGuiTableColumnFlags_WidthStretch, 0.15f);
                ImGui::TableHeadersRow();
                for (const auto& st : bd.stages) {
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0); ImGui::TextColored(c.textPrimary, "%s", st.title.c_str());
                    ImGui::TableSetColumnIndex(1); ImGui::TextColored(c.textMuted, "%s", st.passType.c_str());
                    ImGui::TableSetColumnIndex(2); pushMonoFont(); ImGui::TextColored(c.textSecondary, "%.2f ms", st.timeMs); popMonoFont();
                    ImGui::TableSetColumnIndex(3); pushMonoFont(); ImGui::TextColored(c.cyanAccent, "%.1f MRays/s", st.mrays); popMonoFont();
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
