#include "GuiApp.h"
#include "GuiUiHelpers.h"
#include "core/VulkanContext.h"
#include "yyjson.h"

#include <SDL3/SDL.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace gpubench::gui {

namespace {

struct SceneMeta {
    const char* title;
    const char* tag;      // render_<tag>_*
    const char* shortCap; // CAPTURE <shortCap>
    const char* geometryNote;
};

constexpr SceneMeta kScenes[4] = {
    {"Showroom Studio",    "showroom", "SHOWROOM", "High-poly automotive clear-coat studio with area lights & glossy reflections"},
    {"Indoor Atrium",      "indoor",   "INDOOR",   "First-party MIT/CC0 Renaissance Colonnade Atrium (426K triangles, multi-bounce GI)"},
    {"Open-World Forest",  "forest",   "FOREST",   "Dense instanced alpha-cutout foliage canopy & bark geometry"},
    {"Outdoor Landscape",  "outdoor",  "OUTDOOR",  "Procedural eroded heightfield terrain with directional sun & sky atmosphere"}
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
#ifdef _WIN32
    if (const char* localApp = std::getenv("LOCALAPPDATA")) {
        if (*localApp) roots.push_back(std::string(localApp) + "/gpubench/renders/");
    }
#else
    if (const char* xdg = std::getenv("XDG_CACHE_HOME")) {
        if (*xdg) roots.push_back(std::string(xdg) + "/gpubench/renders/");
    }
    if (const char* home = std::getenv("HOME")) {
        if (*home) roots.push_back(std::string(home) + "/.cache/gpubench/renders/");
    }
#endif

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
    bool stageTimingMeasured{false};
    double tradFps{0.0};
    double tradMrays{0.0};
    double tradMs{0.0};
    double tradBvhMs{0.0};
    double tradShadingMs{0.0};
    double workFps{0.0};
    double workMrays{0.0};
    double workMs{0.0};
    double workBvhMs{0.0};
    double workCompactionMs{0.0};
    double workShadingMs{0.0};
    bool hasLegacyIndirect{false};
    double legacyFps{0.0};
    double legacyMrays{0.0};
    double legacyMs{0.0};
    double legacyBvhMs{0.0};
    double legacyCompactionMs{0.0};
    double legacyShadingMs{0.0};
    bool hasMaterialShading{false};
    double matMkMhits{0.0};
    double matLegacyMhits{0.0};
    double matDgcMhits{0.0};
    double matDgcVsMk{1.0};
    double matDgcVsLegacy{1.0};
    bool hasDgcTelemetry{false};
    uint32_t activeSequences{0};
    uint32_t maxSequences{8};
    bool usesExecutionSet{false};
    uint32_t mkVgprs{0};
    uint32_t mkSgprs{0};
    uint32_t mkMaxWaves{0};
    uint32_t mkCodeBytes{0};
    uint32_t dgcVgprsMin{0};
    uint32_t dgcVgprsMax{0};
    double dgcVgprsAvg{0.0};
    uint32_t dgcWavesMin{0};
    uint32_t dgcWavesMax{0};
    uint32_t dgcCodeBytesMin{0};
    uint32_t dgcCodeBytesMax{0};
    double dgcCodeBytesAvg{0.0};
    uint32_t classifyVgprs{0};
    uint32_t classifyMaxWaves{0};
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
        if (yyjson_val* v = yyjson_obj_get(root, "stage_timing_measured"); yyjson_is_bool(v)) {
            p.stageTimingMeasured = yyjson_get_bool(v);
        }

        if (yyjson_val* trad = yyjson_obj_get(root, "traditional"); yyjson_is_obj(trad)) {
            if (yyjson_val* v = yyjson_obj_get(trad, "fps"); yyjson_is_num(v)) p.tradFps = yyjson_get_num(v);
            if (yyjson_val* v = yyjson_obj_get(trad, "mrays"); yyjson_is_num(v)) p.tradMrays = yyjson_get_num(v);
            if (yyjson_val* v = yyjson_obj_get(trad, "frame_ms"); yyjson_is_num(v)) p.tradMs = yyjson_get_num(v);
            if (yyjson_val* v = yyjson_obj_get(trad, "bvh_ms"); yyjson_is_num(v)) p.tradBvhMs = yyjson_get_num(v);
            if (yyjson_val* v = yyjson_obj_get(trad, "shading_ms"); yyjson_is_num(v)) p.tradShadingMs = yyjson_get_num(v);
        }
        if (yyjson_val* leg = yyjson_obj_get(root, "legacy_indirect"); yyjson_is_obj(leg)) {
            p.hasLegacyIndirect = true;
            if (yyjson_val* v = yyjson_obj_get(leg, "fps"); yyjson_is_num(v)) p.legacyFps = yyjson_get_num(v);
            if (yyjson_val* v = yyjson_obj_get(leg, "mrays"); yyjson_is_num(v)) p.legacyMrays = yyjson_get_num(v);
            if (yyjson_val* v = yyjson_obj_get(leg, "frame_ms"); yyjson_is_num(v)) p.legacyMs = yyjson_get_num(v);
            if (yyjson_val* v = yyjson_obj_get(leg, "bvh_ms"); yyjson_is_num(v)) p.legacyBvhMs = yyjson_get_num(v);
            if (yyjson_val* v = yyjson_obj_get(leg, "compaction_ms"); yyjson_is_num(v)) p.legacyCompactionMs = yyjson_get_num(v);
            if (yyjson_val* v = yyjson_obj_get(leg, "shading_ms"); yyjson_is_num(v)) p.legacyShadingMs = yyjson_get_num(v);
        }
        if (yyjson_val* work = yyjson_obj_get(root, "worklist"); yyjson_is_obj(work)) {
            if (yyjson_val* v = yyjson_obj_get(work, "fps"); yyjson_is_num(v)) p.workFps = yyjson_get_num(v);
            if (yyjson_val* v = yyjson_obj_get(work, "mrays"); yyjson_is_num(v)) p.workMrays = yyjson_get_num(v);
            if (yyjson_val* v = yyjson_obj_get(work, "frame_ms"); yyjson_is_num(v)) p.workMs = yyjson_get_num(v);
            if (yyjson_val* v = yyjson_obj_get(work, "bvh_ms"); yyjson_is_num(v)) p.workBvhMs = yyjson_get_num(v);
            if (yyjson_val* v = yyjson_obj_get(work, "compaction_ms"); yyjson_is_num(v)) p.workCompactionMs = yyjson_get_num(v);
            if (yyjson_val* v = yyjson_obj_get(work, "shading_ms"); yyjson_is_num(v)) p.workShadingMs = yyjson_get_num(v);
        }
        if (yyjson_val* mat = yyjson_obj_get(root, "material_shading"); yyjson_is_obj(mat)) {
            p.hasMaterialShading = true;
            if (yyjson_val* v = yyjson_obj_get(mat, "mk_mhits"); yyjson_is_num(v)) p.matMkMhits = yyjson_get_num(v);
            if (yyjson_val* v = yyjson_obj_get(mat, "legacy_indirect_mhits"); yyjson_is_num(v)) p.matLegacyMhits = yyjson_get_num(v);
            if (yyjson_val* v = yyjson_obj_get(mat, "dgc_mhits"); yyjson_is_num(v)) p.matDgcMhits = yyjson_get_num(v);
            if (yyjson_val* v = yyjson_obj_get(mat, "dgc_vs_mk_speedup"); yyjson_is_num(v)) p.matDgcVsMk = yyjson_get_num(v);
            if (yyjson_val* v = yyjson_obj_get(mat, "dgc_vs_legacy_speedup"); yyjson_is_num(v)) p.matDgcVsLegacy = yyjson_get_num(v);
        }
        if (yyjson_val* tel = yyjson_obj_get(root, "dgc_telemetry"); yyjson_is_obj(tel)) {
            p.hasDgcTelemetry = true;
            if (yyjson_val* v = yyjson_obj_get(tel, "active_sequences"); yyjson_is_uint(v)) p.activeSequences = static_cast<uint32_t>(yyjson_get_uint(v));
            if (yyjson_val* v = yyjson_obj_get(tel, "max_sequences"); yyjson_is_uint(v)) p.maxSequences = static_cast<uint32_t>(yyjson_get_uint(v));
            if (yyjson_val* v = yyjson_obj_get(tel, "uses_execution_set"); yyjson_is_bool(v)) p.usesExecutionSet = yyjson_get_bool(v);
            if (yyjson_val* v = yyjson_obj_get(tel, "mk_vgprs"); yyjson_is_uint(v)) p.mkVgprs = static_cast<uint32_t>(yyjson_get_uint(v));
            if (yyjson_val* v = yyjson_obj_get(tel, "mk_sgprs"); yyjson_is_uint(v)) p.mkSgprs = static_cast<uint32_t>(yyjson_get_uint(v));
            if (yyjson_val* v = yyjson_obj_get(tel, "mk_max_waves"); yyjson_is_uint(v)) p.mkMaxWaves = static_cast<uint32_t>(yyjson_get_uint(v));
            if (yyjson_val* v = yyjson_obj_get(tel, "mk_code_bytes"); yyjson_is_uint(v)) p.mkCodeBytes = static_cast<uint32_t>(yyjson_get_uint(v));
            if (yyjson_val* v = yyjson_obj_get(tel, "dgc_vgprs_min"); yyjson_is_uint(v)) p.dgcVgprsMin = static_cast<uint32_t>(yyjson_get_uint(v));
            if (yyjson_val* v = yyjson_obj_get(tel, "dgc_vgprs_avg"); yyjson_is_num(v)) p.dgcVgprsAvg = yyjson_get_num(v);
            if (yyjson_val* v = yyjson_obj_get(tel, "dgc_vgprs_max"); yyjson_is_uint(v)) p.dgcVgprsMax = static_cast<uint32_t>(yyjson_get_uint(v));
            if (yyjson_val* v = yyjson_obj_get(tel, "dgc_max_waves_min"); yyjson_is_uint(v)) p.dgcWavesMin = static_cast<uint32_t>(yyjson_get_uint(v));
            if (yyjson_val* v = yyjson_obj_get(tel, "dgc_max_waves_max"); yyjson_is_uint(v)) p.dgcWavesMax = static_cast<uint32_t>(yyjson_get_uint(v));
            if (yyjson_val* v = yyjson_obj_get(tel, "dgc_code_bytes_min"); yyjson_is_uint(v)) p.dgcCodeBytesMin = static_cast<uint32_t>(yyjson_get_uint(v));
            if (yyjson_val* v = yyjson_obj_get(tel, "dgc_code_bytes_avg"); yyjson_is_num(v)) p.dgcCodeBytesAvg = yyjson_get_num(v);
            if (yyjson_val* v = yyjson_obj_get(tel, "dgc_code_bytes_max"); yyjson_is_uint(v)) p.dgcCodeBytesMax = static_cast<uint32_t>(yyjson_get_uint(v));
            if (yyjson_val* v = yyjson_obj_get(tel, "classify_vgprs"); yyjson_is_uint(v)) p.classifyVgprs = static_cast<uint32_t>(yyjson_get_uint(v));
            if (yyjson_val* v = yyjson_obj_get(tel, "classify_max_waves"); yyjson_is_uint(v)) p.classifyMaxWaves = static_cast<uint32_t>(yyjson_get_uint(v));
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

bool GuiApp::getCachedViewportJson(const std::string& bareJsonName, std::string& outJson) {
    outJson.clear();
    const auto now = std::chrono::steady_clock::now();
    auto it = m_viewportJsonCache.find(bareJsonName);
    if (it != m_viewportJsonCache.end() &&
        it->second.epoch == m_viewportJsonEpoch &&
        (it->second.found || (now - it->second.lastChecked) < std::chrono::seconds(1))) {
        if (it->second.found) {
            outJson = it->second.content;
            return true;
        }
        return false;
    }

    CachedViewportJsonEntry entry{};
    entry.epoch = m_viewportJsonEpoch;
    entry.lastChecked = now;
    entry.found = loadViewportJsonContent(m_vulkanContext, bareJsonName, entry.content);
    outJson = entry.content;
    m_viewportJsonCache[bareJsonName] = std::move(entry);
    return !outJson.empty();
}

void GuiApp::renderRayTracingViewport() {
    const BringUpColors c = getBringUpColors(m_palette);
    m_rtSceneIndex = std::clamp(m_rtSceneIndex, 0, 3);
    const SceneMeta& curScene = kScenes[m_rtSceneIndex];
    const std::string tag = curScene.tag;
    const bool isRunning = (m_execState == ExecutionState::Running);

    std::string sceneUpper = curScene.title;
    std::transform(sceneUpper.begin(), sceneUpper.end(), sceneUpper.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::toupper(ch)); });

    drawEtchedHeader("OPT-03", "RAY TRACING PARITY & STAGE INSPECTOR", c, curScene.title);
    ImGui::Spacing();

    // Top Mode Selector
    const char* modeLabels[] = {
        "SPLIT-WIPE PARITY",
        "7-STAGE PIPELINE",
        "PATH TRACING & GI",
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

    // Scene Selector & Direct Capture Toolbar (applicable across all 4 modes)
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

    if (isRunning) ImGui::BeginDisabled();
    {
        std::string capSceneLbl = std::string("CAPTURE ") + curScene.shortCap;
        float capW = ImGui::CalcTextSize(capSceneLbl.c_str()).x + s(16.0f);
        sameLineIfFits(capW, s(10.0f));
        if (tacticalButton("##RtCaptureScene", capSceneLbl.c_str(), ImVec2(0.0f, s(21.0f)), false, c)) {
            startSceneVerificationCapture(tag);
        }
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            ImGui::SetTooltip("Render & verify actual GPU output for %s (stays on F3)", curScene.title);
        }

        const char* capAllLbl = "CAPTURE ALL 4";
        float capAllW = ImGui::CalcTextSize(capAllLbl).x + s(14.0f);
        sameLineIfFits(capAllW, s(4.0f));
        if (tacticalButton("##RtCaptureAll", capAllLbl, ImVec2(0.0f, s(21.0f)), false, c)) {
            startSceneVerificationCapture("all");
        }
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            ImGui::SetTooltip("Render & verify actual GPU output across all 4 scenes");
        }
    }
    if (isRunning) ImGui::EndDisabled();

    const float lineRemain = (ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x) -
                           ImGui::GetItemRectMax().x - s(14.0f);
    if (lineRemain > s(140.0f)) {
        ImGui::SameLine(0.0f, s(8.0f));
        ImGui::TextColored(c.textMuted, "%s", fitTextToPixelWidth(curScene.geometryNote, lineRemain).c_str());
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    auto resolveTex = [&](const std::string& primaryRelPath) -> VulkanContext::TextureResource {
        if (primaryRelPath.empty() || isRunning) {
            return VulkanContext::TextureResource{};
        }
        return getOrLoadTexture(primaryRelPath);
    };

    int emptySlateSeq = 0;
    auto drawUnrenderedSlate = [&](ImDrawList* dl,
                                   ImVec2 p0, ImVec2 p1,
                                   float canvasW, float canvasH,
                                   bool showCaptureAction) {
        dl->AddRectFilled(p0, p1, toU32(c.bgRecessed), 0.0f);
        // Subtle diagnostic crosshair lines
        const ImU32 gridCol = toU32(c.gridMinor);
        dl->AddLine(ImVec2(p0.x + canvasW * 0.5f, p0.y), ImVec2(p0.x + canvasW * 0.5f, p1.y), gridCol, 1.0f);
        dl->AddLine(ImVec2(p0.x, p0.y + canvasH * 0.5f), ImVec2(p1.x, p0.y + canvasH * 0.5f), gridCol, 1.0f);
        dl->AddRect(p0, p1, toU32(c.borderOuter), 0.0f, 0, 1.0f);
        drawCornerReticles(dl, p0, p1, toU32(isRunning ? c.phosphorMid : c.textMuted), s(10.0f));

        // Reserve the outer canvas extent cleanly before placing optional interactive button inside
        ImGui::SetCursorScreenPos(p0);
        ImGui::Dummy(ImVec2(canvasW, canvasH));

        const float padX = s(16.0f);
        const float maxTextW = std::max(s(80.0f), canvasW - padX * 2.0f);
        const float centerX = p0.x + canvasW * 0.5f;
        const float centerY = p0.y + canvasH * 0.5f;

        if (isRunning) {
            const std::string title = fitTextToPixelWidth("RENDERING VERIFICATION OUTPUT...", maxTextW);
            ImVec2 tSz = ImGui::CalcTextSize(title.c_str());
            dl->AddText(ImVec2(centerX - tSz.x * 0.5f, centerY - s(18.0f)), toU32(c.phosphorHi), title.c_str());
            const std::string statusMsg = getStatusMessage();
            if (!statusMsg.empty()) {
                const std::string sub = fitTextToPixelWidth(statusMsg, maxTextW);
                ImVec2 sSz = ImGui::CalcTextSize(sub.c_str());
                dl->AddText(ImVec2(centerX - sSz.x * 0.5f, centerY + s(4.0f)), toU32(c.textSecondary), sub.c_str());
            }
        } else {
            const std::string title = fitTextToPixelWidth("NO RENDERED OUTPUT CAPTURED FOR " + sceneUpper, maxTextW);
            ImVec2 tSz = ImGui::CalcTextSize(title.c_str());
            float titleOffsetY = (showCaptureAction && canvasH >= s(140.0f)) ? s(-34.0f) : s(-18.0f);
            dl->AddText(ImVec2(centerX - tSz.x * 0.5f, centerY + titleOffsetY), toU32(c.phosphorHi), title.c_str());

            const std::string sub = fitTextToPixelWidth(
                "Run RayScheduling on this scene to capture and verify actual GPU output.", maxTextW);
            ImVec2 sSz = ImGui::CalcTextSize(sub.c_str());
            dl->AddText(ImVec2(centerX - sSz.x * 0.5f, centerY + titleOffsetY + tSz.y + s(6.0f)),
                        toU32(c.textMuted), sub.c_str());

            if (showCaptureAction && canvasW >= s(180.0f) && canvasH >= s(110.0f)) {
                const std::string btnLbl = "CAPTURE " + std::string(curScene.shortCap) + " RENDERS";
                float btnW = std::min(maxTextW, ImGui::CalcTextSize(btnLbl.c_str()).x + s(24.0f));
                float btnH = s(24.0f);
                ImVec2 btnPos(std::round(centerX - btnW * 0.5f), std::round(centerY + s(12.0f)));
                ImGui::SetCursorScreenPos(btnPos);
                std::string btnId = "##RtEmptyCap_" + std::to_string(++emptySlateSeq);
                if (tacticalButton(btnId.c_str(), btnLbl.c_str(), ImVec2(btnW, btnH), true, c)) {
                    startSceneVerificationCapture(tag);
                }
            }
        }

        // Restore cursor to the bottom-left of [p0, p1] so ImGui item bounds remain consistent
        ImGui::SetCursorScreenPos(ImVec2(p0.x, p1.y));
        ImGui::Dummy(ImVec2(canvasW, 0.0f));
    };

    auto drawTexFramed = [&](const VulkanContext::TextureResource& tex,
                             const char* caption,
                             float targetW,
                             bool showCaptureAction) {
        float imgH = std::round(targetW * (9.0f / 16.0f));
        if (tex.isValid() && tex.width > 0 && tex.height > 0) {
            imgH = std::round(targetW * (static_cast<float>(tex.height) / static_cast<float>(tex.width)));
        }

        ImGui::BeginGroup();
        ImDrawList* dl = ImGui::GetWindowDrawList();
        ImVec2 p0 = ImGui::GetCursorScreenPos();
        ImVec2 p1(p0.x + targetW, p0.y + imgH);

        if (tex.isValid()) {
            dl->AddRectFilled(p0, p1, toU32(c.bgRecessed), 0.0f);
            ImGui::Image(reinterpret_cast<ImTextureID>(tex.descriptorSet), ImVec2(targetW, imgH));
            dl->AddRect(p0, p1, toU32(c.borderOuter), 0.0f, 0, 1.0f);
            const ImVec4 reticleCol = tex.borrowsExternalImage ? c.okColor : c.cyanAccent;
            drawCornerReticles(dl, p0, p1, toU32(reticleCol), s(8.0f));
            if (tex.borrowsExternalImage) {
                ImGui::TextColored(c.okColor, "[LIVE VRAM %dx%d] %s",
                                   tex.width, tex.height, caption);
            } else {
                ImGui::TextColored(c.cyanAccent, "[SAVED RENDER %dx%d PNG] %s",
                                   tex.width, tex.height, caption);
            }
        } else {
            drawUnrenderedSlate(dl, p0, p1, targetW, imgH, showCaptureAction);
            ImGui::TextColored(c.textMuted, "[NO CAPTURE] %s", caption);
        }
        ImGui::EndGroup();
    };

    auto drawImageFramed = [&](const std::string& relPath,
                               const char* caption,
                               float targetW,
                               bool showCaptureAction) {
        VulkanContext::TextureResource tex = resolveTex(relPath);
        drawTexFramed(tex, caption, targetW, showCaptureAction);
    };

    auto renderRatioBadgeText = [&](double tradMs, double workMs, char* outBuf, size_t outSz) {
        const double ratio = (tradMs > 0.0 && workMs > 0.0) ? (tradMs / workMs) : 1.0;
        if (ratio >= 1.005) {
            std::snprintf(outBuf, outSz, "%.2fx FASTER", ratio);
        } else if (ratio <= 0.995) {
            std::snprintf(outBuf, outSz, "%.2fx DGC SLOWER", ratio);
        } else {
            std::snprintf(outBuf, outSz, "1.00x PARITY");
        }
    };

    auto isSceneDgcCapable = [&]() -> bool {
        for (const auto& cat : m_categories) {
            for (const auto& sg : cat.subgroups) {
                for (const auto& itm : sg.items) {
                    if (itm.id == "RayScheduling" && (itm.configIndex == 2 || itm.configIndex == 5)) {
                        return itm.isSupported;
                    }
                }
            }
        }
        return false;
    };
    const bool activeGpuDgcSupported = isSceneDgcCapable();

    auto wavefrontUsesDgc = [&](const ParsedParityProfile& prof) -> bool {
        if (prof.hasDgcTelemetry) {
            return prof.usesExecutionSet || activeGpuDgcSupported;
        }
        return activeGpuDgcSupported;
    };

    auto renderDgcArchitectureBreakdown = [&](const ParsedParityProfile& prof, float maxW) {
        if (!prof.valid || !prof.stageTimingMeasured || prof.tradMs <= 0.0 || prof.workMs <= 0.0) {
            return;
        }
        ImGui::Spacing();
        const float boxW = std::max(s(300.0f), maxW);
        const float padX = s(10.0f);
        const float padY = s(8.0f);
        const float rowH = s(17.0f);
        const float rowGap = s(5.0f);
        const bool showLegacy = prof.hasLegacyIndirect && prof.legacyMs > 0.0;
        const bool usesDgc = wavefrontUsesDgc(prof);
        const int rowCount = showLegacy ? 3 : 2;
        const bool showTelRow1 = prof.hasDgcTelemetry && (prof.mkVgprs > 0 || prof.maxSequences > 0);
        const bool showMatRow = prof.hasMaterialShading && prof.matMkMhits > 0.0 && prof.matDgcMhits > 0.0;
        const float extraRows = (showTelRow1 ? 1.0f : 0.0f) + (showMatRow ? 1.0f : 0.0f);
        const float boxH = padY * 2.0f + s(20.0f) + rowCount * (rowH + rowGap) +
                           (extraRows > 0.0f ? (s(6.0f) + extraRows * s(17.0f)) : 0.0f);

        ImVec2 bp0 = ImGui::GetCursorScreenPos();
        ImVec2 bp1(bp0.x + boxW, bp0.y + boxH);
        ImDrawList* dl = ImGui::GetWindowDrawList();
        dl->AddRectFilled(bp0, bp1, toU32(c.bgRecessed), 0.0f);
        dl->AddRect(bp0, bp1, toU32(c.borderOuter), 0.0f, 0, 1.0f);

        const float innerW = boxW - padX * 2.0f;
        const ImVec4 colBvh = c.cyanAccent;
        const ImVec4 colCmp = c.phosphorHi;
        const ImVec4 colShd = c.okColor;

        // Header + Legend (measured cleanly to avoid collision at 1.0x and 2.0x HiDPI)
        const char* titleTxt = prof.usesExecutionSet
            ? "DGC EXECUTION SET STAGE BREAKDOWN (REAL GPU TIMESTAMPS)"
            : (usesDgc
                ? "DGC WAVEFRONT STAGE BREAKDOWN (REAL GPU TIMESTAMPS)"
                : "WAVEFRONT INDIRECT STAGE BREAKDOWN (REAL GPU TIMESTAMPS)");
        struct LegItem { const char* lbl; ImVec4 col; };
        const LegItem legs[3] = {
            {"BVH", colBvh},
            {"Vis+Cmp+Res", colCmp},
            {"Shading", colShd}
        };
        float legendTotalW = 0.0f;
        for (int li = 0; li < 3; ++li) {
            legendTotalW += ImGui::CalcTextSize(legs[li].lbl).x + s(24.0f);
        }
        const float maxTitleW = std::max(s(80.0f), innerW - legendTotalW - s(12.0f));
        const std::string fittedTitle = fitTextToPixelWidth(titleTxt, maxTitleW);
        dl->AddText(ImVec2(bp0.x + padX, bp0.y + padY), toU32(c.textPrimary), fittedTitle.c_str());

        if (innerW > legendTotalW + s(90.0f)) {
            float rx = bp1.x - padX;
            for (int li = 2; li >= 0; --li) {
                ImVec2 tSz = ImGui::CalcTextSize(legs[li].lbl);
                rx -= tSz.x;
                dl->AddText(ImVec2(rx, bp0.y + padY), toU32(c.textSecondary), legs[li].lbl);
                rx -= s(14.0f);
                dl->AddRectFilled(ImVec2(rx, bp0.y + padY + s(3.0f)),
                                  ImVec2(rx + s(9.0f), bp0.y + padY + s(12.0f)),
                                  toU32(legs[li].col), 0.0f);
                rx -= s(10.0f);
            }
        }

        const double maxMs = std::max({prof.tradMs, showLegacy ? prof.legacyMs : 0.0, prof.workMs, 0.01});
        const float lblW = std::clamp(innerW * 0.34f, s(150.0f), s(295.0f));
        const float valW = std::clamp(innerW * 0.24f, s(115.0f), s(205.0f));
        const float barX = bp0.x + padX + lblW + s(8.0f);
        const float barW = std::max(s(80.0f), innerW - lblW - valW - s(16.0f));

        struct StageBarRow {
            std::string label;
            double totalMs;
            double bvhMs;
            double cmpMs;
            double shdMs;
            double mrays;
            bool highlight;
        };
        std::vector<StageBarRow> rows;
        rows.push_back({"MEGAKERNEL (1x vkCmdDispatch)", prof.tradMs, prof.tradBvhMs, 0.0, prof.tradShadingMs, prof.tradMrays, false});
        if (showLegacy) {
            rows.push_back({"LEGACY INDIRECT (8x DispatchIndirect)", prof.legacyMs, prof.legacyBvhMs, prof.legacyCompactionMs, prof.legacyShadingMs, prof.legacyMrays, false});
        }
        rows.push_back({
            prof.usesExecutionSet
                ? "DGC EXECUTION SET (1x ExecuteGenCmdsEXT)"
                : (usesDgc
                    ? "DGC WORKLIST (1x ExecuteGenCmdsEXT)"
                    : "WAVEFRONT WORKLIST (DispatchIndirect)"),
            prof.workMs, prof.workBvhMs, prof.workCompactionMs, prof.workShadingMs, prof.workMrays, true
        });

        float curY = bp0.y + padY + s(21.0f);
        for (const auto& r : rows) {
            dl->AddText(ImVec2(bp0.x + padX, curY + s(1.0f)),
                        toU32(r.highlight ? c.phosphorHi : c.textSecondary),
                        fitTextToPixelWidth(r.label, lblW).c_str());

            dl->AddRectFilled(ImVec2(barX, curY + s(2.0f)),
                              ImVec2(barX + barW, curY + rowH - s(2.0f)),
                              toU32(c.bgVoid), 0.0f);

            const float scale = static_cast<float>(barW / maxMs);
            float segX = barX;
            auto drawSeg = [&](double segMs, const ImVec4& col) {
                if (segMs <= 0.0) return;
                float sw = std::max(s(2.0f), static_cast<float>(segMs) * scale);
                float nextX = std::min(barX + barW, segX + sw);
                dl->AddRectFilled(ImVec2(segX, curY + s(2.0f)),
                                  ImVec2(nextX, curY + rowH - s(2.0f)),
                                  toU32(col), 0.0f);
                segX = nextX;
            };
            drawSeg(r.bvhMs, colBvh);
            drawSeg(r.cmpMs, colCmp);
            drawSeg(r.shdMs, colShd);
            dl->AddRect(ImVec2(barX, curY + s(2.0f)),
                        ImVec2(barX + barW, curY + rowH - s(2.0f)),
                        toU32(r.highlight ? c.phosphorMid : c.borderOuter), 0.0f, 0, 1.0f);

            char valBuf[64];
            if (r.highlight && prof.tradMs > 0.0 && prof.workMs > 0.0) {
                std::snprintf(valBuf, sizeof(valBuf), "%.2f ms (%.0f MR/s, %.2fx)",
                              r.totalMs, r.mrays, prof.tradMs / prof.workMs);
            } else {
                std::snprintf(valBuf, sizeof(valBuf), "%.2f ms (%.0f MR/s)", r.totalMs, r.mrays);
            }
            dl->AddText(ImVec2(barX + barW + s(8.0f), curY + s(1.0f)),
                        toU32(r.highlight ? c.okColor : c.textPrimary),
                        fitTextToPixelWidth(valBuf, valW).c_str());

            curY += rowH + rowGap;
        }

        if (showTelRow1 || showMatRow) {
            dl->AddLine(ImVec2(bp0.x + padX, curY + s(1.0f)),
                        ImVec2(bp1.x - padX, curY + s(1.0f)), toU32(c.gridMinor), 1.0f);
            curY += s(5.0f);
        }
        if (showTelRow1) {
            char telBuf[240];
            uint32_t pruned = (prof.maxSequences >= prof.activeSequences) ? (prof.maxSequences - prof.activeSequences) : 0u;
            if (prof.mkVgprs > 0 && prof.dgcVgprsMax > 0) {
                char vgprStr[32];
                if (prof.dgcVgprsMin == prof.dgcVgprsMax) {
                    std::snprintf(vgprStr, sizeof(vgprStr), "%u", prof.dgcVgprsMin);
                } else {
                    std::snprintf(vgprStr, sizeof(vgprStr), "%u-%u", prof.dgcVgprsMin, prof.dgcVgprsMax);
                }
                char waveStr[32];
                if (prof.dgcWavesMin == prof.dgcWavesMax) {
                    std::snprintf(waveStr, sizeof(waveStr), "%u", prof.dgcWavesMin);
                } else {
                    std::snprintf(waveStr, sizeof(waveStr), "%u-%u", prof.dgcWavesMin, prof.dgcWavesMax);
                }
                if (prof.mkCodeBytes > 0 && prof.dgcCodeBytesAvg > 0.0) {
                    std::snprintf(telBuf, sizeof(telBuf),
                                  "ISA: MK %u VGPR (%uW, %.0f KB) -> DGC %s VGPR (%sW, %.0f KB)  |  PRUNING: %u/%u ACTIVE (%u PRUNED)",
                                  prof.mkVgprs, prof.mkMaxWaves, static_cast<double>(prof.mkCodeBytes) / 1024.0,
                                  vgprStr, waveStr, prof.dgcCodeBytesAvg / 1024.0,
                                  prof.activeSequences, prof.maxSequences, pruned);
                } else {
                    std::snprintf(telBuf, sizeof(telBuf),
                                  "ISA OCCUPANCY: MK %u VGPR (%uW/SIMD) -> DGC %s VGPR (%sW/SIMD)  |  GPU PRUNING: %u/%u ACTIVE (%u PRUNED)",
                                  prof.mkVgprs, prof.mkMaxWaves,
                                  vgprStr, waveStr,
                                  prof.activeSequences, prof.maxSequences, pruned);
                }
            } else {
                std::snprintf(telBuf, sizeof(telBuf),
                              "GPU SEQUENCE PRUNING: %u/%u PIPELINE SEQUENCES ACTIVE (%u EMPTY BINS PRUNED ON GPU)",
                              prof.activeSequences, prof.maxSequences, pruned);
            }
            dl->AddText(ImVec2(bp0.x + padX, curY), toU32(c.cyanAccent),
                        fitTextToPixelWidth(telBuf, innerW).c_str());
            curY += s(17.0f);
        }
        if (showMatRow) {
            char matBuf[220];
            if (prof.matLegacyMhits > 0.0) {
                std::snprintf(matBuf, sizeof(matBuf),
                              "ISOLATED MATERIAL SHADING STAGE: MK %.0f vs LEGACY %.0f vs DGC %.0f MHits/s (%.2fx vs MK | %.2fx vs LEGACY)",
                              prof.matMkMhits, prof.matLegacyMhits, prof.matDgcMhits,
                              prof.matDgcVsMk, prof.matDgcVsLegacy);
            } else {
                std::snprintf(matBuf, sizeof(matBuf),
                              "ISOLATED MATERIAL SHADING STAGE: MK %.0f vs DGC %.0f MHits/s (%.2fx FASTER)",
                              prof.matMkMhits, prof.matDgcMhits, prof.matDgcVsMk);
            }
            dl->AddText(ImVec2(bp0.x + padX, curY), toU32(c.okColor),
                        fitTextToPixelWidth(matBuf, innerW).c_str());
        }

        ImGui::SetCursorScreenPos(bp0);
        ImGui::Dummy(ImVec2(boxW, boxH));
    };

    auto renderParityBannerIfMeasured = [&](const ParsedParityProfile& prof) {
        double speedup = (prof.tradMs > 0.0 && prof.workMs > 0.0) ? (prof.tradMs / prof.workMs) : 1.0;
        const char* wfTag = wavefrontUsesDgc(prof) ? "DGC" : "WAVEFRONT";
        ImVec4 statusCol = (prof.psnr >= 45.0) ? c.okColor : c.errColor;
        pushMonoFont();
        ImGui::TextColored(statusCol,
                           "%s  |  PSNR: %.2f dB  |  EXACT: %.2f%%  |  MEGAKERNEL: %.2f ms (%.0f MR/s)  |  %s: %.2f ms (%.0f MR/s, %.2fx)",
                           prof.status.empty() ? "PARITY TELEMETRY" : prof.status.c_str(),
                           prof.psnr, prof.exactPct,
                           prof.tradMs, prof.tradMrays,
                           wfTag, prof.workMs, prof.workMrays, speedup);
        popMonoFont();
        ImGui::Spacing();
    };

    if (m_rtViewportMode == 0) {
        // Scenes & Visual Parity Mode
        const char* wfViewLbl = activeGpuDgcSupported ? "DGC WAVEFRONT" : "WAVEFRONT WORKLIST";
        const char* parityViews[] = {
            "SPLIT-WIPE COMPARATOR",
            wfViewLbl,
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

        const bool isPtPair = (m_rtParityViewMode == 0 && m_rtComparatorPair == 2);
        std::string jsonKey = isPtPair
            ? ("render_" + tag + "_pt16_profile.json")
            : ("render_" + tag + "_profile.json");
        ParsedParityProfile prof{};
        if (!isRunning) {
            std::string profileJsonStr;
            static std::unordered_map<std::string, std::pair<std::string, ParsedParityProfile>> sParityCache;
            if (getCachedViewportJson(jsonKey, profileJsonStr)) {
                auto& cached = sParityCache[jsonKey];
                if (cached.first != profileJsonStr) {
                    cached.first = profileJsonStr;
                    cached.second = parseParityProfileJson(profileJsonStr);
                }
                prof = cached.second;
            }
        }

        const bool wfIsDgc = wavefrontUsesDgc(prof);
        const char* wfShort = wfIsDgc ? "DGC" : "WAVEFRONT";

        if (m_rtParityViewMode == 0) {
            // Interactive Split-Wipe Reticle Comparator
            const char* pairLabels[] = {
                wfIsDgc ? "DGC vs MEGAKERNEL" : "WAVEFRONT vs MEGAKERNEL",
                wfIsDgc ? "DGC vs STAGE-1 BVH" : "WAVEFRONT vs STAGE-1 BVH",
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
            ImGui::Spacing();

            std::string leftRel, leftBadge;
            std::string rightRel, rightBadge;
            switch (m_rtComparatorPair) {
            case 1:
                leftRel    = "renders/render_" + tag + "_worklist_dgc.png";
                leftBadge  = wfIsDgc ? "LEFT: DGC FINAL" : "LEFT: WAVEFRONT FINAL";
                rightRel   = "renders/render_" + tag + "_stage1_bvh.png";
                rightBadge = "RIGHT: STAGE-1 BVH";
                break;
            case 2:
                leftRel    = "renders/render_" + tag + "_pathtracing_16spp_worklist.png";
                leftBadge  = "LEFT: 16 SPP CONVERGED";
                rightRel   = "renders/render_" + tag + "_pathtracing_1spp_worklist.png";
                rightBadge = "RIGHT: 1 SPP RAW";
                break;
            case 3:
                leftRel    = "renders/render_" + tag + "_stage7_final.png";
                leftBadge  = "LEFT: 7. FINAL PBR";
                rightRel   = "renders/render_" + tag + "_stage2_primary.png";
                rightBadge = "RIGHT: 2. G-BUF NORMALS";
                break;
            case 0:
            default:
                leftRel    = "renders/render_" + tag + "_worklist_dgc.png";
                leftBadge  = wfIsDgc ? "LEFT: DGC WAVEFRONT" : "LEFT: WAVEFRONT WORKLIST";
                rightRel   = "renders/render_" + tag + "_traditional_megakernel.png";
                rightBadge = "RIGHT: MEGAKERNEL";
                break;
            }

            VulkanContext::TextureResource texL = resolveTex(leftRel);
            VulkanContext::TextureResource texR = resolveTex(rightRel);

            const bool bothValid = texL.isValid() && texR.isValid();
            const bool pairMeasured = (m_rtComparatorPair == 0 || m_rtComparatorPair == 2)
                                          ? bothValid
                                          : texL.isValid();
            if (prof.valid && pairMeasured) {
                renderParityBannerIfMeasured(prof);
            }

            float aspect = 16.0f / 9.0f;
            if (texL.isValid() && texL.width > 0 && texL.height > 0) {
                aspect = static_cast<float>(texL.width) / static_cast<float>(texL.height);
            } else if (texR.isValid() && texR.width > 0 && texR.height > 0) {
                aspect = static_cast<float>(texR.width) / static_cast<float>(texR.height);
            }

            float availW = std::max(s(320.0f), ImGui::GetContentRegionAvail().x - s(4.0f));
            const float reserveBelow = (prof.valid && prof.stageTimingMeasured && m_rtComparatorPair == 0) ? s(156.0f) : s(28.0f);
            float availH = std::max(s(160.0f), ImGui::GetContentRegionAvail().y - reserveBelow);
            float canvasW = std::min({availW, availH * aspect, s(1060.0f)});
            float canvasH = std::round(canvasW / std::max(0.5f, aspect));

            ImVec2 p0 = ImGui::GetCursorScreenPos();
            ImVec2 p1(p0.x + canvasW, p0.y + canvasH);
            ImDrawList* dl = ImGui::GetWindowDrawList();

            if (!texL.isValid() && !texR.isValid()) {
                drawUnrenderedSlate(dl, p0, p1, canvasW, canvasH, true);
            } else {
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

                dl->AddRectFilled(p0, p1, toU32(c.bgRecessed), 0.0f);

                auto formatSourceBadge = [&](const std::string& base, const VulkanContext::TextureResource& t) {
                    if (!t.isValid()) return base + " [NO CAPTURE]";
                    if (t.borrowsExternalImage) {
                        return base + " [VRAM " + std::to_string(t.width) + "x" + std::to_string(t.height) + "]";
                    }
                    return base + " [PNG " + std::to_string(t.width) + "x" + std::to_string(t.height) + "]";
                };

                // Left Half
                dl->PushClipRect(p0, ImVec2(splitX, p1.y), true);
                if (texL.isValid()) {
                    dl->AddImage(reinterpret_cast<ImTextureID>(texL.descriptorSet), p0, p1);
                } else {
                    dl->AddText(ImVec2(p0.x + s(16.0f), p0.y + canvasH * 0.5f), toU32(c.textMuted), "LEFT CHANNEL: NOT CAPTURED");
                }
                dl->PopClipRect();

                // Right Half
                dl->PushClipRect(ImVec2(splitX, p0.y), p1, true);
                if (texR.isValid()) {
                    dl->AddImage(reinterpret_cast<ImTextureID>(texR.descriptorSet), p0, p1);
                } else {
                    dl->AddText(ImVec2(splitX + s(16.0f), p0.y + canvasH * 0.5f), toU32(c.textMuted), "RIGHT CHANNEL: NOT CAPTURED");
                }
                dl->PopClipRect();

                const float maxBadgeW = std::max(s(80.0f), canvasW * 0.5f - s(28.0f));
                {
                    const ImVec4 lCol = !texL.isValid() ? c.textMuted : (texL.borrowsExternalImage ? c.okColor : c.cyanAccent);
                    std::string lFull = fitTextToPixelWidth(formatSourceBadge(leftBadge, texL), maxBadgeW);
                    ImVec2 bSz = ImGui::CalcTextSize(lFull.c_str());
                    ImVec2 bp0(p0.x + s(10.0f), p0.y + s(10.0f));
                    ImVec2 bp1(bp0.x + bSz.x + s(12.0f), bp0.y + bSz.y + s(6.0f));
                    dl->AddRectFilled(bp0, bp1, toU32(c.bgVoid, 0.88f), 0.0f);
                    dl->AddRect(bp0, bp1, toU32(lCol), 0.0f, 0, 1.0f);
                    dl->AddText(ImVec2(bp0.x + s(6.0f), bp0.y + s(3.0f)), toU32(lCol), lFull.c_str());
                }
                {
                    const ImVec4 rCol = !texR.isValid() ? c.textMuted : (texR.borrowsExternalImage ? c.okColor : c.cyanAccent);
                    std::string rFull = fitTextToPixelWidth(formatSourceBadge(rightBadge, texR), maxBadgeW);
                    ImVec2 bSz = ImGui::CalcTextSize(rFull.c_str());
                    ImVec2 bp1(p1.x - s(10.0f), p0.y + s(10.0f) + bSz.y + s(6.0f));
                    ImVec2 bp0(bp1.x - (bSz.x + s(12.0f)), p0.y + s(10.0f));
                    dl->AddRectFilled(bp0, bp1, toU32(c.bgVoid, 0.88f), 0.0f);
                    dl->AddRect(bp0, bp1, toU32(rCol), 0.0f, 0, 1.0f);
                    dl->AddText(ImVec2(bp0.x + s(6.0f), bp0.y + s(3.0f)), toU32(rCol), rFull.c_str());
                }

                // Outer frame + Corner reticles + Vertical wipe hairline & center reticle handle
                dl->AddRect(p0, p1, toU32(c.borderOuter), 0.0f, 0, 1.0f);
                const ImVec4 frameRetCol = (texL.borrowsExternalImage || texR.borrowsExternalImage) ? c.okColor : c.cyanAccent;
                drawCornerReticles(dl, p0, p1, toU32(frameRetCol), s(10.0f));

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
                if (prof.valid && pairMeasured && prof.tradMs > 0.0 && prof.workMs > 0.0) {
                    const double ratio = prof.tradMs / prof.workMs;
                    if (ratio >= 1.005) {
                        std::snprintf(footerBuf, sizeof(footerBuf),
                                      "WIPE %.0f%%  |  DELTA: %.2fx FASTER (%s %.2fms vs MEGA %.2fms)  |  PSNR %.2f dB  |  DRAG HANDLE OR DOUBLE-CLICK CENTER",
                                      m_rtSplitRatio * 100.0f, ratio, wfShort, prof.workMs, prof.tradMs, prof.psnr);
                    } else if (ratio <= 0.995) {
                        std::snprintf(footerBuf, sizeof(footerBuf),
                                      "WIPE %.0f%%  |  DELTA: %.2fx (%s SLOWER: %.2fms vs MEGA %.2fms)  |  PSNR %.2f dB  |  DRAG HANDLE OR DOUBLE-CLICK CENTER",
                                      m_rtSplitRatio * 100.0f, ratio, wfShort, prof.workMs, prof.tradMs, prof.psnr);
                    } else {
                        std::snprintf(footerBuf, sizeof(footerBuf),
                                      "WIPE %.0f%%  |  DELTA: 1.00x PARITY (%s %.2fms vs MEGA %.2fms)  |  PSNR %.2f dB  |  DRAG HANDLE OR DOUBLE-CLICK CENTER",
                                      m_rtSplitRatio * 100.0f, wfShort, prof.workMs, prof.tradMs, prof.psnr);
                    }
                } else {
                    std::snprintf(footerBuf, sizeof(footerBuf),
                                  "WIPE %.0f%%  |  %s (%dx%d)  <->  %s (%dx%d)  |  DRAG RETICLE TO COMPARE",
                                  m_rtSplitRatio * 100.0f,
                                  leftBadge.c_str(), texL.width, texL.height,
                                  rightBadge.c_str(), texR.width, texR.height);
                }
                ImGui::TextColored(c.textSecondary, "%s", fitTextToPixelWidth(footerBuf, availW).c_str());
                popMonoFont();
                if (m_rtComparatorPair == 0) {
                    renderDgcArchitectureBreakdown(prof, availW);
                }
            }
        } else if (m_rtParityViewMode == 1) {
            VulkanContext::TextureResource tex = resolveTex("renders/render_" + tag + "_worklist_dgc.png");
            if (prof.valid && tex.isValid()) renderParityBannerIfMeasured(prof);
            float availW = std::max(s(320.0f), ImGui::GetContentRegionAvail().x - s(6.0f));
            drawTexFramed(tex,
                          wfIsDgc ? "DGC Wavefront Material-Sorted Render"
                                  : "Wavefront Material-Sorted Render (DispatchIndirect)",
                          std::min(availW, s(980.0f)), true);
        } else if (m_rtParityViewMode == 2) {
            VulkanContext::TextureResource tex = resolveTex("renders/render_" + tag + "_traditional_megakernel.png");
            if (prof.valid && tex.isValid()) renderParityBannerIfMeasured(prof);
            float availW = std::max(s(320.0f), ImGui::GetContentRegionAvail().x - s(6.0f));
            drawTexFramed(tex, "Traditional Monolithic Megakernel Render", std::min(availW, s(980.0f)), true);
        } else if (m_rtParityViewMode == 3) {
            VulkanContext::TextureResource texTrad = resolveTex("renders/render_" + tag + "_traditional_megakernel.png");
            VulkanContext::TextureResource texWork = resolveTex("renders/render_" + tag + "_worklist_dgc.png");
            VulkanContext::TextureResource texDiff = resolveTex("renders/render_" + tag + "_difference_heatmap.png");
            if (prof.valid && texTrad.isValid() && texWork.isValid()) renderParityBannerIfMeasured(prof);
            float availW = std::max(s(320.0f), ImGui::GetContentRegionAvail().x - s(6.0f));
            float colW = (availW - s(20.0f)) / 3.0f;
            drawTexFramed(texTrad, "Megakernel", colW, true);
            ImGui::SameLine(0.0f, s(10.0f));
            drawTexFramed(texWork, wfIsDgc ? "DGC Wavefront" : "Wavefront Worklist", colW, false);
            ImGui::SameLine(0.0f, s(10.0f));
            drawTexFramed(texDiff, "10x Difference Heatmap", colW, false);
        } else {
            VulkanContext::TextureResource texMlTrad = resolveTex("renders/render_" + tag + "_multilight_128_traditional.png");
            VulkanContext::TextureResource texMlDgc = resolveTex("renders/render_" + tag + "_multilight_128_dgc.png");
            if (texMlTrad.isValid() && texMlDgc.isValid()) {
                ImGui::TextColored(c.cyanAccent,
                                   "128-LIGHT SPATIAL BINNING VERIFICATION  |  MEGAKERNEL vs %s",
                                   activeGpuDgcSupported ? "DGC WAVEFRONT" : "WAVEFRONT INDIRECT");
                ImGui::Spacing();
            }
            float availW = std::max(s(320.0f), ImGui::GetContentRegionAvail().x - s(6.0f));
            float colW = (availW - s(12.0f)) / 2.0f;
            drawTexFramed(texMlTrad, "128-Light Megakernel", colW, true);
            ImGui::SameLine(0.0f, s(12.0f));
            drawTexFramed(texMlDgc,
                          activeGpuDgcSupported ? "128-Light DGC Spatial Binning"
                                                : "128-Light Wavefront Spatial Binning",
                          colW, false);
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
        const std::string bdKey = "render_" + tag + "_pipeline_breakdown.json";
        if (!isRunning && getCachedViewportJson(bdKey, bdJson)) {
            static std::unordered_map<std::string, std::pair<std::string, ParsedPipelineBreakdown>> sBdCache;
            auto& cached = sBdCache[bdKey];
            if (cached.first != bdJson) {
                cached.first = bdJson;
                cached.second = parsePipelineBreakdownJson(bdJson);
            }
            bd = cached.second;
        }

        float availW = std::max(s(320.0f), ImGui::GetContentRegionAvail().x - s(8.0f));
        std::string sid = stageIds[m_rtPassIndex];
        VulkanContext::TextureResource stageTex = resolveTex("renders/render_" + tag + "_" + sid + ".png");
        drawTexFramed(stageTex, stageLabels[m_rtPassIndex], std::min(availW, s(760.0f)), true);

        if (bd.valid && !bd.stages.empty() && stageTex.isValid()) {
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
        // Multi-Bounce Monte Carlo Path Tracing & Convergence Verification
        VulkanContext::TextureResource texPt16 = resolveTex("renders/render_" + tag + "_pathtracing_16spp_worklist.png");
        VulkanContext::TextureResource texPt1 = resolveTex("renders/render_" + tag + "_pathtracing_1spp_worklist.png");
        VulkanContext::TextureResource texPtDiff = resolveTex("renders/render_" + tag + "_pathtracing_16spp_difference.png");

        std::string pt16Json;
        ParsedParityProfile profPt16{};
        const std::string pt16Key = "render_" + tag + "_pt16_profile.json";
        if (!isRunning && getCachedViewportJson(pt16Key, pt16Json)) {
            static std::unordered_map<std::string, std::pair<std::string, ParsedParityProfile>> sPt16Cache;
            auto& cached = sPt16Cache[pt16Key];
            if (cached.first != pt16Json) {
                cached.first = pt16Json;
                cached.second = parseParityProfileJson(pt16Json);
            }
            profPt16 = cached.second;
        }
        if (profPt16.valid && texPt16.isValid() && texPt1.isValid()) {
            renderParityBannerIfMeasured(profPt16);
        }

        const bool ptUsesDgc = wavefrontUsesDgc(profPt16);
        float availW = std::max(s(320.0f), ImGui::GetContentRegionAvail().x - s(16.0f));
        float colW = (availW - s(20.0f)) / 3.0f;
        drawTexFramed(texPt16,
                      ptUsesDgc ? "16 SPP Converged Path Tracing (DGC)"
                                : "16 SPP Converged Path Tracing (Wavefront)",
                      colW, true);
        ImGui::SameLine(0.0f, s(10.0f));
        drawTexFramed(texPt1,
                      ptUsesDgc ? "1 SPP Unbiased Path Tracing (DGC)"
                                : "1 SPP Unbiased Path Tracing (Wavefront)",
                      colW, false);
        ImGui::SameLine(0.0f, s(10.0f));
        drawTexFramed(texPtDiff,
                      ptUsesDgc ? "16 SPP Megakernel vs DGC 10x Heatmap"
                                : "16 SPP Megakernel vs Wavefront 10x Heatmap",
                      colW, false);

        if (profPt16.valid && profPt16.tradMs > 0.0 && profPt16.workMs > 0.0 && texPt16.isValid()) {
            ImGui::Spacing();
            const float boxW = std::max(s(300.0f), availW);
            const float padX = s(10.0f);
            const float padY = s(8.0f);
            const float rowH = s(17.0f);
            const float boxH = padY * 2.0f + s(21.0f) + 2.0f * rowH + s(5.0f);
            ImVec2 bp0 = ImGui::GetCursorScreenPos();
            ImVec2 bp1(bp0.x + boxW, bp0.y + boxH);
            ImDrawList* dl = ImGui::GetWindowDrawList();
            dl->AddRectFilled(bp0, bp1, toU32(c.bgRecessed), 0.0f);
            dl->AddRect(bp0, bp1, toU32(c.borderOuter), 0.0f, 0, 1.0f);

            const float innerW = boxW - padX * 2.0f;
            char badge[48];
            renderRatioBadgeText(profPt16.tradMs, profPt16.workMs, badge, sizeof(badge));
            std::string ptHdr = std::string(ptUsesDgc
                ? "16-SPP CONVERGED PATH TRACING: MEGAKERNEL vs DGC WAVEFRONT  ("
                : "16-SPP CONVERGED PATH TRACING: MEGAKERNEL vs WAVEFRONT INDIRECT  (") + badge + ")";
            dl->AddText(ImVec2(bp0.x + padX, bp0.y + padY), toU32(c.textPrimary),
                        fitTextToPixelWidth(ptHdr, innerW).c_str());

            const double maxPtMs = std::max({profPt16.tradMs, profPt16.workMs, 0.01});
            const float lblW = std::clamp(innerW * 0.30f, s(140.0f), s(250.0f));
            const float valW = std::clamp(innerW * 0.24f, s(115.0f), s(195.0f));
            const float barX = bp0.x + padX + lblW + s(8.0f);
            const float barW = std::max(s(80.0f), innerW - lblW - valW - s(16.0f));

            auto drawPtRow = [&](float y, const char* lbl, double ms, double mrays, const ImVec4& barCol, bool hl) {
                dl->AddText(ImVec2(bp0.x + padX, y + s(1.0f)), toU32(hl ? c.okColor : c.textSecondary),
                            fitTextToPixelWidth(lbl, lblW).c_str());
                dl->AddRectFilled(ImVec2(barX, y + s(2.0f)), ImVec2(barX + barW, y + rowH - s(2.0f)), toU32(c.bgVoid), 0.0f);
                float fw = std::max(s(4.0f), static_cast<float>(ms / maxPtMs) * barW);
                dl->AddRectFilled(ImVec2(barX, y + s(2.0f)), ImVec2(barX + fw, y + rowH - s(2.0f)), toU32(barCol), 0.0f);
                dl->AddRect(ImVec2(barX, y + s(2.0f)), ImVec2(barX + barW, y + rowH - s(2.0f)), toU32(c.borderOuter), 0.0f, 0, 1.0f);
                char vb[64];
                std::snprintf(vb, sizeof(vb), "%.2f ms (%.0f MR/s)", ms, mrays);
                dl->AddText(ImVec2(barX + barW + s(8.0f), y + s(1.0f)), toU32(c.textPrimary),
                            fitTextToPixelWidth(vb, valW).c_str());
            };
            float ry = bp0.y + padY + s(21.0f);
            drawPtRow(ry, "MEGAKERNEL PATH TRACER", profPt16.tradMs, profPt16.tradMrays, c.cyanAccent, false);
            drawPtRow(ry + rowH + s(5.0f),
                      ptUsesDgc ? "DGC OCTANT-COMPACTED PT" : "WAVEFRONT OCTANT-COMPACTED PT",
                      profPt16.workMs, profPt16.workMrays, c.okColor, true);

            ImGui::SetCursorScreenPos(bp0);
            ImGui::Dummy(ImVec2(boxW, boxH));
        }
    } else {
        // Geometry & BVH Inspection (Actual Rendered Passes)
        float availW = std::max(s(320.0f), ImGui::GetContentRegionAvail().x - s(20.0f));
        float colW = (availW - s(20.0f)) / 3.0f;
        drawImageFramed("renders/render_" + tag + "_stage2_primary.png",
                        "Stage 2: Primary Hit G-Buffer Surface Normals", colW, true);
        ImGui::SameLine(0.0f, s(10.0f));
        drawImageFramed("renders/render_" + tag + "_alpha_cutout.png",
                        "Alpha-Cutout Any-Hit Traversal", colW, false);
        ImGui::SameLine(0.0f, s(10.0f));
        drawImageFramed("renders/render_" + tag + "_stage1_bvh.png",
                        "Stage 1: BVH Traversal Step Heatmap", colW, false);
    }
}

} // namespace gpubench::gui
