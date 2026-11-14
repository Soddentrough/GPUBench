#pragma once

#include <imgui.h>
#include <implot.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>

namespace gpubench::gui {

enum class BringUpPalette : int {
    AmberP3 = 0,
    GreenP1 = 1,
    PaperDatasheet = 2
};

struct BringUpColors {
    bool isLightTheme{false};

    // Surface ladder
    ImVec4 bgVoid;
    ImVec4 bgPanel;
    ImVec4 bgElevated;
    ImVec4 bgRecessed;
    ImVec4 bgHover;
    ImVec4 bgActive;

    // Hairlines & grids
    ImVec4 borderOuter;
    ImVec4 borderInner;
    ImVec4 gridMajor;
    ImVec4 gridMinor;

    // Phosphor accent ramp
    ImVec4 phosphorHi;
    ImVec4 phosphorMid;
    ImVec4 phosphorDim;
    ImVec4 phosphorGlow;

    // Typography
    ImVec4 textPrimary;
    ImVec4 textSecondary;
    ImVec4 textMuted;
    ImVec4 textInverse;

    // Telemetry, ISA syntax & delta indicators
    ImVec4 okColor;
    ImVec4 warnColor;
    ImVec4 errColor;
    ImVec4 cyanAccent;
    ImVec4 violetAccent;
    ImVec4 ledOff;
};

inline ImU32 toU32(const ImVec4& col, float alphaScale = 1.0f) {
    ImVec4 c = col;
    c.w = std::clamp(c.w * alphaScale, 0.0f, 1.0f);
    return ImGui::ColorConvertFloat4ToU32(c);
}

inline ImVec4 rgbaHex(uint32_t hexRgb, float a = 1.0f) {
    return ImVec4(
        static_cast<float>((hexRgb >> 16) & 0xFF) / 255.0f,
        static_cast<float>((hexRgb >> 8) & 0xFF) / 255.0f,
        static_cast<float>(hexRgb & 0xFF) / 255.0f,
        a
    );
}

inline BringUpColors getBringUpColors(BringUpPalette p) {
    BringUpColors c{};
    switch (p) {
    case BringUpPalette::GreenP1:
        c.isLightTheme  = false;
        c.bgVoid        = rgbaHex(0x060A07);
        c.bgPanel       = rgbaHex(0x0B120D);
        c.bgElevated    = rgbaHex(0x111C14);
        c.bgRecessed    = rgbaHex(0x080E0A);
        c.bgHover       = rgbaHex(0x18281D);
        c.bgActive      = rgbaHex(0x1F3526);
        c.borderOuter   = rgbaHex(0x1F4F30);
        c.borderInner   = rgbaHex(0x172E1F);
        c.gridMajor     = rgbaHex(0x1B4A2B, 0.55f);
        c.gridMinor     = rgbaHex(0x14301D, 0.38f);
        c.phosphorHi    = rgbaHex(0x33FF66);
        c.phosphorMid   = rgbaHex(0x22C55E);
        c.phosphorDim   = rgbaHex(0x147036);
        c.phosphorGlow  = rgbaHex(0x33FF66, 0.16f);
        c.textPrimary   = rgbaHex(0xE2FCE8);
        c.textSecondary = rgbaHex(0x9BC9A7);
        c.textMuted     = rgbaHex(0x5E8569);
        c.textInverse   = rgbaHex(0x060A07);
        c.okColor       = rgbaHex(0x33FF66);
        c.warnColor     = rgbaHex(0xF5C43D);
        c.errColor      = rgbaHex(0xFF5A52);
        c.cyanAccent    = rgbaHex(0x3AE8E8);
        c.violetAccent  = rgbaHex(0xA98CFF);
        c.ledOff        = rgbaHex(0x152219);
        break;

    case BringUpPalette::PaperDatasheet:
        c.isLightTheme  = true;
        c.bgVoid        = rgbaHex(0xEFECE6);
        c.bgPanel       = rgbaHex(0xFAF9F6);
        c.bgElevated    = rgbaHex(0xFFFFFF);
        c.bgRecessed    = rgbaHex(0xF2F1EC);
        c.bgHover       = rgbaHex(0xE6E4DC);
        c.bgActive      = rgbaHex(0xD8D5CB);
        c.borderOuter   = rgbaHex(0x4A525A);
        c.borderInner   = rgbaHex(0xC8C4B8);
        c.gridMajor     = rgbaHex(0x9BA0A6, 0.42f);
        c.gridMinor     = rgbaHex(0xC2C6CA, 0.35f);
        c.phosphorHi    = rgbaHex(0xB84A00);
        c.phosphorMid   = rgbaHex(0x8E3800);
        c.phosphorDim   = rgbaHex(0xC97A42);
        c.phosphorGlow  = rgbaHex(0xB84A00, 0.14f);
        c.textPrimary   = rgbaHex(0x14181D);
        c.textSecondary = rgbaHex(0x384048);
        c.textMuted     = rgbaHex(0x646C74);
        c.textInverse   = rgbaHex(0xFFFFFF);
        c.okColor       = rgbaHex(0x117A3E);
        c.warnColor     = rgbaHex(0xB55E00);
        c.errColor      = rgbaHex(0xC62828);
        c.cyanAccent    = rgbaHex(0x006B8F);
        c.violetAccent  = rgbaHex(0x5B3AA8);
        c.ledOff        = rgbaHex(0xDFDCD2);
        break;

    case BringUpPalette::AmberP3:
    default:
        c.isLightTheme  = false;
        c.bgVoid        = rgbaHex(0x080908);
        c.bgPanel       = rgbaHex(0x0E110E);
        c.bgElevated    = rgbaHex(0x151A15);
        c.bgRecessed    = rgbaHex(0x0B0D0B);
        c.bgHover       = rgbaHex(0x1D241C);
        c.bgActive      = rgbaHex(0x283022);
        c.borderOuter   = rgbaHex(0x473514);
        c.borderInner   = rgbaHex(0x232A23);
        c.gridMajor     = rgbaHex(0x4D3915, 0.50f);
        c.gridMinor     = rgbaHex(0x262E26, 0.36f);
        c.phosphorHi    = rgbaHex(0xFFB000);
        c.phosphorMid   = rgbaHex(0xD98A00);
        c.phosphorDim   = rgbaHex(0x7D4F00);
        c.phosphorGlow  = rgbaHex(0xFFB000, 0.16f);
        c.textPrimary   = rgbaHex(0xECE6D6);
        c.textSecondary = rgbaHex(0xB3AB96);
        c.textMuted     = rgbaHex(0x767468);
        c.textInverse   = rgbaHex(0x080908);
        c.okColor       = rgbaHex(0x48E08A);
        c.warnColor     = rgbaHex(0xFFC038);
        c.errColor      = rgbaHex(0xFF5247);
        c.cyanAccent    = rgbaHex(0x3CD6D0);
        c.violetAccent  = rgbaHex(0xB58BFF);
        c.ledOff        = rgbaHex(0x1E231E);
        break;
    }
    return c;
}

inline BringUpPalette& activeBringUpPaletteStorage() {
    static BringUpPalette s_palette = BringUpPalette::AmberP3;
    return s_palette;
}

inline BringUpPalette activeBringUpPalette() {
    return activeBringUpPaletteStorage();
}

inline void setActiveBringUpPalette(BringUpPalette p) {
    activeBringUpPaletteStorage() = p;
}

inline BringUpColors activeBringUpColors() {
    return getBringUpColors(activeBringUpPalette());
}

inline const char* paletteShortName(BringUpPalette p) {
    switch (p) {
    case BringUpPalette::GreenP1:        return "GREEN";
    case BringUpPalette::PaperDatasheet: return "PAPER";
    case BringUpPalette::AmberP3:
    default:                             return "AMBER";
    }
}

inline void applyBringUpStyle(ImGuiStyle& style, ImPlotStyle& plotStyle,
                                BringUpPalette p, float scale) {
    setActiveBringUpPalette(p);
    const BringUpColors c = getBringUpColors(p);

    style = ImGuiStyle{};
    plotStyle = ImPlotStyle{};

    // Strict 0px geometric sharp aesthetic
    style.WindowPadding     = ImVec2(10.0f, 8.0f);
    style.FramePadding      = ImVec2(8.0f, 4.0f);
    style.CellPadding       = ImVec2(6.0f, 3.0f);
    style.ItemSpacing       = ImVec2(6.0f, 5.0f);
    style.ItemInnerSpacing  = ImVec2(5.0f, 4.0f);
    style.IndentSpacing     = 14.0f;
    style.ScrollbarSize     = 11.0f;
    style.GrabMinSize       = 8.0f;

    style.WindowRounding    = 0.0f;
    style.ChildRounding     = 0.0f;
    style.FrameRounding     = 0.0f;
    style.PopupRounding     = 0.0f;
    style.ScrollbarRounding = 0.0f;
    style.GrabRounding      = 0.0f;
    style.TabRounding       = 0.0f;

    style.WindowBorderSize  = 1.0f;
    style.ChildBorderSize   = 1.0f;
    style.FrameBorderSize   = 1.0f;
    style.PopupBorderSize   = 1.0f;
    style.TabBorderSize     = 1.0f;
    style.AntiAliasedLines  = true;
    style.AntiAliasedFill   = true;

    ImVec4* col = style.Colors;
    col[ImGuiCol_Text]                  = c.textPrimary;
    col[ImGuiCol_TextDisabled]          = c.textMuted;
    col[ImGuiCol_WindowBg]              = c.bgVoid;
    col[ImGuiCol_ChildBg]               = c.bgPanel;
    col[ImGuiCol_PopupBg]               = c.bgElevated;
    col[ImGuiCol_Border]                = c.borderOuter;
    col[ImGuiCol_BorderShadow]          = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
    col[ImGuiCol_FrameBg]               = c.bgRecessed;
    col[ImGuiCol_FrameBgHovered]        = c.bgHover;
    col[ImGuiCol_FrameBgActive]         = c.bgActive;
    col[ImGuiCol_TitleBg]               = c.bgPanel;
    col[ImGuiCol_TitleBgActive]         = c.bgElevated;
    col[ImGuiCol_TitleBgCollapsed]      = c.bgPanel;
    col[ImGuiCol_MenuBarBg]             = c.bgPanel;
    col[ImGuiCol_ScrollbarBg]           = c.bgRecessed;
    col[ImGuiCol_ScrollbarGrab]         = c.borderOuter;
    col[ImGuiCol_ScrollbarGrabHovered]  = c.phosphorMid;
    col[ImGuiCol_ScrollbarGrabActive]   = c.phosphorHi;
    col[ImGuiCol_CheckMark]             = c.phosphorHi;
    col[ImGuiCol_SliderGrab]            = c.phosphorMid;
    col[ImGuiCol_SliderGrabActive]      = c.phosphorHi;
    col[ImGuiCol_Button]                = c.bgElevated;
    col[ImGuiCol_ButtonHovered]         = c.bgHover;
    col[ImGuiCol_ButtonActive]          = c.bgActive;
    col[ImGuiCol_Header]                = c.bgElevated;
    col[ImGuiCol_HeaderHovered]         = c.bgHover;
    col[ImGuiCol_HeaderActive]          = c.bgActive;
    col[ImGuiCol_Separator]             = c.borderInner;
    col[ImGuiCol_SeparatorHovered]      = c.phosphorMid;
    col[ImGuiCol_SeparatorActive]       = c.phosphorHi;
    col[ImGuiCol_ResizeGrip]            = c.borderOuter;
    col[ImGuiCol_ResizeGripHovered]     = c.phosphorMid;
    col[ImGuiCol_ResizeGripActive]      = c.phosphorHi;
    col[ImGuiCol_Tab]                   = c.bgRecessed;
    col[ImGuiCol_TabHovered]            = c.bgHover;
    col[ImGuiCol_TabSelected]           = c.bgElevated;
    col[ImGuiCol_TabSelectedOverline]   = c.phosphorHi;
    col[ImGuiCol_TabDimmed]             = c.bgRecessed;
    col[ImGuiCol_TabDimmedSelected]     = c.bgPanel;
    col[ImGuiCol_PlotLines]             = c.phosphorHi;
    col[ImGuiCol_PlotLinesHovered]      = c.cyanAccent;
    col[ImGuiCol_PlotHistogram]         = c.phosphorMid;
    col[ImGuiCol_PlotHistogramHovered]  = c.phosphorHi;
    col[ImGuiCol_TableHeaderBg]         = c.bgElevated;
    col[ImGuiCol_TableBorderStrong]     = c.borderOuter;
    col[ImGuiCol_TableBorderLight]      = c.borderInner;
    col[ImGuiCol_TableRowBg]            = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
    col[ImGuiCol_TableRowBgAlt]         = c.isLightTheme
        ? ImVec4(0.0f, 0.0f, 0.0f, 0.03f)
        : ImVec4(1.0f, 1.0f, 1.0f, 0.02f);
    col[ImGuiCol_TextSelectedBg]        = c.phosphorGlow;
    col[ImGuiCol_NavCursor]             = c.phosphorHi;
    col[ImGuiCol_ModalWindowDimBg]      = ImVec4(0.0f, 0.0f, 0.0f, c.isLightTheme ? 0.35f : 0.72f);

    // ImPlot styling synchronized to active instrument palette
    plotStyle.PlotBorderSize        = 1.0f;
    plotStyle.PlotPadding           = ImVec2(8.0f, 8.0f);
    plotStyle.LabelPadding          = ImVec2(4.0f, 3.0f);
    plotStyle.LegendPadding         = ImVec2(6.0f, 4.0f);
    plotStyle.LegendInnerPadding    = ImVec2(5.0f, 3.0f);
    plotStyle.LegendSpacing         = ImVec2(5.0f, 3.0f);
    plotStyle.MousePosPadding       = ImVec2(6.0f, 6.0f);

    plotStyle.Colors[ImPlotCol_FrameBg]       = c.isLightTheme ? c.bgPanel : c.bgVoid;
    plotStyle.Colors[ImPlotCol_PlotBg]        = c.isLightTheme ? c.bgElevated : c.bgRecessed;
    plotStyle.Colors[ImPlotCol_PlotBorder]    = c.borderOuter;
    plotStyle.Colors[ImPlotCol_LegendBg]      = c.bgPanel;
    plotStyle.Colors[ImPlotCol_LegendBorder]  = c.borderOuter;
    plotStyle.Colors[ImPlotCol_LegendText]    = c.textPrimary;
    plotStyle.Colors[ImPlotCol_TitleText]     = c.phosphorHi;
    plotStyle.Colors[ImPlotCol_InlayText]     = c.textPrimary;
    plotStyle.Colors[ImPlotCol_AxisText]      = c.textSecondary;
    plotStyle.Colors[ImPlotCol_AxisGrid]      = c.gridMajor;
    plotStyle.Colors[ImPlotCol_AxisTick]      = c.borderOuter;
    plotStyle.Colors[ImPlotCol_AxisBgHovered] = c.bgHover;
    plotStyle.Colors[ImPlotCol_AxisBgActive]  = c.bgActive;
    plotStyle.Colors[ImPlotCol_Selection]     = c.phosphorGlow;
    plotStyle.Colors[ImPlotCol_Crosshairs]    = c.phosphorMid;

    float safeScale = (scale > 0.1f) ? scale : 1.0f;
    style.ScaleAllSizes(safeScale);
    style.FontScaleDpi = safeScale;
}

inline std::string resolveDiagramIdForItem(const std::string& benchId,
                                           const std::string& subcat,
                                           const std::string& itemName,
                                           const std::string& existingDiagramId) {
    if (!existingDiagramId.empty()) return existingDiagramId;
    auto hasI = [](const std::string& haystack, const char* needle) {
        std::string h = haystack;
        std::string n = needle;
        std::transform(h.begin(), h.end(), h.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
        std::transform(n.begin(), n.end(), n.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
        return h.find(n) != std::string::npos;
    };
    const std::string combo = benchId + " " + subcat + " " + itemName;
    if (hasI(combo, "matrix") || hasI(combo, "fp8") || hasI(combo, "fp6") ||
        hasI(combo, "fp4") || hasI(combo, "int4") || hasI(combo, "coop")) {
        return "wmma_matrix_core";
    }
    if (hasI(combo, "interconnect") || hasI(combo, "pcie") || hasI(combo, "rebar") ||
        hasI(combo, "unified") || hasI(combo, "system") || hasI(combo, "host")) {
        return "interconnect_fabric";
    }
    if (hasI(combo, "bandwidth") || hasI(combo, "cache") || hasI(combo, "latency") || hasI(combo, "lds")) {
        return "memory_hierarchy_bus";
    }
    if (hasI(combo, "bvh") || hasI(combo, "ray")) {
        return "bvh_cutout_layers";
    }
    return "simd_vector_alu";
}

// ============================================================================
// Silicon Bring-Up Custom ImDrawList Hardware Instrument Primitives
// ============================================================================

inline void drawCornerReticles(ImDrawList* dl, ImVec2 p0, ImVec2 p1, ImU32 col, float arm = 6.0f) {
    float a = std::min({arm, (p1.x - p0.x) * 0.25f, (p1.y - p0.y) * 0.25f});
    if (a <= 1.0f) return;
    // Top-left
    dl->AddLine(p0, ImVec2(p0.x + a, p0.y), col, 1.5f);
    dl->AddLine(p0, ImVec2(p0.x, p0.y + a), col, 1.5f);
    // Top-right
    dl->AddLine(ImVec2(p1.x, p0.y), ImVec2(p1.x - a, p0.y), col, 1.5f);
    dl->AddLine(ImVec2(p1.x, p0.y), ImVec2(p1.x, p0.y + a), col, 1.5f);
    // Bottom-left
    dl->AddLine(ImVec2(p0.x, p1.y), ImVec2(p0.x + a, p1.y), col, 1.5f);
    dl->AddLine(ImVec2(p0.x, p1.y), ImVec2(p0.x, p1.y - a), col, 1.5f);
    // Bottom-right
    dl->AddLine(p1, ImVec2(p1.x - a, p1.y), col, 1.5f);
    dl->AddLine(p1, ImVec2(p1.x, p1.y - a), col, 1.5f);
}

inline std::string fitBringUpText(const char* text, float maxWidth) {
    if (!text || !*text || maxWidth <= 0.0f) return {};
    std::string s(text);
    if (ImGui::CalcTextSize(s.c_str()).x <= maxWidth) return s;

    static constexpr const char* kEllipsis = "...";
    const float ellW = ImGui::CalcTextSize(kEllipsis).x;
    if (maxWidth <= ellW) {
        while (!s.empty() && ImGui::CalcTextSize(s.c_str()).x > maxWidth) {
            s.pop_back();
            while (!s.empty() && (static_cast<unsigned char>(s.back()) & 0xC0) == 0x80) {
                s.pop_back();
            }
        }
        return s;
    }

    size_t lo = 0;
    size_t hi = s.size();
    while (lo < hi) {
        size_t mid0 = lo + (hi - lo + 1) / 2;
        size_t mid = mid0;
        while (mid > lo && (static_cast<unsigned char>(s[mid]) & 0xC0) == 0x80) {
            --mid;
        }
        if (mid == lo) {
            mid = mid0;
            while (mid < s.size() && (static_cast<unsigned char>(s[mid]) & 0xC0) == 0x80) {
                ++mid;
            }
            if (mid > hi) break;
        }
        std::string cand = s.substr(0, mid) + kEllipsis;
        if (ImGui::CalcTextSize(cand.c_str()).x <= maxWidth) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    if (lo == 0) return kEllipsis;
    return s.substr(0, lo) + kEllipsis;
}

inline void drawEtchedHeader(const char* sectionCode, const char* title,
                             const BringUpColors& c, const char* rightTag = nullptr) {
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 pos = ImGui::GetCursorScreenPos();
    float availW = ImGui::GetContentRegionAvail().x;
    float h = ImGui::GetTextLineHeight() + 6.0f;
    ImVec2 p1(pos.x + availW, pos.y + h);

    dl->AddRectFilled(pos, p1, toU32(c.bgElevated), 0.0f);
    dl->AddLine(ImVec2(pos.x, p1.y), p1, toU32(c.borderOuter), 1.0f);
    dl->AddRectFilled(pos, ImVec2(pos.x + 3.0f, p1.y), toU32(c.phosphorHi), 0.0f);

    float textX = pos.x + 9.0f;
    float textY = pos.y + 3.0f;
    float rightEdge = pos.x + availW - 6.0f;
    bool truncated = false;

    dl->PushClipRect(pos, p1, true);
    if (sectionCode && *sectionCode) {
        dl->AddText(ImVec2(textX, textY), toU32(c.phosphorHi), sectionCode);
        textX += ImGui::CalcTextSize(sectionCode).x + 8.0f;
    }
    if (title && *title) {
        float maxTitleW = rightEdge - textX;
        std::string fittedTitle = fitBringUpText(title, maxTitleW);
        if (!fittedTitle.empty()) {
            dl->AddText(ImVec2(textX, textY), toU32(c.textPrimary), fittedTitle.c_str());
            textX += ImGui::CalcTextSize(fittedTitle.c_str()).x;
        }
        if (fittedTitle != title) {
            truncated = true;
        }
    }
    if (rightTag && *rightTag) {
        float maxRightW = rightEdge - (textX + 12.0f);
        if (maxRightW >= 42.0f) {
            std::string fittedTag = fitBringUpText(rightTag, maxRightW);
            if (!fittedTag.empty()) {
                float rw = ImGui::CalcTextSize(fittedTag.c_str()).x;
                dl->AddText(ImVec2(rightEdge - rw, textY), toU32(c.textSecondary), fittedTag.c_str());
            }
            if (fittedTag != rightTag) {
                truncated = true;
            }
        } else {
            truncated = true;
        }
    }
    dl->PopClipRect();

    ImGui::Dummy(ImVec2(availW > 0.0f ? availW : 0.0f, h + 2.0f));
    if (truncated && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        ImGui::BeginTooltip();
        if (sectionCode && *sectionCode) {
            ImGui::TextColored(c.phosphorHi, "%s", sectionCode);
            if (title && *title) ImGui::SameLine();
        }
        if (title && *title) {
            ImGui::TextUnformatted(title);
        }
        if (rightTag && *rightTag) {
            ImGui::TextColored(c.textSecondary, "%s", rightTag);
        }
        ImGui::EndTooltip();
    }
}

inline bool tacticalButton(const char* id, const char* label, ImVec2 size, bool active,
                           const BringUpColors& c, bool danger = false, bool primaryFill = false) {
    ImVec2 pos = ImGui::GetCursorScreenPos();
    ImVec2 txtSz = ImGui::CalcTextSize(label);
    ImVec2 resolvedSz = size;
    if (resolvedSz.x <= 0.0f) resolvedSz.x = txtSz.x + 16.0f;
    if (resolvedSz.y <= 0.0f) resolvedSz.y = txtSz.y + 8.0f;

    bool pressed = ImGui::InvisibleButton(id, resolvedSz);
    bool hovered = ImGui::IsItemHovered();
    bool held = ImGui::IsItemActive();

    ImVec2 p0 = pos;
    ImVec2 p1(pos.x + resolvedSz.x, pos.y + resolvedSz.y);
    ImDrawList* dl = ImGui::GetWindowDrawList();

    ImVec4 bg = c.bgElevated;
    ImVec4 border = c.borderOuter;
    ImVec4 fg = c.textPrimary;

    if (danger) {
        bg = held ? c.errColor : (hovered ? rgbaHex(0x611A16) : rgbaHex(0x3D1210));
        if (c.isLightTheme && !held) bg = hovered ? rgbaHex(0xF6D2D0) : rgbaHex(0xFCE8E6);
        border = c.errColor;
        fg = (held || !c.isLightTheme) ? rgbaHex(0xFFECEB) : c.errColor;
    } else if (primaryFill) {
        bg = held ? c.phosphorMid : (hovered ? c.phosphorHi : c.phosphorMid);
        border = c.phosphorHi;
        fg = c.textInverse;
    } else if (active) {
        bg = c.phosphorGlow;
        border = c.phosphorHi;
        fg = c.phosphorHi;
    } else if (held) {
        bg = c.bgActive;
        border = c.phosphorHi;
        fg = c.phosphorHi;
    } else if (hovered) {
        bg = c.bgHover;
        border = c.phosphorMid;
        fg = c.textPrimary;
    }

    dl->AddRectFilled(p0, p1, toU32(bg), 0.0f);
    dl->AddRect(p0, p1, toU32(border), 0.0f, 0, active || primaryFill ? 1.5f : 1.0f);
    if (active && !primaryFill && !danger) {
        dl->AddRectFilled(p0, ImVec2(p0.x + 2.5f, p1.y), toU32(c.phosphorHi), 0.0f);
    }

    ImVec2 textPos(std::floor(p0.x + (resolvedSz.x - txtSz.x) * 0.5f),
                   std::floor(p0.y + (resolvedSz.y - txtSz.y) * 0.5f));
    dl->AddText(textPos, toU32(fg), label);
    return pressed;
}

inline bool hardwareArmToggle(const char* id, bool* v, const BringUpColors& c, float scale = 1.0f) {
    const float w = std::round(15.0f * scale);
    const float h = std::round(15.0f * scale);
    ImVec2 pos = ImGui::GetCursorScreenPos();
    bool clicked = ImGui::InvisibleButton(id, ImVec2(w, h));
    bool hovered = ImGui::IsItemHovered();
    if (clicked && v) {
        *v = !(*v);
    }
    const bool armed = (v && *v);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 p1(pos.x + w, pos.y + h);

    dl->AddRectFilled(pos, p1, toU32(c.bgRecessed), 0.0f);
    dl->AddRect(pos, p1, toU32(armed ? c.phosphorHi : (hovered ? c.phosphorMid : c.borderOuter)), 0.0f, 0, 1.0f);
    if (armed) {
        float pad = std::max(2.0f, std::round(3.0f * scale));
        dl->AddRectFilled(ImVec2(pos.x + pad, pos.y + pad),
                          ImVec2(p1.x - pad, p1.y - pad),
                          toU32(c.phosphorHi), 0.0f);
    }
    return clicked;
}

inline void drawSteppedLedBar(ImDrawList* dl, ImVec2 p0, ImVec2 p1, float fraction,
                              const BringUpColors& c, int segments = 16, bool unsupported = false) {
    float f = std::clamp(fraction, 0.0f, 1.0f);
    dl->AddRectFilled(p0, p1, toU32(c.bgRecessed), 0.0f);
    dl->AddRect(p0, p1, toU32(c.borderInner), 0.0f, 0, 1.0f);

    if (segments <= 0) return;
    float innerX0 = p0.x + 2.0f;
    float innerX1 = p1.x - 2.0f;
    float innerY0 = p0.y + 2.0f;
    float innerY1 = p1.y - 2.0f;
    float totalW = innerX1 - innerX0;
    if (totalW <= 4.0f || innerY1 <= innerY0) return;

    float gap = 1.5f;
    float segW = (totalW - gap * static_cast<float>(segments - 1)) / static_cast<float>(segments);
    if (segW < 1.0f) segW = 1.0f;

    int lit = unsupported ? 0 : static_cast<int>(std::round(f * static_cast<float>(segments)));
    for (int i = 0; i < segments; ++i) {
        float sx0 = innerX0 + static_cast<float>(i) * (segW + gap);
        float sx1 = std::min(innerX1, sx0 + segW);
        ImVec4 segCol = c.ledOff;
        if (i < lit) {
            float ratio = static_cast<float>(i) / static_cast<float>(std::max(1, segments - 1));
            if (ratio > 0.82f) {
                segCol = c.okColor;
            } else if (ratio > 0.45f) {
                segCol = c.phosphorHi;
            } else {
                segCol = c.phosphorMid;
            }
        } else if (unsupported && (i % 4 == 0)) {
            segCol = ImVec4(c.errColor.x, c.errColor.y, c.errColor.z, 0.35f);
        }
        dl->AddRectFilled(ImVec2(sx0, innerY0), ImVec2(sx1, innerY1), toU32(segCol), 0.0f);
    }
}

inline void drawVgprOccupancyGrid(ImDrawList* dl, ImVec2 p0, ImVec2 p1,
                                  uint32_t vgprs, uint32_t sgprs,
                                  uint32_t ldsBytes, uint32_t maxWaves,
                                  const BringUpColors& c) {
    dl->AddRectFilled(p0, p1, toU32(c.bgRecessed), 0.0f);
    dl->AddRect(p0, p1, toU32(c.borderOuter), 0.0f, 0, 1.0f);
    dl->PushClipRect(p0, p1, true);

    float w = p1.x - p0.x;
    float pad = 8.0f;
    float lineH = ImGui::GetTextLineHeight();

    char hdr[160];
    std::snprintf(hdr, sizeof(hdr), "VGPR %u/256  SGPR %u/128  LDS %u B  WAVES %u/16",
                  vgprs, sgprs, ldsBytes, maxWaves);
    float maxHdrW = (w - 2.0f * pad) > 0.0f ? (w - 2.0f * pad) : 0.0f;
    std::string fittedHdr = fitBringUpText(hdr, maxHdrW);
    dl->AddText(ImVec2(p0.x + pad, p0.y + 5.0f), toU32(c.textPrimary), fittedHdr.c_str());

    // 64 pips (4 rows x 16 cols) representing 256 VGPR budget (each pip = 4 VGPRs)
    const int cols = 16;
    const int rows = 4;
    float gridX0 = p0.x + pad;
    float gridY0 = p0.y + 8.0f + lineH;
    float gridW = std::max(80.0f, w - 2.0f * pad);
    float gridH = std::max(24.0f, (p1.y - 6.0f) - gridY0);
    float gap = 2.0f;
    float cellW = (gridW - gap * (cols - 1)) / static_cast<float>(cols);
    float cellH = (gridH - gap * (rows - 1)) / static_cast<float>(rows);

    int activePips = static_cast<int>(std::ceil(static_cast<float>(std::min<uint32_t>(vgprs, 256u)) / 4.0f));
    for (int r = 0; r < rows; ++r) {
        for (int col = 0; col < cols; ++col) {
            int idx = r * cols + col;
            ImVec2 cp0(gridX0 + col * (cellW + gap), gridY0 + r * (cellH + gap));
            ImVec2 cp1(cp0.x + cellW, cp0.y + cellH);
            ImVec4 fill = c.ledOff;
            if (idx < activePips) {
                if (idx >= 48)      fill = c.errColor;
                else if (idx >= 32) fill = c.warnColor;
                else if (idx >= 16) fill = c.phosphorHi;
                else                fill = c.okColor;
            }
            dl->AddRectFilled(cp0, cp1, toU32(fill), 0.0f);
        }
    }
    dl->PopClipRect();
}

inline void drawInstructionMixStrip(ImDrawList* dl, ImVec2 p0, ImVec2 p1,
                                    int wmma, int valu, int vmem, int lds, int bvh, int salu, int other,
                                    const BringUpColors& c) {
    dl->AddRectFilled(p0, p1, toU32(c.bgRecessed), 0.0f);
    dl->AddRect(p0, p1, toU32(c.borderInner), 0.0f, 0, 1.0f);

    int total = wmma + valu + vmem + lds + bvh + salu + other;
    if (total <= 0) {
        dl->AddText(ImVec2(p0.x + 6.0f, p0.y + 3.0f), toU32(c.textMuted), "NO ISA STATS CAPTURED");
        return;
    }

    struct Seg { const char* name; int count; ImVec4 col; };
    Seg segs[] = {
        {"WMMA", wmma, c.phosphorHi},
        {"BVH",  bvh,  c.cyanAccent},
        {"VMEM", vmem, c.okColor},
        {"LDS",  lds,  c.violetAccent},
        {"VALU", valu, c.phosphorMid},
        {"SALU", salu, c.textSecondary},
        {"CTRL", other, c.textMuted}
    };

    float innerX = p0.x + 2.0f;
    float barW = (p1.x - p0.x) - 4.0f;
    float barY0 = p0.y + 2.0f;
    float barY1 = p1.y - 2.0f;

    for (const auto& sg : segs) {
        if (sg.count <= 0) continue;
        float frac = static_cast<float>(sg.count) / static_cast<float>(total);
        float segW = std::max(2.0f, barW * frac);
        if (innerX + segW > p1.x - 2.0f) segW = (p1.x - 2.0f) - innerX;
        if (segW <= 0.5f) continue;
        dl->AddRectFilled(ImVec2(innerX, barY0), ImVec2(innerX + segW - 1.0f, barY1), toU32(sg.col), 0.0f);
        innerX += segW;
    }
}

} // namespace gpubench::gui
