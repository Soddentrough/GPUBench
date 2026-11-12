#pragma once

#include "core/ResultFormatter.h"
#include <imgui.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <string>

namespace gpubench::gui {

inline std::string fitTextToPixelWidth(const std::string& text, float maxWidth) {
    if (text.empty() || maxWidth <= 0.0f) return text;
    if (ImGui::CalcTextSize(text.c_str()).x <= maxWidth) return text;
    const std::string ellipsis = "...";
    float ellW = ImGui::CalcTextSize(ellipsis.c_str()).x;
    if (maxWidth <= ellW) return ellipsis;
    size_t len = text.size();
    while (len > 0) {
        do {
            --len;
        } while (len > 0 && (static_cast<unsigned char>(text[len]) & 0xC0) == 0x80);
        std::string candidate = text.substr(0, len);
        while (!candidate.empty() && candidate.back() == ' ') {
            candidate.pop_back();
        }
        if (ImGui::CalcTextSize((candidate + ellipsis).c_str()).x <= maxWidth) {
            return candidate + ellipsis;
        }
    }
    return ellipsis;
}

inline bool sameLineIfFits(float nextItemW, float spacing = -1.0f) {
    float sp = (spacing >= 0.0f) ? spacing : ImGui::GetStyle().ItemSpacing.x;
    float limitRight = ImGui::GetCursorScreenPos().x + ImGui::GetContentRegionAvail().x;
    float nextRight = ImGui::GetItemRectMax().x + sp + nextItemW;
    if (nextRight <= limitRight) {
        ImGui::SameLine(0.0f, spacing);
        return true;
    }
    return false;
}

inline bool containsCaseInsensitive(const std::string& haystack, const std::string& needle) {
    if (needle.empty()) return true;
    auto it = std::search(haystack.begin(), haystack.end(), needle.begin(), needle.end(),
        [](char c1, char c2) {
            return std::tolower(static_cast<unsigned char>(c1)) ==
                   std::tolower(static_cast<unsigned char>(c2));
        });
    return (it != haystack.end());
}

inline bool equalsCaseInsensitive(const std::string& a, const std::string& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::tolower(static_cast<unsigned char>(a[i])) !=
            std::tolower(static_cast<unsigned char>(b[i]))) {
            return false;
        }
    }
    return true;
}

inline bool isLowerBetterMetric(const std::string& metric) {
    return metric == "ns" || metric == "us" || metric == "ms" ||
           metric.find("ns") != std::string::npos;
}

inline double computeThroughputOrInvLatency(const ResultData& r) {
    if (r.isUnsupported || r.time_ms <= 0.0) return 0.0;
    if (r.metric == "us" || r.metric == "ms") {
        const double ops = (r.operations > 0) ? static_cast<double>(r.operations) : 1.0;
        return (ops * 1000.0) / r.time_ms;
    }
    if (r.metric.find("ns") != std::string::npos) {
        if (r.operations == 0) return 0.0;
        double nsPerOp = (r.time_ms * 1e6) / static_cast<double>(r.operations);
        return (nsPerOp > 0.0) ? (1e9 / nsPerOp) : 0.0;
    }
    if (r.operations == 0) return 0.0;
    return (static_cast<double>(r.operations) / r.time_ms) * 1000.0;
}

inline std::string formatResultScoreString(const ResultData& r, bool compactPrecision = false) {
    if (r.time_ms == -3.0) return "ABORTED";
    if (r.time_ms < 0.0) return "FAILED";
    if (r.time_ms == 0.0 || (r.operations == 0 && r.metric != "us" && r.metric != "ms")) return "-";

    double opsPerSec = (r.operations > 0)
        ? ((static_cast<double>(r.operations) / r.time_ms) * 1000.0)
        : 0.0;

    char buf[64];
    const char* f1 = compactPrecision ? "%.1f %s" : "%.2f %s";
    if (r.metric.find("TFLOPS") != std::string::npos || r.metric.find("TOPS") != std::string::npos) {
        std::snprintf(buf, sizeof(buf), "%.2f %s", opsPerSec / 1e12, r.metric.c_str());
    } else if (r.metric.find("GFLOPS") != std::string::npos || r.metric.find("GOPS") != std::string::npos) {
        std::snprintf(buf, sizeof(buf), "%.2f %s", opsPerSec / 1e9, r.metric.c_str());
    } else if (r.metric.find("TB/s") != std::string::npos) {
        std::snprintf(buf, sizeof(buf), "%.2f TB/s", opsPerSec / 1e12);
    } else if (r.metric.find("GB/s") != std::string::npos) {
        std::snprintf(buf, sizeof(buf), compactPrecision ? "%.1f GB/s" : "%.2f GB/s", opsPerSec / 1e9);
    } else if (r.metric.find("GIS/s") != std::string::npos) {
        std::snprintf(buf, sizeof(buf), "%.2f GIS/s", opsPerSec / 1e9);
    } else if (r.metric.find("GPixels/s") != std::string::npos) {
        std::snprintf(buf, sizeof(buf), "%.2f GPixels/s", opsPerSec / 1e9);
    } else if (r.metric.find("MRays/s") != std::string::npos ||
               r.metric.find("MBVH/s") != std::string::npos ||
               r.metric.find("MTris/s") != std::string::npos ||
               r.metric.find("MInst/s") != std::string::npos ||
               r.metric.find("MHits/s") != std::string::npos ||
               r.metric.find("MItems/s") != std::string::npos ||
               r.metric.find("MRecords/s") != std::string::npos) {
        std::snprintf(buf, sizeof(buf), f1, opsPerSec / 1e6, r.metric.c_str());
    } else if (r.metric.find("ns") != std::string::npos) {
        double nsVal = (r.operations > 0) ? ((r.time_ms * 1e6) / static_cast<double>(r.operations)) : r.time_ms;
        std::snprintf(buf, sizeof(buf), compactPrecision ? "%.1f ns" : "%.2f ns", nsVal);
    } else if (r.metric == "us") {
        double usVal = (r.operations > 1) ? ((r.time_ms * 1000.0) / static_cast<double>(r.operations)) : (r.time_ms * 1000.0);
        std::snprintf(buf, sizeof(buf), compactPrecision ? "%.1f us" : "%.2f us", usVal);
    } else if (r.metric == "ms") {
        std::snprintf(buf, sizeof(buf), "%.2f ms", r.time_ms);
    } else {
        std::snprintf(buf, sizeof(buf), f1, opsPerSec, r.metric.c_str());
    }
    return std::string(buf);
}

} // namespace gpubench::gui
