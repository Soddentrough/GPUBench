#include "GuiApp.h"
#include "GuiUiHelpers.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <string>
#include <vector>

namespace gpubench::gui {

namespace {

std::string toLowerAscii(const std::string& s) {
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return out;
}

ImVec4 classifyIsaLineColor(const std::string& line, const BringUpColors& c, bool& isLabelOut) {
    isLabelOut = false;
    size_t first = line.find_first_not_of(" \t");
    if (first == std::string::npos) return c.textMuted;

    std::string trimmed = line.substr(first);
    while (!trimmed.empty() && (trimmed.back() == ' ' || trimmed.back() == '\t' || trimmed.back() == '\r')) {
        trimmed.pop_back();
    }
    if (trimmed.empty()) return c.textMuted;

    if (trimmed[0] == ';' || trimmed.rfind("//", 0) == 0 || trimmed.rfind("/*", 0) == 0) {
        return c.textMuted;
    }
    if (trimmed.back() == ':' && trimmed.find(' ') == std::string::npos) {
        isLabelOut = true;
        return c.cyanAccent;
    }
    if (trimmed.find('<') != std::string::npos && trimmed.back() == ':') {
        isLabelOut = true;
        return c.cyanAccent;
    }

    size_t sp = trimmed.find_first_of(" \t");
    std::string op = toLowerAscii(sp == std::string::npos ? trimmed : trimmed.substr(0, sp));

    if (op.find("wmma") != std::string::npos || op.find("mfma") != std::string::npos ||
        op.find("dot") != std::string::npos || op.find("fmac") != std::string::npos) {
        return c.phosphorHi;
    }
    if (op.find("bvh") != std::string::npos) {
        return c.cyanAccent;
    }
    if (op.rfind("ds_", 0) == 0) {
        return c.violetAccent;
    }
    if (op.rfind("global_", 0) == 0 || op.rfind("buffer_", 0) == 0 ||
        op.rfind("flat_", 0) == 0 || op.rfind("image_", 0) == 0 ||
        op.rfind("tbuffer_", 0) == 0 || op.rfind("scratch_", 0) == 0) {
        return c.okColor;
    }
    if (op.rfind("s_cbranch", 0) == 0 || op.rfind("s_branch", 0) == 0 ||
        op.rfind("s_wait", 0) == 0 || op.rfind("s_barrier", 0) == 0 ||
        op.rfind("s_endpgm", 0) == 0) {
        return c.warnColor;
    }
    if (op.rfind("v_", 0) == 0) {
        return c.textPrimary;
    }
    if (op.rfind("s_", 0) == 0) {
        return c.textSecondary;
    }
    return c.textSecondary;
}

} // namespace

void GuiApp::renderIsaInspectorPanel() {
    const BringUpColors c = getBringUpColors(m_palette);
    const bool isRunning = (m_execState == ExecutionState::Running);

    ensureDefaultInspectedBenchmark();
    const BenchmarkItem* item = getInspectedItem();
    if (!item) {
        ImGui::TextColored(c.textMuted, "NO WORKLOAD SLOT SELECTED IN ROM MATRIX.");
        return;
    }

    const std::string cacheKey = makeIsaCacheKey(*item);
    IsaInspectionResult cachedIsa;
    bool hasCachedIsa = false;
    {
        std::lock_guard<std::mutex> lock(m_isaMutex);
        auto it = m_isaCache.find(cacheKey);
        if (it != m_isaCache.end()) {
            cachedIsa = it->second;
            hasCachedIsa = true;
        }
    }
    const bool isLoadingThis = m_isaLoading.load() && (m_isaActiveKey == cacheKey);

    if (!hasCachedIsa && !isLoadingThis && !isRunning) {
        requestIsaInspection(*item, false);
    }

    const IsaArtifactReport* activeArt = nullptr;
    if (hasCachedIsa && !cachedIsa.artifacts.empty()) {
        for (const auto& a : cachedIsa.artifacts) {
            if (a.configIndex == item->configIndex) {
                activeArt = &a;
                break;
            }
        }
        if (!activeArt) {
            for (const auto& a : cachedIsa.artifacts) {
                if (!a.executables.empty()) {
                    activeArt = &a;
                    break;
                }
            }
        }
        if (!activeArt) {
            activeArt = &cachedIsa.artifacts.front();
        }
    }

    const IsaExecutableReport* activeExec = nullptr;
    if (activeArt && !activeArt->executables.empty()) {
        if (m_isaExecStageIdx < 0 ||
            m_isaExecStageIdx >= static_cast<int>(activeArt->executables.size())) {
            m_isaExecStageIdx = 0;
        }
        activeExec = &activeArt->executables[static_cast<size_t>(m_isaExecStageIdx)];
    }

    // ------------------------------------------------------------------------
    // F1 Top Slot Header Bar
    // ------------------------------------------------------------------------
    ImGui::BeginChild("F1TopSlotBar", ImVec2(0, s(42.0f)), ImGuiChildFlags_Borders);
    {
        const float btnH = s(24.0f);
        const float execW = s(98.0f);
        const float reprobeW = s(104.0f);
        const float rightBtnsW = execW + reprobeW + s(20.0f);

        pushBoldFont();
        ImGui::TextColored(c.phosphorHi, "[SLOT // %s]", item->id.c_str());
        popBoldFont();
        ImGui::SameLine(0.0f, s(8.0f));

        std::string infoStr = item->name;
        if (activeArt && !activeArt->artifactFile.empty()) {
            infoStr += "  |  " + activeArt->artifactFile;
        }
        if (hasCachedIsa && !cachedIsa.resolvedAsic.empty()) {
            infoStr += "  |  " + cachedIsa.resolvedAsic + " [" + m_selectedBackend + "]";
        } else {
            infoStr += "  |  [" + m_selectedBackend + "]";
        }

        const float maxTextW = std::max(s(120.0f), ImGui::GetContentRegionAvail().x - rightBtnsW);
        std::string fitted = fitTextToPixelWidth(infoStr, maxTextW);
        ImGui::TextColored(c.textPrimary, "%s", fitted.c_str());
        if (fitted != infoStr && ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s", infoStr.c_str());
        }

        const float rightEdge = ImGui::GetWindowWidth() - s(10.0f);
        ImGui::SameLine(std::max(ImGui::GetCursorPosX() + s(6.0f), rightEdge - execW - reprobeW - s(6.0f)));

        ImGui::BeginDisabled(isRunning || isLoadingThis);
        if (tacticalButton("##F1Reprobe", "RE-PROBE ISA", ImVec2(reprobeW, btnH), false, c)) {
            requestIsaInspection(*item, true);
        }
        ImGui::EndDisabled();

        ImGui::SameLine(0.0f, s(6.0f));
        ImGui::BeginDisabled(isRunning || !item->isSupported);
        if (tacticalButton("##F1ExecSlot", "RUN TEST", ImVec2(execW, btnH), false, c, false, true)) {
            runSingleBenchmarkItem(*item);
        }
        ImGui::EndDisabled();
    }
    ImGui::EndChild();

    ImGui::Spacing();

    // ------------------------------------------------------------------------
    // 2-Column F1 Analyzer Stage
    // ------------------------------------------------------------------------
    const float availW = ImGui::GetContentRegionAvail().x;
    const float availH = std::max(s(220.0f), ImGui::GetContentRegionAvail().y);
    const float leftW = std::clamp(std::floor(availW * 0.43f), s(290.0f), std::max(s(300.0f), availW * 0.48f));
    const float rightW = std::max(s(260.0f), availW - leftW - s(8.0f));

    // LEFT COLUMN: KPIs, 64-Pip VGPR Matrix, Instruction Mix, Microarchitecture Schematic
    ImGui::BeginChild("F1LeftTelemetry", ImVec2(leftW, availH), ImGuiChildFlags_Borders);
    {
        BenchmarkDisplayInfo disp = getBenchmarkDisplayInfo(*item, m_telemetryGpuIndex);

        drawEtchedHeader("01", "MEASURED KPI & VARIANCE", c, item->metricType.c_str());
        ImGui::Spacing();

        if (disp.hasResult) {
            pushBoldFont();
            ImGui::TextColored(c.phosphorHi, "%s", disp.scoreText.c_str());
            popBoldFont();
            if (disp.hasComparison && !disp.deltaText.empty()) {
                ImGui::SameLine(0.0f, s(12.0f));
                ImGui::TextColored(disp.deltaColor, "[VS BASE %s]", disp.deltaText.c_str());
            }

            if (disp.primaryResult.sample_count > 0 || disp.primaryResult.median_time_ms > 0.0) {
                ImGui::TextColored(c.textSecondary, "MIN %.3f ms   MED %.3f ms   P95 %.3f ms   N=%u",
                                   disp.primaryResult.min_time_ms,
                                   disp.primaryResult.median_time_ms,
                                   disp.primaryResult.p95_time_ms,
                                   disp.primaryResult.sample_count);
            } else {
                ImGui::TextColored(c.textSecondary, "TIME %.3f ms   OI %.2f FLOP/B",
                                   disp.primaryResult.time_ms,
                                   disp.primaryResult.operationalIntensity);
            }

            if (disp.primaryResult.hasPowerTelemetry && disp.primaryResult.powerWatts > 0.0) {
                ImGui::TextColored(c.okColor, "PWR %.1f W   ENERGY %.2f J   EFF %.2f %s",
                                   disp.primaryResult.powerWatts,
                                   disp.primaryResult.energyJoules,
                                   disp.primaryResult.unitPerWatt,
                                   disp.primaryResult.efficiencyUnit.c_str());
            } else if (!disp.subResults.empty()) {
                for (size_t si = 0; si < std::min<size_t>(disp.subResults.size(), 2); ++si) {
                    const auto& sr = disp.subResults[si];
                    ImGui::TextColored(c.textSecondary, "  %s: %s", sr.label.c_str(), sr.score.c_str());
                }
            }
        } else {
            ImGui::TextColored(c.textMuted,
                               isRunning ? "RUN ACTIVE — TELEMETRY UPDATING ON COMPLETION"
                                         : "UNMEASURED — CLICK [RUN TEST] OR [RUN TESTS]");
        }

        ImGui::Spacing();
        drawEtchedHeader("02", "REGISTER FILE & INSTRUCTION MIX", c,
                         activeExec ? activeExec->name.c_str() : "HARDWARE PROBE");
        ImGui::Spacing();

        uint32_t vgprs = activeExec ? activeExec->vgprCount : 0;
        uint32_t sgprs = activeExec ? activeExec->sgprCount : 0;
        uint32_t ldsB = activeExec ? activeExec->ldsBytes : 0;
        uint32_t scratchB = activeExec ? activeExec->scratchBytes : 0;
        uint32_t subgrpSz = activeExec ? activeExec->subgroupSize : 0;
        uint32_t waves = 0;
        if (disp.hasResult && disp.primaryResult.hasRegisterTelemetry) {
            if (vgprs == 0) vgprs = static_cast<uint32_t>(disp.primaryResult.vgprCount);
            if (sgprs == 0) sgprs = static_cast<uint32_t>(disp.primaryResult.sgprCount);
            if (ldsB == 0) ldsB = static_cast<uint32_t>(disp.primaryResult.ldsSizeBytes);
            if (scratchB == 0) scratchB = static_cast<uint32_t>(disp.primaryResult.scratchSizeBytes);
            waves = static_cast<uint32_t>(disp.primaryResult.maxWavesPerSimd);
        }
        if (waves == 0 && vgprs > 0) {
            waves = std::clamp(1536u / std::max(vgprs, 1u), 1u, 16u);
        }

        ImDrawList* dl = ImGui::GetWindowDrawList();
        const float gridW = std::max(s(180.0f), ImGui::GetContentRegionAvail().x);
        const float gridH = s(74.0f);
        ImVec2 gp0 = ImGui::GetCursorScreenPos();
        ImVec2 gp1(gp0.x + gridW, gp0.y + gridH);
        ImGui::Dummy(ImVec2(gridW, gridH));
        drawVgprOccupancyGrid(dl, gp0, gp1, vgprs, sgprs, ldsB, waves, c);
        if (ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            pushBoldFont();
            ImGui::TextColored(c.phosphorHi, "REGISTER FILE & WAVE OCCUPANCY");
            popBoldFont();
            ImGui::Separator();
            const float occPct = (static_cast<float>(waves) / 16.0f) * 100.0f;
            ImGui::TextColored(c.okColor, "VGPR (Vector Registers): %u / 256 per lane", vgprs);
            ImGui::TextWrapped("  32-bit registers private to each thread/lane. Each block in the 4x16 grid = 4 VGPRs (Green <=64, Accent <=128, Warn <=192, Red >192). Scratch spill: %u B/lane.", scratchB);
            ImGui::TextColored(c.cyanAccent, "SGPR (Scalar Registers): %u / 128 per wave", sgprs);
            ImGui::TextWrapped("  32-bit registers shared across all lanes in a wave for descriptors, loop counters, and uniform constants.");
            ImGui::TextColored(c.violetAccent, "LDS (Local Data Share): %u B", ldsB);
            ImGui::TextWrapped("  On-CU shared scratchpad memory allocated per workgroup for thread cooperation and matrix/shared tiling.");
            if (subgrpSz > 0) {
                ImGui::TextColored(c.phosphorHi, "WAVES / SIMD: %u / 16 (%.0f%% occupancy | Wave%u)", waves, occPct, subgrpSz);
            } else {
                ImGui::TextColored(c.phosphorHi, "WAVES / SIMD: %u / 16 (%.0f%% occupancy)", waves, occPct);
            }
            ImGui::TextWrapped("  Max concurrent waves resident per SIMD. Higher occupancy hides memory latency; dense matrix tiles trade waves for register reuse.");
            ImGui::EndTooltip();
        }

        ImGui::Spacing();
        const float stripH = s(18.0f);
        ImVec2 sp0 = ImGui::GetCursorScreenPos();
        ImVec2 sp1(sp0.x + gridW, sp0.y + stripH);
        ImGui::Dummy(ImVec2(gridW, stripH));
        if (activeExec) {
            const auto& mx = activeExec->mix;
            drawInstructionMixStrip(dl, sp0, sp1,
                                    mx.wmma, mx.valu, mx.vmem, mx.lds, mx.bvh, mx.salu, mx.other, c);
            if (ImGui::IsItemHovered()) {
                ImGui::BeginTooltip();
                pushBoldFont();
                ImGui::TextColored(c.phosphorHi, "COMPILED HARDWARE INSTRUCTION MIX");
                popBoldFont();
                ImGui::Separator();
                if (mx.total > 0) {
                    const double invTot = 100.0 / static_cast<double>(mx.total);
                    ImGui::Text("Total Compiled Instructions: %d", mx.total);
                    ImGui::TextColored(c.phosphorHi,    "  WMMA (Matrix / Tensor): %4d (%.1f%%)", mx.wmma, mx.wmma * invTot);
                    ImGui::TextColored(c.phosphorMid,   "  VALU (Vector ALU):      %4d (%.1f%%)", mx.valu, mx.valu * invTot);
                    ImGui::TextColored(c.okColor,       "  VMEM (Vector Memory):   %4d (%.1f%%)", mx.vmem, mx.vmem * invTot);
                    ImGui::TextColored(c.violetAccent,  "  LDS  (Shared Memory):   %4d (%.1f%%)", mx.lds,  mx.lds  * invTot);
                    ImGui::TextColored(c.cyanAccent,    "  BVH  (Ray Intersection):%4d (%.1f%%)", mx.bvh,  mx.bvh  * invTot);
                    ImGui::TextColored(c.textSecondary, "  SALU (Scalar ALU):      %4d (%.1f%%)", mx.salu, mx.salu * invTot);
                    ImGui::TextColored(c.textMuted,     "  CTRL (Branch / Sync):   %4d (%.1f%%)", mx.other, mx.other * invTot);
                    if (!activeExec->topOpcodes.empty()) {
                        ImGui::Separator();
                        ImGui::TextColored(c.textMuted, "Dominant Opcodes:");
                        const size_t opLimit = std::min<size_t>(activeExec->topOpcodes.size(), 5);
                        for (size_t oi = 0; oi < opLimit; ++oi) {
                            ImGui::Text("  %s x%d", activeExec->topOpcodes[oi].first.c_str(), activeExec->topOpcodes[oi].second);
                        }
                    }
                } else {
                    ImGui::TextColored(c.textMuted, "Instruction mix breakdown populates once compiled hardware ISA is probed.");
                }
                ImGui::EndTooltip();
            }

            char leg[160];
            if (mx.bvh > 0) {
                std::snprintf(leg, sizeof(leg),
                              "INS:%d WMMA:%d VALU:%d VMEM:%d LDS:%d BVH:%d SALU:%d",
                              mx.total, mx.wmma, mx.valu, mx.vmem, mx.lds, mx.bvh, mx.salu);
            } else {
                std::snprintf(leg, sizeof(leg),
                              "INS:%d WMMA:%d VALU:%d VMEM:%d LDS:%d SALU:%d",
                              mx.total, mx.wmma, mx.valu, mx.vmem, mx.lds, mx.salu);
            }
            ImGui::PushStyleColor(ImGuiCol_Text, c.textSecondary);
            ImGui::TextWrapped("%s", leg);
            ImGui::PopStyleColor();

            if (!activeExec->topOpcodes.empty()) {
                std::string topLine = "TOP:";
                const size_t limit = std::min<size_t>(activeExec->topOpcodes.size(), 4);
                for (size_t oi = 0; oi < limit; ++oi) {
                    topLine += "  " + activeExec->topOpcodes[oi].first + "(" +
                               std::to_string(activeExec->topOpcodes[oi].second) + ")";
                }
                ImGui::TextColored(c.textMuted, "%s", fitTextToPixelWidth(topLine, gridW).c_str());
            }
        } else {
            drawInstructionMixStrip(dl, sp0, sp1, 0, 0, 0, 0, 0, 0, 0, c);
        }

        ImGui::Spacing();
        drawEtchedHeader("03", "MICROARCHITECTURE BLOCK DIAGRAM", c,
                         item->subcategory.c_str());
        ImGui::Spacing();

        const std::string diagramId = resolveDiagramIdForItem(
            item->id, item->subcategory, item->name, item->viz.diagramId);
        const float remainH = ImGui::GetContentRegionAvail().y;
        const float descReserve = s(52.0f);
        const float diagH = std::clamp(remainH - descReserve, s(104.0f), s(190.0f));
        ImVec2 dp0 = ImGui::GetCursorScreenPos();
        ImVec2 dp1(dp0.x + gridW, dp0.y + diagH);
        ImGui::Dummy(ImVec2(gridW, diagH));
        renderProceduralDiagram(diagramId, dp0, dp1);

        if (!item->description.empty()) {
            ImGui::Spacing();
            ImGui::PushStyleColor(ImGuiCol_Text, c.textSecondary);
            ImGui::TextWrapped("%s", item->description.c_str());
            ImGui::PopStyleColor();
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("%s", item->description.c_str());
            }
        }
    }
    ImGui::EndChild();

    ImGui::SameLine(0.0f, s(8.0f));

    // RIGHT COLUMN: Hardware Shader / Kernel ISA Disassembly
    ImGui::BeginChild("F1RightIsaDisassembly", ImVec2(rightW, availH), ImGuiChildFlags_Borders);
    {
        drawEtchedHeader("04", "LIVE HARDWARE ISA DISASSEMBLY", c,
                         activeExec && activeExec->hotLoopStartLine > 0 ? "HOT-LOOP TAGGED" : "NATIVE PROBE");

        // Controls Toolbar
        ImGui::Spacing();
        if (activeArt && activeArt->executables.size() > 1) {
            for (size_t ei = 0; ei < activeArt->executables.size(); ++ei) {
                if (ei > 0) ImGui::SameLine(0.0f, s(4.0f));
                const auto& ex = activeArt->executables[ei];
                std::string lbl = ex.name.empty() ? ("STAGE " + std::to_string(ei)) : ex.name;
                if (lbl.size() > 18) lbl = lbl.substr(0, 15) + "...";
                std::string bid = "##IsaStage" + std::to_string(ei);
                if (tacticalButton(bid.c_str(), lbl.c_str(), ImVec2(0, s(22.0f)),
                                   m_isaExecStageIdx == static_cast<int>(ei), c)) {
                    m_isaExecStageIdx = static_cast<int>(ei);
                }
            }
        }

        const float modeBtnH = s(22.0f);
        if (tacticalButton("##IsaHotLoopBtn", "HOT LOOP", ImVec2(s(84.0f), modeBtnH),
                           m_isaHotLoopOnly, c)) {
            m_isaHotLoopOnly = true;
        }
        ImGui::SameLine(0.0f, s(4.0f));
        if (tacticalButton("##IsaFullBtn", "FULL ISA", ImVec2(s(84.0f), modeBtnH),
                           !m_isaHotLoopOnly, c)) {
            m_isaHotLoopOnly = false;
        }
        ImGui::SameLine(0.0f, s(10.0f));
        ImGui::SetNextItemWidth(std::min(s(240.0f), std::max(s(120.0f), ImGui::GetContentRegionAvail().x - s(130.0f))));
        ImGui::InputTextWithHint("##IsaFilterInput", "FILTER OPCODE / REG...", m_isaFilterBuf, sizeof(m_isaFilterBuf));

        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_ChildBg, c.isLightTheme ? c.bgElevated : c.bgVoid);
        ImGui::BeginChild("IsaDisasmViewport", ImVec2(0, 0), ImGuiChildFlags_Borders,
                          ImGuiWindowFlags_HorizontalScrollbar);
        {
            if (isLoadingThis) {
                ImGui::TextColored(c.phosphorHi, "PROBING HARDWARE PIPELINE EXECUTABLES...");
            } else if (!hasCachedIsa && isRunning) {
                ImGui::TextColored(c.warnColor,
                                   "ZERO-CONTENTION INTERLOCK: HARDWARE ISA PROBE DEFERRED DURING ACTIVE RUN.");
            } else if (activeArt && !activeArt->skipReason.empty()) {
                ImGui::TextColored(c.warnColor, "[SKIPPED] %s", activeArt->skipReason.c_str());
            } else if (activeArt && !activeArt->errorMessage.empty()) {
                ImGui::TextColored(c.errColor, "[ERROR] %s", activeArt->errorMessage.c_str());
            } else if (hasCachedIsa && !cachedIsa.ok && !cachedIsa.errorSummary.empty()) {
                ImGui::TextColored(c.errColor, "[PROBE INFO] %s", cachedIsa.errorSummary.c_str());
            } else if (!activeExec || activeExec->fullIsaLines.empty()) {
                ImGui::TextColored(c.textMuted,
                                   "NO COMPILED HARDWARE ISA ARTIFACT FOR THIS SLOT (HOST / CAPABILITY SLOT).");
            } else {
                const bool useHotLoop = m_isaHotLoopOnly &&
                                        !activeExec->hotLoopLines.empty() &&
                                        activeExec->hotLoopStartLine > 0;
                const std::vector<std::string>& srcLines =
                    useHotLoop ? activeExec->hotLoopLines : activeExec->fullIsaLines;
                const int startLineOneBased = useHotLoop ? activeExec->hotLoopStartLine : 1;

                std::string filterNeedle(m_isaFilterBuf);
                std::vector<int> visibleIndices;
                visibleIndices.reserve(srcLines.size());
                for (int i = 0; i < static_cast<int>(srcLines.size()); ++i) {
                    if (filterNeedle.empty() ||
                        containsCaseInsensitive(srcLines[static_cast<size_t>(i)], filterNeedle)) {
                        visibleIndices.push_back(i);
                    }
                }

                const std::string modeRangeLabel = useHotLoop
                    ? ("HOT LOOP LINES " + std::to_string(activeExec->hotLoopStartLine) + "-" +
                       std::to_string(activeExec->hotLoopEndLine))
                    : "FULL SHADER";
                ImGui::TextColored(c.textMuted,
                                   "SHOWING %zu / %zu LINES (%s)",
                                   visibleIndices.size(),
                                   activeExec->fullIsaLines.size(),
                                   modeRangeLabel.c_str());
                ImGui::Separator();

                pushMonoFont();
                ImGuiListClipper clipper;
                clipper.Begin(static_cast<int>(visibleIndices.size()));
                while (clipper.Step()) {
                    for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; ++row) {
                        const int srcIdx = visibleIndices[static_cast<size_t>(row)];
                        const int lineNo = startLineOneBased + srcIdx;
                        const std::string& lineText = srcLines[static_cast<size_t>(srcIdx)];

                        const bool inHotLoop = (activeExec->hotLoopStartLine > 0 &&
                                                lineNo >= activeExec->hotLoopStartLine &&
                                                lineNo <= activeExec->hotLoopEndLine);

                        bool isLabel = false;
                        const ImVec4 textCol = classifyIsaLineColor(lineText, c, isLabel);

                        char lnBuf[20];
                        std::snprintf(lnBuf, sizeof(lnBuf), "%04d", lineNo);
                        ImGui::TextColored(c.textMuted, "%s", lnBuf);
                        ImGui::SameLine(0.0f, s(6.0f));
                        ImGui::TextColored(inHotLoop ? c.phosphorHi : c.borderInner, "%s", inHotLoop ? "|" : " ");
                        ImGui::SameLine(0.0f, s(6.0f));
                        ImGui::TextColored(textCol, "%s", lineText.c_str());
                    }
                }
                popMonoFont();
            }
        }
        ImGui::EndChild();
        ImGui::PopStyleColor();
    }
    ImGui::EndChild();
}

} // namespace gpubench::gui
