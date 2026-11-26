#pragma once

#include "core/ResultFormatter.h"
#include "core/RunnerAPI.h"
#include <cstdint>
#include <string>

namespace gpubench::gui {

inline std::string fallbackDescriptionForBenchmark(const BenchmarkConfigMetadata& meta) {
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
    if (bn == "DualIssue" || bn.find("Dual-Issue") != std::string::npos) return "Instruction-level parallelism (ILP) & FP32/INT32 co-issue scaling from serialized chains to peak wave occupancy.";
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

inline std::string subgroupTitleForCatalogEntry(const BenchmarkConfigMetadata& meta,
                                               std::string& outCategory,
                                               std::string& outSubgroupDesc) {
    if (!meta.isDeviceDependent || meta.component == "System") {
        outCategory = "Host System";
        if (meta.benchmarkName.find("Bandwidth") != std::string::npos) {
            outSubgroupDesc = "Measures host CPU-to-RAM streaming bandwidth. Affects game asset streaming, decompression, physics/animation prep, and directly caps GPU throughput on unified-memory APUs.";
            return "CPU System RAM Bandwidth";
        }
        outSubgroupDesc = "Measures host CPU-to-RAM access latency using large pages. Affects CPU game-thread simulation, draw-call submission, and 1% low frame-time stability.";
        return "CPU System RAM Latency";
    }

    if (meta.component == "Scheduling" ||
        meta.benchmarkName == "GPU-Driven Compute Dispatch" ||
        meta.benchmarkName == "In-Shader Indirect Synthesis") {
        outCategory = "Scheduling";
#ifdef GPUBENCH_ENABLE_EXPERIMENTAL_WORKGRAPHS
        if (meta.benchmarkName.find("Work Graphs") != std::string::npos) {
            outSubgroupDesc = "Tests GPU-autonomous shader-to-shader work graphs (producer kernels enqueueing consumer work directly on chip without CPU roundtrips or intermediate VRAM buffers).";
            return "GPU Work Graphs (Shader-to-Shader Enqueue)";
        }
#endif
        if (meta.benchmarkName.find("Async") != std::string::npos) {
            outSubgroupDesc = "Tests whether the GPU can overlap Async Compute (ACE) or PCIe DMA transfers alongside active 3D rendering or compute—hiding shadow/post-FX/culling passes and texture streaming behind frame work.";
            return "Async Compute (ACE) & DMA Transfer Overlap";
        }
        outSubgroupDesc = "Tests how efficiently the GPU schedules its own compute shader dispatches (not raster draw calls or ray traces) when active work varies per frame—as in visibility-buffer material shading, GPU culling, light binning, and sparse VFX.";
        return "GPU-Driven Compute Dispatch (Indirect / DGC)";
    }

    if (meta.component == "Raster" || meta.component == "Graphics" || meta.component == "Rasterization & ROP") {
        outCategory = "Raster";
        outSubgroupDesc = "Tests fixed-function rasterizer and Render Output Unit (ROP) pixel/blend fill throughput. Directly limits high-resolution/high-refresh color output, HDR targets, and heavy particle/smoke overdraw.";
        return "Rasterizer & ROP Pixel Fill Rate";
    }

    if (meta.component == "Memory") {
        outCategory = "Memory";
        if (meta.benchmarkName.find("Interconnect") != std::string::npos) {
            outSubgroupDesc = "Tests PCIe DMA, Resizable BAR CPU-to-VRAM writes, APU unified memory coherency, and multi-GPU peer transfers—gating texture/mesh streaming, dynamic frame uploads, and capture readbacks.";
            return "PCIe, Resizable BAR & Multi-GPU Interconnect";
        }
        if (meta.benchmarkName == "Device Memory Bandwidth") {
            outSubgroupDesc = "Tests sustained GDDR/HBM VRAM streaming throughput across workgroup sizes—the primary bottleneck for 4K framebuffers, G-buffer lighting, shadow maps, and post-processing.";
            return "GPU VRAM Streaming Bandwidth";
        }
        if (meta.benchmarkName.find("Cache Bandwidth") != std::string::npos) {
            outSubgroupDesc = "Tests peak bandwidth of each GPU cache tier (L0 -> L3/Infinity Cache). Determines how well textures, BVH nodes, and framebuffers avoid hitting slower VRAM.";
            return "GPU Cache Bandwidth (L0 -> L3)";
        }
        outSubgroupDesc = "Measures memory round-trip latency across 16 KB to 256 MB working sets, exposing cache size cliffs that stall irregular shader fetches in BVH traversal and indirection tables.";
        return "GPU Cache & VRAM Access Latency";
    }

    if (meta.component == "Ray Tracing") {
        outCategory = "Ray Tracing";
        if (meta.benchmarkName.rfind("RayScheduling", 0) == 0) {
            const std::string sub = meta.subcategory.empty() ? "Scene Ray Scheduling" : meta.subcategory;
            if (sub == "Material Shading") {
                outSubgroupDesc = "Compares how different GPU scheduling strategies (Megakernel, Legacy Indirect, DGC ExecutionSet, RTP+SER) handle divergent 8-material PBR shading after ray hits.";
            } else if (sub == "Scene Path Tracing (Multi-Bounce)") {
                outSubgroupDesc = "Full multi-bounce path tracing benchmark comparing Megakernel, RTP+SER, and Wavefront DGC across 3D PBR scenes—measuring real-world path-traced frame throughput.";
            } else if (sub == "Scene Path Tracing (16 SPP)") {
                outSubgroupDesc = "16 samples-per-pixel path tracing reference comparing Megakernel vs. octant-sorted Wavefront DGC under sustained multi-bounce accumulation.";
            } else if (sub == "Scene Ray Tracing (PBR)") {
                outSubgroupDesc = "Primary camera ray cast & G-buffer evaluation comparing Compute RayQuery, Ray Tracing Pipeline (RTP), and RTP+SER.";
            } else if (sub == "Incoherent Ray Tracing") {
                outSubgroupDesc = "Multi-bounce diffuse global illumination (GI) rays scattering across hemispheres—stressing incoherent BVH traversal and ray sorting.";
            } else if (sub == "Multi-Light Evaluation") {
                outSubgroupDesc = "Direct lighting across 128 stochastic lights, comparing brute-force shader loops against GPU-sorted DGC light-cone binning.";
            } else if (sub == "Alpha Cutout Divergence") {
                outSubgroupDesc = "Measures ray-tracing slowdown when rays pass through transparent foliage/cutouts requiring mid-traversal texture alpha checks.";
            } else if (sub == "Traversal Ordering & Coherence") {
                outSubgroupDesc = "Measures how screen-space ray launch order (1D scanline vs. 2D Morton Z-order) impacts L1/L2 BVH cache locality.";
            } else if (sub == "Wavefront Stream Compaction") {
                outSubgroupDesc = "Measures the cost of compacting active ray hit records in wavefront path tracers (128B full payloads vs. 16B wave-ballot bins).";
            } else {
                outSubgroupDesc = "Evaluates ray-tracing scheduling and execution efficiency across complex 3D PBR scenes.";
            }
            return sub;
        }
        if (meta.benchmarkName.find("Opacity Micromap") != std::string::npos) {
            outSubgroupDesc = "Tests hardware Opacity Micromaps (OMM) on layered alpha-cutout foliage—measuring how well BVH hardware avoids costly Any-Hit shader stalls.";
            return "Opacity Micromaps (OMM)";
        }
        if (meta.benchmarkName == "RayASBuild") {
            outSubgroupDesc = "Tests hardware BVH acceleration structure build speed for meshes (BLAS) and dynamic scene instances (TLAS)—gating per-frame RT CPU/GPU prep for animated characters, destruction, and moving objects.";
            return "BVH Construction (BLAS / TLAS)";
        }
        if (meta.benchmarkName == "RayDivergence" || meta.benchmarkName == "RayPayload") {
            outSubgroupDesc = "Tests two major real-world RT bottlenecks: bounce cone divergence (diffuse GI vs. mirror reflections) and ray payload size (VGPR pressure and scratch spills).";
            return "Ray Divergence & Payload";
        }
        outSubgroupDesc = "Tests raw fixed-function RT unit throughput: ray-box traversal, ray-triangle intersection, Any-Hit alpha interruption, and custom procedural AABB intersections.";
        return "BVH Traversal & Intersection";
    }

    outCategory = "Compute";
    if (meta.benchmarkName.find("Sparse") != std::string::npos ||
        meta.benchmarkName.find("Cooperative Vector") != std::string::npos) {
        outSubgroupDesc = "Tests 2026 neural rendering and 2:4 sparse tensor hardware—powering per-pixel neural materials/BRDFs, neural radiance caches, and pruned AI denoisers/upscalers.";
        return "2:4 Sparse Matrix & Neural Shading";
    }
    if (meta.benchmarkName == "DualIssue" || meta.benchmarkName.find("Dual-Issue") != std::string::npos) {
        outSubgroupDesc = "Tests Instruction-Level Parallelism (ILP) and dual-issue VALU hardware—revealing how sensitive shader throughput is to dependency chains and whether FP32 + INT32 can co-issue.";
        return "Shader ILP & Dual-Issue ALU Scaling";
    }
    if (meta.benchmarkName == "LDS Bank Conflicts") {
        outSubgroupDesc = "Tests on-chip workgroup shared memory (LDS) bank conflict penalties—critical for tile-based deferred lighting, screen-space filters, reductions, and matrix tiling.";
        return "On-Chip Shared Memory (LDS) Bank Conflicts";
    }
    outSubgroupDesc = "Tests core GPU shader vector ALU and hardware AI/tensor matrix core throughput across precisions from FP64 down to INT4.";
    return "Shader & AI Matrix Compute (FP64 -> INT4)";
}

inline const char* shortCategoryTag(const std::string& name) {
    if (name == "Ray Tracing") return "RT";
    if (name == "Scheduling") return "SCHED";
    if (name == "Raster" || name == "Graphics") return "RASTER";
    if (name == "Host System") return "HOST";
    if (name == "Compute") return "COMP";
    if (name == "Memory") return "MEM";
    return name.c_str();
}

/// Produce a compact sidebar slot label by stripping subgroup-redundant prefixes
/// while preserving full names in tooltips, inspectors, and exported charts.
inline std::string compactWorkloadSlotLabel(const std::string& fullName, const std::string& subgroupName) {
    std::string out = fullName;

    static constexpr const char* kLeadingTags[] = {
        "Stage: ",
        "Scene Render: ",
        "Stage Breakdown - "
    };
    for (const char* tag : kLeadingTags) {
        const size_t len = std::char_traits<char>::length(tag);
        if (out.rfind(tag, 0) == 0) {
            out.erase(0, len);
            break;
        }
    }

    auto stripPrefixWithSep = [](std::string& s, const std::string& prefix) -> bool {
        if (prefix.empty() || s.size() <= prefix.size() || s.rfind(prefix, 0) != 0) {
            return false;
        }
        const size_t n = prefix.size();
        for (const char* sep : {" - ", ": ", " "}) {
            const size_t sepLen = std::char_traits<char>::length(sep);
            if (s.compare(n, sepLen, sep) == 0 && s.size() > n + sepLen) {
                s.erase(0, n + sepLen);
                return true;
            }
        }
        return false;
    };

    if (!stripPrefixWithSep(out, subgroupName)) {
        static constexpr const char* kGroupPrefixes[] = {
            "Multi-Light Evaluation",
            "Multi-Light",
            "BVH Traversal",
            "Queue Compaction",
            "Traversal Divergence",
            "Alpha-Cutout Traversal Divergence",
            "Incoherent Ray Tracing",
            "Primary Ray Tracing",
            "Full Scene Path Tracing",
            "Material Shading"
        };
        for (const char* gp : kGroupPrefixes) {
            if (stripPrefixWithSep(out, gp)) {
                break;
            }
        }
    }

    struct ReplaceRule { const char* from; const char* to; };
    static constexpr ReplaceRule kRules[] = {
        {"FP32 Serialized (1 Chain / Low ILP)", "FP32 Serialized (1ch)"},
        {"FP32 Baseline (Ping-Pong)", "FP32 Baseline"},
        {"FP32 Moderate ILP (8 Chains)", "FP32 Mid ILP (8ch)"},
        {"FP32 Peak ILP / Dual-Issue (16 Chains)", "FP32 Peak ILP (16ch)"},
        {"FP32 Peak ILP / Co-Issue (16 Chains)", "FP32 Peak ILP (16ch)"},
        {"INT32 Serialized (1 Chain / Low ILP)", "INT32 Serialized (1ch)"},
        {"INT32 Baseline (Ping-Pong)", "INT32 Baseline"},
        {"INT32 Moderate ILP (8 Chains)", "INT32 Mid ILP (8ch)"},
        {"INT32 Peak ILP / Dual-Issue (16 Chains)", "INT32 Peak ILP (16ch)"},
        {"INT32 Peak ILP (16 Chains)", "INT32 Peak ILP (16ch)"},
        {"Concurrent Mixed (FP32 + INT32)", "FP32+INT32 Mixed"},
        {"Concurrent Mixed (FP32+INT32)", "FP32+INT32 Mixed"},
        {"Batched BLAS Build (5M Tris, 256 Meshes)", "Batched BLAS (5M, 256m)"},
        {"TLAS: Indoor Corridor (20K Inst)", "TLAS: Indoor (20K Inst)"},
        {"TLAS: Dense Jungle (50K Inst)", "TLAS: Jungle (50K Inst)"},
        {"TLAS: Massive Open World (200K Inst)", "TLAS: Open World (200K)"},
        {"Raw BVH - Coherent Triangle Traversal (Max Occupancy)", "Raw BVH: Coherent Triangles"},
        {"Raw BVH - Deep Multi-Layer Box Stress", "Raw BVH: Deep Box Stress"},
        {"100% Solid (Opaque Fast-Path Baseline)", "Any-Hit: 100% Opaque"},
        {"50% Solid (Cutout Stress)", "Any-Hit: 50% Cutout"},
        {"Spheres (AABB)", "Procedural AABB (Spheres)"},
        {"Bounce Divergence: ", ""},
        {"16B Payload (Shadow / AO Visibility Probe)", "16B Payload (Shadow/AO)"},
        {"128B Payload (Real-Time Path Tracing & GI)", "128B Payload (Path Trace)"},
        {"256B Payload (Production / Film BSDF & MIS)", "256B Payload (Film BSDF)"},
        {"(Device-Generated Commands (DGC))", "(DGC)"},
        {"Device-Generated Commands (DGC)", "DGC"},
        {"(DGC Light Binning)", "(DGC)"},
        {"(DGC Indirect)", "(DGC)"},
        {"(DGC Specialized)", "(DGC Spec)"},
        {"DGC Specialized", "DGC Spec"},
        {"(Traditional Megakernel)", "(MK)"},
        {"Traditional Megakernel", "MK"},
        {"(Megakernel)", "(MK)"},
        {"Wavefront Compaction (Material Sorting)", "Wavefront (Mat)"},
        {"Wavefront Compaction (Directional Binning)", "Wavefront (Dir)"},
        {"Wavefront Compaction (Ray Reordering)", "Wavefront (Reorder)"},
        {"RTP + Hardware SER", "RTP + SER"},
        {"Single Light", "1 Light"},
        {"1024-Shader Sparse Buckets (CPU Fixed Grid)", "1024-Bucket (CPU Grid)"},
        {"1024-Shader Sparse Buckets (1024x Indirect)", "1024-Bucket (Indirect)"},
        {"1024-Shader Sparse Buckets (Single DGC Call)", "1024-Bucket (DGC)"},
        {"Compute Fixed Grid - CPU Direct (100% Active)", "CPU Fixed Grid (100%)"},
        {"Compute Fixed Grid - CPU Direct (10% Active)", "CPU Fixed Grid (10%)"},
        {"GPU Indirect Compute Dispatch (100% Active)", "GPU Indirect (100%)"},
        {"GPU Indirect Compute Dispatch (10% Active)", "GPU Indirect (10%)"},
        {"GPU Zero-Dispatch Pruning (0% Active)", "Zero-Dispatch (0%)"},
        {"Persistent Compute Work-Queue (100% Active)", "Work-Queue (100%)"},
        {"Persistent Compute Work-Queue (10% Active)", "Work-Queue (10%)"},
        {"Persistent Work-Queue Early-Exit (0% Active)", "Work-Queue (0%)"},
        {"Multi-Bucket 1024x Sparse (CPU Direct Fixed Grid)", "Multi-Bucket 1024x (CPU)"},
        {"Multi-Bucket 1024x Sparse (Legacy vkCmdDispatchIndirect)", "Multi-Bucket 1024x (Indirect)"},
        {"Multi-Bucket 1024x Sparse (DGC ExecutionSet + Pruning)", "Multi-Bucket 1024x (DGC)"},
        {"Legacy Indirect Dispatch", "Legacy Indirect"},
        {"Legacy Zero-Dispatch Pruning (0% Active)", "Zero-Dispatch (0% Active)"},
        {"Persistent Work-Queue Compaction", "Work-Queue Compaction"},
        {"In-Shader Indirect Synthesis", "Indirect"},
        {"In-Shader Zero-Dispatch ", "Zero-Dispatch "},
        {"CPU Direct Fixed Grid", "CPU Fixed Grid"}
    };
    for (const auto& rule : kRules) {
        const size_t fromLen = std::char_traits<char>::length(rule.from);
        const size_t toLen = std::char_traits<char>::length(rule.to);
        size_t pos = 0;
        while ((pos = out.find(rule.from, pos)) != std::string::npos) {
            out.replace(pos, fromLen, rule.to);
            pos += toLen;
        }
    }

    if (out.size() >= 2 && out.front() == '(' && out.back() == ')') {
        std::string inner = out.substr(1, out.size() - 2);
        if (inner.find('(') == std::string::npos && inner.find(')') == std::string::npos) {
            out = std::move(inner);
        }
    }
    return out;
}

inline bool matchesWorkloadItem(const ResultData& r,
                                const std::string& itemId,
                                const std::string& itemName,
                                const std::string& itemCategory,
                                int itemConfigIndex,
                                uint32_t activeDev) {
    const bool rIsHost = (r.component == "System" || r.backendName == "CPU" ||
                          r.backendName == "System" || r.deviceIndex == UINT32_MAX ||
                          r.deviceName == "Host CPU");
    if (itemCategory == "System") {
        if (!rIsHost) return false;
    } else {
        if (rIsHost || r.deviceIndex != activeDev) return false;
    }

    if (itemId == "Cache Latency Curve") {
        return r.benchmarkName.rfind("Cache Latency Curve", 0) == 0;
    }

    if (itemId == "RayScheduling") {
        if (r.benchmarkName.rfind("RayScheduling", 0) != 0) return false;
        return (itemConfigIndex >= 0) && (static_cast<int>(r.configIndex) == itemConfigIndex);
    }

    // Normalized multi-precision Vector / Matrix records from ResultImporter
    if (r.benchmarkName == "Vector" || r.benchmarkName == "Matrix") {
        const bool subMatches = (r.subcategory == itemId) ||
                                (r.subcategory.find("(" + itemId + ")") != std::string::npos);
        if (!subMatches) return false;
        if (itemName.find("(Matrix)") != std::string::npos) {
            return r.benchmarkName == "Matrix";
        }
        if (itemName.find("(Vector)") != std::string::npos) {
            return r.benchmarkName == "Vector";
        }
        return (itemConfigIndex >= 0) && (static_cast<int>(r.configIndex) == itemConfigIndex);
    }

    // Normalized Device Memory Bandwidth records from ResultImporter
    if (itemId == "Device Memory Bandwidth") {
        if (r.benchmarkName == "GPU VRAM Bandwidth" && itemConfigIndex == 3) {
            return true;
        }
        if ((r.benchmarkName.rfind("Device Memory Bandwidth", 0) == 0 ||
             r.benchmarkName.rfind("VRAM Bandwidth", 0) == 0) &&
            static_cast<int>(r.configIndex) == itemConfigIndex) {
            return true;
        }
    }

    // Normalized RayIntersect & Pixel Fill Rate records from ResultImporter
    if (itemId == "RayIntersect" && static_cast<int>(r.configIndex) == itemConfigIndex) {
        if ((itemConfigIndex == 0 && r.benchmarkName == "Ray-Triangle Intersection") ||
            (itemConfigIndex == 1 && r.benchmarkName == "Ray-Box Intersection") ||
            r.benchmarkName.rfind("RayTracing (", 0) == 0) {
            return true;
        }
    }
    if (itemId == "Pixel Fill Rate" && r.subcategory == "Pixel Fill Rate" &&
        static_cast<int>(r.configIndex) == itemConfigIndex) {
        return true;
    }

    if (itemConfigIndex >= 0 && static_cast<int>(r.configIndex) == itemConfigIndex) {
        if (r.benchmarkName == itemId || r.benchmarkName.rfind(itemId + " (", 0) == 0 ||
            r.benchmarkName.rfind(itemId + " -", 0) == 0) {
            return true;
        }
    }

    return r.benchmarkName == itemId;
}

} // namespace gpubench::gui
