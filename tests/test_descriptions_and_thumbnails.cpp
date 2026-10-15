#include "test_harness.h"
#include "benchmarks/Fp64Bench.h"
#include "benchmarks/Fp32Bench.h"
#include "benchmarks/DualIssueBench.h"
#include "benchmarks/LdsBankConflictBench.h"
#include "benchmarks/InShaderIndirectBench.h"
#include "benchmarks/Fp16Bench.h"
#include "benchmarks/Bf16Bench.h"
#include "benchmarks/Fp8Bench.h"
#include "benchmarks/Fp6Bench.h"
#include "benchmarks/Fp4Bench.h"
#include "benchmarks/Int8Bench.h"
#include "benchmarks/Int4Bench.h"
#include "benchmarks/CacheLatencyCurveBench.h"
#include "benchmarks/PixelFillRateBench.h"
#include "benchmarks/SysMemBandwidthBench.h"
#include "benchmarks/SysMemLatencyBench.h"
#include "benchmarks/RayASBuildBench.h"
#include "benchmarks/RayRawTraversalBench.h"
#include "benchmarks/RayIntersectBench.h"
#include "benchmarks/RayAnyHitBench.h"
#include "benchmarks/RayProceduralBench.h"
#include "benchmarks/RayDivergenceBench.h"
#include "benchmarks/RaySchedulingBench.h"

#include <filesystem>
#include <fstream>
#include <vector>
#include <string>

using namespace gpubench;

TEST_CASE(DescriptionsAndThumbnails, QuantizedAndFloatPrecisionConfigs) {
  // FP64
  Fp64Bench fp64;
  ASSERT_EQ(fp64.GetNumConfigs(), 1u);
  ASSERT_EQ(std::string(fp64.GetMetric()), "TFLOPS");

  // FP32
  Fp32Bench fp32;
  ASSERT_EQ(fp32.GetNumConfigs(), 1u);
  ASSERT_EQ(std::string(fp32.GetMetric()), "TFLOPS");

  // FP16: Vector (0), Matrix (1)
  Fp16Bench fp16;
  ASSERT_EQ(fp16.GetNumConfigs(), 2u);
  ASSERT_EQ(fp16.GetConfigName(0), "Vector");
  ASSERT_EQ(fp16.GetConfigName(1), "Matrix");

  // BF16: Vector (0), Matrix (1)
  Bf16Bench bf16;
  ASSERT_EQ(bf16.GetNumConfigs(), 2u);
  ASSERT_EQ(bf16.GetConfigName(0), "Vector");
  ASSERT_EQ(bf16.GetConfigName(1), "Matrix");

  // FP8: Matrix only (0)
  Fp8Bench fp8;
  ASSERT_EQ(fp8.GetNumConfigs(), 1u);
  ASSERT_EQ(fp8.GetConfigName(0), "Matrix");

  // FP6: Vector only (0)
  Fp6Bench fp6;
  ASSERT_EQ(fp6.GetNumConfigs(), 1u);
  ASSERT_EQ(std::string(fp6.GetMetric()), "TOPS");

  // FP4: Vector only (0)
  Fp4Bench fp4;
  ASSERT_EQ(fp4.GetNumConfigs(), 1u);
  ASSERT_EQ(fp4.GetConfigName(0), "Vector");
  ASSERT_EQ(std::string(fp4.GetMetric()), "TOPS");

  // INT8: Vector (0), Matrix (1)
  Int8Bench int8;
  ASSERT_EQ(int8.GetNumConfigs(), 2u);
  ASSERT_EQ(int8.GetConfigName(0), "Vector");
  ASSERT_EQ(int8.GetConfigName(1), "Matrix");
  ASSERT_EQ(std::string(int8.GetMetric()), "TOPS");

  // INT4: Matrix only (0) — no emulated vector
  Int4Bench int4;
  ASSERT_EQ(int4.GetNumConfigs(), 1u);
  ASSERT_EQ(int4.GetConfigName(0), "Matrix");
  ASSERT_EQ(std::string(int4.GetMetric()), "TOPS");
}

TEST_CASE(DescriptionsAndThumbnails, DualIssueConfigs) {
  DualIssueBench di;
  ASSERT_EQ(di.GetNumConfigs(), 7u);
  ASSERT_EQ(di.GetConfigName(0), "FP32 Baseline (Ping-Pong)");
  ASSERT_EQ(di.GetConfigName(1), "FP32 Moderate ILP (8 Chains)");
  ASSERT_EQ(di.GetConfigName(2), "FP32 Peak ILP / Co-Issue (16 Chains)");
  ASSERT_EQ(di.GetConfigName(3), "INT32 Baseline (Ping-Pong)");
  ASSERT_EQ(di.GetConfigName(4), "INT32 Moderate ILP (8 Chains)");
  ASSERT_EQ(di.GetConfigName(5), "INT32 Peak ILP (16 Chains)");
  ASSERT_EQ(di.GetConfigName(6), "Concurrent Mixed (FP32+INT32)");
}

TEST_CASE(DescriptionsAndThumbnails, LdsBankConflictConfigs) {
  LdsBankConflictBench lds;
  ASSERT_EQ(lds.GetNumConfigs(), 8u);
  ASSERT_EQ(lds.GetConfigName(0), "Stride 1 (Conflict-Free Baseline)");
  ASSERT_EQ(lds.GetConfigName(1), "Stride 2 (2-Way Conflict)");
  ASSERT_EQ(lds.GetConfigName(2), "Stride 4 (4-Way Conflict)");
  ASSERT_EQ(lds.GetConfigName(3), "Stride 8 (8-Way Conflict)");
  ASSERT_EQ(lds.GetConfigName(4), "Stride 16 (16-Way Conflict)");
  ASSERT_EQ(lds.GetConfigName(5), "Stride 32 (32-Way Full Serialization)");
  ASSERT_EQ(lds.GetConfigName(6), "Stride 3 (Odd Stride Control)");
  ASSERT_EQ(lds.GetConfigName(7), "Stride 5 (Odd Stride Control)");
}

TEST_CASE(DescriptionsAndThumbnails, IndirectSynthesisConfigs) {
  InShaderIndirectBench ind;
  ASSERT_EQ(ind.GetNumConfigs(), 7u);
  ASSERT_EQ(ind.GetConfigName(0), "CPU Direct Fixed Grid (100% Active Baseline)");
  ASSERT_EQ(ind.GetConfigName(1), "In-Shader Indirect Synthesis (100% Active)");
  ASSERT_EQ(ind.GetConfigName(2), "In-Shader Indirect Synthesis (50% Active)");
  ASSERT_EQ(ind.GetConfigName(3), "In-Shader Indirect Synthesis (10% Active)");
  ASSERT_EQ(ind.GetConfigName(4), "In-Shader Indirect Synthesis (1% Active)");
  ASSERT_EQ(ind.GetConfigName(5), "In-Shader Zero-Dispatch Pruning (0% Active)");
  ASSERT_EQ(ind.GetConfigName(6), "CPU Direct Fixed Grid (10% Active)");
}

TEST_CASE(DescriptionsAndThumbnails, RaySchedulingConfigsCount) {
  RaySchedulingBench rs(RaySchedulingBench::SceneType::Showroom);
  ASSERT_EQ(rs.GetNumConfigs(), 35u);
  ASSERT_EQ(rs.GetConfigName(0), "Stage: Material Shading (Megakernel)");
  ASSERT_EQ(rs.GetConfigName(9), "Scene Render: Full Frame (Megakernel)");
  ASSERT_EQ(rs.GetConfigName(17), "Primary Rays (Compute Megakernel)");
  ASSERT_EQ(rs.GetConfigName(19), "Stage: Directional Shadows - Single Light (Megakernel)");
  ASSERT_EQ(rs.GetConfigName(27), "Stage: Traversal Divergence - Alpha Cutout Geometry");
  ASSERT_EQ(rs.GetConfigName(31), "Stage: Multi-Light Evaluation - Single Light (Megakernel)");
  ASSERT_EQ(rs.GetConfigName(33), "Stage: Multi-Light Evaluation - 128 Lights (Megakernel)");
  ASSERT_EQ(rs.GetConfigName(34), "Stage: Multi-Light Evaluation - 128 Lights (DGC Light Binning)");
}

TEST_CASE(DescriptionsAndThumbnails, RayHardwareTraversalAndDivergenceConfigs) {
  RayRawTraversalBench raw;
  ASSERT_EQ(raw.GetNumConfigs(), 2u);
  ASSERT_EQ(raw.GetConfigName(0), "Raw BVH - Coherent Triangle Traversal (Max Occupancy)");
  ASSERT_EQ(raw.GetConfigName(1), "Raw BVH - Deep Multi-Layer Box Stress");

  RayIntersectBench isect;
  ASSERT_EQ(isect.GetNumConfigs(), 2u);
  ASSERT_EQ(isect.GetConfigName(0), "Ray-Triangle");
  ASSERT_EQ(isect.GetConfigName(1), "Ray-Box");

  RayAnyHitBench anyhit;
  ASSERT_EQ(anyhit.GetNumConfigs(), 2u);
  ASSERT_EQ(anyhit.GetConfigName(0), "100% Solid (Any-Hit Baseline)");
  ASSERT_EQ(anyhit.GetConfigName(1), "50% Solid (Cutout Stress)");

  RayProceduralBench proc;
  ASSERT_EQ(proc.GetNumConfigs(), 1u);
  ASSERT_EQ(proc.GetConfigName(0), "Spheres (AABB)");

  RayDivergenceBench div;
  ASSERT_EQ(div.GetNumConfigs(), 5u);
  ASSERT_EQ(div.GetConfigName(0), "Bounce Divergence: Rough Diffuse (r=1.0, 90° Spread)");
  ASSERT_EQ(div.GetConfigName(1), "Bounce Divergence: Matte Satin (r=0.75, 67.5° Spread)");
  ASSERT_EQ(div.GetConfigName(2), "Bounce Divergence: Brushed Metal (r=0.5, 45° Spread)");
  ASSERT_EQ(div.GetConfigName(3), "Bounce Divergence: Glossy Coat (r=0.25, 22.5° Spread)");
  ASSERT_EQ(div.GetConfigName(4), "Bounce Divergence: Mirror Reflection (r=0.0, 0° Coherent)");
}

TEST_CASE(DescriptionsAndThumbnails, RasterizerAndMemoryConfigs) {
  PixelFillRateBench rop;
  ASSERT_EQ(rop.GetNumConfigs(), 3u);
  ASSERT_EQ(rop.GetConfigName(0), "RGBA8 Color Fill");
  ASSERT_EQ(rop.GetConfigName(1), "RGBA16F HDR Fill");
  ASSERT_EQ(rop.GetConfigName(2), "Alpha Blending Fill");

  CacheLatencyCurveBench curve;
  ASSERT_EQ(curve.GetNumConfigs(), 15u);
  ASSERT_EQ(curve.GetConfigName(0), "16 KB");
  ASSERT_EQ(curve.GetConfigName(14), "256 MB");

  SysMemBandwidthBench sys;
  ASSERT_EQ(sys.GetNumConfigs(), 6u);
  ASSERT_EQ(sys.GetConfigName(0), "Read");
  ASSERT_EQ(sys.GetConfigName(3), "Read (1 Thread)");
}

TEST_CASE(DescriptionsAndThumbnails, ThumbnailPngAssetsIntegrity) {
  // Verify all 48 shipped PNG thumbnails exist and have PNG magic headers
  const std::vector<std::string> scenes = {"showroom", "indoor", "forest", "outdoor"};
  const std::vector<std::string> stages = {
    "stage1_bvh", "stage2_primary", "stage3_shadow", "stage4_rtao",
    "stage5_direct", "stage6_indirect", "stage7_final", "multilight_128_dgc",
    "alpha_cutout", "pathtracing_1spp", "pathtracing_16spp"
  };
  const std::vector<std::string> statics = {
    "thumb_blas_wireframe", "thumb_alpha_layers", "thumb_material_lineup", "thumb_material_range"
  };

  std::vector<std::string> allThumbs;
  for (const auto &scene : scenes) {
    for (const auto &stage : stages) {
      allThumbs.push_back("thumb_" + scene + "_" + stage + ".png");
    }
  }
  for (const auto &stat : statics) {
    allThumbs.push_back(stat + ".png");
  }

  ASSERT_EQ(allThumbs.size(), 48u);

  std::vector<std::string> probeDirs = {
    "assets/thumbnails",
    "../assets/thumbnails",
    "../../assets/thumbnails"
  };

  std::string foundDir;
  for (const auto &d : probeDirs) {
    if (std::filesystem::exists(d)) {
      foundDir = d;
      break;
    }
  }

  ASSERT_FALSE(foundDir.empty());

  for (const auto &name : allThumbs) {
    std::filesystem::path p = std::filesystem::path(foundDir) / name;
    ASSERT_TRUE(std::filesystem::exists(p));
    ASSERT_TRUE(std::filesystem::file_size(p) > 1024u); // At least 1 KB

    // Verify 8-byte PNG header: 89 50 4E 47 0D 0A 1A 0A
    std::ifstream file(p, std::ios::binary);
    ASSERT_TRUE(file.is_open());
    unsigned char header[8] = {0};
    file.read(reinterpret_cast<char*>(header), 8);
    ASSERT_EQ(header[0], 0x89);
    ASSERT_EQ(header[1], 0x50); // 'P'
    ASSERT_EQ(header[2], 0x4E); // 'N'
    ASSERT_EQ(header[3], 0x47); // 'G'
  }
}
