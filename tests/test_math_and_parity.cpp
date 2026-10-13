#include "test_harness.h"
#include "benchmarks/DualIssueBench.h"
#include "benchmarks/LdsBankConflictBench.h"
#include "benchmarks/RaySchedulingBench.h"
#include <cmath>
#include <cstdint>
#include <vector>

TEST_CASE(MathAndParity, Fp32RecurrenceConvergence) {
  // Analytical verification of x_{n+1} = fma(m, x_n, c) with |m| < 1.0:
  // spectral radius < 1.0, converges to the unique fixed point x* = c / (1 - m).
  const float m = 0.999f;
  const float c = 0.001f;
  const float fixedPoint = c / (1.0f - m); // 1.0f

  float x = 0.5f;
  constexpr int kIterations = 16384;

  for (int i = 0; i < kIterations; ++i) {
    x = std::fma(m, x, c);
    ASSERT_TRUE(std::isfinite(x));
    ASSERT_FALSE(std::isnan(x));
    ASSERT_FALSE(std::isinf(x));
  }

  ASSERT_NEAR(x, fixedPoint, 1e-4);
}

TEST_CASE(MathAndParity, Fp16AndDualIssueRecurrenceBounds) {
  // 1. Verify FP16 self-recurrence: a = fma(a, -0.9990234375f, 0.5f)
  // remains strictly within IEEE-754 FP16 range (max 65504.0) across 65,536 iterations.
  const float m16 = -0.9990234375f;
  const float c16 = 0.5f;
  float a16 = 0.03125f;
  for (int i = 0; i < 65536; ++i) {
    a16 = std::fma(a16, m16, c16);
    ASSERT_TRUE(std::isfinite(a16));
    ASSERT_LT(std::abs(a16), 65504.0f);
  }
  ASSERT_GT(std::abs(a16), 0.0f);

  // 2. Verify DualIssue ILP4 coupled recurrence with scaled m = 0.999f * 0.00001f
  // stays finite across 131,072 iterations.
  const float mIlp4 = 0.999f * 0.00001f;
  float val0 = 1.0f;
  float val1 = 1.1f;
  for (int i = 0; i < 131072; ++i) {
    val0 = std::fma(val0, mIlp4, val1);
    val1 = std::fma(val1, mIlp4, val0);
    ASSERT_TRUE(std::isfinite(val0));
    ASSERT_TRUE(std::isfinite(val1));
  }
}

TEST_CASE(MathAndParity, LdsBankConflictStridePreservation) {
  // Verify that initializing LDS elements to 32u preserves the exact bank stride
  // (base_addr(tid) - base_addr(0)) mod 32 == (tid * stride) mod 32 across iterations.
  constexpr uint32_t kLdsSize = 1024;
  std::vector<uint32_t> lds(kLdsSize, 32u);

  const uint32_t strides[] = {1u, 2u, 4u, 8u, 16u, 32u, 3u, 5u};
  for (uint32_t stride : strides) {
    std::vector<uint32_t> baseAddr(32);
    std::vector<uint32_t> acc0(32, 0u);
    for (uint32_t tid = 0; tid < 32; ++tid) {
      baseAddr[tid] = (tid * stride) & 1023u;
    }

    for (int iter = 0; iter < 256; ++iter) {
      for (uint32_t tid = 0; tid < 32; ++tid) {
        uint32_t bank = baseAddr[tid] & 31u;
        uint32_t expectedBank = ( (baseAddr[0] + tid * stride) & 31u );
        ASSERT_EQ(bank, expectedBank);

        acc0[tid] += lds[(baseAddr[tid] + 0u * stride) & 1023u];
        baseAddr[tid] = (baseAddr[tid] + acc0[tid]) & 1023u;
      }
    }
  }
}

TEST_CASE(MathAndParity, BenchmarkBaselineAndSortWeightIntegrity) {
  RaySchedulingBench rtBench;
  // Verify DGC Path Tracing and GI configs map to their Megakernel baselines
  ASSERT_EQ(rtBench.GetBaselineConfigIndex(2), 0);
  ASSERT_EQ(rtBench.GetBaselineConfigIndex(5), 3);
  ASSERT_EQ(rtBench.GetBaselineConfigIndex(8), 6);
  ASSERT_EQ(rtBench.GetBaselineConfigIndex(11), 9);
  ASSERT_EQ(rtBench.GetBaselineConfigIndex(18), 17);
  ASSERT_EQ(rtBench.GetBaselineConfigIndex(24), 23);

  // Verify Config 2 (Material Shading DGC) and Config 6 (Incoherent GI Megakernel) do not share sortWeight
  ASSERT_NE(rtBench.GetSortWeight(2), rtBench.GetSortWeight(6));

  DualIssueBench diBench;
  ASSERT_EQ(std::string(diBench.GetName()), "ILP & Dual-Issue");
  ASSERT_EQ(diBench.GetNumConfigs(), 7u);

  LdsBankConflictBench ldsBench;
  ASSERT_EQ(ldsBench.GetNumConfigs(), 8u);
  ASSERT_EQ(ldsBench.GetBaselineConfigIndex(0), -1);
  ASSERT_EQ(ldsBench.GetBaselineConfigIndex(1), 0);
}
