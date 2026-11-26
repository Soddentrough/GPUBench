#include "test_harness.h"
#include "benchmarks/DualIssueBench.h"
#include "benchmarks/LdsBankConflictBench.h"
#include "benchmarks/RaySchedulingBench.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace {

uint16_t floatToHalfBits(float val) {
  uint32_t bits = 0;
  std::memcpy(&bits, &val, sizeof(bits));
  uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000u);
  int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
  uint32_t mant = bits & 0x007FFFFFu;

  if ((bits & 0x7FFFFFFFu) > 0x7F800000u) {
    return static_cast<uint16_t>(sign | 0x7E00u | (mant >> 13)); // NaN
  }
  if (exp >= 31) {
    return static_cast<uint16_t>(sign | 0x7C00u); // Inf / overflow
  }
  if (exp <= 0) {
    if (exp < -10) {
      return sign; // Underflow to zero
    }
    mant = (mant | 0x00800000u) >> (1 - exp);
    if (mant & 0x00001000u) {
      mant += 0x00002000u;
    }
    return static_cast<uint16_t>(sign | (mant >> 13));
  }
  uint32_t rounded = mant + 0x00000FFFu + ((mant >> 13) & 1u);
  if (rounded & 0x00800000u) {
    rounded = 0;
    ++exp;
    if (exp >= 31) {
      return static_cast<uint16_t>(sign | 0x7C00u);
    }
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (rounded >> 13));
}

float halfBitsToFloat(uint16_t h) {
  uint32_t sign = (static_cast<uint32_t>(h & 0x8000u)) << 16;
  uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t mant = h & 0x03FFu;
  uint32_t out = 0;
  if (exp == 0) {
    if (mant == 0) {
      out = sign;
    } else {
      int32_t e = -14;
      while ((mant & 0x0400u) == 0) {
        mant <<= 1;
        --e;
      }
      mant &= 0x03FFu;
      out = sign | (static_cast<uint32_t>(e + 127) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    out = sign | 0x7F800000u | (mant << 13);
  } else {
    out = sign | ((exp - 15u + 127u) << 23) | (mant << 13);
  }
  float res = 0.0f;
  std::memcpy(&res, &out, sizeof(res));
  return res;
}

float roundToFp16(float v) {
  return halfBitsToFloat(floatToHalfBits(v));
}

} // namespace

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
  // 1. Verify FP16 self-recurrence under genuine IEEE-754 binary16 rounding:
  // a = fp16(fma(a, -0.9990234375, 0.5)) remains strictly within IEEE-754 FP16 range (max 65504.0).
  const float m16 = roundToFp16(-0.9990234375f);
  const float c16 = roundToFp16(0.5f);
  float a16 = roundToFp16(0.03125f);
  for (int i = 0; i < 65536; ++i) {
    a16 = roundToFp16(std::fma(a16, m16, c16));
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
  ASSERT_EQ(rtBench.GetNumConfigs(), 21u);

  // Verify Legacy Indirect, DGC, RTP, and SER configs map to their respective baselines
  ASSERT_EQ(rtBench.GetBaselineConfigIndex(0), -1);
  ASSERT_EQ(rtBench.GetBaselineConfigIndex(1), 0);
  ASSERT_EQ(rtBench.GetBaselineConfigIndex(2), 0);
  ASSERT_EQ(rtBench.GetBaselineConfigIndex(3), 0);

  ASSERT_EQ(rtBench.GetBaselineConfigIndex(4), -1);
  ASSERT_EQ(rtBench.GetBaselineConfigIndex(5), 4);
  ASSERT_EQ(rtBench.GetBaselineConfigIndex(6), 4);

  ASSERT_EQ(rtBench.GetBaselineConfigIndex(7), -1);
  ASSERT_EQ(rtBench.GetBaselineConfigIndex(8), 7);

  ASSERT_EQ(rtBench.GetBaselineConfigIndex(9), -1);
  ASSERT_EQ(rtBench.GetBaselineConfigIndex(10), 9);
  ASSERT_EQ(rtBench.GetBaselineConfigIndex(11), 9);

  ASSERT_EQ(rtBench.GetBaselineConfigIndex(12), -1);
  ASSERT_EQ(rtBench.GetBaselineConfigIndex(13), 12);

  ASSERT_EQ(rtBench.GetBaselineConfigIndex(14), -1);
  ASSERT_EQ(rtBench.GetBaselineConfigIndex(15), 14);

  // Verify Config 2 (Material Shading DGC) and Config 12 (Incoherent GI Megakernel) do not share sortWeight
  ASSERT_NE(rtBench.GetSortWeight(2), rtBench.GetSortWeight(12));

  // Verify BVH Traversal (17->18) and Queue Compaction (19->20) baselines and ordering
  ASSERT_EQ(rtBench.GetBaselineConfigIndex(17), -1);
  ASSERT_EQ(rtBench.GetBaselineConfigIndex(18), 17);
  ASSERT_EQ(rtBench.GetBaselineConfigIndex(19), -1);
  ASSERT_EQ(rtBench.GetBaselineConfigIndex(20), 19);
  ASSERT_LT(rtBench.GetSortWeight(17), rtBench.GetSortWeight(18));
  ASSERT_LT(rtBench.GetSortWeight(19), rtBench.GetSortWeight(20));
  ASSERT_LT(rtBench.GetSortWeight(14), rtBench.GetSortWeight(15));

  DualIssueBench diBench;
  ASSERT_EQ(std::string(diBench.GetName()), "ILP & Dual-Issue");
  ASSERT_EQ(diBench.GetNumConfigs(), 7u);

  LdsBankConflictBench ldsBench;
  ASSERT_EQ(ldsBench.GetNumConfigs(), 8u);
  ASSERT_EQ(ldsBench.GetBaselineConfigIndex(0), -1);
  ASSERT_EQ(ldsBench.GetBaselineConfigIndex(1), 0);
}
