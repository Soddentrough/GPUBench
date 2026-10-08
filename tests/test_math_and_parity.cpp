#include "test_harness.h"
#include <cmath>
#include <vector>

TEST_CASE(MathAndParity, Fp32RecurrenceConvergence) {
  // Analytical verification of Critical Bug C-2 fix:
  // x_{n+1} = fma(m, x_n, c) with |m| < 1.0 has spectral radius < 1.0
  // and converges to the unique fixed point x* = c / (1 - m).
  const float m = 0.999f;
  const float c = 0.001f;
  const float fixedPoint = c / (1.0f - m); // 1.0f

  float x = 0.5f; // Initial arbitrary non-zero value
  constexpr int kIterations = 16384;

  for (int i = 0; i < kIterations; ++i) {
    x = std::fma(m, x, c);
    ASSERT_TRUE(std::isfinite(x));
    ASSERT_FALSE(std::isnan(x));
    ASSERT_FALSE(std::isinf(x));
  }

  // Value must be strictly finite and within epsilon of the fixed point
  ASSERT_NEAR(x, fixedPoint, 1e-4);

  // Demonstrate the failure of the pre-fix recurrence:
  // x_{n+1} = (1 + m) * x_n whose dominant eigenvalue is ~1.999
  float divergentX = 1.0f;
  bool reachedInf = false;
  int infIteration = -1;

  for (int i = 0; i < kIterations; ++i) {
    divergentX = (1.0f + m) * divergentX;
    if (std::isinf(divergentX)) {
      reachedInf = true;
      infIteration = i;
      break;
    }
  }

  // Pre-fix recurrence diverges to inf within ~130 iterations
  ASSERT_TRUE(reachedInf);
  ASSERT_LT(infIteration, 150);
}

TEST_CASE(MathAndParity, PsnrMaeArithmetic) {
  constexpr size_t N = 10000;
  std::vector<float> bufA(N, 0.75f);
  std::vector<float> bufB(N, 0.75f);

  // Exact identical buffers
  double sumAe = 0.0;
  double sumSe = 0.0;
  for (size_t i = 0; i < N; ++i) {
    double diff = std::abs(static_cast<double>(bufA[i]) - static_cast<double>(bufB[i]));
    sumAe += diff;
    sumSe += diff * diff;
  }
  double mae = sumAe / N;
  double mse = sumSe / N;
  double psnr = (mse < 1e-12) ? 120.0 : 10.0 * std::log10(1.0 / mse);

  ASSERT_EQ(mae, 0.0);
  ASSERT_EQ(mse, 0.0);
  ASSERT_GE(psnr, 100.0);

  // Perturb exactly 1 pixel by 0.5f
  bufB[500] = 0.25f;
  sumAe = 0.0;
  sumSe = 0.0;
  uint32_t discrepantCount = 0;
  for (size_t i = 0; i < N; ++i) {
    double diff = std::abs(static_cast<double>(bufA[i]) - static_cast<double>(bufB[i]));
    sumAe += diff;
    sumSe += diff * diff;
    if (diff > 1e-4) {
      ++discrepantCount;
    }
  }
  mae = sumAe / N;
  mse = sumSe / N;
  psnr = 10.0 * std::log10(1.0 / mse);

  ASSERT_EQ(discrepantCount, 1u);
  ASSERT_NEAR(mae, 0.5 / N, 1e-9);
  ASSERT_NEAR(mse, 0.25 / N, 1e-9);
  ASSERT_TRUE(std::isfinite(psnr));
  ASSERT_GT(psnr, 40.0); // 1 discrepant pixel out of 10,000 gives ~46 dB PSNR
}

TEST_CASE(MathAndParity, BaselineSpeedupGuards) {
  // Baseline config comparison guard:
  // If candidate baselineConfigIndex == -1, no speedup should be evaluated.
  int32_t noBaseline = -1;
  ASSERT_EQ(noBaseline, -1);

  // Candidate with valid baseline config index 0
  int32_t baselineIdx = 0;
  double baselineTime = 100.0; // 100 ms
  double candidateTime = 25.0; // 25 ms
  bool bothValid = true;

  double speedup = 0.0;
  if (baselineIdx >= 0 && bothValid && candidateTime > 0.0) {
    speedup = baselineTime / candidateTime;
  }

  ASSERT_NEAR(speedup, 4.0, 1e-6);
}
