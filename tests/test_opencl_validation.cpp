#include "test_harness.h"
#include "benchmarks/Fp32Bench.h"
#include "benchmarks/LdsBankConflictBench.h"
#include "core/ComputeBackendFactory.h"
#include "utils/KernelPath.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace {

std::unique_ptr<IComputeContext> createOpenCLContext() {
  if (!ComputeBackendFactory::isAvailable(ComputeBackend::OpenCL)) {
    return nullptr;
  }
  try {
    auto ctx = ComputeBackendFactory::create(ComputeBackend::OpenCL, false, false);
    if (!ctx || ctx->getDevices().empty()) {
      return nullptr;
    }
    const uint32_t devIdx =
        ::gpubench::test::preferredTestDeviceIndex(ctx->getDevices().size());
    ctx->pickDevice(devIdx);
    return ctx;
  } catch (...) {
    return nullptr;
  }
}

} // namespace

TEST_CASE(OpenCLValidation, OpenCL_ContextDiscoveryAndBufferRoundTrip) {
  auto ctx = createOpenCLContext();
  if (!ctx) {
    SKIP_TEST("OpenCL backend or devices unavailable");
  }

  ASSERT_TRUE(ctx->getBackend() == ComputeBackend::OpenCL);
  const DeviceInfo info = ctx->getCurrentDeviceInfo();
  ASSERT_FALSE(info.name.empty());

  constexpr size_t kCount = 1024u;
  constexpr size_t kBytes = kCount * sizeof(uint32_t);
  std::vector<uint32_t> src(kCount);
  for (size_t i = 0; i < kCount; ++i) {
    src[i] = static_cast<uint32_t>(i * 2654435761u + 17u);
  }

  ComputeBuffer buf = ctx->createBuffer(kBytes);
  ASSERT_TRUE(buf != nullptr);
  ctx->writeBuffer(buf, 0, kBytes, src.data());

  std::vector<uint32_t> dst(kCount, 0u);
  ctx->readBuffer(buf, 0, kBytes, dst.data());
  for (size_t i = 0; i < kCount; ++i) {
    ASSERT_EQ(dst[i], src[i]);
  }
  ctx->releaseBuffer(buf);
}

TEST_CASE(OpenCLValidation, OpenCL_Fp32AndLds_PositiveAndNegativeValidation) {
  auto ctx = createOpenCLContext();
  if (!ctx) {
    SKIP_TEST("OpenCL backend or devices unavailable");
  }

  const std::string kdir = KernelPath::find();
  ASSERT_FALSE(kdir.empty());
  const DeviceInfo info = ctx->getCurrentDeviceInfo();

  // 1. FP32 positive validation on OpenCL
  {
    Fp32Bench fp32;
    if (fp32.IsSupported(info, ctx.get())) {
      fp32.Setup(*ctx, kdir);
      fp32.SetWorkScale(0, 0.25);
      fp32.Run(0);
      ctx->waitIdle();
      ASSERT_TRUE(fp32.ValidateResults(0));
      fp32.Teardown();
    }
  }

  // 2. LDS Bank Conflicts config 0 positive validation on OpenCL
  {
    LdsBankConflictBench lds;
    if (lds.IsSupported(info, ctx.get())) {
      lds.Setup(*ctx, kdir);
      lds.Run(0);
      ctx->waitIdle();
      ASSERT_TRUE(lds.ValidateResults(0));
      lds.Teardown();
    }
  }
}
