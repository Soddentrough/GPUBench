#include "test_harness.h"
#include "core/DeviceDatabase.h"

using namespace gpubench;

TEST_CASE(DeviceDatabase, StrixHaloGpuExactMatch) {
  // 0x1586 is the AMD Radeon 8060S Graphics (Strix Halo iGPU)
  const auto prof = DeviceDatabase::lookup(0x1002, 0x1586);
  ASSERT_EQ(prof.vendorId, 0x1002u);
  ASSERT_EQ(prof.deviceId, 0x1586u);
  ASSERT_EQ(prof.archName, "gfx1151 (RDNA 3.5)");
  ASSERT_EQ(prof.archFamily, "RDNA 3.5");
  ASSERT_TRUE(prof.isApu);
  ASSERT_NEAR(prof.theoreticalFp32Tflops, 29.7, 0.1);
  ASSERT_NEAR(prof.theoreticalBandwidthGBps, 273.0, 1.0);
}

TEST_CASE(DeviceDatabase, NpuNotConflatedWithGpu) {
  // 0x17F0 is the Strix/Krackan/Strix Halo NPU, NOT a GPU device ID
  const auto prof = DeviceDatabase::lookup(0x1002, 0x17F0);
  // Must NOT match the 8060S GPU row
  ASSERT_NE(prof.deviceId, 0x1586u);
  ASSERT_NE(prof.archName, "gfx1151 (RDNA 3.5)");
}

TEST_CASE(DeviceDatabase, Navi48ArchitectureSeparation) {
  // R9700 (54 CUs cut-down Navi 48)
  const auto r9700 = DeviceDatabase::lookup(0x1002, 0x7551);
  ASSERT_EQ(r9700.archName, "gfx1201 (RDNA 4)");
  ASSERT_EQ(r9700.archFamily, "RDNA 4");
  ASSERT_FALSE(r9700.isApu);
  // Verify 48 MB Infinity Cache (Navi 48 specification), NOT 64 MB
  ASSERT_EQ(r9700.l3CacheBytes, 48u * 1024u * 1024u);
  ASSERT_NEAR(r9700.theoreticalFp32Tflops, 41.06, 0.5);

  // RX 9070 XT (Full 64 CUs Navi 48)
  const auto rx9070xt = DeviceDatabase::lookup(0x1002, 0x7550);
  ASSERT_EQ(rx9070xt.archName, "gfx1201 (RDNA 4)");
  ASSERT_EQ(rx9070xt.archFamily, "RDNA 4");
  ASSERT_FALSE(rx9070xt.isApu);
  ASSERT_EQ(rx9070xt.l3CacheBytes, 48u * 1024u * 1024u);
  ASSERT_NEAR(rx9070xt.theoreticalFp32Tflops, 48.66, 0.1);

  // Verify peak separation: R9700 and 9070 XT must NOT have identical peak TFLOPS
  ASSERT_NE(r9700.theoreticalFp32Tflops, rx9070xt.theoreticalFp32Tflops);
}

TEST_CASE(DeviceDatabase, CdnaGenerations) {
  // MI300X (0x74A1 -> gfx942 (CDNA 3))
  const auto mi300x = DeviceDatabase::lookup(0x1002, 0x74A1);
  ASSERT_EQ(mi300x.archName, "gfx942 (CDNA 3)");
  ASSERT_EQ(mi300x.archFamily, "CDNA 3");

  // MI210 (0x740F -> gfx90a (CDNA 2))
  const auto mi210 = DeviceDatabase::lookup(0x1002, 0x740F);
  ASSERT_EQ(mi210.archName, "gfx90a (CDNA 2)");
  ASSERT_EQ(mi210.archFamily, "CDNA 2");
}

TEST_CASE(DeviceDatabase, NvidiaAdaLovelace) {
  // RTX 4090 (0x10DE, 0x2684)
  const auto rtx4090 = DeviceDatabase::lookup(0x10DE, 0x2684);
  ASSERT_EQ(rtx4090.archFamily, "Ada Lovelace");
  ASSERT_FALSE(rtx4090.isApu);
  ASSERT_GT(rtx4090.theoreticalFp32Tflops, 80.0);
}

TEST_CASE(DeviceDatabase, NameFallbackMatching) {
  std::string name = "AMD Radeon 8060S Graphics";
  const auto prof = DeviceDatabase::lookup(0x1002, 0x0000, name);
  ASSERT_EQ(prof.archName, "gfx1151 (RDNA 3.5)");
  ASSERT_EQ(prof.archFamily, "RDNA 3.5");
  ASSERT_TRUE(prof.isApu);
}

TEST_CASE(DeviceDatabase, UnknownDeviceSynthesisSafe) {
  std::string name = "Fictional Quantum Accelerator";
  const auto prof = DeviceDatabase::lookup(0x9999, 0x8888, name);
  ASSERT_FALSE(prof.archName.empty());
  ASSERT_EQ(prof.archFamily, "Generic");
}

TEST_CASE(DeviceDatabase, ApuPredicate) {
  ASSERT_TRUE(DeviceDatabase::isApuDevice(0x1002, 0x1586, "AMD Radeon 8060S Graphics"));
  ASSERT_FALSE(DeviceDatabase::isApuDevice(0x1002, 0x7550, "AMD Radeon RX 9070 XT"));
}

TEST_CASE(DeviceDatabase, EnrichDeviceInfo) {
  DeviceInfo info;
  info.vendorID = 0x1002;
  info.deviceID = 0x1586;
  info.name = "AMD Radeon 8060S Graphics";
  info.archName = "";
  info.l1CacheSize = 0;
  info.l2CacheSize = 0;
  info.l3CacheSize = 0;

  DeviceDatabase::enrichDeviceInfo(info);

  ASSERT_EQ(info.archName, "gfx1151 (RDNA 3.5)");
  ASSERT_TRUE(info.isApu);
  ASSERT_GT(info.l1CacheSize, 0u);
  ASSERT_GT(info.l2CacheSize, 0u);
}
