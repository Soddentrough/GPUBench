#include "test_harness.h"
#include "utils/KernelPath.h"
#include <cstdlib>
#include <filesystem>

TEST_CASE(KernelPath, EnvironmentVariableOverride) {
  // Set custom path in environment pointing to an existing directory
  std::string tempDir = std::filesystem::temp_directory_path().string();
  setenv("GPUBENCH_KERNEL_PATH", tempDir.c_str(), 1);

  std::string found = KernelPath::find();
  ASSERT_EQ(found, tempDir);

  // Unset environment variable
  unsetenv("GPUBENCH_KERNEL_PATH");
}

TEST_CASE(KernelPath, DefaultPathNonEmpty) {
  unsetenv("GPUBENCH_KERNEL_PATH");
  std::string found = KernelPath::find();
  ASSERT_FALSE(found.empty());
}
