#include "test_harness.h"
#include "utils/KernelPath.h"
#include <cstdlib>
#include <filesystem>

#if defined(_WIN32)
static void set_env_var(const char* name, const char* value) {
  _putenv_s(name, value);
}
static void unset_env_var(const char* name) {
  _putenv_s(name, "");
}
#else
static void set_env_var(const char* name, const char* value) {
  setenv(name, value, 1);
}
static void unset_env_var(const char* name) {
  unsetenv(name);
}
#endif

namespace {
class EnvVarGuard {
public:
  explicit EnvVarGuard(const char* name) : name_(name) {
    if (const char* cur = std::getenv(name_)) {
      hadOld_ = true;
      oldVal_ = cur;
    }
  }
  ~EnvVarGuard() {
    if (hadOld_) {
      set_env_var(name_, oldVal_.c_str());
    } else {
      unset_env_var(name_);
    }
  }
private:
  const char* name_;
  bool hadOld_ = false;
  std::string oldVal_;
};
} // namespace

TEST_CASE(KernelPath, EnvironmentVariableOverride) {
  EnvVarGuard guard("GPUBENCH_KERNEL_PATH");
  // Set custom path in environment pointing to an existing directory
  std::string tempDir = std::filesystem::temp_directory_path().string();
  set_env_var("GPUBENCH_KERNEL_PATH", tempDir.c_str());

  std::string found = KernelPath::find();
  ASSERT_EQ(found, tempDir);
}

TEST_CASE(KernelPath, DefaultPathNonEmpty) {
  EnvVarGuard guard("GPUBENCH_KERNEL_PATH");
  unset_env_var("GPUBENCH_KERNEL_PATH");
  std::string found = KernelPath::find();
  ASSERT_FALSE(found.empty());
}
