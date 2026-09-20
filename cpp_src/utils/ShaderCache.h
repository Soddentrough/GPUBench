#pragma once

#include "core/IComputeContext.h"
#include <filesystem>
#include <string>
#include <vector>

namespace utils {

class ShaderCache {
public:
  static std::filesystem::path getCacheDir(const DeviceInfo &device);

  static bool loadVulkanCache(const std::string &kernel_name,
                              const DeviceInfo &device,
                              std::vector<uint32_t> &spirv);
  static void saveVulkanCache(const std::string &kernel_name,
                              const DeviceInfo &device,
                              const std::vector<uint32_t> &spirv);

  static bool loadROCmCache(const std::string &kernel_name,
                            const DeviceInfo &device, std::vector<char> &code);
  static void saveROCmCache(const std::string &kernel_name,
                            const DeviceInfo &device,
                            const std::vector<char> &code);

  static bool loadOpenCLCache(const std::string &kernel_name,
                              const DeviceInfo &device,
                              std::vector<char> &binary);
  static void saveOpenCLCache(const std::string &kernel_name,
                              const DeviceInfo &device,
                              const std::vector<char> &binary);

private:
  static std::string getSafeName(const std::string &name);
};

} // namespace utils
