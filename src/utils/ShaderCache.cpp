#include "ShaderCache.h"
#include <fstream>
#include <iostream>
#include <sstream>

namespace utils {

std::filesystem::path ShaderCache::getCacheDir(const DeviceInfo &device) {
  std::filesystem::path home = std::getenv("HOME") ? std::getenv("HOME") : ".";
  std::filesystem::path cache_base = home / ".cache" / "gpubench";

  // Create a unique directory for this driver/device combination
  std::string sig =
      device.driverUUID + "_" + std::to_string(device.driverVersion);
  std::filesystem::path dir = cache_base / sig;

  if (!std::filesystem::exists(dir)) {
    std::filesystem::create_directories(dir);
  }

  return dir;
}

std::string ShaderCache::getSafeName(const std::string &name) {
  std::filesystem::path p(name);
  return p.filename().string();
}

bool ShaderCache::loadVulkanCache(const std::string &kernel_name,
                                  const DeviceInfo &device,
                                  std::vector<uint32_t> &spirv) {
  std::filesystem::path cache_file =
      getCacheDir(device) / (getSafeName(kernel_name) + ".spv");

  if (!std::filesystem::exists(cache_file)) {
    return false;
  }

  std::ifstream file(cache_file, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    return false;
  }

  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  spirv.resize(size / sizeof(uint32_t));
  if (file.read(reinterpret_cast<char *>(spirv.data()), size)) {
    return true;
  }

  return false;
}

void ShaderCache::saveVulkanCache(const std::string &kernel_name,
                                  const DeviceInfo &device,
                                  const std::vector<uint32_t> &spirv) {
  std::filesystem::path cache_file =
      getCacheDir(device) / (getSafeName(kernel_name) + ".spv");

  std::ofstream file(cache_file, std::ios::binary);
  if (file.is_open()) {
    file.write(reinterpret_cast<const char *>(spirv.data()),
               spirv.size() * sizeof(uint32_t));
  }
}

bool ShaderCache::loadROCmCache(const std::string &kernel_name,
                                const DeviceInfo &device,
                                std::vector<char> &code) {
  std::filesystem::path cache_file =
      getCacheDir(device) / (getSafeName(kernel_name) + ".co");

  if (!std::filesystem::exists(cache_file)) {
    return false;
  }

  std::ifstream file(cache_file, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    return false;
  }

  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  code.resize(size);
  if (file.read(code.data(), size)) {
    return true;
  }

  return false;
}

void ShaderCache::saveROCmCache(const std::string &kernel_name,
                                const DeviceInfo &device,
                                const std::vector<char> &code) {
  std::filesystem::path cache_file =
      getCacheDir(device) / (getSafeName(kernel_name) + ".co");

  std::ofstream file(cache_file, std::ios::binary);
  if (file.is_open()) {
    file.write(code.data(), code.size());
  }
}

bool ShaderCache::loadOpenCLCache(const std::string &kernel_name,
                                  const DeviceInfo &device,
                                  std::vector<char> &binary) {
  std::filesystem::path cache_file =
      getCacheDir(device) / (getSafeName(kernel_name) + ".clbin");

  if (!std::filesystem::exists(cache_file)) {
    return false;
  }

  std::ifstream file(cache_file, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    return false;
  }

  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  binary.resize(size);
  if (file.read(binary.data(), size)) {
    return true;
  }

  return false;
}

void ShaderCache::saveOpenCLCache(const std::string &kernel_name,
                                  const DeviceInfo &device,
                                  const std::vector<char> &binary) {
  std::filesystem::path cache_file =
      getCacheDir(device) / (getSafeName(kernel_name) + ".clbin");

  std::ofstream file(cache_file, std::ios::binary);
  if (file.is_open()) {
    file.write(binary.data(), binary.size());
  }
}

} // namespace utils
