#include "ShaderCache.h"
#include <atomic>
#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>
#include <system_error>
#include <thread>
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace utils {

namespace {

std::atomic<uint64_t> g_cacheTmpCounter{0};

std::string sanitizeToken(const std::string &in, const std::string &fallback) {
  std::string out;
  out.reserve(in.size());
  for (unsigned char c : in) {
    if (std::isalnum(c) || c == '_' || c == '-' || c == '.') {
      out.push_back(static_cast<char>(c));
    } else {
      out.push_back('_');
    }
  }
  if (out.empty() || out == "." || out == "..") {
    return fallback;
  }
  return out;
}

bool isCacheFresh(const std::string &kernelName,
                  const std::filesystem::path &cacheFile) {
  std::error_code ec;
  if (!std::filesystem::exists(cacheFile, ec) || ec) {
    return false;
  }
  if (std::filesystem::exists(kernelName, ec) && !ec) {
    auto srcTime = std::filesystem::last_write_time(kernelName, ec);
    if (!ec) {
      auto cacheTime = std::filesystem::last_write_time(cacheFile, ec);
      if (!ec && srcTime > cacheTime) {
        return false;
      }
    }
  }
  return true;
}

template <typename T>
bool loadCacheTyped(const std::string &kernelName,
                    const std::filesystem::path &cacheFile,
                    std::vector<T> &out) {
  if (!isCacheFresh(kernelName, cacheFile)) {
    return false;
  }
  std::ifstream file(cacheFile, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    return false;
  }
  const std::streampos pos = file.tellg();
  if (pos <= 0) {
    return false;
  }
  constexpr uint64_t kMaxCacheBytes = 256ULL * 1024ULL * 1024ULL;
  const uint64_t byteSize = static_cast<uint64_t>(pos);
  if (byteSize > kMaxCacheBytes || (byteSize % sizeof(T)) != 0) {
    return false;
  }
  file.seekg(0, std::ios::beg);
  out.resize(static_cast<size_t>(byteSize / sizeof(T)));
  if (file.read(reinterpret_cast<char *>(out.data()),
                static_cast<std::streamsize>(byteSize))) {
    return true;
  }
  out.clear();
  return false;
}

template <typename T>
void saveCacheAtomic(const std::filesystem::path &cacheFile,
                     const std::vector<T> &data) {
  if (data.empty()) {
    return;
  }
  std::error_code ec;
  std::filesystem::create_directories(cacheFile.parent_path(), ec);
  if (ec) {
    return;
  }
#ifdef _WIN32
  const unsigned long pid = static_cast<unsigned long>(_getpid());
#else
  const unsigned long pid = static_cast<unsigned long>(getpid());
#endif
  const uint64_t seq = g_cacheTmpCounter.fetch_add(1, std::memory_order_relaxed);
  const size_t tidHash = std::hash<std::thread::id>{}(std::this_thread::get_id());
  std::filesystem::path tmpFile = cacheFile;
  tmpFile += ".tmp." + std::to_string(pid) + "." + std::to_string(tidHash) + "." + std::to_string(seq);

  bool writeOk = false;
  {
    std::ofstream file(tmpFile, std::ios::binary | std::ios::trunc);
    if (file.is_open()) {
      file.write(reinterpret_cast<const char *>(data.data()),
                 static_cast<std::streamsize>(data.size() * sizeof(T)));
      file.flush();
      writeOk = file.good();
    }
  }
  if (!writeOk) {
    std::filesystem::remove(tmpFile, ec);
    return;
  }
  std::filesystem::rename(tmpFile, cacheFile, ec);
  if (ec) {
    std::filesystem::remove(tmpFile, ec);
  }
}

} // namespace

std::filesystem::path ShaderCache::getCacheDir(const DeviceInfo &device) {
  std::filesystem::path cache_base;
#if defined(_WIN32)
  const char *localAppData = std::getenv("LOCALAPPDATA");
  if (localAppData && *localAppData != '\0') {
    cache_base = std::filesystem::path(localAppData) / "gpubench" / "cache";
  } else {
    const char *userProfile = std::getenv("USERPROFILE");
    cache_base = (userProfile && *userProfile != '\0'
                      ? std::filesystem::path(userProfile)
                      : std::filesystem::path(".")) /
                 ".cache" / "gpubench";
  }
#else
  const char *xdgCache = std::getenv("XDG_CACHE_HOME");
  if (xdgCache && *xdgCache != '\0') {
    cache_base = std::filesystem::path(xdgCache) / "gpubench";
  } else {
    const char *home = std::getenv("HOME");
    cache_base = (home && *home != '\0' ? std::filesystem::path(home)
                                        : std::filesystem::path(".")) /
                 ".cache" / "gpubench";
  }
#endif

  // Create a unique sanitized directory for this driver/device combination
  std::string sig = sanitizeToken(device.driverUUID, "unknown_uuid") + "_" +
                    std::to_string(device.driverVersion);
  std::filesystem::path dir = cache_base / sig;

  std::error_code ec;
  std::filesystem::create_directories(dir, ec);

  return dir;
}

std::string ShaderCache::getSafeName(const std::string &name) {
  std::filesystem::path p(name);
  return sanitizeToken(p.filename().string(), "kernel");
}

bool ShaderCache::loadVulkanCache(const std::string &kernel_name,
                                  const DeviceInfo &device,
                                  std::vector<uint32_t> &spirv) {
  std::filesystem::path cache_file =
      getCacheDir(device) / (getSafeName(kernel_name) + ".spv");
  return loadCacheTyped(kernel_name, cache_file, spirv);
}

void ShaderCache::saveVulkanCache(const std::string &kernel_name,
                                  const DeviceInfo &device,
                                  const std::vector<uint32_t> &spirv) {
  std::filesystem::path cache_file =
      getCacheDir(device) / (getSafeName(kernel_name) + ".spv");
  saveCacheAtomic(cache_file, spirv);
}

bool ShaderCache::loadROCmCache(const std::string &kernel_name,
                                const DeviceInfo &device,
                                std::vector<char> &code) {
  std::filesystem::path cache_file =
      getCacheDir(device) / (getSafeName(kernel_name) + ".co");
  return loadCacheTyped(kernel_name, cache_file, code);
}

void ShaderCache::saveROCmCache(const std::string &kernel_name,
                                const DeviceInfo &device,
                                const std::vector<char> &code) {
  std::filesystem::path cache_file =
      getCacheDir(device) / (getSafeName(kernel_name) + ".co");
  saveCacheAtomic(cache_file, code);
}

bool ShaderCache::loadOpenCLCache(const std::string &kernel_name,
                                  const DeviceInfo &device,
                                  std::vector<char> &binary) {
  std::filesystem::path cache_file =
      getCacheDir(device) / (getSafeName(kernel_name) + ".clbin");
  return loadCacheTyped(kernel_name, cache_file, binary);
}

void ShaderCache::saveOpenCLCache(const std::string &kernel_name,
                                  const DeviceInfo &device,
                                  const std::vector<char> &binary) {
  std::filesystem::path cache_file =
      getCacheDir(device) / (getSafeName(kernel_name) + ".clbin");
  saveCacheAtomic(cache_file, binary);
}

} // namespace utils
