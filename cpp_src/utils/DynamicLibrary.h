#pragma once
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace utils {

class DynamicLibrary {
public:
  explicit DynamicLibrary(const std::string &name) : name(name) {
#ifdef _WIN32
    handle = LoadLibraryA(name.c_str());
#else
    handle = dlopen(name.c_str(), RTLD_LAZY | RTLD_LOCAL);
#endif
  }

  DynamicLibrary(const DynamicLibrary &) = delete;
  DynamicLibrary &operator=(const DynamicLibrary &) = delete;

  DynamicLibrary(DynamicLibrary &&other) noexcept
      : handle(std::exchange(other.handle, nullptr)),
        name(std::move(other.name)) {}

  DynamicLibrary &operator=(DynamicLibrary &&other) noexcept {
    if (this != &other) {
      close();
      handle = std::exchange(other.handle, nullptr);
      name = std::move(other.name);
    }
    return *this;
  }

  ~DynamicLibrary() { close(); }

  void close() noexcept {
    if (handle) {
#ifdef _WIN32
      FreeLibrary((HMODULE)handle);
#else
      dlclose(handle);
#endif
      handle = nullptr;
    }
  }

  bool isValid() const { return handle != nullptr; }

  template <typename T> T getFunction(const std::string &funcName) {
    if (!handle)
      return nullptr;

#ifdef _WIN32
    T func =
        reinterpret_cast<T>(GetProcAddress((HMODULE)handle, funcName.c_str()));
#else
    T func = reinterpret_cast<T>(dlsym(handle, funcName.c_str()));
#endif
    return func;
  }

  const std::string &getName() const { return name; }

private:
  void *handle = nullptr;
  std::string name;
};

} // namespace utils
