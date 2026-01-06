#pragma once

#include "IComputeContext.h"
#include "utils/DynamicLibrary.h"
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

class ROCmContext : public IComputeContext {
public:
  ROCmContext(bool verbose = false);
  ~ROCmContext() override;

  ComputeBackend getBackend() const override { return ComputeBackend::ROCm; }
  bool isAvailable() const override { return available; }
  const std::vector<DeviceInfo> &getDevices() const override { return devices; }
  void pickDevice(uint32_t index) override;
  DeviceInfo getCurrentDeviceInfo() const override;
  uint32_t getSelectedDeviceIndex() const override {
    return static_cast<uint32_t>(selectedDeviceIndex);
  }

  // Buffer management
  ComputeBuffer createBuffer(size_t size,
                             const void *host_ptr = nullptr) override;
  void writeBuffer(ComputeBuffer buffer, size_t offset, size_t size,
                   const void *host_ptr) override;
  void readBuffer(ComputeBuffer buffer, size_t offset, size_t size,
                  void *host_ptr) const override;
  void releaseBuffer(ComputeBuffer buffer) override;

  // Kernel management
  ComputeKernel createKernel(const std::string &file_name,
                             const std::string &kernel_name,
                             uint32_t num_args) override;
  void setKernelArg(ComputeKernel kernel, uint32_t arg_index,
                    ComputeBuffer buffer) override;
  void setKernelArg(ComputeKernel kernel, uint32_t arg_index, size_t arg_size,
                    const void *arg_value) override;
  void dispatch(ComputeKernel kernel, uint32_t grid_x, uint32_t grid_y,
                uint32_t grid_z, uint32_t block_x, uint32_t block_y,
                uint32_t block_z) override;
  void releaseKernel(ComputeKernel kernel) override;
  void waitIdle() override;

  hipDevice_t getROCmDevice() const override { return device; }

private:
  struct ROCmKernel {
    hipFunction_t function;
    std::map<uint32_t, std::vector<char>> args;
  };

  static std::unique_ptr<utils::DynamicLibrary> hipLib;
  static std::unique_ptr<utils::DynamicLibrary> hiprtcLib;
  static bool librariesLoaded;
  static bool loadLibraries();
  void enumerateDevices();

  std::vector<DeviceInfo> devices;
  hipDevice_t device;
  int selectedDeviceIndex = -1;

  std::unordered_map<std::string, hipModule_t> modules;
  std::unordered_map<ComputeKernel, ROCmKernel> kernels;
  bool verbose = false;
  bool available = false;
};
