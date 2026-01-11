#pragma once

#include "IComputeContext.h"
#define CL_TARGET_OPENCL_VERSION 300
#include "utils/DynamicLibrary.h"
#include <CL/cl.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

class OpenCLContext : public IComputeContext {
public:
  OpenCLContext(bool verbose = false);
  ~OpenCLContext();

  OpenCLContext(const OpenCLContext &) = delete;
  OpenCLContext &operator=(const OpenCLContext &) = delete;

  // IComputeContext interface
  ComputeBackend getBackend() const override { return ComputeBackend::OpenCL; }
  bool isAvailable() const override { return available; }
  const std::vector<DeviceInfo> &getDevices() const override;
  void pickDevice(uint32_t index) override;
  DeviceInfo getCurrentDeviceInfo() const override;
  uint32_t getSelectedDeviceIndex() const override {
    return selectedDeviceIndex;
  }

  cl_platform_id getOpenCLPlatform() const override { return platform; }
  cl_device_id getOpenCLDevice() const override { return device; }
  cl_context getOpenCLContext() const override { return context; }

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
  void setKernelAS(ComputeKernel kernel, uint32_t arg_index,
                   AccelerationStructure as) override {
    throw std::runtime_error("setKernelAS not supported on OpenCL backend");
  }
  void setKernelArg(ComputeKernel kernel, uint32_t arg_index, size_t arg_size,
                    const void *arg_value) override;
  void dispatch(ComputeKernel kernel, uint32_t grid_x, uint32_t grid_y,
                uint32_t grid_z, uint32_t block_x, uint32_t block_y,
                uint32_t block_z) override;
  void releaseKernel(ComputeKernel kernel) override;
  void waitIdle() override;

  // OpenCL-specific accessors
  cl_command_queue getCommandQueue() const { return commandQueue; }

private:
  struct ComputeBuffer_cl {
    cl_mem buffer;
  };

  struct ComputeKernel_cl {
    cl_program program;
    cl_kernel kernel;
  };

  void enumeratePlatformsAndDevices();
  void createContext();
  void createCommandQueue();

  static std::unique_ptr<utils::DynamicLibrary> openclLib;
  static bool librariesLoaded;
  static bool loadLibraries();

  cl_platform_id platform = nullptr;
  std::vector<cl_device_id> devices;
  cl_device_id device = nullptr;
  cl_context context = nullptr;
  cl_command_queue commandQueue = nullptr;

  mutable std::vector<DeviceInfo> deviceInfos;
  uint32_t selectedDeviceIndex = 0;
  bool verbose = false;
  bool available = false;
};
