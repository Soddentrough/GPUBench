#pragma once

#include "ComputeBackend.h"
#include <string>
#include <vector>
#include <cstdint>

// Forward declarations for backend-specific types
struct VkPhysicalDevice_T;
struct VkDevice_T;
typedef VkPhysicalDevice_T* VkPhysicalDevice;
typedef VkDevice_T* VkDevice;

struct _cl_platform_id;
struct _cl_device_id;
struct _cl_context;
typedef _cl_platform_id* cl_platform_id;
typedef _cl_device_id* cl_device_id;
typedef _cl_context* cl_context;

// Forward declarations for ROCm/HIP types
struct ihipCtx_t;
struct ihipModule_t;
struct ihipModuleSymbol_t;
typedef int hipDevice_t;
typedef struct ihipCtx_t* hipCtx_t;
typedef struct ihipModule_t* hipModule_t;
typedef struct ihipModuleSymbol_t* hipFunction_t;

struct DeviceInfo {
    std::string name;
    uint64_t memorySize;
    uint32_t maxWorkGroupSize;
    uint32_t maxComputeWorkGroupCountX;
    uint32_t maxComputeWorkGroupCountY;
    uint32_t maxComputeWorkGroupCountZ;
    uint32_t maxComputeSharedMemorySize;
    uint32_t subgroupSize;
    uint32_t l1CacheSize = 0;
    uint32_t l2CacheSize = 0;
    uint32_t l3CacheSize = 0;
    bool fp64Support = false;
    bool fp16Support = false;
    bool fp8Support = false;
    bool fp6Support = false;
    bool fp4Support = false;
    bool int8Support = false;
    bool int4Support = false;
    bool cooperativeMatrixSupport = false;
    bool structuredSparsitySupport = false;
    bool verbose = false;
};

// Opaque handles for compute resources
using ComputeBuffer = void*;
using ComputeKernel = void*;

class IComputeContext {
public:
    virtual ~IComputeContext() = default;

    virtual ComputeBackend getBackend() const = 0;
    virtual const std::vector<DeviceInfo>& getDevices() const = 0;
    virtual void pickDevice(uint32_t index) = 0;
    virtual DeviceInfo getCurrentDeviceInfo() const = 0;
    virtual uint32_t getSelectedDeviceIndex() const = 0;

    // Buffer management
    virtual ComputeBuffer createBuffer(size_t size, const void* host_ptr = nullptr) = 0;
    virtual void writeBuffer(ComputeBuffer buffer, size_t offset, size_t size, const void* host_ptr) = 0;
    virtual void readBuffer(ComputeBuffer buffer, size_t offset, size_t size, void* host_ptr) const = 0;
    virtual void releaseBuffer(ComputeBuffer buffer) = 0;

    // Kernel management
    virtual ComputeKernel createKernel(const std::string& file_name, const std::string& kernel_name, uint32_t num_args) = 0;
    virtual void setKernelArg(ComputeKernel kernel, uint32_t arg_index, ComputeBuffer buffer) = 0;
    virtual void setKernelArg(ComputeKernel kernel, uint32_t arg_index, size_t arg_size, const void* arg_value) = 0;
    virtual void dispatch(ComputeKernel kernel, uint32_t grid_x, uint32_t grid_y, uint32_t grid_z, uint32_t block_x, uint32_t block_y, uint32_t block_z) = 0;
    virtual void releaseKernel(ComputeKernel kernel) = 0;
    virtual void waitIdle() = 0;
    
    // Backend-specific accessors (returns nullptr if not applicable)
    virtual VkPhysicalDevice getVulkanPhysicalDevice() const { return nullptr; }
    virtual VkDevice getVulkanDevice() const { return nullptr; }
    virtual void* getVulkanContext() const { return nullptr; }
    
    virtual cl_platform_id getOpenCLPlatform() const { return nullptr; }
    virtual cl_device_id getOpenCLDevice() const { return nullptr; }
    virtual cl_context getOpenCLContext() const { return nullptr; }

    virtual hipDevice_t getROCmDevice() const { return -1; }
    virtual hipCtx_t getROCmContext() const { return nullptr; }
};
