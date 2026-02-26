#include "OpenCLContext.h"
#include "utils/ShaderCache.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

// Static members
std::unique_ptr<utils::DynamicLibrary> OpenCLContext::openclLib;
bool OpenCLContext::librariesLoaded = false;

// Function pointers for OpenCL
typedef cl_int (*p_clGetPlatformIDs)(cl_uint, cl_platform_id *, cl_uint *);
typedef cl_int (*p_clGetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint,
                                   cl_device_id *, cl_uint *);
typedef cl_int (*p_clGetDeviceInfo)(cl_device_id, cl_device_info, size_t,
                                    void *, size_t *);
typedef cl_context (*p_clCreateContext)(
    const cl_context_properties *, cl_uint, const cl_device_id *,
    void(CL_CALLBACK *)(const char *, const void *, size_t, void *), void *,
    cl_int *);
typedef cl_int (*p_clReleaseContext)(cl_context);
typedef cl_command_queue (*p_clCreateCommandQueueWithProperties)(
    cl_context, cl_device_id, const cl_queue_properties *, cl_int *);
typedef cl_int (*p_clReleaseCommandQueue)(cl_command_queue);
typedef cl_mem (*p_clCreateBuffer)(cl_context, cl_mem_flags, size_t, void *,
                                   cl_int *);
typedef cl_int (*p_clReleaseMemObject)(cl_mem);
typedef cl_int (*p_clEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_bool,
                                         size_t, size_t, const void *, cl_uint,
                                         const cl_event *, cl_event *);
typedef cl_int (*p_clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_bool,
                                        size_t, size_t, void *, cl_uint,
                                        const cl_event *, cl_event *);
typedef cl_program (*p_clCreateProgramWithSource)(cl_context, cl_uint,
                                                  const char **, const size_t *,
                                                  cl_int *);
typedef cl_program (*p_clCreateProgramWithBinary)(cl_context, cl_uint,
                                                  const cl_device_id *,
                                                  const size_t *,
                                                  const unsigned char **,
                                                  cl_int *, cl_int *);
typedef cl_int (*p_clBuildProgram)(cl_program, cl_uint, const cl_device_id *,
                                   const char *,
                                   void(CL_CALLBACK *)(cl_program, void *),
                                   void *);
typedef cl_int (*p_clGetProgramBuildInfo)(cl_program, cl_device_id,
                                          cl_program_build_info, size_t, void *,
                                          size_t *);
typedef cl_int (*p_clGetProgramInfo)(cl_program, cl_program_info, size_t,
                                     void *, size_t *);
typedef cl_int (*p_clReleaseProgram)(cl_program);
typedef cl_kernel (*p_clCreateKernel)(cl_program, const char *, cl_int *);
typedef cl_int (*p_clReleaseKernel)(cl_kernel);
typedef cl_int (*p_clSetKernelArg)(cl_kernel, cl_uint, size_t, const void *);
typedef cl_int (*p_clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint,
                                           const size_t *, const size_t *,
                                           const size_t *, cl_uint,
                                           const cl_event *, cl_event *);
typedef cl_int (*p_clFinish)(cl_command_queue);

static p_clGetPlatformIDs f_clGetPlatformIDs;
static p_clGetDeviceIDs f_clGetDeviceIDs;
static p_clGetDeviceInfo f_clGetDeviceInfo;
static p_clCreateContext f_clCreateContext;
static p_clReleaseContext f_clReleaseContext;
static p_clCreateCommandQueueWithProperties
    f_clCreateCommandQueueWithProperties;
static p_clReleaseCommandQueue f_clReleaseCommandQueue;
static p_clCreateBuffer f_clCreateBuffer;
static p_clReleaseMemObject f_clReleaseMemObject;
static p_clEnqueueWriteBuffer f_clEnqueueWriteBuffer;
static p_clEnqueueReadBuffer f_clEnqueueReadBuffer;
static p_clCreateProgramWithSource f_clCreateProgramWithSource;
static p_clCreateProgramWithBinary f_clCreateProgramWithBinary;
static p_clBuildProgram f_clBuildProgram;
static p_clGetProgramBuildInfo f_clGetProgramBuildInfo;
static p_clGetProgramInfo f_clGetProgramInfo;
static p_clReleaseProgram f_clReleaseProgram;
static p_clCreateKernel f_clCreateKernel;
static p_clReleaseKernel f_clReleaseKernel;
static p_clSetKernelArg f_clSetKernelArg;
static p_clEnqueueNDRangeKernel f_clEnqueueNDRangeKernel;
static p_clFinish f_clFinish;

bool OpenCLContext::loadLibraries() {
  if (librariesLoaded)
    return openclLib && openclLib->isValid();

#ifdef _WIN32
  openclLib = std::make_unique<utils::DynamicLibrary>("OpenCL.dll");
#else
  openclLib = std::make_unique<utils::DynamicLibrary>("libOpenCL.so.1");
  if (!openclLib->isValid()) {
    openclLib = std::make_unique<utils::DynamicLibrary>("libOpenCL.so");
  }
#endif

  if (openclLib->isValid()) {
    f_clGetPlatformIDs =
        openclLib->getFunction<p_clGetPlatformIDs>("clGetPlatformIDs");
    f_clGetDeviceIDs =
        openclLib->getFunction<p_clGetDeviceIDs>("clGetDeviceIDs");
    f_clGetDeviceInfo =
        openclLib->getFunction<p_clGetDeviceInfo>("clGetDeviceInfo");
    f_clCreateContext =
        openclLib->getFunction<p_clCreateContext>("clCreateContext");
    f_clReleaseContext =
        openclLib->getFunction<p_clReleaseContext>("clReleaseContext");
    f_clCreateCommandQueueWithProperties =
        openclLib->getFunction<p_clCreateCommandQueueWithProperties>(
            "clCreateCommandQueueWithProperties");
    f_clReleaseCommandQueue = openclLib->getFunction<p_clReleaseCommandQueue>(
        "clReleaseCommandQueue");
    f_clCreateBuffer =
        openclLib->getFunction<p_clCreateBuffer>("clCreateBuffer");
    f_clReleaseMemObject =
        openclLib->getFunction<p_clReleaseMemObject>("clReleaseMemObject");
    f_clEnqueueWriteBuffer =
        openclLib->getFunction<p_clEnqueueWriteBuffer>("clEnqueueWriteBuffer");
    f_clEnqueueReadBuffer =
        openclLib->getFunction<p_clEnqueueReadBuffer>("clEnqueueReadBuffer");
    f_clCreateProgramWithSource =
        openclLib->getFunction<p_clCreateProgramWithSource>(
            "clCreateProgramWithSource");
    f_clCreateProgramWithBinary =
        openclLib->getFunction<p_clCreateProgramWithBinary>(
            "clCreateProgramWithBinary");
    f_clBuildProgram =
        openclLib->getFunction<p_clBuildProgram>("clBuildProgram");
    f_clGetProgramBuildInfo = openclLib->getFunction<p_clGetProgramBuildInfo>(
        "clGetProgramBuildInfo");
    f_clGetProgramInfo =
        openclLib->getFunction<p_clGetProgramInfo>("clGetProgramInfo");
    f_clReleaseProgram =
        openclLib->getFunction<p_clReleaseProgram>("clReleaseProgram");
    f_clCreateKernel =
        openclLib->getFunction<p_clCreateKernel>("clCreateKernel");
    f_clReleaseKernel =
        openclLib->getFunction<p_clReleaseKernel>("clReleaseKernel");
    f_clSetKernelArg =
        openclLib->getFunction<p_clSetKernelArg>("clSetKernelArg");
    f_clEnqueueNDRangeKernel = openclLib->getFunction<p_clEnqueueNDRangeKernel>(
        "clEnqueueNDRangeKernel");
    f_clFinish = openclLib->getFunction<p_clFinish>("clFinish");
  }

  librariesLoaded = true;
  return openclLib && openclLib->isValid();
}

void OpenCLContext::waitIdle() {
  if (available)
    f_clFinish(commandQueue);
}

OpenCLContext::OpenCLContext(bool verbose)
    : platform(nullptr), device(nullptr), context(nullptr),
      commandQueue(nullptr), selectedDeviceIndex(0), verbose(verbose) {
  if (!loadLibraries()) {
    available = false;
    return;
  }

  available = true;
  try {
    enumeratePlatformsAndDevices();
  } catch (const std::exception &e) {
    if (verbose)
      std::cerr << "OpenCL initialization failed: " << e.what() << std::endl;
    available = false;
  }
}

OpenCLContext::~OpenCLContext() {
  if (commandQueue) {
    f_clReleaseCommandQueue(commandQueue);
  }
  if (context) {
    f_clReleaseContext(context);
  }
}

void OpenCLContext::enumeratePlatformsAndDevices() {
  if (!available)
    return;

  cl_uint platformCount = 0;
  cl_int err = f_clGetPlatformIDs(0, nullptr, &platformCount);
  if (err != CL_SUCCESS || platformCount == 0) {
    throw std::runtime_error("Failed to find OpenCL platforms");
  }

  std::vector<cl_platform_id> platforms(platformCount);
  f_clGetPlatformIDs(platformCount, platforms.data(), nullptr);

  for (const auto &p : platforms) {
    cl_uint deviceCount = 0;
    err = f_clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &deviceCount);
    if (err == CL_SUCCESS && deviceCount > 0) {
      platform = p;
      devices.resize(deviceCount);
      f_clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, deviceCount,
                       devices.data(), nullptr);
      break;
    }
  }

  if (devices.empty()) {
    for (const auto &p : platforms) {
      cl_uint deviceCount = 0;
      err = f_clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceCount);
      if (err == CL_SUCCESS && deviceCount > 0) {
        platform = p;
        devices.resize(deviceCount);
        f_clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, deviceCount,
                         devices.data(), nullptr);
        break;
      }
    }
  }

  if (devices.empty()) {
    throw std::runtime_error("Failed to find OpenCL devices");
  }
}

const std::vector<DeviceInfo> &OpenCLContext::getDevices() const {
  if (!available)
    return deviceInfos;

  if (deviceInfos.empty()) {
    for (const auto &dev : devices) {
      char name[256];
      f_clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name), name, nullptr);
      std::string deviceName = name;

      if (deviceName.find("llvmpipe") != std::string::npos)
        continue;

      cl_device_type type;
      f_clGetDeviceInfo(dev, CL_DEVICE_TYPE, sizeof(type), &type, nullptr);
      if (type & CL_DEVICE_TYPE_CPU)
        continue;

      DeviceInfo info;
      info.name = deviceName;

      cl_ulong memSize;
      f_clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(memSize),
                        &memSize, nullptr);
      info.memorySize = memSize;

      size_t maxWorkGroupSize;
      f_clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                        sizeof(maxWorkGroupSize), &maxWorkGroupSize, nullptr);
      info.maxWorkGroupSize = static_cast<uint32_t>(maxWorkGroupSize);

      cl_ulong localMemSize;
      f_clGetDeviceInfo(dev, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(localMemSize),
                        &localMemSize, nullptr);
      info.maxComputeSharedMemorySize = static_cast<uint32_t>(localMemSize);

      cl_ulong cacheSize;
      f_clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(cacheSize),
                        &cacheSize, nullptr);
      info.l2CacheSize = static_cast<uint32_t>(cacheSize);

      size_t ext_size;
      f_clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, 0, nullptr, &ext_size);
      std::vector<char> extensions(ext_size);
      f_clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, ext_size, extensions.data(),
                        nullptr);
      std::string ext_str(extensions.data());

      info.fp64Support = (ext_str.find("cl_khr_fp64") != std::string::npos ||
                          ext_str.find("cl_amd_fp64") != std::string::npos);
      info.fp16Support = (ext_str.find("cl_khr_fp16") != std::string::npos);
      info.fp8Support = false;
      info.fp6Support = false;
      info.fp4Support = false;
      info.int8Support = true;
      info.int4Support = false;

      char driverVersion[256];
      f_clGetDeviceInfo(dev, CL_DRIVER_VERSION, sizeof(driverVersion),
                        driverVersion, nullptr);
      info.driverVersion = 0;
      try {
        info.driverVersion = std::stoul(driverVersion);
      } catch (...) {
      }

      char uuid[16];
      cl_int uuid_err =
          f_clGetDeviceInfo(dev, 0x106A, sizeof(uuid), uuid, nullptr);
      if (uuid_err == CL_SUCCESS) {
        char uuid_str[33];
        for (int i = 0; i < 16; ++i) {
          sprintf(&uuid_str[i * 2], "%02x", (unsigned char)uuid[i]);
        }
        info.driverUUID = std::string(uuid_str);
      } else {
        char vendor[256];
        f_clGetDeviceInfo(dev, CL_DEVICE_VENDOR, sizeof(vendor), vendor,
                          nullptr);
        std::string hash_input =
            deviceName + std::string(vendor) + std::string(driverVersion);
        info.driverUUID = std::to_string(std::hash<std::string>{}(hash_input));
      }

      deviceInfos.push_back(info);
    }
  }
  return deviceInfos;
}

void OpenCLContext::pickDevice(uint32_t index) {
  if (!available || index >= devices.size()) {
    throw std::runtime_error("Invalid device index or OpenCL not available");
  }
  selectedDeviceIndex = index;
  device = devices[index];
  createContext();
  createCommandQueue();
}

DeviceInfo OpenCLContext::getCurrentDeviceInfo() const {
  if (!device) {
    throw std::runtime_error("No device selected");
  }

  DeviceInfo info;
  char name[256];
  f_clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, nullptr);
  info.name = name;

  cl_ulong memSize;
  f_clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(memSize),
                    &memSize, nullptr);
  info.memorySize = memSize;

  size_t maxWorkGroupSize;
  f_clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                    sizeof(maxWorkGroupSize), &maxWorkGroupSize, nullptr);
  info.maxWorkGroupSize = static_cast<uint32_t>(maxWorkGroupSize);

  cl_ulong localMemSize;
  f_clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(localMemSize),
                    &localMemSize, nullptr);
  info.maxComputeSharedMemorySize = static_cast<uint32_t>(localMemSize);

  cl_ulong cacheSize;
  f_clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(cacheSize),
                    &cacheSize, nullptr);
  info.l2CacheSize = static_cast<uint32_t>(cacheSize);

  size_t ext_size;
  f_clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, nullptr, &ext_size);
  std::vector<char> extensions(ext_size);
  f_clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, ext_size, extensions.data(),
                    nullptr);
  std::string ext_str(extensions.data());

  info.fp64Support = (ext_str.find("cl_khr_fp64") != std::string::npos ||
                      ext_str.find("cl_amd_fp64") != std::string::npos);
  info.fp16Support = (ext_str.find("cl_khr_fp16") != std::string::npos);
  info.fp8Support = false;
  info.fp6Support = false;
  info.fp4Support = false;
  info.int8Support = true;
  info.int4Support = false;

  return info;
}

void OpenCLContext::createContext() {
  cl_int err;
  context = f_clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
  if (err != CL_SUCCESS) {
    throw std::runtime_error("Failed to create OpenCL context");
  }
}

void OpenCLContext::createCommandQueue() {
  cl_int err;
  commandQueue = f_clCreateCommandQueueWithProperties(context, device, 0, &err);
  if (err != CL_SUCCESS) {
    throw std::runtime_error("Failed to create OpenCL command queue");
  }
}

ComputeBuffer OpenCLContext::createBuffer(size_t size, const void *host_ptr) {
  if (!available)
    throw std::runtime_error("OpenCL not available");
  if (size == 0) {
    throw std::runtime_error("Cannot create OpenCL buffer with size 0");
  }

  cl_int err;
  cl_mem_flags flags = CL_MEM_READ_WRITE;
  if (host_ptr) {
    flags |= CL_MEM_USE_HOST_PTR;
  }
  cl_mem buffer = f_clCreateBuffer(context, flags, size,
                                   const_cast<void *>(host_ptr), &err);
  if (err != CL_SUCCESS) {
    throw std::runtime_error("Failed to create OpenCL buffer");
  }
  return new ComputeBuffer_cl{buffer};
}

void OpenCLContext::writeBuffer(ComputeBuffer buffer, size_t offset,
                                size_t size, const void *host_ptr) {
  auto *buffer_cl = static_cast<ComputeBuffer_cl *>(buffer);
  cl_int err =
      f_clEnqueueWriteBuffer(commandQueue, buffer_cl->buffer, CL_TRUE, offset,
                             size, host_ptr, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    throw std::runtime_error("Failed to write to OpenCL buffer");
  }
}

void OpenCLContext::readBuffer(ComputeBuffer buffer, size_t offset, size_t size,
                               void *host_ptr) const {
  const auto *buffer_cl = static_cast<const ComputeBuffer_cl *>(buffer);
  cl_int err =
      f_clEnqueueReadBuffer(commandQueue, buffer_cl->buffer, CL_TRUE, offset,
                            size, host_ptr, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    throw std::runtime_error("Failed to read from OpenCL buffer");
  }
}

void OpenCLContext::releaseBuffer(ComputeBuffer buffer) {
  if (buffer) {
    auto *buffer_cl = static_cast<ComputeBuffer_cl *>(buffer);
    f_clReleaseMemObject(buffer_cl->buffer);
    delete buffer_cl;
  }
}

ComputeKernel OpenCLContext::createKernel(const std::string &file_name,
                                          const std::string &kernel_name,
                                          uint32_t num_args) {
  if (!available)
    throw std::runtime_error("OpenCL not available");
  cl_int err;
  cl_program program;
  std::vector<char> program_binary;
  if (utils::ShaderCache::loadOpenCLCache(
          file_name, getDevices()[selectedDeviceIndex], program_binary)) {
    const unsigned char *binary_ptr =
        reinterpret_cast<const unsigned char *>(program_binary.data());
    size_t binary_size = program_binary.size();
    cl_int binary_status;
    program = f_clCreateProgramWithBinary(context, 1, &device, &binary_size,
                                          &binary_ptr, &binary_status, &err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Failed to create OpenCL program from binary");
    }
  } else {
    std::ifstream file(file_name);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open kernel file: " + file_name);
    }
    std::string source((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());

    std::cout << "--- OPENCL COMPILER READING KERNEL ---" << std::endl;
    std::cout << "File: " << file_name << std::endl;
    std::cout << "First 150 chars: "
              << source.substr(0, std::min(source.size(), (size_t)150))
              << std::endl;

    const char *source_ptr = source.c_str();
    size_t source_size = source.length();

    program = f_clCreateProgramWithSource(context, 1, &source_ptr, &source_size,
                                          &err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("Failed to create OpenCL program");
    }
  }

  err = f_clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    size_t log_size;
    f_clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr,
                            &log_size);
    std::vector<char> log(log_size);
    f_clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size,
                            log.data(), nullptr);
    std::string log_str(log.begin(), log.end());
    throw std::runtime_error("Failed to build OpenCL program: " + log_str);
  }

  if (program_binary.empty()) {
    size_t binary_size;
    f_clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t),
                       &binary_size, nullptr);
    if (binary_size > 0) {
      program_binary.resize(binary_size);
      char *bin_ptr = program_binary.data();
      f_clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(char *), &bin_ptr,
                         nullptr);
      utils::ShaderCache::saveOpenCLCache(
          file_name, getDevices()[selectedDeviceIndex], program_binary);
    }
  }

  cl_kernel kernel = f_clCreateKernel(program, kernel_name.c_str(), &err);
  if (err != CL_SUCCESS) {
    f_clReleaseProgram(program);
    throw std::runtime_error("Failed to create OpenCL kernel");
  }

  return new ComputeKernel_cl{program, kernel};
}

void OpenCLContext::setKernelArg(ComputeKernel kernel, uint32_t arg_index,
                                 ComputeBuffer buffer) {
  auto *kernel_cl = static_cast<ComputeKernel_cl *>(kernel);
  auto *buffer_cl = static_cast<ComputeBuffer_cl *>(buffer);
  cl_int err = f_clSetKernelArg(kernel_cl->kernel, arg_index, sizeof(cl_mem),
                                &buffer_cl->buffer);
  if (err != CL_SUCCESS) {
    throw std::runtime_error(
        "Failed to set OpenCL kernel buffer argument at index " +
        std::to_string(arg_index) + ". Error: " + std::to_string(err));
  }
}

void OpenCLContext::setKernelArg(ComputeKernel kernel, uint32_t arg_index,
                                 size_t arg_size, const void *arg_value) {
  auto *kernel_cl = static_cast<ComputeKernel_cl *>(kernel);
  cl_int err =
      f_clSetKernelArg(kernel_cl->kernel, arg_index, arg_size, arg_value);
  if (err != CL_SUCCESS) {
    throw std::runtime_error("Failed to set OpenCL kernel value argument " +
                             std::to_string(arg_index) + " with size " +
                             std::to_string(arg_size) +
                             ". Error: " + std::to_string(err));
  }
}

void OpenCLContext::dispatch(ComputeKernel kernel, uint32_t grid_x,
                             uint32_t grid_y, uint32_t grid_z, uint32_t block_x,
                             uint32_t block_y, uint32_t block_z) {
  auto *kernel_cl = static_cast<ComputeKernel_cl *>(kernel);
  size_t global_work_size[3] = {(size_t)grid_x * block_x,
                                (size_t)grid_y * block_y,
                                (size_t)grid_z * block_z};
  size_t local_work_size[3] = {(size_t)block_x, (size_t)block_y,
                               (size_t)block_z};
  cl_int err = f_clEnqueueNDRangeKernel(commandQueue, kernel_cl->kernel, 3,
                                        nullptr, global_work_size,
                                        local_work_size, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    throw std::runtime_error("Failed to dispatch OpenCL kernel");
  }
}

void OpenCLContext::releaseKernel(ComputeKernel kernel) {
  if (kernel) {
    auto *kernel_cl = static_cast<ComputeKernel_cl *>(kernel);
    f_clReleaseKernel(kernel_cl->kernel);
    f_clReleaseProgram(kernel_cl->program);
    delete kernel_cl;
  }
}
