#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif

#include "ROCmContext.h"
#include "DeviceDatabase.h"
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <stdexcept>
#ifdef HAVE_HIPRTC
#include <hip/hiprtc.h>
#endif
#include "utils/ShaderCache.h"
#include <cstring>
#include <fstream>
#include <algorithm>
#include <cctype>

// Static members
std::unique_ptr<utils::DynamicLibrary> ROCmContext::hipLib;
std::unique_ptr<utils::DynamicLibrary> ROCmContext::hiprtcLib;
bool ROCmContext::librariesLoaded = false;

// Function pointers for HIP
typedef hipError_t (*p_hipInit)(unsigned int);
typedef hipError_t (*p_hipGetDeviceCount)(int *);
typedef hipError_t (*p_hipGetDeviceProperties)(hipDeviceProp_t *, int);
typedef hipError_t (*p_hipRuntimeGetVersion)(int *);
typedef hipError_t (*p_hipSetDevice)(int);
typedef const char *(*p_hipGetErrorString)(hipError_t);
typedef hipError_t (*p_hipMalloc)(void **, size_t);
typedef hipError_t (*p_hipMemcpy)(void *, const void *, size_t, hipMemcpyKind);
typedef hipError_t (*p_hipFree)(void *);
typedef hipError_t (*p_hipModuleLoadData)(hipModule_t *, const void *);
typedef hipError_t (*p_hipModuleLoad)(hipModule_t *, const char *);
typedef hipError_t (*p_hipModuleGetFunction)(hipFunction_t *, hipModule_t,
                                             const char *);
typedef hipError_t (*p_hipModuleLaunchKernel)(hipFunction_t, unsigned int,
                                              unsigned int, unsigned int,
                                              unsigned int, unsigned int,
                                              unsigned int, unsigned int,
                                              hipStream_t, void **, void **);
typedef hipError_t (*p_hipModuleUnload)(hipModule_t);
typedef hipError_t (*p_hipDeviceSynchronize)(void);
typedef hipError_t (*p_hipMemGetInfo)(size_t *, size_t *);
typedef hipError_t (*p_hipEventCreate)(hipEvent_t *);
typedef hipError_t (*p_hipEventDestroy)(hipEvent_t);
typedef hipError_t (*p_hipEventRecord)(hipEvent_t, hipStream_t);
typedef hipError_t (*p_hipEventSynchronize)(hipEvent_t);
typedef hipError_t (*p_hipEventElapsedTime)(float *, hipEvent_t, hipEvent_t);

// Function pointers for HIPRTC
#ifdef HAVE_HIPRTC
typedef hiprtcResult (*p_hiprtcCreateProgram)(hiprtcProgram *, const char *,
                                              const char *, int, const char **,
                                              const char **);
typedef hiprtcResult (*p_hiprtcCompileProgram)(hiprtcProgram, int,
                                               const char **);
typedef hiprtcResult (*p_hiprtcGetProgramLogSize)(hiprtcProgram, size_t *);
typedef hiprtcResult (*p_hiprtcGetProgramLog)(hiprtcProgram, char *);
typedef hiprtcResult (*p_hiprtcGetCodeSize)(hiprtcProgram, size_t *);
typedef hiprtcResult (*p_hiprtcGetCode)(hiprtcProgram, char *);
typedef hiprtcResult (*p_hiprtcDestroyProgram)(hiprtcProgram *);

static p_hiprtcCreateProgram f_hiprtcCreateProgram;
static p_hiprtcCompileProgram f_hiprtcCompileProgram;
static p_hiprtcGetProgramLogSize f_hiprtcGetProgramLogSize;
static p_hiprtcGetProgramLog f_hiprtcGetProgramLog;
static p_hiprtcGetCodeSize f_hiprtcGetCodeSize;
static p_hiprtcGetCode f_hiprtcGetCode;
static p_hiprtcDestroyProgram f_hiprtcDestroyProgram;
#endif

static p_hipInit f_hipInit;
static p_hipGetDeviceCount f_hipGetDeviceCount;
static p_hipGetDeviceProperties f_hipGetDeviceProperties;
static p_hipRuntimeGetVersion f_hipRuntimeGetVersion;
static p_hipSetDevice f_hipSetDevice;
static p_hipGetErrorString f_hipGetErrorString;
static p_hipMalloc f_hipMalloc;
static p_hipMemcpy f_hipMemcpy;
static p_hipFree f_hipFree;
static p_hipModuleLoadData f_hipModuleLoadData;
static p_hipModuleLoad f_hipModuleLoad;
static p_hipModuleGetFunction f_hipModuleGetFunction;
static p_hipModuleLaunchKernel f_hipModuleLaunchKernel;
static p_hipModuleUnload f_hipModuleUnload;
static p_hipDeviceSynchronize f_hipDeviceSynchronize;
static p_hipMemGetInfo f_hipMemGetInfo;
static p_hipEventCreate f_hipEventCreate;
static p_hipEventDestroy f_hipEventDestroy;
static p_hipEventRecord f_hipEventRecord;
static p_hipEventSynchronize f_hipEventSynchronize;
static p_hipEventElapsedTime f_hipEventElapsedTime;
typedef hipError_t (*p_hipFuncGetAttribute)(int *, hipFunction_attribute, hipFunction_t);
static p_hipFuncGetAttribute f_hipFuncGetAttribute;

bool ROCmContext::loadLibraries() {
  if (librariesLoaded)
    return hipLib && hipLib->isValid();

#ifdef _WIN32
  hipLib = std::make_unique<utils::DynamicLibrary>("amdhip64.dll");
#else
  hipLib = std::make_unique<utils::DynamicLibrary>("libamdhip64.so.7");
  if (!hipLib->isValid()) {
    hipLib = std::make_unique<utils::DynamicLibrary>("libamdhip64.so.6");
  }
  if (!hipLib->isValid()) {
    hipLib = std::make_unique<utils::DynamicLibrary>("libamdhip64.so");
  }
  if (!hipLib->isValid()) {
    hipLib = std::make_unique<utils::DynamicLibrary>("/opt/rocm/core-10.1/lib/libamdhip64.so.7");
  }
  if (!hipLib->isValid()) {
    hipLib = std::make_unique<utils::DynamicLibrary>("/opt/rocm/core-10.1/lib/libamdhip64.so.6");
  }
  if (!hipLib->isValid()) {
    hipLib = std::make_unique<utils::DynamicLibrary>("/opt/rocm/core-10.1/lib/libamdhip64.so");
  }
  if (!hipLib->isValid()) {
    hipLib = std::make_unique<utils::DynamicLibrary>("/opt/rocm/lib/libamdhip64.so.7");
  }
  if (!hipLib->isValid()) {
    hipLib = std::make_unique<utils::DynamicLibrary>("/opt/rocm/lib/libamdhip64.so.6");
  }
  if (!hipLib->isValid()) {
    hipLib = std::make_unique<utils::DynamicLibrary>("/opt/rocm/lib/libamdhip64.so");
  }
#endif

  if (hipLib->isValid()) {
    f_hipInit = hipLib->getFunction<p_hipInit>("hipInit");
    f_hipGetDeviceCount =
        hipLib->getFunction<p_hipGetDeviceCount>("hipGetDeviceCount");
    f_hipGetDeviceProperties =
        hipLib->getFunction<p_hipGetDeviceProperties>("hipGetDeviceProperties");
    f_hipRuntimeGetVersion =
        hipLib->getFunction<p_hipRuntimeGetVersion>("hipRuntimeGetVersion");
    f_hipSetDevice = hipLib->getFunction<p_hipSetDevice>("hipSetDevice");
    f_hipGetErrorString =
        hipLib->getFunction<p_hipGetErrorString>("hipGetErrorString");
    f_hipMalloc = hipLib->getFunction<p_hipMalloc>("hipMalloc");
    f_hipMemcpy = hipLib->getFunction<p_hipMemcpy>("hipMemcpy");
    f_hipFree = hipLib->getFunction<p_hipFree>("hipFree");
    f_hipModuleLoadData =
        hipLib->getFunction<p_hipModuleLoadData>("hipModuleLoadData");
    f_hipModuleLoad = hipLib->getFunction<p_hipModuleLoad>("hipModuleLoad");
    f_hipModuleGetFunction =
        hipLib->getFunction<p_hipModuleGetFunction>("hipModuleGetFunction");
    f_hipModuleLaunchKernel =
        hipLib->getFunction<p_hipModuleLaunchKernel>("hipModuleLaunchKernel");
    f_hipModuleUnload =
        hipLib->getFunction<p_hipModuleUnload>("hipModuleUnload");
    f_hipDeviceSynchronize =
        hipLib->getFunction<p_hipDeviceSynchronize>("hipDeviceSynchronize");
    f_hipMemGetInfo = hipLib->getFunction<p_hipMemGetInfo>("hipMemGetInfo");
    f_hipEventCreate = hipLib->getFunction<p_hipEventCreate>("hipEventCreate");
    f_hipEventDestroy = hipLib->getFunction<p_hipEventDestroy>("hipEventDestroy");
    f_hipEventRecord = hipLib->getFunction<p_hipEventRecord>("hipEventRecord");
    f_hipEventSynchronize = hipLib->getFunction<p_hipEventSynchronize>("hipEventSynchronize");
    f_hipEventElapsedTime = hipLib->getFunction<p_hipEventElapsedTime>("hipEventElapsedTime");
    f_hipFuncGetAttribute = hipLib->getFunction<p_hipFuncGetAttribute>("hipFuncGetAttribute");

#ifdef HAVE_HIPRTC
#ifdef _WIN32
    hiprtcLib = std::make_unique<utils::DynamicLibrary>("hiprtc.dll");
    if (!hiprtcLib->isValid()) {
      hiprtcLib = std::make_unique<utils::DynamicLibrary>("hiprtc64.dll");
    }
#else
    hiprtcLib = std::make_unique<utils::DynamicLibrary>("libhiprtc.so.7");
    if (!hiprtcLib->isValid()) {
      hiprtcLib = std::make_unique<utils::DynamicLibrary>("libhiprtc.so.6");
    }
    if (!hiprtcLib->isValid()) {
      hiprtcLib = std::make_unique<utils::DynamicLibrary>("libhiprtc.so");
    }
    if (!hiprtcLib->isValid()) {
      hiprtcLib = std::make_unique<utils::DynamicLibrary>("/opt/rocm/lib/libhiprtc.so.7");
    }
    if (!hiprtcLib->isValid()) {
      hiprtcLib = std::make_unique<utils::DynamicLibrary>("/opt/rocm/lib/libhiprtc.so.6");
    }
    if (!hiprtcLib->isValid()) {
      hiprtcLib = std::make_unique<utils::DynamicLibrary>("/opt/rocm/lib/libhiprtc.so");
    }
#endif

    if (hiprtcLib->isValid()) {
      f_hiprtcCreateProgram =
          hiprtcLib->getFunction<p_hiprtcCreateProgram>("hiprtcCreateProgram");
      f_hiprtcCompileProgram = hiprtcLib->getFunction<p_hiprtcCompileProgram>(
          "hiprtcCompileProgram");
      f_hiprtcGetProgramLogSize =
          hiprtcLib->getFunction<p_hiprtcGetProgramLogSize>(
              "hiprtcGetProgramLogSize");
      f_hiprtcGetProgramLog =
          hiprtcLib->getFunction<p_hiprtcGetProgramLog>("hiprtcGetProgramLog");
      f_hiprtcGetCodeSize =
          hiprtcLib->getFunction<p_hiprtcGetCodeSize>("hiprtcGetCodeSize");
      f_hiprtcGetCode =
          hiprtcLib->getFunction<p_hiprtcGetCode>("hiprtcGetCode");
      f_hiprtcDestroyProgram = hiprtcLib->getFunction<p_hiprtcDestroyProgram>(
          "hiprtcDestroyProgram");
    }
#endif
  }

  librariesLoaded = true;
  return hipLib && hipLib->isValid();
}

ROCmContext::ROCmContext(bool verbose)
    : device(-1), selectedDeviceIndex(-1), verbose(verbose) {
  if (!loadLibraries()) {
    available = false;
    return;
  }

  if (f_hipInit) {
    (void)f_hipInit(0);
  }

  available = true;
  enumerateDevices();
  if (devices.empty()) {
    available = false;
  }
}

ROCmContext::~ROCmContext() {
  try {
    waitIdle();
  } catch (...) {
  }

  for (auto &pair : kernels) {
    delete static_cast<ROCmKernel *>(pair.first);
  }
  kernels.clear();

  if (f_hipModuleUnload) {
    for (auto &pair : modules) {
      if (pair.second) {
        (void)f_hipModuleUnload(pair.second);
      }
    }
  }
  modules.clear();

  if (f_hipEventDestroy) {
    if (timingStartEvent) {
      (void)f_hipEventDestroy(timingStartEvent);
      timingStartEvent = nullptr;
    }
    if (timingStopEvent) {
      (void)f_hipEventDestroy(timingStopEvent);
      timingStopEvent = nullptr;
    }
  }
}

void ROCmContext::enumerateDevices() {
  if (!available)
    return;

  int deviceCount = 0;
  if (f_hipGetDeviceCount(&deviceCount) != hipSuccess) {
    return;
  }

  for (int i = 0; i < deviceCount; ++i) {
    hipDeviceProp_t prop;
    if (f_hipGetDeviceProperties(&prop, i) == hipSuccess) {
      DeviceInfo info;
      info.name = prop.name;
      info.archName = prop.gcnArchName;

      int runtimeVersion = 0;
      (void)f_hipRuntimeGetVersion(&runtimeVersion);
      info.driverVersion = static_cast<uint32_t>(runtimeVersion);

      char uuid_str[64];
      snprintf(uuid_str, sizeof(uuid_str), "%s_%04x:%02x:%02x.%d",
               prop.gcnArchName, prop.pciDomainID, prop.pciBusID,
               prop.pciDeviceID, i);
      info.driverUUID = std::string(uuid_str);

      info.memorySize = prop.totalGlobalMem;
      if (f_hipMemGetInfo && f_hipSetDevice) {
        if (f_hipSetDevice(i) == hipSuccess) {
          size_t freeBytes = 0, totalBytes = 0;
          if (f_hipMemGetInfo(&freeBytes, &totalBytes) == hipSuccess && totalBytes > 0) {
            info.memorySize = totalBytes;
          }
        }
      }
      info.verbose = verbose;
      info.maxWorkGroupSize = prop.maxThreadsPerBlock;
      info.maxComputeWorkGroupCountX = prop.maxGridSize[0];
      info.maxComputeWorkGroupCountY = prop.maxGridSize[1];
      info.maxComputeWorkGroupCountZ = prop.maxGridSize[2];
      info.maxComputeSharedMemorySize = prop.sharedMemPerBlock;
      info.subgroupSize = prop.warpSize;
      info.l2CacheSize = prop.l2CacheSize;

      std::string archNameStr = prop.gcnArchName;
      std::string deviceNameStr = prop.name;
      std::transform(archNameStr.begin(), archNameStr.end(), archNameStr.begin(), ::tolower);
      std::transform(deviceNameStr.begin(), deviceNameStr.end(), deviceNameStr.begin(), ::tolower);

      DeviceDatabase::enrichDeviceInfo(info);
      const auto &prof = DeviceDatabase::lookup(info.vendorID, info.deviceID, info.name);
      bool is_rdna4 = (prof.archFamily == "RDNA 4") || (archNameStr.find("gfx12") != std::string::npos);
      bool is_rdna3 = (prof.archFamily == "RDNA 3" || prof.archFamily == "RDNA 3.5") || (archNameStr.find("gfx11") != std::string::npos);
      bool is_cdna3 = (prof.archFamily == "CDNA 3") || (archNameStr.find("gfx942") != std::string::npos);

      info.fp8Support = (is_cdna3 || is_rdna4);
      info.fp6Support = false;
      // RDNA4 (gfx12) has no FP4 hardware; FP4 arrived with CDNA4 (gfx950).
      info.fp4Support = false;
      info.fp64Support = true;
      info.fp16Support = true;
      info.bf16Support = true;
      info.int8Support = true;
      info.int4Support = is_rdna4;

      info.cooperativeMatrixSupport = (is_cdna3 || is_rdna3 || is_rdna4);
      devices.push_back(info);
    }
  }
}

void ROCmContext::pickDevice(uint32_t index) {
  if (!available || index >= devices.size()) {
    throw std::runtime_error("Invalid device index or ROCm not available");
  }

  hipError_t err = f_hipSetDevice(index);
  if (err != hipSuccess) {
    if (verbose) {
      std::cerr << "hipSetDevice error: " << f_hipGetErrorString(err)
                << std::endl;
    }
    throw std::runtime_error("Failed to set HIP device: " +
                             std::string(f_hipGetErrorString(err)));
  }

  device = index;
  selectedDeviceIndex = index;

  if (f_hipEventCreate) {
    if (timingStartEvent && f_hipEventDestroy) {
      (void)f_hipEventDestroy(timingStartEvent);
      timingStartEvent = nullptr;
    }
    if (timingStopEvent && f_hipEventDestroy) {
      (void)f_hipEventDestroy(timingStopEvent);
      timingStopEvent = nullptr;
    }
    hipError_t e1 = f_hipEventCreate(&timingStartEvent);
    hipError_t e2 = f_hipEventCreate(&timingStopEvent);
    timingSupported = (e1 == hipSuccess && e2 == hipSuccess);
  }

  if (verbose) {
    std::cout << "Successfully selected HIP device " << index << ": "
              << devices[index].name << std::endl;
  }
}

DeviceInfo ROCmContext::getCurrentDeviceInfo() const {
  if (selectedDeviceIndex < 0 ||
      selectedDeviceIndex >= static_cast<int>(devices.size())) {
    throw std::runtime_error("No device selected. Call pickDevice() first.");
  }
  return devices[selectedDeviceIndex];
}

ComputeBuffer ROCmContext::createBuffer(size_t size, const void *host_ptr) {
  if (!available || selectedDeviceIndex < 0) {
    throw std::runtime_error("No device selected or ROCm not available.");
  }

  void *device_ptr;
  hipError_t err = f_hipMalloc(&device_ptr, size);
  if (err != hipSuccess) {
    if (verbose) {
      std::cerr << "hipMalloc error: " << f_hipGetErrorString(err) << std::endl;
    }
    throw std::runtime_error("Failed to allocate device memory: " +
                             std::string(f_hipGetErrorString(err)));
  }
  if (host_ptr) {
    err = f_hipMemcpy(device_ptr, host_ptr, size, hipMemcpyHostToDevice);
    if (err != hipSuccess) {
      if (verbose) {
        std::cerr << "hipMemcpy error: " << f_hipGetErrorString(err)
                  << std::endl;
      }
      (void)f_hipFree(device_ptr);
      throw std::runtime_error("Failed to copy data to device: " +
                               std::string(f_hipGetErrorString(err)));
    }
  }
  return device_ptr;
}

void ROCmContext::writeBuffer(ComputeBuffer buffer, size_t offset, size_t size,
                              const void *host_ptr) {
  void *device_ptr = static_cast<char *>(buffer) + offset;
  if (f_hipMemcpy(device_ptr, host_ptr, size, hipMemcpyHostToDevice) !=
      hipSuccess) {
    throw std::runtime_error("Failed to write to buffer");
  }
}

void ROCmContext::readBuffer(ComputeBuffer buffer, size_t offset, size_t size,
                             void *host_ptr) const {
  const void *device_ptr = static_cast<const char *>(buffer) + offset;
  if (f_hipMemcpy(host_ptr, device_ptr, size, hipMemcpyDeviceToHost) !=
      hipSuccess) {
    throw std::runtime_error("Failed to read from buffer");
  }
}

void ROCmContext::releaseBuffer(ComputeBuffer buffer) {
  if (buffer) {
    if (f_hipFree(buffer) != hipSuccess) {
      std::cerr << "hipFree failed" << std::endl;
    }
  }
}

ComputeKernel ROCmContext::createKernel(const std::string &file_name,
                                        const std::string &kernel_name,
                                        uint32_t num_args) {
  notifyKernelCreated(file_name);
  if (!available || selectedDeviceIndex < 0) {
    throw std::runtime_error("No device selected or ROCm not available.");
  }

  bool is_hip = false;
  if (file_name.size() > 4 &&
      file_name.substr(file_name.size() - 4) == ".hip") {
    is_hip = true;
  }

  hipModule_t module;
  if (modules.find(file_name) == modules.end()) {
    bool loaded_co_successfully = false;
    if (is_hip) {
      std::string co_file_name =
          file_name.substr(0, file_name.size() - 4) + ".co";
      hipError_t err = f_hipModuleLoad(&module, co_file_name.c_str());
      if (err == hipSuccess) {
        if (verbose) {
          std::cout << "Successfully loaded pre-compiled HIP module: "
                    << co_file_name << std::endl;
        }
        loaded_co_successfully = true;
        is_hip = false; // Bypass HIPRTC entirely
      } else {
        std::cout << "Failed to load pre-compiled .co file " << co_file_name
                  << " error: " << err << " (" << f_hipGetErrorString(err)
                  << ")" << std::endl;
      }
    }

#ifdef HAVE_HIPRTC
    if (is_hip && hiprtcLib && hiprtcLib->isValid() &&
        !loaded_co_successfully) {
      std::vector<char> code;
      if (utils::ShaderCache::loadROCmCache(
              file_name, devices[selectedDeviceIndex], code)) {
        if (verbose) {
          std::cout << "Loaded HIP kernel from cache: " << file_name
                    << std::endl;
        }
      } else {
        if (verbose) {
          std::cout << "Attempting to compile HIP source: " << file_name
                    << std::endl;
        }
        std::ifstream file(file_name);
        if (!file.is_open()) {
          throw std::runtime_error("Failed to open HIP source file: " +
                                   file_name);
        }
        std::string source((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());

        hiprtcProgram prog;
        f_hiprtcCreateProgram(&prog, source.c_str(), file_name.c_str(), 0,
                              nullptr, nullptr);

        std::string offload_arch =
            "--offload-arch=" + devices[selectedDeviceIndex].archName;
        const char *opts[] = {offload_arch.c_str(), "-O3",
                              "-I/usr/include", "-I/opt/rocm/include",
                              "-I/usr/local/include"};
        hiprtcResult compileResult = f_hiprtcCompileProgram(prog, 5, opts);

        if (compileResult != HIPRTC_SUCCESS) {
          std::cout << "HIPRTC compilation failed with code: " << compileResult
                    << std::endl;
          std::cout << "Source snippet: "
                    << source.substr(0, std::min(source.length(), (size_t)150))
                    << std::endl;
          size_t logSize = 0;
          hiprtcResult logSizeRes = f_hiprtcGetProgramLogSize(prog, &logSize);
          std::cout << "HIPRTC logSizeRes=" << logSizeRes
                    << " logSize=" << logSize << std::endl;
          std::string log_str = "No log available";
          if (logSize > 0) {
            std::vector<char> log(logSize);
            hiprtcResult logRes = f_hiprtcGetProgramLog(prog, log.data());
            std::cout << "HIPRTC get log res: " << logRes << std::endl;
            log_str = std::string(log.data(), logSize);
          }
          f_hiprtcDestroyProgram(&prog);
          throw std::runtime_error("Failed to compile HIP kernel " + file_name +
                                   ":\n" + log_str);
        }

        size_t codeSize;
        f_hiprtcGetCodeSize(prog, &codeSize);
        code.resize(codeSize);
        f_hiprtcGetCode(prog, code.data());
        f_hiprtcDestroyProgram(&prog);

        utils::ShaderCache::saveROCmCache(file_name,
                                          devices[selectedDeviceIndex], code);
      }

      hipError_t err = f_hipModuleLoadData(&module, code.data());
      if (err != hipSuccess) {
        throw std::runtime_error("Failed to load compiled HIP module: " +
                                 std::string(f_hipGetErrorString(err)));
      }
    } else
#endif
    {
      if (!loaded_co_successfully) {
        if (verbose) {
          std::cout << "Attempting to load HIP module from binary: "
                    << file_name << std::endl;
        }
        hipError_t err = f_hipModuleLoad(&module, file_name.c_str());
        if (err != hipSuccess) {
          if (verbose) {
            std::cerr << "hipModuleLoad error: " << f_hipGetErrorString(err)
                      << std::endl;
            std::cerr << "Module file: " << file_name << std::endl;
          }
          if (err == hipErrorNoBinaryForGpu) {
            throw std::runtime_error("Failed to load HIP module from " +
                                     file_name + ": No binary for GPU");
          }
          throw std::runtime_error("Failed to load HIP module from " +
                                   file_name + ": " +
                                   std::string(f_hipGetErrorString(err)));
        }
      }
    }
    modules[file_name] = module;
  } else {
    module = modules[file_name];
  }

  hipFunction_t function;
  hipError_t err =
      f_hipModuleGetFunction(&function, module, kernel_name.c_str());
  if (err != hipSuccess) {
    if (verbose) {
      std::cerr << "hipModuleGetFunction error: " << f_hipGetErrorString(err)
                << std::endl;
    }
    throw std::runtime_error("Failed to get HIP function " + kernel_name +
                             " from module " + file_name + ": " +
                             std::string(f_hipGetErrorString(err)));
  }

  ROCmKernel kernel_obj;
  kernel_obj.function = function;

  ComputeKernel kernel_handle = new ROCmKernel(kernel_obj);
  kernels[kernel_handle] = kernel_obj;

  if (verbose) {
    std::cout << "Successfully created kernel: " << kernel_name << std::endl;
  }

  return kernel_handle;
}

void ROCmContext::setKernelArg(ComputeKernel kernel, uint32_t arg_index,
                               ComputeBuffer buffer) {
  setKernelArg(kernel, arg_index, sizeof(void *), &buffer);
}

void ROCmContext::setKernelArg(ComputeKernel kernel, uint32_t arg_index,
                               size_t arg_size, const void *arg_value) {
  auto it = kernels.find(kernel);
  if (it == kernels.end()) {
    throw std::runtime_error("Invalid kernel handle");
  }

  auto &arg_vec = it->second.args[arg_index];
  arg_vec.resize(arg_size);
  std::memcpy(arg_vec.data(), arg_value, arg_size);
}

void ROCmContext::dispatch(ComputeKernel kernel, uint32_t grid_x,
                           uint32_t grid_y, uint32_t grid_z, uint32_t block_x,
                           uint32_t block_y, uint32_t block_z) {
  auto it = kernels.find(kernel);
  if (it == kernels.end()) {
    throw std::runtime_error("Invalid kernel handle");
  }

  std::vector<void *> arg_pointers;
  if (!it->second.args.empty()) {
    arg_pointers.resize(it->second.args.rbegin()->first + 1, nullptr);
    for (auto const &[key, val] : it->second.args) {
      arg_pointers[key] = (void *)val.data();
    }
  }

  lastDispatchedKernel = kernel;

  if (f_hipModuleLaunchKernel(it->second.function, grid_x, grid_y, grid_z,
                              block_x, block_y, block_z, 0, nullptr,
                              arg_pointers.data(), nullptr) != hipSuccess) {
    throw std::runtime_error("Failed to launch kernel");
  }
}

KernelResourceUsage ROCmContext::getKernelResourceUsage(ComputeKernel kernel) const {
  KernelResourceUsage usage;
  if (!kernel || !available || !f_hipFuncGetAttribute) {
    return usage;
  }
  auto it = kernels.find(kernel);
  if (it == kernels.end() || !it->second.function) {
    return usage;
  }

  usage.available = true;
  usage.compilerNotes = "ROCm LLVM Backend";

  int vgprs = 0;
  if (f_hipFuncGetAttribute(&vgprs, HIP_FUNC_ATTRIBUTE_NUM_REGS, it->second.function) == hipSuccess) {
    usage.vgprCount = static_cast<uint32_t>(vgprs);
  }

  int lds = 0;
  if (f_hipFuncGetAttribute(&lds, HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, it->second.function) == hipSuccess) {
    usage.ldsSizeBytes = static_cast<uint32_t>(lds);
  }

  int scratch = 0;
  if (f_hipFuncGetAttribute(&scratch, HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, it->second.function) == hipSuccess) {
    usage.scratchSizeBytes = static_cast<uint32_t>(scratch);
  }

  if (usage.vgprCount > 0) {
    // AMD RDNA architecture SIMD32 budget (1536 physical VGPRs per SIMD32, max 16 waves)
    usage.maxWavesPerSimd = std::min(16u, 1536u / usage.vgprCount);
  }

  return usage;
}

void ROCmContext::releaseKernel(ComputeKernel kernel) {
  auto it = kernels.find(kernel);
  if (it != kernels.end()) {
    delete static_cast<ROCmKernel *>(kernel);
    kernels.erase(it);
  }
}

void ROCmContext::waitIdle() {
  if (available && f_hipDeviceSynchronize() != hipSuccess) {
    throw std::runtime_error("hipDeviceSynchronize failed");
  }
}

void ROCmContext::setExpectedKernelCount(uint32_t count) {
  expectedKernelCount = count;
  createdKernelCount = 0;
  if (verbose && count > 0) {
    std::cout << "Starting setup for " << count << " kernels..." << std::endl;
#ifdef HAVE_HIPRTC
    std::cout << "Using compiler: hiprtc (ROCm)" << std::endl;
#endif
  }
}

void ROCmContext::notifyKernelCreated(const std::string &file_name) {
  createdKernelCount++;
  if (!quiet && !verbose && expectedKernelCount > 0) {
    printProgressBar(createdKernelCount, expectedKernelCount, file_name);
  }
}

void ROCmContext::printProgressBar(uint32_t current, uint32_t total,
                                   const std::string &kernel_name) {
  const int barWidth = 30;
  float progress = static_cast<float>(current) / total;
  int pos = static_cast<int>(barWidth * progress);

  std::string short_name = kernel_name;
  size_t last_slash = kernel_name.find_last_of("/\\");
  if (last_slash != std::string::npos) {
    short_name = kernel_name.substr(last_slash + 1);
  }

  std::cout << "\r\033[K[";
  for (int i = 0; i < barWidth; ++i) {
    if (i < pos)
      std::cout << "#";
    else if (i == pos)
      std::cout << ">";
    else
      std::cout << " ";
  }
  std::cout << "] " << int(progress * 100.0) << "% Compiling " << short_name
            << (current == total ? "\n" : "") << std::flush;
}

void ROCmContext::startTiming() {
  if (!timingSupported || !timingStartEvent || !f_hipEventRecord) {
    return;
  }
  waitIdle();
  hipError_t err = f_hipEventRecord(timingStartEvent, 0);
  isTimingActive = (err == hipSuccess);
}

double ROCmContext::stopTiming() {
  if (!isTimingActive || !timingSupported || !timingStopEvent || !f_hipEventRecord ||
      !f_hipEventElapsedTime) {
    isTimingActive = false;
    return 0.0;
  }
  isTimingActive = false;
  hipError_t err = f_hipEventRecord(timingStopEvent, 0);
  if (err != hipSuccess) {
    return 0.0;
  }
  if (f_hipEventSynchronize) {
    (void)f_hipEventSynchronize(timingStopEvent);
  } else {
    waitIdle();
  }
  float elapsed_ms = 0.0f;
  err = f_hipEventElapsedTime(&elapsed_ms, timingStartEvent, timingStopEvent);
  if (err != hipSuccess || elapsed_ms < 0.0f) {
    return 0.0;
  }
  return static_cast<double>(elapsed_ms);
}

bool ROCmContext::hasGpuTiming() const {
  return available && timingSupported && (f_hipEventRecord != nullptr) &&
         (f_hipEventElapsedTime != nullptr);
}

