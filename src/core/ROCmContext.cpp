#include "ROCmContext.h"
#include <iostream>
#include <stdexcept>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <cstring>

ROCmContext::ROCmContext(bool verbose) : device(-1), selectedDeviceIndex(-1), verbose(verbose) {
    if (hipInit(0) != hipSuccess) {
        throw std::runtime_error("Failed to initialize HIP");
    }
    enumerateDevices();
}

ROCmContext::~ROCmContext() = default;

void ROCmContext::enumerateDevices() {
    int deviceCount = 0;
    if (hipGetDeviceCount(&deviceCount) != hipSuccess) {
        throw std::runtime_error("Failed to get HIP device count");
    }

    for (int i = 0; i < deviceCount; ++i) {
        hipDeviceProp_t prop;
        if (hipGetDeviceProperties(&prop, i) == hipSuccess) {
            DeviceInfo info;
            info.name = prop.name;
            info.memorySize = prop.totalGlobalMem;
            info.verbose = verbose;
            info.maxWorkGroupSize = prop.maxThreadsPerBlock;
            info.maxComputeWorkGroupCountX = prop.maxGridSize[0];
            info.maxComputeWorkGroupCountY = prop.maxGridSize[1];
            info.maxComputeWorkGroupCountZ = prop.maxGridSize[2];
            info.maxComputeSharedMemorySize = prop.sharedMemPerBlock;
            info.subgroupSize = prop.warpSize;
            devices.push_back(info);
        }
    }
}

void ROCmContext::pickDevice(uint32_t index) {
    if (index >= devices.size()) {
        throw std::runtime_error("Invalid device index");
    }

    hipError_t err = hipSetDevice(index);
    if (err != hipSuccess) {
        if (verbose) {
            std::cerr << "hipSetDevice error: " << hipGetErrorString(err) << std::endl;
        }
        throw std::runtime_error("Failed to set HIP device: " + std::string(hipGetErrorString(err)));
    }

    device = index;
    selectedDeviceIndex = index;
    
    if (verbose) {
        std::cout << "Successfully selected HIP device " << index << ": " << devices[index].name << std::endl;
    }
}

DeviceInfo ROCmContext::getCurrentDeviceInfo() const {
    if (selectedDeviceIndex < 0 || selectedDeviceIndex >= static_cast<int>(devices.size())) {
        throw std::runtime_error("No device selected. Call pickDevice() first.");
    }
    return devices[selectedDeviceIndex];
}

ComputeBuffer ROCmContext::createBuffer(size_t size, const void* host_ptr) {
    if (selectedDeviceIndex < 0) {
        throw std::runtime_error("No device selected. Call pickDevice() before creating buffers.");
    }
    
    void* device_ptr;
    hipError_t err = hipMalloc(&device_ptr, size);
    if (err != hipSuccess) {
        if (verbose) {
            std::cerr << "hipMalloc error: " << hipGetErrorString(err) << std::endl;
        }
        throw std::runtime_error("Failed to allocate device memory: " + std::string(hipGetErrorString(err)));
    }
    if (host_ptr) {
        err = hipMemcpy(device_ptr, host_ptr, size, hipMemcpyHostToDevice);
        if (err != hipSuccess) {
            if (verbose) {
                std::cerr << "hipMemcpy error: " << hipGetErrorString(err) << std::endl;
            }
            (void)hipFree(device_ptr);
            throw std::runtime_error("Failed to copy data to device: " + std::string(hipGetErrorString(err)));
        }
    }
    return device_ptr;
}

void ROCmContext::writeBuffer(ComputeBuffer buffer, size_t offset, size_t size, const void* host_ptr) {
    void* device_ptr = static_cast<char*>(buffer) + offset;
    if (hipMemcpy(device_ptr, host_ptr, size, hipMemcpyHostToDevice) != hipSuccess) {
        throw std::runtime_error("Failed to write to buffer");
    }
}

void ROCmContext::readBuffer(ComputeBuffer buffer, size_t offset, size_t size, void* host_ptr) const {
    const void* device_ptr = static_cast<const char*>(buffer) + offset;
    if (hipMemcpy(host_ptr, device_ptr, size, hipMemcpyDeviceToHost) != hipSuccess) {
        throw std::runtime_error("Failed to read from buffer");
    }
}

void ROCmContext::releaseBuffer(ComputeBuffer buffer) {
    if (buffer) {
        if (hipFree(buffer) != hipSuccess) {
            std::cerr << "hipFree failed" << std::endl;
        }
    }
}

ComputeKernel ROCmContext::createKernel(const std::string& file_name, const std::string& kernel_name, uint32_t num_args) {
    if (selectedDeviceIndex < 0) {
        throw std::runtime_error("No device selected. Call pickDevice() before creating kernels.");
    }
    
    if (verbose) {
        std::cout << "Attempting to load HIP module from: " << file_name << std::endl;
    }
    hipModule_t module;
    if (modules.find(file_name) == modules.end()) {
        hipError_t err = hipModuleLoad(&module, file_name.c_str());
        if (err != hipSuccess) {
            if (verbose) {
                std::cerr << "hipModuleLoad error: " << hipGetErrorString(err) << std::endl;
                std::cerr << "Module file: " << file_name << std::endl;
            }
            if (err == hipErrorNoBinaryForGpu) {
                throw std::runtime_error("Failed to load HIP module from " + file_name + ": No binary for GPU");
            }
            throw std::runtime_error("Failed to load HIP module from " + file_name + ": " + std::string(hipGetErrorString(err)));
        }
        modules[file_name] = module;
        if (verbose) {
            std::cout << "Successfully loaded HIP module: " << file_name << std::endl;
        }
    } else {
        module = modules[file_name];
        if (verbose) {
            std::cout << "Using cached HIP module: " << file_name << std::endl;
        }
    }

    hipFunction_t function;
    hipError_t err = hipModuleGetFunction(&function, module, kernel_name.c_str());
    if (err != hipSuccess) {
        if (verbose) {
            std::cerr << "hipModuleGetFunction error: " << hipGetErrorString(err) << std::endl;
        }
        throw std::runtime_error("Failed to get HIP function " + kernel_name + " from module " + file_name + ": " + std::string(hipGetErrorString(err)));
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

void ROCmContext::setKernelArg(ComputeKernel kernel, uint32_t arg_index, ComputeBuffer buffer) {
    // For buffers, the argument is the device pointer itself.
    setKernelArg(kernel, arg_index, sizeof(void*), &buffer);
}

void ROCmContext::setKernelArg(ComputeKernel kernel, uint32_t arg_index, size_t arg_size, const void* arg_value) {
    auto it = kernels.find(kernel);
    if (it == kernels.end()) {
        throw std::runtime_error("Invalid kernel handle");
    }

    auto& arg_vec = it->second.args[arg_index];
    arg_vec.resize(arg_size);
    std::memcpy(arg_vec.data(), arg_value, arg_size);
}

void ROCmContext::dispatch(ComputeKernel kernel, uint32_t grid_x, uint32_t grid_y, uint32_t grid_z, uint32_t block_x, uint32_t block_y, uint32_t block_z) {
    auto it = kernels.find(kernel);
    if (it == kernels.end()) {
        throw std::runtime_error("Invalid kernel handle");
    }

    std::vector<void*> arg_pointers;
    if (!it->second.args.empty()) {
        arg_pointers.resize(it->second.args.rbegin()->first + 1, nullptr);
        for (auto const& [key, val] : it->second.args) {
            arg_pointers[key] = (void*)val.data();
        }
    }

    if (hipModuleLaunchKernel(it->second.function,
                              grid_x, grid_y, grid_z,
                              block_x, block_y, block_z,
                              0, nullptr,
                              arg_pointers.data(), nullptr) != hipSuccess) {
        throw std::runtime_error("Failed to launch kernel");
    }
}

void ROCmContext::releaseKernel(ComputeKernel kernel) {
    auto it = kernels.find(kernel);
    if (it != kernels.end()) {
        delete static_cast<ROCmKernel*>(kernel);
        kernels.erase(it);
    }
}

void ROCmContext::waitIdle() {
    if (hipDeviceSynchronize() != hipSuccess) {
        throw std::runtime_error("hipDeviceSynchronize failed");
    }
}
