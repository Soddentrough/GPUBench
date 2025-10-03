#include "OpenCLContext.h"
#include <iostream>
#include <vector>
#include <stdexcept>
#include <fstream>
#include <sstream>

void OpenCLContext::waitIdle() {
    clFinish(commandQueue);
}

OpenCLContext::OpenCLContext() {
    try {
        enumeratePlatformsAndDevices();
    } catch (const std::exception& e) {
        std::cerr << "OpenCL initialization failed: " << e.what() << std::endl;
        throw;
    }
}

OpenCLContext::~OpenCLContext() {
    if (commandQueue) {
        clReleaseCommandQueue(commandQueue);
    }
    if (context) {
        clReleaseContext(context);
    }
}

void OpenCLContext::enumeratePlatformsAndDevices() {
    cl_uint platformCount = 0;
    cl_int err = clGetPlatformIDs(0, nullptr, &platformCount);
    if (err != CL_SUCCESS || platformCount == 0) {
        throw std::runtime_error("Failed to find OpenCL platforms");
    }

    std::vector<cl_platform_id> platforms(platformCount);
    clGetPlatformIDs(platformCount, platforms.data(), nullptr);
    
    // Try to find a platform with GPU devices
    for (const auto& p : platforms) {
        cl_uint deviceCount = 0;
        err = clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &deviceCount);
        if (err == CL_SUCCESS && deviceCount > 0) {
            platform = p;
            devices.resize(deviceCount);
            clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, deviceCount, devices.data(), nullptr);
            break;
        }
    }
    
    // Fallback to any device type if no GPU found
    if (devices.empty()) {
        for (const auto& p : platforms) {
            cl_uint deviceCount = 0;
            err = clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceCount);
            if (err == CL_SUCCESS && deviceCount > 0) {
                platform = p;
                devices.resize(deviceCount);
                clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, deviceCount, devices.data(), nullptr);
                break;
            }
        }
    }
    
    if (devices.empty()) {
        throw std::runtime_error("Failed to find OpenCL devices");
    }
}

const std::vector<DeviceInfo>& OpenCLContext::getDevices() const {
    if (deviceInfos.empty()) {
        for (const auto& dev : devices) {
            DeviceInfo info;
            
            char name[256];
            clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name), name, nullptr);
            info.name = name;
            
            cl_ulong memSize;
            clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(memSize), &memSize, nullptr);
            info.memorySize = memSize;
            
            cl_uint computeUnits;
            clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(computeUnits), &computeUnits, nullptr);
            info.computeUnits = computeUnits;
            
            size_t maxWorkGroupSize;
            clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, nullptr);
            info.maxWorkGroupSize = static_cast<uint32_t>(maxWorkGroupSize);
            
            deviceInfos.push_back(info);
        }
    }
    return deviceInfos;
}

void OpenCLContext::pickDevice(uint32_t index) {
    if (index >= devices.size()) {
        throw std::runtime_error("Invalid device index");
    }
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
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, nullptr);
    info.name = name;
    
    cl_ulong memSize;
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(memSize), &memSize, nullptr);
    info.memorySize = memSize;
    
    cl_uint computeUnits;
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(computeUnits), &computeUnits, nullptr);
    info.computeUnits = computeUnits;
    
    size_t maxWorkGroupSize;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, nullptr);
    info.maxWorkGroupSize = static_cast<uint32_t>(maxWorkGroupSize);
    
    return info;
}

void OpenCLContext::createContext() {
    cl_int err;
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create OpenCL context");
    }
}

void OpenCLContext::createCommandQueue() {
    cl_int err;
    commandQueue = clCreateCommandQueueWithProperties(context, device, 0, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create OpenCL command queue");
    }
}

ComputeBuffer OpenCLContext::createBuffer(size_t size, const void* host_ptr) {
    if (size == 0) {
        throw std::runtime_error("Cannot create OpenCL buffer with size 0");
    }
    
    cl_int err;
    cl_mem_flags flags = CL_MEM_READ_WRITE;
    if (host_ptr) {
        flags |= CL_MEM_COPY_HOST_PTR;
    }
    cl_mem buffer = clCreateBuffer(context, flags, size, const_cast<void*>(host_ptr), &err);
    if (err != CL_SUCCESS) {
        std::string error_msg = "Failed to create OpenCL buffer of size " + std::to_string(size) + " bytes";
        if (err == CL_MEM_OBJECT_ALLOCATION_FAILURE) {
            error_msg += " (allocation failure - out of memory)";
        } else if (err == CL_INVALID_BUFFER_SIZE) {
            error_msg += " (invalid buffer size)";
        }
        throw std::runtime_error(error_msg);
    }
    return new ComputeBuffer_cl{buffer};
}

void OpenCLContext::writeBuffer(ComputeBuffer buffer, size_t offset, size_t size, const void* host_ptr) {
    auto* buffer_cl = static_cast<ComputeBuffer_cl*>(buffer);
    cl_int err = clEnqueueWriteBuffer(commandQueue, buffer_cl->buffer, CL_TRUE, offset, size, host_ptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to write to OpenCL buffer");
    }
}

void OpenCLContext::readBuffer(ComputeBuffer buffer, size_t offset, size_t size, void* host_ptr) const {
    const auto* buffer_cl = static_cast<const ComputeBuffer_cl*>(buffer);
    cl_int err = clEnqueueReadBuffer(commandQueue, buffer_cl->buffer, CL_TRUE, offset, size, host_ptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to read from OpenCL buffer");
    }
}

void OpenCLContext::releaseBuffer(ComputeBuffer buffer) {
    if (buffer) {
        auto* buffer_cl = static_cast<ComputeBuffer_cl*>(buffer);
        clReleaseMemObject(buffer_cl->buffer);
        delete buffer_cl;
    }
}

ComputeKernel OpenCLContext::createKernel(const std::string& file_name, const std::string& kernel_name, uint32_t num_args) {
    std::ifstream file(file_name);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open kernel file: " + file_name);
    }
    std::string source((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    const char* source_ptr = source.c_str();
    size_t source_size = source.length();

    cl_int err;
    cl_program program = clCreateProgramWithSource(context, 1, &source_ptr, &source_size, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create OpenCL program");
    }

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::string log_str(log.begin(), log.end());
        throw std::runtime_error("Failed to build OpenCL program: " + log_str);
    }

    cl_kernel kernel = clCreateKernel(program, kernel_name.c_str(), &err);
    if (err != CL_SUCCESS) {
        std::string error_msg = "Failed to create OpenCL kernel '" + kernel_name + "' from file '" + file_name + "'";
        if (err == CL_INVALID_KERNEL_NAME) {
            error_msg += " (invalid kernel name - kernel not found in program)";
        } else if (err == CL_INVALID_PROGRAM) {
            error_msg += " (invalid program)";
        }
        clReleaseProgram(program);
        throw std::runtime_error(error_msg);
    }

    return new ComputeKernel_cl{program, kernel};
}

void OpenCLContext::setKernelArg(ComputeKernel kernel, uint32_t arg_index, ComputeBuffer buffer) {
    auto* kernel_cl = static_cast<ComputeKernel_cl*>(kernel);
    auto* buffer_cl = static_cast<ComputeBuffer_cl*>(buffer);
    cl_int err = clSetKernelArg(kernel_cl->kernel, arg_index, sizeof(cl_mem), &buffer_cl->buffer);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to set OpenCL kernel buffer argument");
    }
}

void OpenCLContext::setKernelArg(ComputeKernel kernel, uint32_t arg_index, size_t arg_size, const void* arg_value) {
    auto* kernel_cl = static_cast<ComputeKernel_cl*>(kernel);
    cl_int err = clSetKernelArg(kernel_cl->kernel, arg_index, arg_size, arg_value);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to set OpenCL kernel value argument");
    }
}

void OpenCLContext::dispatch(ComputeKernel kernel, uint32_t grid_x, uint32_t grid_y, uint32_t grid_z, uint32_t block_x, uint32_t block_y, uint32_t block_z) {
    auto* kernel_cl = static_cast<ComputeKernel_cl*>(kernel);
    size_t global_work_size[3] = { (size_t)grid_x * block_x, (size_t)grid_y * block_y, (size_t)grid_z * block_z };
    size_t local_work_size[3] = { (size_t)block_x, (size_t)block_y, (size_t)block_z };
    cl_int err = clEnqueueNDRangeKernel(commandQueue, kernel_cl->kernel, 3, nullptr, global_work_size, local_work_size, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::string error_msg = "Failed to dispatch OpenCL kernel with workgroup size [" + 
                               std::to_string(block_x) + "," + std::to_string(block_y) + "," + std::to_string(block_z) + "]";
        if (err == CL_INVALID_WORK_GROUP_SIZE) {
            error_msg += " (work group size exceeds device limit)";
        } else if (err == CL_INVALID_WORK_ITEM_SIZE) {
            error_msg += " (work item size exceeds device limit)";
        }
        throw std::runtime_error(error_msg);
    }
}

void OpenCLContext::releaseKernel(ComputeKernel kernel) {
    if (kernel) {
        auto* kernel_cl = static_cast<ComputeKernel_cl*>(kernel);
        clReleaseKernel(kernel_cl->kernel);
        clReleaseProgram(kernel_cl->program);
        delete kernel_cl;
    }
}
