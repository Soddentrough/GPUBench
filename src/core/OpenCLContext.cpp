#include "OpenCLContext.h"
#include <iostream>
#include <vector>
#include <stdexcept>

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
