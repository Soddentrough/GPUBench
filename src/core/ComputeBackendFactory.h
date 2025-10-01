#pragma once

#include "IComputeContext.h"
#include "ComputeBackend.h"
#include <memory>
#include <stdexcept>

#ifdef HAVE_VULKAN
#include "VulkanContext.h"
#endif

#ifdef HAVE_OPENCL
#include "OpenCLContext.h"
#endif

#ifdef HAVE_ROCM
#include "ROCmContext.h"
#endif

class ComputeBackendFactory {
public:
    // Try to create the specified backend, throw if not available
    static std::unique_ptr<IComputeContext> create(ComputeBackend backend, bool verbose = false) {
        switch (backend) {
            case ComputeBackend::Vulkan:
#ifdef HAVE_VULKAN
                return std::make_unique<VulkanContext>();
#else
                throw std::runtime_error("Vulkan backend not available (not compiled with HAVE_VULKAN)");
#endif
            
            case ComputeBackend::OpenCL:
#ifdef HAVE_OPENCL
                return std::make_unique<OpenCLContext>();
#else
                throw std::runtime_error("OpenCL backend not available (not compiled with HAVE_OPENCL)");
#endif

            case ComputeBackend::ROCm:
#ifdef HAVE_ROCM
                return std::make_unique<ROCmContext>(verbose);
#else
                throw std::runtime_error("ROCm backend not available (not compiled with HAVE_ROCM)");
#endif
            
            default:
                throw std::runtime_error("Unknown backend");
        }
    }
    
    // Try to create a backend with automatic fallback
    // Priority: Vulkan > ROCm > OpenCL
    static std::unique_ptr<IComputeContext> createWithFallback() {
#ifdef HAVE_VULKAN
        try {
            return std::make_unique<VulkanContext>();
        } catch (const std::exception& e) {
            // Vulkan failed, fall through
        }
#endif

#ifdef HAVE_ROCM
        try {
            return std::make_unique<ROCmContext>();
        } catch (const std::exception& e) {
            // ROCm failed, fall through
        }
#endif

#ifdef HAVE_OPENCL
        try {
            return std::make_unique<OpenCLContext>();
        } catch (const std::exception& e) {
            // OpenCL failed, fall through
        }
#endif

        throw std::runtime_error("No compute backend available or all failed to initialize");
    }
    
    // Check if a backend is available
    static bool isAvailable(ComputeBackend backend) {
        switch (backend) {
            case ComputeBackend::Vulkan:
#ifdef HAVE_VULKAN
                return true;
#else
                return false;
#endif
            
            case ComputeBackend::OpenCL:
#ifdef HAVE_OPENCL
                return true;
#else
                return false;
#endif

            case ComputeBackend::ROCm:
#ifdef HAVE_ROCM
                return true;
#else
                return false;
#endif
            
            default:
                return false;
        }
    }
    
    // Get the name of a backend
    static const char* getBackendName(ComputeBackend backend) {
        switch (backend) {
            case ComputeBackend::Vulkan: return "Vulkan";
            case ComputeBackend::OpenCL: return "OpenCL";
            case ComputeBackend::ROCm: return "ROCm";
            default: return "Unknown";
        }
    }
};
