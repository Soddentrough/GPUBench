import glob
import re

# 1. Fix CacheBench.cpp to pass individual arguments for OpenCL and ROCm
with open('src/benchmarks/CacheBench.cpp', 'r') as f:
    cache_bench = f.read()

replacement = """    if (context.getBackend() == ComputeBackend::Vulkan) {
      struct PushConstants {
        uint32_t stride;
        uint32_t mask;
        uint32_t iterations;
        uint32_t padding;
      } pc = {stride, mask, iterations, 0};
      context.setKernelArg(kernel, 1, sizeof(pc), &pc);
    } else {
      context.setKernelArg(kernel, 1, sizeof(uint32_t), &stride);
      context.setKernelArg(kernel, 2, sizeof(uint32_t), &mask);
      context.setKernelArg(kernel, 3, sizeof(uint32_t), &iterations);
    }"""

cache_bench = re.sub(r'struct PushConstants \{.*?\} pc = \{.*?\};\s*context\.setKernelArg\(kernel, 1, sizeof\(pc\), &pc\);', replacement, cache_bench, flags=re.DOTALL)

with open('src/benchmarks/CacheBench.cpp', 'w') as f:
    f.write(cache_bench)

# 2. Fix OpenCL and HIP kernels to accept uint stride, uint mask, uint iterations
for filepath in glob.glob("kernels/opencl/cachebw*.cl") + glob.glob("kernels/opencl/cache_latency*.cl") + \
                glob.glob("hip_kernels/cachebw*.hip") + glob.glob("hip_kernels/cache_latency*.hip") + \
                glob.glob("hip_kernels/cache_bw_robust*.hip") + glob.glob("kernels/opencl/cache_bw_robust*.cl"):
    with open(filepath, 'r') as f:
        content = f.read()

    # OpenCL
    content = re.sub(r'typedef struct \{.*?\} PushConstants;\s*', '', content, flags=re.DOTALL)
    content = content.replace("PushConstants pc", "uint stride, uint mask, uint iterations")
    
    # ROCm HIP
    content = re.sub(r'struct PushConstants \{.*?uint32_t padding;\n\};\s*', '', content, flags=re.DOTALL)
    content = content.replace("uint32_t stride, uint32_t mask, uint32_t iterations", "uint stride, uint mask, uint iterations")
    
    content = content.replace("pc.stride", "stride")
    content = content.replace("pc.mask", "mask")
    content = content.replace("pc.iterations", "iterations")

    with open(filepath, 'w') as f:
        f.write(content)

print("Applied CacheBench argument fixes")
