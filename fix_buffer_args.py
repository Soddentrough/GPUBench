import glob
import re

# 1. Update CacheBench.cpp to use pcBuffer for OpenCL/ROCm
with open('src/benchmarks/CacheBench.cpp', 'r') as f:
    bench = f.read()

# Replace the parameter setting block
old_block_regex = r'if \(context\.getBackend\(\) == ComputeBackend::Vulkan\) \{.*?\} else \{.*?context\.setKernelArg\(kernel, 3, sizeof\(uint32_t\), &iterations\);\s*\}'
new_block = """struct PushConstants {
      uint32_t stride;
      uint32_t mask;
      uint32_t iterations;
      uint32_t padding;
    } pc = {stride, mask, iterations, 0};

    if (context.getBackend() == ComputeBackend::Vulkan) {
      context.setKernelArg(kernel, 1, sizeof(pc), &pc);
    } else {
      if (!pcBuffer) {
        pcBuffer = context.createBuffer(sizeof(PushConstants));
      }
      context.writeBuffer(pcBuffer, 0, sizeof(PushConstants), &pc);
      context.setKernelArg(kernel, 1, pcBuffer);
    }"""

bench = re.sub(old_block_regex, new_block, bench, flags=re.DOTALL)

with open('src/benchmarks/CacheBench.cpp', 'w') as f:
    f.write(bench)


# 2. Update OpenCL kernels
for fpath in glob.glob("kernels/opencl/*.cl"):
    with open(fpath, 'r') as f:
        content = f.read()
    
    # Change signature
    content = re.sub(r'__kernel void run_benchmark\(([^,]+)[^)]+\)', r'__kernel void run_benchmark(\1, __global uint* pc)', content)
    
    # Add variable extraction
    extraction = """
    uint stride = pc[0];
    uint mask = pc[1];
    uint iterations = pc[2];
"""
    # Insert extraction at start of kernel body
    content = re.sub(r'\{', r'{' + extraction, content, count=1)
    
    with open(fpath, 'w') as f:
        f.write(content)

# 3. Update HIP kernels
for fpath in glob.glob("hip_kernels/*.hip"):
    with open(fpath, 'r') as f:
        content = f.read()
        
    # Change signature
    content = re.sub(r'run_benchmark\(([^,]+)[^)]+\)', r'run_benchmark(\1, uint* pc)', content)

    # Add variable extraction
    extraction = """
    uint stride = pc[0];
    uint mask = pc[1];
    uint iterations = pc[2];
"""
    # Insert extraction at start of kernel body
    content = re.sub(r'\{', r'{' + extraction, content, count=1)
    
    with open(fpath, 'w') as f:
        f.write(content)

print("Applied Buffer-based Argument Passing Fixes")
