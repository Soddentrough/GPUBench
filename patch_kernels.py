import glob
import os
import re

o_struct = """typedef struct {
    uint stride;
    uint mask;
    uint iterations;
} PushConstants;

"""
h_struct = """struct PushConstants {
    uint32_t stride;
    uint32_t mask;
    uint32_t iterations;
};

"""

def patch_cl(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    if 'PushConstants' in content: return
    content = o_struct + content
    content = re.sub(r'__kernel void run_benchmark\((__global [^*]+\* data(?:, [^)]*)?)\)', r'__kernel void run_benchmark(\1, PushConstants pc)', content)
    # replace parameter usages of mask -> pc.mask
    # In some: (..., uint mask) was there? 
    # Let me check if there was a uint mask argument. In OpenCL, `cachebw` had NO mask arguments in my previous read of `cachebw_l1.cl`!
    # Wait, `cachebw_l1.cl` didn't have mask argument? No, wait: baseIndex = workgroupOffset + (localId % 2);
    content = re.sub(r'(baseIndex .*?);', r'\1 & pc.mask;', content)
    
    with open(filepath, 'w') as f:
        f.write(content)

def patch_hip(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    if 'PushConstants' in content: return
    content = content.replace('#include <hip/hip_runtime.h>\n', '#include <hip/hip_runtime.h>\n\n' + h_struct)
    content = re.sub(r'extern "C" __global__ void run_benchmark\(([^,]+)(, uint32_t mask)?\)', r'extern "C" __global__ void run_benchmark(\1, PushConstants pc)', content)
    content = content.replace('& mask', '& pc.mask')
    with open(filepath, 'w') as f:
        f.write(content)

for f in glob.glob("kernels/opencl/cachebw*.cl") + glob.glob("kernels/opencl/cache_latency.cl") + glob.glob("kernels/opencl/cache_bandwidth.cl"):
    patch_cl(f)

for f in glob.glob("hip_kernels/cachebw*.hip") + glob.glob("hip_kernels/cache_latency.hip") + glob.glob("hip_kernels/cache_bandwidth.hip"):
    patch_hip(f)

