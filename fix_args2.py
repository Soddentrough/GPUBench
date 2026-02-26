import glob
import re

# 1. Fix OpenCL and HIP kernels to accept uint stride, uint mask, uint iterations
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

    # Force variables usage for DCE
    if "cache_latency" in filepath:
        if "uint i = 0; i < 1024" in content or "int i = 0; i < 1024" in content or "int i = 0; i < 1000000" in content:
            content = re.sub(r'(unroll\n\s*)?for\s*\([^;]+;\s*i\s*<\s*\d+;\s*\+\+i\)', r'\1for (uint i = 0; i < iterations; i++)', content)
        if "stride == 0xFFFFFFFF" not in content:
            if "if (val > 0)" in content:
                content = content.replace("if (val > 0)", "if (stride == 0xFFFFFFFF) { data[1] = mask; }\n    if (val > 0)")
            elif "data[0] = index;" in content:
                content = content.replace("data[0] = index;", "if (stride == 0xFFFFFFFF) { data[1] = mask; }\n    data[0] = index;")
            elif "buffer[0] = val;" in content:
                content = content.replace("buffer[0] = val;", "if (stride == 0xFFFFFFFF) { buffer[1] = mask; }\n    buffer[0] = val;")
    elif "cache_bw_robust" in filepath:
        if "if (final_sum.x < -1e30f)" in content:
            content = content.replace("if (final_sum.x < -1e30f)", "if (stride == 0xFFFFFFFF)")
        if "if (final_x < -1e30f)" in content:
            content = content.replace("if (final_x < -1e30f)", "if (stride == 0xFFFFFFFF)")

    with open(filepath, 'w') as f:
        f.write(content)

print("Applied CacheBench arg splits and DCE fixes to kernels")
