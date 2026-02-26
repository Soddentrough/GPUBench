import glob, re, os

# Files to patch
hip_files = [f for f in glob.glob("hip_kernels/cache*.hip") if "cache_bw_robust" not in f]
ocl_files = [f for f in glob.glob("kernels/opencl/cache*.cl") if "cache_bw_robust" not in f]
all_files = hip_files + ocl_files

extraction_hip = """    uint stride = pc[0];
    uint mask = pc[1];
    uint iterations = pc[2];
"""

extraction_ocl = """    uint stride = pc[0];
    uint mask = pc[1];
    uint iterations = pc[2];
"""

for fpath in all_files:
    with open(fpath, 'r') as f:
        content = f.read()

    # Determine if OpenCL or HIP
    is_ocl = ".cl" in fpath
    pc_type = "__global uint* pc" if is_ocl else "uint* pc"

    # Remove the struct PushConstants block if it exists (for HIP and OpenCL)
    content = re.sub(r'struct\s+PushConstants\s*\{[^}]+\};', '', content)

    # Replace the run_benchmark signature.
    # We want to replace `struct PushConstants pc` or `PushConstants pc` or `uint stride, uint mask, uint iterations`
    # Let's find the signature.
    sig_match = re.search(r'(__global__ void __kernel void|__kernel void|extern "C" __global__ void) run_benchmark\(([^)]+)\)', content)
    if not sig_match:
        print(f"Could not find signature in {fpath}")
        continue

    sig = sig_match.group(2)
    # The first arg is the data pointer (e.g. float4* data). The remaining are pc.
    args = sig.split(',')
    new_sig = f"{args[0].strip()}, {pc_type}"
    
    content = content[:sig_match.start(2)] + new_sig + content[sig_match.end(2):]

    # Insert extraction immediately after the `{` of run_benchmark
    body_start_match = re.search(r'run_benchmark[^{]+\{', content)
    if body_start_match:
        pos = body_start_match.end()
        content = content[:pos] + "\n" + (extraction_ocl if is_ocl else extraction_hip) + content[pos:]

    with open(fpath, 'w') as f:
        f.write(content)
    print(f"Patched {fpath}")
