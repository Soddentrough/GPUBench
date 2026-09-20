import glob

pc_struct = """typedef struct {
    uint stride;
    uint mask;
    uint iterations;
    uint padding;
} PushConstants;

"""

for f in glob.glob("kernels/opencl/*.cl"):
    with open(f, 'r') as file:
        content = file.read()
    
    if content.find("PushConstants pc") != -1: continue
    
    content = pc_struct + content
    content = content.replace("uint4 pc", "PushConstants pc")
    
    content = content.replace("pc.x", "pc.stride")
    content = content.replace("pc.y", "pc.mask")
    content = content.replace("pc.z", "pc.iterations")

    # If it was still completely without pc, fix that too:
    if "run_benchmark(__global uint* data)" in content:
        content = content.replace("run_benchmark(__global uint* data)", "run_benchmark(__global uint* data, PushConstants pc)")
    if "run_benchmark(__global float4* data)" in content:
        content = content.replace("run_benchmark(__global float4* data)", "run_benchmark(__global float4* data, PushConstants pc)")

    with open(f, 'w') as file:
        file.write(content)
