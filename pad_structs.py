import glob
import re

for filepath in glob.glob("kernels/opencl/*.cl") + glob.glob("hip_kernels/*.hip") + glob.glob("kernels/vulkan/*.comp"):
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Add padding to Vulkan
    content = re.sub(r'uint iterations;\n\} pushConstants;', r'uint iterations;\n    uint padding;\n} pushConstants;', content)
    
    # Add padding to OpenCL/HIP
    content = re.sub(r'uint iterations;\n\} PushConstants;', r'uint iterations;\n    uint padding;\n} PushConstants;', content)
    content = re.sub(r'uint32_t iterations;\n\};', r'uint32_t iterations;\n    uint32_t padding;\n};', content)

    with open(filepath, 'w') as f:
        f.write(content)

