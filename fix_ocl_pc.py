import glob
import re

for filepath in glob.glob("kernels/opencl/*.cl"):
    with open(filepath, 'r') as f:
        content = f.read()

    # Remove typedef struct
    content = re.sub(r'typedef struct \{[^\}]+\} PushConstants;', '', content)

    # Change signature
    content = content.replace('PushConstants pc', 'uint4 pc')

    # Replace pc.stride -> pc.x, pc.mask -> pc.y, pc.iterations -> pc.z
    content = content.replace('pc.stride', 'pc.x')
    content = content.replace('pc.mask', 'pc.y')
    content = content.replace('pc.iterations', 'pc.z')

    with open(filepath, 'w') as f:
        f.write(content)

print("Replaced PushConstants with uint4 in OpenCL kernels")
