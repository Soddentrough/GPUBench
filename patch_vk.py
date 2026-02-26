import glob

repl = """layout(push_constant) uniform PushConstants {
    uint stride;
    uint mask;
    uint iterations;
} pushConstants;"""

for f in glob.glob("kernels/vulkan/cachebw*.comp") + glob.glob("kernels/vulkan/cache_latency.comp"):
    with open(f, 'r') as file:
        content = file.read()
    if 'uint stride;' in content: continue
    
    # Simple replacement for the block
    import re
    content = re.sub(r'layout\(push_constant\) uniform PushConstants \{[^}]*\} pushConstants;', repl, content)
    
    with open(f, 'w') as file:
        file.write(content)

