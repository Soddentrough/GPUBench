#!/usr/bin/env python3
"""
Patch assets/models/toycar.glb to fix the Glass material:
- Replace raw lime green baseColorFactor [0.3, 0.8, 0.3, 1] with [0.92, 0.95, 0.96, 1.0] (clear automotive safety glass)
- Add KHR_materials_ior extension with ior: 1.52
- Ensure KHR_materials_transmission transmissionFactor: 1.0
- Add KHR_materials_ior to extensionsUsed
"""

import os
import shutil
import struct
import json

src_path = "assets/models/toycar.glb"
bak_path = "assets/models/toycar.glb.bak"

if not os.path.exists(bak_path):
    shutil.copy2(src_path, bak_path)
    print(f"Created backup: {bak_path}")
else:
    print(f"Backup already exists: {bak_path}")

with open(src_path, "rb") as f:
    header = f.read(12)
    magic, version, length = struct.unpack("<4sII", header)
    assert magic == b"glTF", "Not a valid GLB file"
    assert version == 2, "Unsupported glTF version"
    
    chunk0_len, chunk0_type = struct.unpack("<II", f.read(8))
    assert chunk0_type == 0x4E4F534A, "Chunk 0 is not JSON"
    json_bytes = f.read(chunk0_len)
    gltf = json.loads(json_bytes.decode("utf-8"))
    
    chunk1_header = f.read(8)
    chunk1_len, chunk1_type = struct.unpack("<II", chunk1_header)
    assert chunk1_type == 0x004E4942, "Chunk 1 is not BIN"
    bin_bytes = f.read(chunk1_len)

print("Original Glass material:")
print(json.dumps(gltf["materials"][2], indent=2))

# 1. Update Glass Material
glass_mat = gltf["materials"][2]
assert glass_mat["name"] == "Glass"
glass_mat["pbrMetallicRoughness"]["baseColorFactor"] = [0.92, 0.95, 0.96, 1.0]
glass_mat["pbrMetallicRoughness"]["metallicFactor"] = 0.0
glass_mat["pbrMetallicRoughness"]["roughnessFactor"] = 0.005

if "extensions" not in glass_mat:
    glass_mat["extensions"] = {}

glass_mat["extensions"]["KHR_materials_transmission"] = {
    "transmissionFactor": 1.0
}
glass_mat["extensions"]["KHR_materials_ior"] = {
    "ior": 1.52
}

# 2. Update extensionsUsed
if "extensionsUsed" not in gltf:
    gltf["extensionsUsed"] = []
if "KHR_materials_ior" not in gltf["extensionsUsed"]:
    gltf["extensionsUsed"].append("KHR_materials_ior")

print("\nPatched Glass material:")
print(json.dumps(glass_mat, indent=2))
print(f"extensionsUsed: {gltf['extensionsUsed']}")

# 3. Serialize JSON and align to 4-byte boundary with trailing spaces (0x20)
new_json_str = json.dumps(gltf, separators=(",", ":"))
new_json_bytes = new_json_str.encode("utf-8")
pad_len = (4 - (len(new_json_bytes) % 4)) % 4
new_json_bytes += b" " * pad_len
new_chunk0_len = len(new_json_bytes)

# 4. Compute new total file length
new_total_len = 12 + 8 + new_chunk0_len + 8 + chunk1_len

# 5. Write patched GLB
with open(src_path, "wb") as f:
    # Header
    f.write(struct.pack("<4sII", b"glTF", 2, new_total_len))
    # Chunk 0 (JSON)
    f.write(struct.pack("<II", new_chunk0_len, 0x4E4F534A))
    f.write(new_json_bytes)
    # Chunk 1 (BIN)
    f.write(struct.pack("<II", chunk1_len, 0x004E4942))
    f.write(bin_bytes)

print(f"\nSuccessfully wrote patched {src_path} (new size: {new_total_len} bytes)")
