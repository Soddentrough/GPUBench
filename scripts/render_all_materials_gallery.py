#!/usr/bin/env python3
"""
GPUBench Complete Master Material Showcase - 4K Orthographic Blender Cycles Render
Renders ALL 42 material shaders used across the entire GPUBench benchmark suite:
- 25 Crytek Sponza Atrium Materials (assets/models/sponza.glb)
- 3 Khronos ToyCar Showcase Materials (assets/models/toycar.glb)
- 6 Showroom Studio & Atrium Procedural Materials (Car Paint, Jade SSS, Gold, Chrome, Velvet, Rust)
- 8 Open-World Forest Nature PBR Materials (Canopy Leaves, Bark, Granite Rock, Dirt/Mud, Grass, River Water, Snow, Timber)
Arranged in a 7x6 orthographic grid on Suzanne heads with display plinths and typography cards.
"""

import os
import sys
import math
import time
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

BLENDER_SCRIPT = """
import bpy
import math
import time
import os

print("=== Starting GPUBench 4K 42-Material Complete Showcase Render ===")
t_start = time.time()

# 1. Reset scene
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

# 2. Configure Preferences for Cycles HIP on GPU 1 ONLY (0000:4d:00)
cpref = bpy.context.preferences.addons['cycles'].preferences
cpref.compute_device_type = 'HIP'
cpref.get_devices_for_type('HIP')

gpu1_found = False
for d in cpref.devices:
    if "4d:00" in d.id and d.type == 'HIP':
        d.use = True
        gpu1_found = True
        print(f"Enabling Target GPU 1: {d.name} ({d.id})")
    else:
        d.use = False
        print(f"Disabling device: {d.name} ({d.id})")

if not gpu1_found:
    print("WARNING: GPU 1 (4d:00) not found by PCI ID, checking for second HIP device...")
    hip_devs = [d for d in cpref.devices if d.type == 'HIP']
    if len(hip_devs) > 1:
        hip_devs[1].use = True
        print(f"Enabling second HIP device: {hip_devs[1].name}")
    elif len(hip_devs) > 0:
        hip_devs[0].use = True
        print(f"Enabling available HIP device: {hip_devs[0].name}")

if hasattr(cpref, 'use_hiprt'):
    cpref.use_hiprt = True
    print("Enabled HIP-RT hardware acceleration")

scene.render.engine = 'CYCLES'
scene.cycles.device = 'GPU'
scene.cycles.samples = 128
scene.cycles.use_denoising = True
scene.cycles.denoiser = 'OPENIMAGEDENOISE'

scene.cycles.max_bounces = 12
scene.cycles.transmission_bounces = 12
scene.cycles.transparent_max_bounces = 12

scene.render.resolution_x = 3840
scene.render.resolution_y = 2160
scene.render.resolution_percentage = 100

# Color Management
scene.view_settings.view_transform = 'Filmic' if 'Filmic' in [t.name for t in bpy.types.ColorManagedViewSettings.bl_rna.properties['view_transform'].enum_items] else 'Standard'
scene.view_settings.look = 'High Contrast' if 'High Contrast' in [l.name for l in bpy.types.ColorManagedViewSettings.bl_rna.properties['look'].enum_items] else 'None'

def set_input(node, name, val):
    for inp in node.inputs:
        if inp.name == name:
            inp.default_value = val
            return True
    return False

# 3. Import Sponza and ToyCar glTF Models to extract their full PBR material graphs
print("Importing Sponza glTF asset...")
bpy.ops.import_scene.gltf(filepath='assets/models/sponza.glb')
print("Importing ToyCar glTF asset...")
bpy.ops.import_scene.gltf(filepath='assets/models/toycar.glb')

# Delete all imported mesh geometry; materials remain preserved in bpy.data.materials
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
print(f"Retained {len(bpy.data.materials)} glTF materials in memory")

# Apply GPUBench automotive safety glass shader override
# In toycar.glb, raw baseColorFactor is [0.3, 0.8, 0.3] (lime green).
# GPUBench's pbr_common.glsl maps this to realistic clear automotive safety glass:
# mix(vec3(0.92, 0.95, 0.96), mat.baseColorFactor.rgb, 0.08)
mat_glass = bpy.data.materials.get("Glass")
if mat_glass and mat_glass.node_tree:
    b_glass = mat_glass.node_tree.nodes.get("Principled BSDF")
    if b_glass:
        set_input(b_glass, "Base Color", (0.92, 0.95, 0.96, 1.0))
        set_input(b_glass, "Transmission Weight", 1.0)
        set_input(b_glass, "Roughness", 0.005)
        set_input(b_glass, "IOR", 1.52)
        print("Configured Glass with GPUBench clear automotive safety glass shader (IOR 1.52, tint [0.92, 0.95, 0.96])")

# Helper to create procedural materials
def create_mat(name, setup_fn):
    mat = bpy.data.materials.new(name=name)
    setup_fn(mat, mat.node_tree)
    return mat

# Procedural Showroom & Nature PBR Material Setups
def setup_car_paint(mat, tree):
    b = tree.nodes.get("Principled BSDF")
    set_input(b, "Base Color", (0.68, 0.015, 0.025, 1.0))
    set_input(b, "Metallic", 0.85)
    set_input(b, "Roughness", 0.22)
    set_input(b, "Coat Weight", 1.0)
    set_input(b, "Coat Roughness", 0.03)

def setup_jade_sss(mat, tree):
    b = tree.nodes.get("Principled BSDF")
    set_input(b, "Base Color", (0.08, 0.88, 0.74, 1.0))
    set_input(b, "Subsurface Weight", 0.85)
    set_input(b, "Subsurface Radius", (0.7, 1.2, 0.9))
    set_input(b, "Subsurface Scale", 0.25)
    set_input(b, "Roughness", 0.08)
    set_input(b, "IOR", 1.61)

def setup_gold(mat, tree):
    b = tree.nodes.get("Principled BSDF")
    set_input(b, "Base Color", (1.00, 0.78, 0.32, 1.0))
    set_input(b, "Metallic", 1.0)
    set_input(b, "Roughness", 0.04)

def setup_chrome(mat, tree):
    b = tree.nodes.get("Principled BSDF")
    set_input(b, "Base Color", (0.96, 0.96, 0.98, 1.0))
    set_input(b, "Metallic", 1.0)
    set_input(b, "Roughness", 0.002)

def setup_velvet(mat, tree):
    b = tree.nodes.get("Principled BSDF")
    set_input(b, "Base Color", (0.60, 0.02, 0.42, 1.0))
    set_input(b, "Roughness", 0.85)
    set_input(b, "Sheen Weight", 1.0)
    set_input(b, "Sheen Roughness", 0.35)
    set_input(b, "Sheen Tint", (0.95, 0.35, 0.85, 1.0))

def setup_rust(mat, tree):
    b = tree.nodes.get("Principled BSDF")
    tex = tree.nodes.new(type='ShaderNodeTexNoise')
    tex.inputs['Scale'].default_value = 7.5
    tex.inputs['Detail'].default_value = 8.0
    ramp = tree.nodes.new(type='ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].position = 0.42
    ramp.color_ramp.elements[0].color = (0.24, 0.26, 0.30, 1.0)
    ramp.color_ramp.elements[1].position = 0.58
    ramp.color_ramp.elements[1].color = (0.85, 0.36, 0.12, 1.0)
    bump = tree.nodes.new(type='ShaderNodeBump')
    bump.inputs['Strength'].default_value = 0.45
    tree.links.new(tex.outputs['Fac'], ramp.inputs['Fac'])
    tree.links.new(ramp.outputs['Color'], b.inputs['Base Color'])
    tree.links.new(tex.outputs['Fac'], bump.inputs['Height'])
    tree.links.new(bump.outputs['Normal'], b.inputs['Normal'])
    set_input(b, "Roughness", 0.65)
    set_input(b, "Metallic", 0.45)

def setup_showroom_floor(mat, tree):
    b = tree.nodes.get("Principled BSDF")
    set_input(b, "Base Color", (0.20, 0.22, 0.26, 1.0))
    set_input(b, "Roughness", 0.42)
    set_input(b, "Coat Weight", 0.35)
    set_input(b, "Coat Roughness", 0.18)

def setup_foliage(mat, tree):
    b = tree.nodes.get("Principled BSDF")
    set_input(b, "Base Color", (0.08, 0.32, 0.12, 1.0))
    set_input(b, "Subsurface Weight", 0.65)
    set_input(b, "Subsurface Radius", (0.25, 0.65, 0.15))
    set_input(b, "Subsurface Scale", 0.20)
    set_input(b, "Roughness", 0.40)

def setup_pine_bark(mat, tree):
    b = tree.nodes.get("Principled BSDF")
    tex = tree.nodes.new(type='ShaderNodeTexWave')
    tex.inputs['Scale'].default_value = 6.0
    tex.inputs['Distortion'].default_value = 12.0
    ramp = tree.nodes.new(type='ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].color = (0.12, 0.08, 0.05, 1.0)
    ramp.color_ramp.elements[1].color = (0.28, 0.19, 0.12, 1.0)
    bump = tree.nodes.new(type='ShaderNodeBump')
    bump.inputs['Strength'].default_value = 0.75
    tree.links.new(tex.outputs['Color'], ramp.inputs['Fac'])
    tree.links.new(ramp.outputs['Color'], b.inputs['Base Color'])
    tree.links.new(tex.outputs['Fac'], bump.inputs['Height'])
    tree.links.new(bump.outputs['Normal'], b.inputs['Normal'])
    set_input(b, "Roughness", 0.88)

def setup_granite_rock(mat, tree):
    b = tree.nodes.get("Principled BSDF")
    tex = tree.nodes.new(type='ShaderNodeTexNoise')
    tex.inputs['Scale'].default_value = 18.0
    tex.inputs['Detail'].default_value = 10.0
    ramp = tree.nodes.new(type='ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].color = (0.18, 0.19, 0.21, 1.0)
    ramp.color_ramp.elements[1].color = (0.34, 0.35, 0.38, 1.0)
    bump = tree.nodes.new(type='ShaderNodeBump')
    bump.inputs['Strength'].default_value = 0.65
    tree.links.new(tex.outputs['Fac'], ramp.inputs['Fac'])
    tree.links.new(ramp.outputs['Color'], b.inputs['Base Color'])
    tree.links.new(tex.outputs['Fac'], bump.inputs['Height'])
    tree.links.new(bump.outputs['Normal'], b.inputs['Normal'])
    set_input(b, "Roughness", 0.92)

def setup_topsoil_mud(mat, tree):
    b = tree.nodes.get("Principled BSDF")
    tex = tree.nodes.new(type='ShaderNodeTexNoise')
    tex.inputs['Scale'].default_value = 12.0
    tex.inputs['Detail'].default_value = 8.0
    ramp = tree.nodes.new(type='ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].color = (0.14, 0.10, 0.08, 1.0)
    ramp.color_ramp.elements[1].color = (0.28, 0.19, 0.14, 1.0)
    bump = tree.nodes.new(type='ShaderNodeBump')
    bump.inputs['Strength'].default_value = 0.50
    tree.links.new(tex.outputs['Fac'], ramp.inputs['Fac'])
    tree.links.new(ramp.outputs['Color'], b.inputs['Base Color'])
    tree.links.new(tex.outputs['Fac'], bump.inputs['Height'])
    tree.links.new(bump.outputs['Normal'], b.inputs['Normal'])
    set_input(b, "Roughness", 0.38)
    set_input(b, "Coat Weight", 0.50)

def setup_meadow_grass(mat, tree):
    b = tree.nodes.get("Principled BSDF")
    set_input(b, "Base Color", (0.18, 0.45, 0.14, 1.0))
    set_input(b, "Roughness", 0.55)
    set_input(b, "Sheen Weight", 0.85)
    set_input(b, "Sheen Tint", (0.6, 0.9, 0.4, 1.0))

def setup_river_water(mat, tree):
    b = tree.nodes.get("Principled BSDF")
    set_input(b, "Base Color", (0.08, 0.28, 0.42, 1.0))
    set_input(b, "Transmission Weight", 0.95)
    set_input(b, "Roughness", 0.02)
    set_input(b, "IOR", 1.333)

def setup_snow(mat, tree):
    b = tree.nodes.get("Principled BSDF")
    set_input(b, "Base Color", (0.92, 0.95, 0.98, 1.0))
    set_input(b, "Subsurface Weight", 0.45)
    set_input(b, "Subsurface Radius", (1.0, 1.0, 1.0))
    set_input(b, "Subsurface Scale", 0.15)
    set_input(b, "Roughness", 0.45)

def setup_timber_bridge(mat, tree):
    b = tree.nodes.get("Principled BSDF")
    tex = tree.nodes.new(type='ShaderNodeTexWave')
    tex.inputs['Scale'].default_value = 5.0
    tex.inputs['Distortion'].default_value = 6.0
    ramp = tree.nodes.new(type='ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].color = (0.24, 0.18, 0.12, 1.0)
    ramp.color_ramp.elements[1].color = (0.42, 0.32, 0.22, 1.0)
    bump = tree.nodes.new(type='ShaderNodeBump')
    bump.inputs['Strength'].default_value = 0.60
    tree.links.new(tex.outputs['Color'], ramp.inputs['Fac'])
    tree.links.new(ramp.outputs['Color'], b.inputs['Base Color'])
    tree.links.new(tex.outputs['Fac'], bump.inputs['Height'])
    tree.links.new(bump.outputs['Normal'], b.inputs['Normal'])
    set_input(b, "Roughness", 0.82)

# Build list of 42 material objects
materials_42 = []
# 0-24: Sponza materials
for i in range(25):
    m = bpy.data.materials.get(f"Material_{i}")
    materials_42.append(m)

# 25-27: ToyCar materials
materials_42.append(bpy.data.materials.get("ToyCar"))
materials_42.append(bpy.data.materials.get("Fabric"))
materials_42.append(bpy.data.materials.get("Glass"))

# 28-33: Showroom & Atrium Procedural
materials_42.append(create_mat("Proc_JadeSSS", setup_jade_sss))
materials_42.append(create_mat("Proc_Gold", setup_gold))
materials_42.append(create_mat("Proc_Chrome", setup_chrome))
materials_42.append(create_mat("Proc_Velvet", setup_velvet))
materials_42.append(create_mat("Proc_Rust", setup_rust))
materials_42.append(create_mat("Proc_ShowroomFloor", setup_showroom_floor))

# 34-41: Open-World Forest Nature PBR
materials_42.append(create_mat("Proc_Foliage", setup_foliage))
materials_42.append(create_mat("Proc_PineBark", setup_pine_bark))
materials_42.append(create_mat("Proc_GraniteRock", setup_granite_rock))
materials_42.append(create_mat("Proc_TopsoilMud", setup_topsoil_mud))
materials_42.append(create_mat("Proc_MeadowGrass", setup_meadow_grass))
materials_42.append(create_mat("Proc_RiverWater", setup_river_water))
materials_42.append(create_mat("Proc_SnowIce", setup_snow))
materials_42.append(create_mat("Proc_TimberBridge", setup_timber_bridge))

print(f"Successfully assembled {len(materials_42)} materials for 7x6 grid")

# 4. Setup World & Studio Lighting Rig
world = bpy.data.worlds.new(name="MasterStudioWorld")
scene.world = world
bg_node = world.node_tree.nodes.get("Background")
if bg_node:
    set_input(bg_node, "Color", (0.015, 0.020, 0.028, 1.0))
    set_input(bg_node, "Strength", 0.45)

# Key Light
bpy.ops.object.light_add(type='AREA', location=(12.0, -22.0, 16.0))
key_light = bpy.context.object
key_light.data.energy = 5400.0
key_light.data.size = 14.0
key_light.data.size_y = 10.0
key_light.data.color = (1.0, 0.96, 0.90)
key_light.rotation_euler = (math.radians(45), math.radians(15), math.radians(-30))

# Fill Light
bpy.ops.object.light_add(type='AREA', location=(-16.0, -18.0, 10.0))
fill_light = bpy.context.object
fill_light.data.energy = 2400.0
fill_light.data.size = 14.0
fill_light.data.size_y = 10.0
fill_light.data.color = (0.85, 0.92, 1.0)
fill_light.rotation_euler = (math.radians(55), math.radians(-20), math.radians(45))

# Rim Light
bpy.ops.object.light_add(type='AREA', location=(0.0, 10.0, 14.0))
rim_light = bpy.context.object
rim_light.data.energy = 3600.0
rim_light.data.size = 20.0
rim_light.data.size_y = 4.0
rim_light.data.color = (1.0, 1.0, 1.0)
rim_light.rotation_euler = (math.radians(-50), 0, 0)

# Low Bounce Light
bpy.ops.object.light_add(type='AREA', location=(0.0, -12.0, -10.0))
bounce_light = bpy.context.object
bounce_light.data.energy = 1100.0
bounce_light.data.size = 22.0
bounce_light.data.size_y = 8.0
bounce_light.data.color = (0.70, 0.78, 0.90)
bounce_light.rotation_euler = (math.radians(130), 0, 0)

# 5. Orthographic Camera
ortho_scale = 22.0
bpy.ops.object.camera_add(location=(0.0, -50.0, 0.0))
cam = bpy.context.object
scene.camera = cam
cam.data.type = 'ORTHO'
cam.data.ortho_scale = ortho_scale
cam.rotation_euler = (math.radians(90.0), 0.0, 0.0)

# Cyclorama Backdrop
bpy.ops.mesh.primitive_plane_add(size=90.0, location=(0.0, 10.0, 0.0))
backdrop = bpy.context.object
backdrop.rotation_euler = (math.radians(90.0), 0.0, 0.0)
mat_backdrop = bpy.data.materials.new(name="BackdropMat")
bsdf_bg = mat_backdrop.node_tree.nodes.get("Principled BSDF")
if bsdf_bg:
    set_input(bsdf_bg, "Base Color", (0.048, 0.062, 0.082, 1.0))
    set_input(bsdf_bg, "Roughness", 0.95)
backdrop.data.materials.append(mat_backdrop)

# Plinth Material (Dark Anodized Titanium)
mat_plinth = bpy.data.materials.new(name="PlinthMat")
bsdf_pl = mat_plinth.node_tree.nodes.get("Principled BSDF")
if bsdf_pl:
    set_input(bsdf_pl, "Base Color", (0.11, 0.12, 0.15, 1.0))
    set_input(bsdf_pl, "Metallic", 0.88)
    set_input(bsdf_pl, "Roughness", 0.25)
    set_input(bsdf_pl, "Coat Weight", 0.6)
    set_input(bsdf_pl, "Coat Roughness", 0.12)

# 6. Build 7x6 Grid of Suzanne Heads and Plinths with Exact Pixel Alignment
W = 3840.0
H = 2160.0
world_w = ortho_scale
world_h = ortho_scale * (H / W) # 12.375

cols = 7
rows = 6

col_px_centers = [274, 823, 1371, 1920, 2469, 3017, 3566]
row_px_centers = [287, 620, 953, 1287, 1620, 1953]

for idx in range(42):
    c = idx % cols
    r = idx // cols
    
    cx = col_px_centers[c]
    ry = row_px_centers[r]
    
    head_px_y = ry - 22
    plinth_px_y = ry + 60
    
    wx = (cx / W - 0.5) * world_w
    wz_head = (0.5 - head_px_y / H) * world_h
    wz_plinth = (0.5 - plinth_px_y / H) * world_h
    
    # Plinth (Cylinder)
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=48,
        radius=0.46,
        depth=0.18,
        location=(wx, 0.0, wz_plinth)
    )
    plinth = bpy.context.object
    plinth.data.materials.append(mat_plinth)
    plinth.data.shade_smooth()
    bm = plinth.modifiers.new(name="Bevel", type='BEVEL')
    bm.width = 0.018
    bm.segments = 3
    
    # Suzanne Head
    bpy.ops.mesh.primitive_monkey_add(location=(wx, 0.0, wz_head))
    suzanne = bpy.context.object
    suzanne.scale = (0.44, 0.44, 0.44)
    suzanne.rotation_euler = (math.radians(-10.0), math.radians(24.0), math.radians(-12.0))
    suzanne.data.shade_smooth()
    
    subsurf = suzanne.modifiers.new(name="Subsurf", type='SUBSURF')
    subsurf.levels = 1
    subsurf.render_levels = 1
    
    mat = materials_42[idx]
    if mat:
        suzanne.data.materials.append(mat)

# 7. Render Output
output_raw = "/home/naoki/Development/GPUBench/renders/raw_all_materials_gallery.png"
os.makedirs(os.path.dirname(output_raw), exist_ok=True)
scene.render.filepath = output_raw
print(f"Rendering 4K 42-material frame to {output_raw}...")
bpy.ops.render.render(write_still=True)
t_end = time.time()
print(f"=== Blender Cycles 42-Material render completed in {t_end - t_start:.2f} s ===")
"""

def generate_blender_render():
    blend_script_path = "/home/naoki/.gemini/antigravity/brain/b26a5e7f-321e-4ad2-a16c-02af680fa03c/scratch/run_all_materials_render.py"
    with open(blend_script_path, "w") as f:
        f.write(BLENDER_SCRIPT)
    
    cmd = f"blender -b --python {blend_script_path}"
    print(f"Executing Blender Cycles render on GPU 1: {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f"Blender render failed with exit code {ret}")

# ------------------------------------------------------------------------------
# 2. Typography & Caption Card Compositor (Pillow)
# ------------------------------------------------------------------------------
def composite_captions():
    raw_path = "renders/raw_all_materials_gallery.png"
    out_path = "renders/render_all_materials_gallery_4k.png"
    artifact_path = "/home/naoki/.gemini/antigravity/brain/b26a5e7f-321e-4ad2-a16c-02af680fa03c/render_all_materials_gallery_4k.png"
    
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Missing raw render file: {raw_path}")
    
    img = Image.open(raw_path).convert("RGBA")
    w, h = img.size
    
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    font_title = ImageFont.truetype("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf", 32)
    font_sub = ImageFont.truetype("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf", 18)
    font_badge = ImageFont.truetype("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf", 16)
    
    font_mat = ImageFont.truetype("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf", 12)
    font_shader = ImageFont.truetype("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf", 9)
    font_scene = ImageFont.truetype("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Oblique.ttf", 8)
    
    tz_abbr = time.strftime("%Z")
    if tz_abbr.startswith("+") or tz_abbr.startswith("-"):
        tz_abbr = "UTC" + tz_abbr
    timestamp_str = datetime.now().astimezone().strftime(f"%Y-%m-%d %H:%M:%S {tz_abbr}")
    
    # Top Banner
    draw.rectangle([(0, 0), (w, 110)], fill=(12, 16, 24, 235))
    draw.line([(0, 110), (w, 110)], fill=(40, 55, 80, 255), width=2)
    
    draw.text((50, 20), "GPUBENCH COMPLETE MATERIAL SHADER SHOWCASE • 4K MASTER GALLERY", fill=(255, 255, 255), font=font_title)
    draw.text((50, 68), f"AMD Radeon AI PRO R9700 (RADV GFX1201 / Vulkan 1.4) • All 42 Shaders Across Sponza, ToyCar, Nature & Studio • Rendered: {timestamp_str}", fill=(160, 185, 220), font=font_sub)
    
    badge_text = f"RENDERED: {timestamp_str}"
    bbox = draw.textbbox((0, 0), badge_text, font=font_badge)
    bw = bbox[2] - bbox[0] + 28
    bh = bbox[3] - bbox[1] + 16
    bx = w - bw - 50
    by = 36
    draw.rectangle([(bx, by), (bx + bw, by + bh)], fill=(20, 30, 48, 220), outline=(80, 140, 220, 255), width=2)
    draw.text((bx + 14, by + 8), badge_text, fill=(180, 220, 255), font=font_badge)
    
    cards = [
        # Row 0: Sponza Architecture & Masonry
        {
            "name": "01. SPONZA PAVING FLAGSTONES",
            "shader": "Textured Cook-Torrance GGX, Sandstone PBR & Normal Map",
            "scene": "Scene: Indoor Atrium (Courtyard Floor, 27.8k Tris)",
            "color": (210, 190, 160)
        },
        {
            "name": "02. SPONZA COLUMN BASE BRICK",
            "shader": "Layered Terracotta Brickwork with Mortar Normal Relief",
            "scene": "Scene: Indoor Atrium (Pillar Bases, 15.5k Tris)",
            "color": (225, 120, 95)
        },
        {
            "name": "03. SPONZA WALL PLASTER",
            "shader": "Matte Mineral Chalk Diffuse with Micro-Pore Roughness",
            "scene": "Scene: Indoor Atrium (Upper Corridor Facade)",
            "color": (230, 230, 225)
        },
        {
            "name": "04. SPONZA CORINTHIAN CAPITALS",
            "shader": "Carved Limestone Relief with High-Frequency Cavity AO",
            "scene": "Scene: Indoor Atrium (Colonnade Capitals, 2.9k Tris)",
            "color": (235, 210, 165)
        },
        {
            "name": "05. SPONZA CORNER ARCHWAYS",
            "shader": "Weathered Calcified Limestone with Edge Chipping",
            "scene": "Scene: Indoor Atrium (Perimeter Niches, 2.7k Tris)",
            "color": (220, 185, 130)
        },
        {
            "name": "06. SPONZA ASHLAR WALL STONE",
            "shader": "Dressed Masonry Blocks with Recessed Bevel Mortar",
            "scene": "Scene: Indoor Atrium (Structural Piers)",
            "color": (175, 180, 190)
        },
        {
            "name": "07. SPONZA GALLERY COLONNADE",
            "shader": "Fluted Pillar Relief with Soft Specular Highlights",
            "scene": "Scene: Indoor Atrium (Upper Gallery, 11.8k Tris)",
            "color": (240, 225, 200)
        },
        # Row 1: Sponza Arches, Balustrades & Ceiling
        {
            "name": "08. SPONZA MAIN ARCH SPANDRELS",
            "shader": "Molded Architrave Stone with Architectural Bas-Relief",
            "scene": "Scene: Indoor Atrium (Ground Arcade, 8.4k Tris)",
            "color": (215, 195, 155)
        },
        {
            "name": "09. SPONZA BALUSTRADE RAILS",
            "shader": "Turned Alabaster Stone Balusters & Molded Handrails",
            "scene": "Scene: Indoor Atrium (Mezzanine Rail, 2.8k Tris)",
            "color": (245, 235, 215)
        },
        {
            "name": "10. SPONZA GALLERY STRINGCOURSE",
            "shader": "Continuous Horizontal Architectural Cornice Stone",
            "scene": "Scene: Indoor Atrium (Floor Division Belt)",
            "color": (205, 190, 150)
        },
        {
            "name": "11. SPONZA VAULTED ARCH RIBS",
            "shader": "Vault Rib Masonry with Tangent Normal Mapping",
            "scene": "Scene: Indoor Atrium (Ceiling Arches, 7.1k Tris)",
            "color": (210, 180, 140)
        },
        {
            "name": "12. SPONZA FACADE ENTABLATURE",
            "shader": "Classic Classical Architrave, Frieze & Cornice Molding",
            "scene": "Scene: Indoor Atrium (Upper Entablature)",
            "color": (230, 215, 185)
        },
        {
            "name": "13. SPONZA UPPER BRICK FACADE",
            "shader": "Venetian Terracotta Brickwork with Mortar Erosion",
            "scene": "Scene: Indoor Atrium (Gallery Facade, 23.2k Tris)",
            "color": (210, 105, 80)
        },
        {
            "name": "14. SPONZA PERIMETER GROUND WALL",
            "shader": "Heavy Foundation Ashlar Stone with Ground Moisture",
            "scene": "Scene: Indoor Atrium (Ground Perimeter, 16.5k Tris)",
            "color": (160, 140, 120)
        },
        # Row 2: Sponza Fabrics, Banners, Chains & Sculptures
        {
            "name": "15. SPONZA FOREST GREEN DRAPE",
            "shader": "Woven Velvet Drape Microfiber (Green Dye Pigment)",
            "scene": "Scene: Indoor Atrium (Archway Portieres, 16.5k Tris)",
            "color": (50, 180, 80)
        },
        {
            "name": "16. SPONZA NAVY BLUE DRAPE",
            "shader": "Indigo Dyed Heavy Cloth Weave with Micro-Sheen",
            "scene": "Scene: Indoor Atrium (Archway Portieres, 16.5k Tris)",
            "color": (70, 140, 230)
        },
        {
            "name": "17. SPONZA CRIMSON ARCH DRAPE",
            "shader": "Carmine Red Velvet with Deep Folds & Grazing Sheen",
            "scene": "Scene: Indoor Atrium (Archway Portieres, 11.0k Tris)",
            "color": (230, 60, 60)
        },
        {
            "name": "18. SPONZA ROYAL BLUE BANNER",
            "shader": "Heraldic Silk Tapestry with Gold Filigree Embroidery",
            "scene": "Scene: Indoor Atrium (Corridor Banners, 14.3k Tris)",
            "color": (80, 160, 255)
        },
        {
            "name": "19. SPONZA IMPERIAL RED BANNER",
            "shader": "Heraldic Imperial Lion Emblem Weave with Gold Border",
            "scene": "Scene: Indoor Atrium (Corridor Banners, 18.9k Tris)",
            "color": (245, 75, 75)
        },
        {
            "name": "20. SPONZA EMERALD GREEN BANNER",
            "shader": "Heraldic Silk Pattern & Fringe with Grazing Sheen",
            "scene": "Scene: Indoor Atrium (Corridor Banners, 14.3k Tris)",
            "color": (60, 210, 100)
        },
        {
            "name": "21. SPONZA CLIMBING IVY LEAF",
            "shader": "Alpha-Cutout Foliage Leaf Translucency & SSS",
            "scene": "Scene: Indoor Atrium (Courtyard Planters)",
            "color": (120, 230, 90)
        },
        # Row 3: Sponza Metal/Sculpture + ToyCar Showcase
        {
            "name": "22. SPONZA WROUGHT IRON CHAIN",
            "shader": "Weathered Ferrous Conductor with Pitted Rust Patina",
            "scene": "Scene: Indoor Atrium (Lantern Chains, 19.8k Tris)",
            "color": (130, 140, 150)
        },
        {
            "name": "23. SPONZA LION GARGOYLE RELIEF",
            "shader": "High-Relief Carved Sandstone Gargoyle Sculpture",
            "scene": "Scene: Indoor Atrium (Wall Sconces, 9.2k Tris)",
            "color": (230, 190, 120)
        },
        {
            "name": "24. SPONZA TERRACOTTA URN",
            "shader": "Porous Clay Ceramic with Mineral Salt Glaze Flakes",
            "scene": "Scene: Indoor Atrium (Courtyard Urns, 3.0k Tris)",
            "color": (220, 130, 80)
        },
        {
            "name": "25. SPONZA ROOF TIMBER BEAMS",
            "shader": "Weathered Structural Timber Plank Grain & Dark Stain",
            "scene": "Scene: Indoor Atrium (Ceiling Trusses, 14.5k Tris)",
            "color": (160, 115, 75)
        },
        {
            "name": "26. TOYCAR RUBY FLAKE CLEARCOAT",
            "shader": "Dual-Lobe GGX Clearcoat & Decals (Base α=0.22, Coat α=0.03)",
            "scene": "Scene: Showroom Studio (Car Body, 108.9k Tris)",
            "color": (255, 65, 85)
        },
        {
            "name": "27. TOYCAR CABIN LEATHER",
            "shader": "Pebbled Anisotropic Automotive Vinyl & Stitching",
            "scene": "Scene: Showroom Studio (Interior Cabin Seats)",
            "color": (210, 150, 100)
        },
        {
            "name": "28. TOYCAR DIELECTRIC GLASS",
            "shader": "Snell Refraction, Bounded Shell Loop, IOR 1.52",
            "scene": "Scene: Showroom Studio (Windshield & Rear Glass)",
            "color": (150, 230, 255)
        },
        # Row 4: Showroom & Atrium Procedural Shaders
        {
            "name": "29. JADE & MARBLE SSS",
            "shader": "Volumetric Subsurface Scatter (Exp Depth, IOR 1.61)",
            "scene": "Scene: Showroom & Atrium (Knot Band 1 & Columns)",
            "color": (90, 255, 210)
        },
        {
            "name": "30. POLISHED 24K GOLD",
            "shader": "High-Conductivity Complex Fresnel (F0 ≈ 0.88, α=0.04)",
            "scene": "Scene: Showroom & Atrium (Center Pedestal & Ceiling)",
            "color": (255, 220, 80)
        },
        {
            "name": "31. POLISHED CHROME MIRROR",
            "shader": "Ideal Specular Metallic Conductor (F0 = 0.98, α < 0.005)",
            "scene": "Scene: Showroom Studio (Knot Band 2)",
            "color": (225, 235, 255)
        },
        {
            "name": "32. IMPERIAL MAGENTA VELVET",
            "shader": "Charlie Sheen Microfiber BRDF (Roughness 0.85)",
            "scene": "Scene: Showroom Studio (Knot Band 3)",
            "color": (255, 110, 220)
        },
        {
            "name": "33. WEATHERED RUST & BRONZE",
            "shader": "Layered 3D FBM Procedural (Pitted Terracotta & Steel)",
            "scene": "Scene: Showroom Studio (Knot Band 4) & Atrium Walls",
            "color": (255, 150, 75)
        },
        {
            "name": "34. SHOWROOM SATIN CYCLORAMA",
            "shader": "Soft Micro-Rough Specular Floor with Curved Horizon",
            "scene": "Scene: Showroom Studio (Stage Backdrop Floor)",
            "color": (150, 165, 185)
        },
        {
            "name": "35. TRANSLUCENT CANOPY FOLIAGE",
            "shader": "Double-Sided Thin Transmission & Needle Subsurface",
            "scene": "Scene: Open-World Forest (Canopy Needles & Leaves)",
            "color": (110, 240, 130)
        },
        # Row 5: Open-World Forest Nature PBR Shaders
        {
            "name": "36. WEATHERED PINE BARK",
            "shader": "Fibrous Longitudinal Grain with Vertical Anisotropy",
            "scene": "Scene: Open-World Forest & Landscape (Trunks & Roots)",
            "color": (195, 140, 95)
        },
        {
            "name": "37. EXPOSED GRANITE CLIFF",
            "shader": "Rough Mineral Lambertian + Crevice Self-Shadowing",
            "scene": "Scene: Open-World Forest & Landscape (Cliffs & Boulders)",
            "color": (190, 200, 215)
        },
        {
            "name": "38. FOREST TOPSOIL & WET MUD",
            "shader": "Porous Mineral Topsoil with Specular Water Puddles",
            "scene": "Scene: Open-World Forest (Pathways & Riverbanks)",
            "color": (180, 125, 85)
        },
        {
            "name": "39. ALPINE MEADOW GRASS",
            "shader": "Fine Blade Translucency + Grazing Charlie Sheen",
            "scene": "Scene: Open-World Forest (Meadows & Clearings)",
            "color": (135, 230, 95)
        },
        {
            "name": "40. RIVER WATER & BATHYMETRY",
            "shader": "Snell Refraction (IOR 1.333) + Beer-Lambert Depth",
            "scene": "Scene: Open-World Forest & Lake (River Plane)",
            "color": (100, 220, 255)
        },
        {
            "name": "41. MOUNTAIN SNOW & ICE",
            "shader": "Subsurface Forward Scatter + Crystalline Micro-Glints",
            "scene": "Scene: Open-World Forest & Landscape (Peaks)",
            "color": (220, 240, 255)
        },
        {
            "name": "42. TIMBER BRIDGE & MASONRY",
            "shader": "Weathered Timber Planks with Structural Wood Grain",
            "scene": "Scene: Open-World Forest (Footbridge & Ruins)",
            "color": (210, 165, 115)
        }
    ]
    
    col_px_centers = [274, 823, 1371, 1920, 2469, 3017, 3566]
    row_px_centers = [287, 620, 953, 1287, 1620, 1953]
    
    card_w = 490
    card_h = 62
    
    for idx, card in enumerate(cards):
        c = idx % 7
        r = idx // 7
        
        cx = col_px_centers[c]
        ry = row_px_centers[r]
        
        card_py = ry + 118
        left = cx - card_w // 2
        top = card_py - card_h // 2
        
        # Plaque Card Background
        draw.rounded_rectangle(
            [(left, top), (left + card_w, top + card_h)],
            radius=8,
            fill=(14, 18, 26, 225),
            outline=(45, 60, 85, 240),
            width=2
        )
        
        # Color accent strip on left
        draw.rounded_rectangle(
            [(left + 4, top + 6), (left + 9, top + card_h - 6)],
            radius=2,
            fill=card["color"]
        )
        
        # Text lines
        draw.text((left + 16, top + 7), card["name"], fill=card["color"], font=font_mat)
        draw.text((left + 16, top + 26), card["shader"], fill=(210, 220, 235), font=font_shader)
        draw.text((left + 16, top + 43), card["scene"], fill=(130, 160, 200), font=font_scene)
        
    final_img = Image.alpha_composite(img, overlay).convert("RGB")
    final_img.save(out_path, quality=95)
    final_img.save(artifact_path, quality=95)
    print(f"Saved complete 42-material 4K showcase image to {out_path} and {artifact_path}")

if __name__ == "__main__":
    generate_blender_render()
    composite_captions()
