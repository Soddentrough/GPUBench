#!/usr/bin/env python3
"""
GPUBench Material Shader Showcase - 4K Orthographic Blender Cycles Render
Renders all 12 material shader formulations on Suzanne heads on AMD GPU 1.
"""

import os
import sys
import math
import time
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

# ------------------------------------------------------------------------------
# 1. Blender Automation Script Generator
# ------------------------------------------------------------------------------
BLENDER_SCRIPT = """
import bpy
import math
import time
import os

print("=== Starting GPUBench 4K Suzanne Material Gallery Render ===")
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

# Ensure deep ray transmission for dielectric glass and water
scene.cycles.max_bounces = 12
scene.cycles.transmission_bounces = 12
scene.cycles.transparent_max_bounces = 12

scene.render.resolution_x = 3840
scene.render.resolution_y = 2160
scene.render.resolution_percentage = 100

# Color Management
scene.view_settings.view_transform = 'Filmic' if 'Filmic' in [t.name for t in bpy.types.ColorManagedViewSettings.bl_rna.properties['view_transform'].enum_items] else 'Standard'
scene.view_settings.look = 'High Contrast' if 'High Contrast' in [l.name for l in bpy.types.ColorManagedViewSettings.bl_rna.properties['look'].enum_items] else 'None'

# Helper to set node inputs safely by name
def set_input(node, name, val):
    for inp in node.inputs:
        if inp.name == name:
            inp.default_value = val
            return True
    return False

# 3. Setup World & Lighting
world = bpy.data.worlds.new(name="StudioWorld")
scene.world = world
bg_node = world.node_tree.nodes.get("Background")
if bg_node:
    set_input(bg_node, "Color", (0.015, 0.020, 0.028, 1.0))
    set_input(bg_node, "Strength", 0.45)

# 3-Point Studio Softbox Lighting Rig + Low Bounce
# Key Light (Top-Right Front, Warm Softbox)
bpy.ops.object.light_add(type='AREA', location=(10.0, -18.0, 14.0))
key_light = bpy.context.object
key_light.data.energy = 4800.0
key_light.data.size = 9.0
key_light.data.size_y = 7.0
key_light.data.color = (1.0, 0.96, 0.90)
key_light.rotation_euler = (math.radians(45), math.radians(15), math.radians(-30))

# Fill Light (Left Front, Cool Softbox)
bpy.ops.object.light_add(type='AREA', location=(-14.0, -16.0, 8.0))
fill_light = bpy.context.object
fill_light.data.energy = 2100.0
fill_light.data.size = 11.0
fill_light.data.size_y = 8.0
fill_light.data.color = (0.85, 0.92, 1.0)
fill_light.rotation_euler = (math.radians(55), math.radians(-20), math.radians(45))

# Rim / Top Accent Light (Rear-Top, Crisp Neutral White)
bpy.ops.object.light_add(type='AREA', location=(0.0, 8.0, 12.0))
rim_light = bpy.context.object
rim_light.data.energy = 3000.0
rim_light.data.size = 16.0
rim_light.data.size_y = 3.5
rim_light.data.color = (1.0, 1.0, 1.0)
rim_light.rotation_euler = (math.radians(-50), 0, 0)

# Low-angle subtle bounce light (illuminates plinths and chin facets softly)
bpy.ops.object.light_add(type='AREA', location=(0.0, -10.0, -8.0))
bounce_light = bpy.context.object
bounce_light.data.energy = 900.0
bounce_light.data.size = 18.0
bounce_light.data.size_y = 6.0
bounce_light.data.color = (0.70, 0.78, 0.90)
bounce_light.rotation_euler = (math.radians(130), 0, 0)

# 4. Setup Orthographic Camera
ortho_scale = 18.0
bpy.ops.object.camera_add(location=(0.0, -40.0, 0.0))
cam = bpy.context.object
scene.camera = cam
cam.data.type = 'ORTHO'
cam.data.ortho_scale = ortho_scale
cam.rotation_euler = (math.radians(90.0), 0.0, 0.0)

# 5. Studio Cyclorama Backdrop
bpy.ops.mesh.primitive_plane_add(size=70.0, location=(0.0, 8.0, 0.0))
backdrop = bpy.context.object
backdrop.rotation_euler = (math.radians(90.0), 0.0, 0.0)
mat_backdrop = bpy.data.materials.new(name="BackdropMat")
bsdf_bg = mat_backdrop.node_tree.nodes.get("Principled BSDF")
if bsdf_bg:
    set_input(bsdf_bg, "Base Color", (0.048, 0.062, 0.082, 1.0))
    set_input(bsdf_bg, "Roughness", 0.95)
backdrop.data.materials.append(mat_backdrop)

# 6. Plinth Material
mat_plinth = bpy.data.materials.new(name="PlinthMat")
bsdf_pl = mat_plinth.node_tree.nodes.get("Principled BSDF")
if bsdf_pl:
    set_input(bsdf_pl, "Base Color", (0.11, 0.12, 0.15, 1.0))
    set_input(bsdf_pl, "Metallic", 0.88)
    set_input(bsdf_pl, "Roughness", 0.25)
    set_input(bsdf_pl, "Coat Weight", 0.6)
    set_input(bsdf_pl, "Coat Roughness", 0.12)

# Helper: Create Shader Materials
def create_material(name, setup_fn):
    mat = bpy.data.materials.new(name=name)
    setup_fn(mat, mat.node_tree)
    return mat

# Material 1: Ruby Metallic Car Paint (Dual-Lobe GGX Clearcoat)
def setup_car_paint(mat, tree):
    b = tree.nodes.get("Principled BSDF")
    set_input(b, "Base Color", (0.68, 0.015, 0.025, 1.0))
    set_input(b, "Metallic", 0.85)
    set_input(b, "Roughness", 0.22)
    set_input(b, "Coat Weight", 1.0)
    set_input(b, "Coat Roughness", 0.03)
    set_input(b, "Coat IOR", 1.50)

# Material 2: Jade & Carrara Marble (Subsurface Scattering)
def setup_jade_sss(mat, tree):
    b = tree.nodes.get("Principled BSDF")
    set_input(b, "Base Color", (0.08, 0.88, 0.74, 1.0))
    set_input(b, "Subsurface Weight", 0.85)
    set_input(b, "Subsurface Radius", (0.7, 1.2, 0.9))
    set_input(b, "Subsurface Scale", 0.25)
    set_input(b, "Subsurface IOR", 1.61)
    set_input(b, "Roughness", 0.08)
    set_input(b, "IOR", 1.61)

# Material 3: Dielectric Safety Glass (Snell Refraction & Fresnel)
def setup_dielectric_glass(mat, tree):
    b = tree.nodes.get("Principled BSDF")
    set_input(b, "Base Color", (0.92, 0.96, 0.95, 1.0))
    set_input(b, "Transmission Weight", 1.0)
    set_input(b, "Roughness", 0.005)
    set_input(b, "IOR", 1.52)

# Material 4: Imperial Magenta Velvet (Charlie Sheen BRDF)
def setup_magenta_velvet(mat, tree):
    b = tree.nodes.get("Principled BSDF")
    set_input(b, "Base Color", (0.60, 0.02, 0.42, 1.0))
    set_input(b, "Roughness", 0.85)
    set_input(b, "Sheen Weight", 1.0)
    set_input(b, "Sheen Roughness", 0.35)
    set_input(b, "Sheen Tint", (0.95, 0.35, 0.85, 1.0))

# Material 5: Weathered Industrial Rust & Metal (Procedural FBM)
def setup_weathered_rust(mat, tree):
    b = tree.nodes.get("Principled BSDF")
    tex = tree.nodes.new(type='ShaderNodeTexNoise')
    tex.inputs['Scale'].default_value = 7.5
    tex.inputs['Detail'].default_value = 8.0
    tex.inputs['Roughness'].default_value = 0.65
    
    ramp = tree.nodes.new(type='ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].position = 0.42
    ramp.color_ramp.elements[0].color = (0.24, 0.26, 0.30, 1.0)
    ramp.color_ramp.elements[1].position = 0.58
    ramp.color_ramp.elements[1].color = (0.85, 0.36, 0.12, 1.0)
    
    bump = tree.nodes.new(type='ShaderNodeBump')
    bump.inputs['Strength'].default_value = 0.45
    bump.inputs['Distance'].default_value = 0.1
    
    tree.links.new(tex.outputs['Fac'], ramp.inputs['Fac'])
    tree.links.new(ramp.outputs['Color'], b.inputs['Base Color'])
    tree.links.new(tex.outputs['Fac'], bump.inputs['Height'])
    tree.links.new(bump.outputs['Normal'], b.inputs['Normal'])
    set_input(b, "Roughness", 0.65)
    set_input(b, "Metallic", 0.45)

# Material 6: Polished 24K Gold (Noble Conductor)
def setup_polished_gold(mat, tree):
    b = tree.nodes.get("Principled BSDF")
    set_input(b, "Base Color", (1.00, 0.78, 0.32, 1.0))
    set_input(b, "Metallic", 1.0)
    set_input(b, "Roughness", 0.04)

# Material 7: Polished Chrome Mirror (Ideal Mirror Conductor)
def setup_chrome_mirror(mat, tree):
    b = tree.nodes.get("Principled BSDF")
    set_input(b, "Base Color", (0.96, 0.96, 0.98, 1.0))
    set_input(b, "Metallic", 1.0)
    set_input(b, "Roughness", 0.002)

# Material 8: Alpine Glacial Water (Liquid Dielectric)
def setup_alpine_water(mat, tree):
    b = tree.nodes.get("Principled BSDF")
    set_input(b, "Base Color", (0.15, 0.42, 0.58, 1.0))
    set_input(b, "Transmission Weight", 0.95)
    set_input(b, "Roughness", 0.02)
    set_input(b, "IOR", 1.333)

# Material 9: High Mountain Snow & Glacial Ice (Subsurface Glint)
def setup_mountain_snow(mat, tree):
    b = tree.nodes.get("Principled BSDF")
    set_input(b, "Base Color", (0.92, 0.95, 0.98, 1.0))
    set_input(b, "Subsurface Weight", 0.45)
    set_input(b, "Subsurface Radius", (1.0, 1.0, 1.0))
    set_input(b, "Subsurface Scale", 0.15)
    set_input(b, "Roughness", 0.45)

# Material 10: Exposed Granite Cliff Rock (Mineral Roughness)
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

# Material 11: Translucent Conifer Pine Foliage (Foliage SSS)
def setup_pine_foliage(mat, tree):
    b = tree.nodes.get("Principled BSDF")
    set_input(b, "Base Color", (0.08, 0.32, 0.12, 1.0))
    set_input(b, "Subsurface Weight", 0.65)
    set_input(b, "Subsurface Radius", (0.25, 0.65, 0.15))
    set_input(b, "Subsurface Scale", 0.20)
    set_input(b, "Roughness", 0.40)

# Material 12: Weathered Pine Bark & Timber (Organic Wood Grain)
def setup_timber_bark(mat, tree):
    b = tree.nodes.get("Principled BSDF")
    tex = tree.nodes.new(type='ShaderNodeTexWave')
    tex.inputs['Scale'].default_value = 6.0
    tex.inputs['Distortion'].default_value = 12.0
    tex.inputs['Detail'].default_value = 8.0
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

material_setups = [
    ("Ruby Metallic Car Paint", setup_car_paint),
    ("Jade & Marble Subsurface", setup_jade_sss),
    ("Dielectric Safety Glass", setup_dielectric_glass),
    ("Imperial Magenta Velvet", setup_magenta_velvet),
    ("Weathered Rust & Bronze", setup_weathered_rust),
    ("Polished 24K Gold", setup_polished_gold),
    ("Polished Chrome Mirror", setup_chrome_mirror),
    ("Alpine Glacial Water", setup_alpine_water),
    ("Mountain Snow & Ice", setup_mountain_snow),
    ("Exposed Granite Rock", setup_granite_rock),
    ("Translucent Pine Foliage", setup_pine_foliage),
    ("Weathered Pine Bark & Timber", setup_timber_bark),
]

# 7. Build 4x3 Grid of Suzanne Heads & Plinths with Exact Pixel Alignment
# Screen Resolution: 3840 x 2160 (16:9)
# World Dimensions: 18.0 wide x 10.125 high
W = 3840.0
H = 2160.0
world_w = ortho_scale
world_h = ortho_scale * (H / W) # 10.125

col_px_centers = [480, 1440, 2400, 3360]
row_px_centers = [460, 1130, 1800]

mat_instances = []
for name, fn in material_setups:
    mat_instances.append(create_material(name, fn))

for idx in range(12):
    c = idx % 4
    r = idx // 4
    
    cx = col_px_centers[c]
    ry = row_px_centers[r]
    
    # Mathematical conversion from screen pixels to 3D world coordinates
    head_px_y = ry - 55
    plinth_px_y = ry + 140
    
    wx = (cx / W - 0.5) * world_w
    wz_head = (0.5 - head_px_y / H) * world_h
    wz_plinth = (0.5 - plinth_px_y / H) * world_h
    
    # Add Display Plinth (Beveled Cylinder)
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=64,
        radius=0.92,
        depth=0.32,
        location=(wx, 0.0, wz_plinth)
    )
    plinth = bpy.context.object
    plinth.data.materials.append(mat_plinth)
    plinth.data.shade_smooth()
    
    # Add subtle bevel to plinth edge for sleek studio rim highlight
    bm = plinth.modifiers.new(name="Bevel", type='BEVEL')
    bm.width = 0.035
    bm.segments = 3
    
    # Add Suzanne Head
    bpy.ops.mesh.primitive_monkey_add(location=(wx, 0.0, wz_head))
    suzanne = bpy.context.object
    suzanne.scale = (0.88, 0.88, 0.88)
    suzanne.rotation_euler = (math.radians(-10.0), math.radians(24.0), math.radians(-12.0))
    suzanne.data.shade_smooth()
    
    # Subdivision surface modifier (level 2)
    subsurf = suzanne.modifiers.new(name="Subsurf", type='SUBSURF')
    subsurf.levels = 2
    subsurf.render_levels = 2
    
    # Assign Material
    suzanne.data.materials.append(mat_instances[idx])

# 8. Render to output PNG
output_raw = "/home/naoki/Development/GPUBench/renders/raw_material_gallery.png"
os.makedirs(os.path.dirname(output_raw), exist_ok=True)
scene.render.filepath = output_raw
print(f"Rendering 4K frame to {output_raw}...")
bpy.ops.render.render(write_still=True)
t_end = time.time()
print(f"=== Blender Cycles render completed in {t_end - t_start:.2f} s ===")
"""

def generate_blender_render():
    blend_script_path = "/home/naoki/.gemini/antigravity/brain/b26a5e7f-321e-4ad2-a16c-02af680fa03c/scratch/run_gallery_render.py"
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
    raw_path = "renders/raw_material_gallery.png"
    out_path = "renders/render_material_gallery_4k.png"
    artifact_path = "/home/naoki/.gemini/antigravity/brain/b26a5e7f-321e-4ad2-a16c-02af680fa03c/render_material_gallery_4k.png"
    
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Missing raw render file: {raw_path}")
    
    img = Image.open(raw_path).convert("RGBA")
    w, h = img.size
    
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Fonts
    font_title = ImageFont.truetype("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf", 36)
    font_sub = ImageFont.truetype("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf", 20)
    font_badge = ImageFont.truetype("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf", 18)
    font_mat = ImageFont.truetype("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf", 21)
    font_shader = ImageFont.truetype("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf", 15)
    font_scene = ImageFont.truetype("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Oblique.ttf", 14)
    
    # Timestamp
    tz_abbr = time.strftime("%Z")
    if tz_abbr.startswith("+") or tz_abbr.startswith("-"):
        tz_abbr = "UTC" + tz_abbr
    timestamp_str = datetime.now().astimezone().strftime(f"%Y-%m-%d %H:%M:%S {tz_abbr}")
    
    # Top Banner Background
    draw.rectangle([(0, 0), (w, 140)], fill=(12, 16, 24, 235))
    draw.line([(0, 140), (w, 140)], fill=(40, 55, 80, 255), width=2)
    
    # Title & Subtitle
    draw.text((60, 26), "GPUBENCH MATERIAL SHADER SHOWCASE • 4K ORTHOGRAPHIC GALLERY", fill=(255, 255, 255), font=font_title)
    draw.text((60, 80), f"AMD Radeon AI PRO R9700 (RADV GFX1201 / Vulkan 1.4) • Cycles HIP Ground-Truth Material References • Rendered: {timestamp_str}", fill=(160, 185, 220), font=font_sub)
    
    # Top-Right Timestamp Badge
    badge_text = f"RENDERED: {timestamp_str}"
    bbox = draw.textbbox((0, 0), badge_text, font=font_badge)
    bw = bbox[2] - bbox[0] + 32
    bh = bbox[3] - bbox[1] + 18
    bx = w - bw - 60
    by = 45
    draw.rectangle([(bx, by), (bx + bw, by + bh)], fill=(20, 30, 48, 220), outline=(80, 140, 220, 255), width=2)
    draw.text((bx + 16, by + 9), badge_text, fill=(180, 220, 255), font=font_badge)
    
    # Card metadata for 12 materials
    cards = [
        {
            "name": "01. RUBY METALLIC CAR PAINT",
            "shader": "Dual-Lobe Cook-Torrance GGX Clearcoat (Base α=0.22, Coat α=0.03)",
            "scene": "Scene: Showroom Studio & Indoor Atrium (Knot Band 0)",
            "color": (255, 90, 100)
        },
        {
            "name": "02. JADE & MARBLE SSS",
            "shader": "Volumetric Subsurface Scattering (Exp Depth Absorb + Soft GGX, IOR 1.61)",
            "scene": "Scene: Indoor Atrium (Knot Band 1, Columns, Satellite Sphere)",
            "color": (90, 255, 210)
        },
        {
            "name": "03. DIELECTRIC SAFETY GLASS",
            "shader": "Snell's Law Refraction, Bounded Shell Loop, Schlick Fresnel (IOR 1.52)",
            "scene": "Scene: Showroom Studio (ToyCar Windshield & Suzanne)",
            "color": (150, 225, 255)
        },
        {
            "name": "04. IMPERIAL MAGENTA VELVET",
            "shader": "Charlie Sheen Microfiber BRDF (Grazing Sheen Tint, Roughness 0.85)",
            "scene": "Scene: Showroom Studio & Indoor Atrium (Knot Band 3)",
            "color": (255, 120, 220)
        },
        {
            "name": "05. WEATHERED RUST & BRONZE",
            "shader": "Layered 3D FBM Procedural (Pitted Terracotta Rust, Steel, Patina)",
            "scene": "Scene: Showroom Studio (Knot Band 4) & Atrium Enclosing Walls",
            "color": (255, 160, 90)
        },
        {
            "name": "06. POLISHED 24K GOLD",
            "shader": "High-Conductivity Complex Fresnel Conductor (F0 ≈ 0.88, Roughness 0.04)",
            "scene": "Scene: Indoor Atrium (Coffered Ceiling & Pedestal)",
            "color": (255, 220, 90)
        },
        {
            "name": "07. POLISHED CHROME MIRROR",
            "shader": "Ideal Specular Metallic Conductor (F0 = 0.98, Roughness α < 0.005)",
            "scene": "Scene: Showroom Studio (Knot Band 2)",
            "color": (220, 230, 255)
        },
        {
            "name": "08. ALPINE GLACIAL WATER",
            "shader": "Fluid Dielectric (Sky Fresnel Reflection, Sun Glint, Volumetric Depth)",
            "scene": "Scene: Outdoor Landscape (Alpine Lake & River Water Plane)",
            "color": (110, 210, 255)
        },
        {
            "name": "09. HIGH MOUNTAIN SNOW & ICE",
            "shader": "Subsurface Wrapped Diffuse with Crystalline Specular Glint",
            "scene": "Scene: Outdoor Landscape (High Peaks & Glacial Ridges)",
            "color": (215, 235, 255)
        },
        {
            "name": "10. EXPOSED GRANITE CLIFF ROCK",
            "shader": "Rough Mineral Lambertian with Micro-facet Crevice Self-Shadowing",
            "scene": "Scene: Outdoor Landscape & Open-World Forest",
            "color": (195, 205, 215)
        },
        {
            "name": "11. TRANSLUCENT PINE FOLIAGE",
            "shader": "Double-Sided Thin Transmission & Needle Subsurface Scattering",
            "scene": "Scene: Outdoor Landscape & Open-World Forest (Pine Needles)",
            "color": (120, 240, 140)
        },
        {
            "name": "12. WEATHERED PINE BARK & TIMBER",
            "shader": "Fibrous Organic Wood Diffuse with Structural Longitudinal Grain",
            "scene": "Scene: Outdoor Landscape & Open-World Forest (Trunk & Bridge)",
            "color": (220, 180, 130)
        }
    ]
    
    col_px_centers = [480, 1440, 2400, 3360]
    row_px_centers = [460, 1130, 1800]
    
    card_w = 860
    card_h = 92
    
    for idx, card in enumerate(cards):
        c = idx % 4
        r = idx // 4
        
        cx = col_px_centers[c]
        ry = row_px_centers[r]
        
        # Exact mathematical position: card top sits just 12px below plinth bottom
        card_py = ry + 235
        left = cx - card_w // 2
        top = card_py - card_h // 2
        
        # Draw translucent plaque card
        draw.rounded_rectangle(
            [(left, top), (left + card_w, top + card_h)],
            radius=10,
            fill=(14, 18, 26, 220),
            outline=(45, 60, 85, 240),
            width=2
        )
        
        # Color accent strip on left
        draw.rounded_rectangle(
            [(left + 6, top + 8), (left + 12, top + card_h - 8)],
            radius=3,
            fill=card["color"]
        )
        
        # Text details
        draw.text((left + 24, top + 10), card["name"], fill=card["color"], font=font_mat)
        draw.text((left + 24, top + 39), card["shader"], fill=(210, 220, 235), font=font_shader)
        draw.text((left + 24, top + 64), card["scene"], fill=(130, 160, 200), font=font_scene)
        
    # Composite overlay
    final_img = Image.alpha_composite(img, overlay).convert("RGB")
    final_img.save(out_path, quality=95)
    final_img.save(artifact_path, quality=95)
    print(f"Saved completed 4K showcase image to {out_path} and {artifact_path}")

if __name__ == "__main__":
    generate_blender_render()
    composite_captions()
