#!/usr/bin/env python3
import os
from PIL import Image, ImageDraw, ImageFont

def make_comparison():
    p_gpu = "renders/render_worklist_dgc.png"
    p_blender = "docs/images/realistic_scene_material_range.png"
    out_path = "renders/render_blender_comparison.png"
    
    im_gpu = Image.open(p_gpu)
    im_blend = Image.open(p_blender).convert("RGB")
    
    # im_gpu is 1920x1215 (has bottom telemetry slate of 135px), extract the 1920x1080 frame
    im_gpu_frame = im_gpu.crop((0, 0, 1920, 1080))
    
    # Target height 720 for side-by-side display
    target_h = 720
    w_gpu, h_gpu = im_gpu_frame.size
    w_blend, h_blend = im_blend.size
    
    target_w_gpu = int(w_gpu * (target_h / h_gpu))
    target_w_blend = int(w_blend * (target_h / h_blend))
    
    im_gpu_s = im_gpu_frame.resize((target_w_gpu, target_h), Image.Resampling.LANCZOS)
    im_blend_s = im_blend.resize((target_w_blend, target_h), Image.Resampling.LANCZOS)
    
    header_h = 56
    border = 6
    total_w = target_w_gpu + target_w_blend + border * 3
    total_h = target_h + header_h + border * 2
    
    comp = Image.new("RGB", (total_w, total_h), (18, 22, 28))
    draw = ImageDraw.Draw(comp)
    
    # Try default font
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf", 22)
        font_sub = ImageFont.truetype("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf", 14)
    except:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()
        
    # Title Left
    draw.text((border + 8, 10), "GPUBench Real-Time Vulkan Pipeline", fill=(52, 211, 153), font=font_title)
    draw.text((border + 8, 36), "Single-Pass Work Lists / Decoupled Ray Scheduling (1,384 FPS, 0.72 ms)", fill=(156, 163, 175), font=font_sub)
    
    # Title Right
    x_right = border * 2 + target_w_gpu
    draw.text((x_right + 8, 10), "Blender Cycles Reference Render", fill=(96, 165, 250), font=font_title)
    draw.text((x_right + 8, 36), "Ground-Truth Offline Path Tracer (Full GI, Multi-Bounce MIS)", fill=(156, 163, 175), font=font_sub)
    
    # Paste images
    comp.paste(im_gpu_s, (border, header_h + border))
    comp.paste(im_blend_s, (border * 2 + target_w_gpu, header_h + border))
    
    # Accent dividing line
    draw.line([(border * 2 + target_w_gpu - 3, 0), (border * 2 + target_w_gpu - 3, total_h)], fill=(40, 48, 60), width=2)
    
    comp.save(out_path, quality=95)
    print(f"Exported Blender comparison to {out_path} ({total_w}x{total_h})")
    
    artifact_dir = os.environ.get("ARTIFACT_DIR")
    if artifact_dir and os.path.isdir(artifact_dir):
        comp.save(os.path.join(artifact_dir, "render_blender_comparison.png"), quality=95)

if __name__ == "__main__":
    make_comparison()
