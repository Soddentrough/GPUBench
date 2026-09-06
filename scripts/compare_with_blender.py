import argparse
import os
from PIL import Image, ImageDraw, ImageFont

def make_comparison(scenario="forest"):
    if scenario.lower() in ["forest", "aaa_forest", "aaa_outdoor_forest"]:
        p_gpu = "renders/render_forest_worklist_dgc.png"
        p_blender = "renders/render_forest_cycles_reference.png"
        out_path = "renders/render_forest_blender_comparison.png"
        title_left = "GPUBench Real-Time Vulkan Pipeline (Open-World Forest)"
        sub_left = "Decoupled Work Lists / DGC (55.2 FPS, 18.13 ms @ 4K 3840x2160)"
        title_right = "Blender Cycles Reference Render (Open-World Forest)"
        sub_right = "Cycles HIP RT on GPU 1 (AMD Radeon AI PRO R9700, 8.43 s @ 64 spp, OIDN)"
    else:
        p_gpu = "renders/render_indoor_worklist_dgc.png"
        p_blender = "renders/render_sponza_cycles_reference.png"
        out_path = "renders/render_blender_comparison.png"
        title_left = "GPUBench Real-Time Vulkan Pipeline (Crytek Sponza)"
        sub_left = "Decoupled Work Lists / DGC (338.3 FPS, 2.96 ms @ 4K 3840x2160)"
        title_right = "Blender Cycles Reference Render (Crytek Sponza)"
        sub_right = "Ground-Truth Offline Path Tracer (Full Multi-Bounce GI, Denoised)"
    
    if not os.path.exists(p_gpu):
        raise FileNotFoundError(f"Missing GPU render file: {p_gpu}")
    if not os.path.exists(p_blender):
        raise FileNotFoundError(f"Missing Blender reference file: {p_blender}")
    
    im_gpu = Image.open(p_gpu)
    im_blend = Image.open(p_blender).convert("RGB")
    
    # im_gpu frame aspect ratio check
    w_gpu_orig, h_gpu_orig = im_gpu.size
    target_aspect = 16.0 / 9.0
    if h_gpu_orig > int(w_gpu_orig / target_aspect) + 10:
        im_gpu_frame = im_gpu.crop((0, 0, w_gpu_orig, int(w_gpu_orig / target_aspect)))
    else:
        im_gpu_frame = im_gpu
    
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
    draw.text((border + 8, 10), title_left, fill=(52, 211, 153), font=font_title)
    draw.text((border + 8, 36), sub_left, fill=(156, 163, 175), font=font_sub)
    
    # Title Right
    x_right = border * 2 + target_w_gpu
    draw.text((x_right + 8, 10), title_right, fill=(96, 165, 250), font=font_title)
    draw.text((x_right + 8, 36), sub_right, fill=(156, 163, 175), font=font_sub)
    
    # Paste images
    comp.paste(im_gpu_s, (border, header_h + border))
    comp.paste(im_blend_s, (border * 2 + target_w_gpu, header_h + border))
    
    # Accent dividing line
    draw.line([(border * 2 + target_w_gpu - 3, 0), (border * 2 + target_w_gpu - 3, total_h)], fill=(40, 48, 60), width=2)
    
    comp.save(out_path, quality=95)
    print(f"Exported Blender comparison to {out_path} ({total_w}x{total_h})")
    
    artifact_dir = os.environ.get("ARTIFACT_DIR")
    if artifact_dir and os.path.isdir(artifact_dir):
        comp.save(os.path.join(artifact_dir, os.path.basename(out_path)), quality=95)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate side-by-side comparison with Blender Cycles ground truth.")
    parser.add_argument("-s", "--scenario", default="forest", help="Scenario name (forest or indoor)")
    args = parser.parse_args()
    make_comparison(args.scenario)
