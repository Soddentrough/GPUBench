#!/usr/bin/env python3
import os
import sys
from PIL import Image

def process_scene(scene_tag):
    if scene_tag == "outdoor":
        p1 = "renders/render_outdoor_traditional_megakernel.png"
        p2 = "renders/render_outdoor_worklist_dgc.png"
        p3 = "renders/render_outdoor_difference_heatmap.png"
        out_names = ["render_outdoor_comparison_triptych.png"]
    elif scene_tag == "indoor":
        p1 = "renders/render_indoor_traditional_megakernel.png"
        p2 = "renders/render_indoor_worklist_dgc.png"
        p3 = "renders/render_indoor_difference_heatmap.png"
        out_names = ["render_indoor_comparison_triptych.png", "render_comparison_triptych.png"]
    else:
        p1 = "renders/render_traditional_megakernel.png"
        p2 = "renders/render_worklist_dgc.png"
        p3 = "renders/render_difference_heatmap.png"
        out_names = ["render_comparison_triptych.png"]

    if not (os.path.exists(p1) and os.path.exists(p2) and os.path.exists(p3)):
        p1 = "renders/render_traditional_megakernel.png"
        p2 = "renders/render_worklist_dgc.png"
        p3 = "renders/render_difference_heatmap.png"
        if not (os.path.exists(p1) and os.path.exists(p2) and os.path.exists(p3)):
            print(f"Skipping triptych for {scene_tag}: source images not found.")
            return

    im1 = Image.open(p1)
    im2 = Image.open(p2)
    im3 = Image.open(p3)

    w, h = im1.size
    target_h = 720
    target_w = int(w * (target_h / h))

    im1_s = im1.resize((target_w, target_h), Image.Resampling.LANCZOS)
    im2_s = im2.resize((target_w, target_h), Image.Resampling.LANCZOS)
    im3_s = im3.resize((target_w, target_h), Image.Resampling.LANCZOS)

    triptych = Image.new("RGB", (target_w * 3 + 16, target_h + 8), (15, 18, 24))
    triptych.paste(im1_s, (4, 4))
    triptych.paste(im2_s, (target_w + 8, 4))
    triptych.paste(im3_s, (target_w * 2 + 12, 4))

    artifact_dir = os.environ.get("ARTIFACT_DIR")
    for name in out_names:
        out_path = os.path.join("renders", name)
        triptych.save(out_path, quality=95)
        print(f"Saved triptych to {out_path} ({triptych.size[0]}x{triptych.size[1]})")
        if artifact_dir and os.path.isdir(artifact_dir):
            triptych.save(os.path.join(artifact_dir, name), quality=95)

if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "indoor"
    if tag == "all":
        process_scene("indoor")
        process_scene("outdoor")
    else:
        process_scene(tag)
