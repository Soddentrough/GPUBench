#!/usr/bin/env python3
"""
GPUBench Tooltip Thumbnail Generator

Pre-renders small 400x225 (16:9) PNG thumbnails of the graphics-pipeline
visualizations shown in benchmark hover tooltips.

Sources (shipped at full resolution for the Ray Tracing Viewport):
  - renders/render_<scene>_<stage>.png        (4K pipeline stage captures)
  - docs/images/*.png                         (geometry / material reference art)

Output (shipped, committed, lazily loaded by the GUI on first hover):
  - assets/thumbnails/thumb_<scene>_<stage>.png
  - assets/thumbnails/thumb_<name>.png

Usage:
  python3 scripts/make_thumbnails.py            # generate all
  python3 scripts/make_thumbnails.py --check    # verify outputs are current
"""

import argparse
import os
import sys

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "assets", "thumbnails")

THUMB_W, THUMB_H = 400, 225

SCENES = ["showroom", "indoor", "forest", "outdoor"]
STAGES = [
    "stage1_bvh",
    "stage2_primary",
    "stage3_shadow",
    "stage4_rtao",
    "stage5_direct",
    "stage6_indirect",
    "stage7_final",
    "multilight_128_dgc",
]

# (source relative path, output thumbnail name)
STATIC_SOURCES = [
    ("docs/images/geometry_showroom_wireframe.png", "thumb_blas_wireframe"),
    ("docs/images/geometry_alpha_layers.png", "thumb_alpha_layers"),
    ("docs/images/material_lineup.png", "thumb_material_lineup"),
    ("docs/images/realistic_scene_material_range.png", "thumb_material_range"),
]


def source_list():
    pairs = []
    for scene in SCENES:
        for stage in STAGES:
            src = os.path.join("renders", f"render_{scene}_{stage}.png")
            dst = f"thumb_{scene}_{stage}"
            pairs.append((src, dst))
    for src, dst in STATIC_SOURCES:
        pairs.append((src, dst))
    return pairs


def make_one(src_rel, dst_name, check_only=False):
    src = os.path.join(ROOT, src_rel)
    dst = os.path.join(OUT_DIR, dst_name + ".png")
    if not os.path.exists(src):
        print(f"  MISSING SOURCE: {src_rel}")
        return False

    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        # All shipped pipeline assets are 16:9; guard against future drift by
        # fitting inside the 400x225 box while preserving aspect ratio.
        scale = min(THUMB_W / w, THUMB_H / h)
        if scale < 1.0:
            new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
            im = im.resize(new_size, Image.LANCZOS)
        elif scale > 1.0:
            new_size = (THUMB_W, THUMB_H)
            im = im.resize(new_size, Image.LANCZOS)

    if check_only:
        if os.path.exists(dst):
            with Image.open(dst) as existing:
                if existing.size == (THUMB_W, THUMB_H):
                    return True
        print(f"  STALE: {dst_name}.png")
        return False

    im.save(dst, "PNG", optimize=True)
    kb = os.path.getsize(dst) / 1024.0
    print(f"  {dst_name}.png  ({im.size[0]}x{im.size[1]}, {kb:.0f} KB)")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="only verify thumbnails exist at the right size")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    pairs = source_list()
    print(f"Generating {len(pairs)} thumbnails -> {os.path.relpath(OUT_DIR, ROOT)}/")
    ok = True
    for src, dst in pairs:
        ok = make_one(src, dst, check_only=args.check) and ok
    if args.check:
        print("All thumbnails present and current." if ok else "Regeneration needed (run without --check).")
    else:
        total = sum(os.path.getsize(os.path.join(OUT_DIR, n + ".png"))
                    for _, n in pairs
                    if os.path.exists(os.path.join(OUT_DIR, n + ".png"))) / (1024 * 1024)
        print(f"Done. Total thumbnail payload: {total:.1f} MB")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
