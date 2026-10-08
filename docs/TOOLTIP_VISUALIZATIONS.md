# Benchmark Tooltip Visualizations — Implementation

Graphics-pipeline benchmarks (ray tracing, BVH, materials, ROP) manipulate
spatial data that is hard to convey with text alone. This document describes
the shipped implementation of hover-tooltip visualizations for those
workloads: a static pre-baked thumbnail, a procedural vector schematic, or a
cross-link into the interactive Ray Tracing Viewport.

> Supersedes the draft architecture spec in
> `docs/archive/TOOLTIP_PIPELINE_VISUALIZATIONS.md`.

---

## 1. Overview

Hovering a benchmark workload in the **Benchmark Suite** tab (or a benchmark
row in the **Results Scorecard**) shows a tooltip:

```
+-------------------------------------------------------------+
| Coherent Triangles                                          |
| Subcategory: Hardware BVH Traversal | Metric: MRays/s       |
|-------------------------------------------------------------|
| Measures raw hardware ray-triangle test speed with          |
| parallel, orderly rays                                      |
|-------------------------------------------------------------|
|  +-------------------------------------------------------+  |
|  |                                                       |  |
|  |          [ VISUALIZATION CANVAS  340 x 191 ]          |  |
|  |        (400x225 thumbnail or vector diagram)          |  |
|  |                                                       |  |
|  +-------------------------------------------------------+  |
|  BVH traversal cost per pixel: blue = shallow, red = deep   |
|  [ Open in Ray Tracing Viewport  -> ]                       |
+-------------------------------------------------------------+
```

Only workloads with graphics-pipeline data get a visualization (57 of the
~150 workloads). Compute, Memory, and Host System workloads keep plain-text
tooltips.

Two rendering tiers are used:

| Tier | Used for | Mechanism | Cost |
|:-----|:---------|:----------|:-----|
| **1 — Static thumbnails** | Scene-dependent captures: BVH heatmaps, G-Buffer passes, shadows, GI, multi-light, materials, mesh wireframes | 400×225 PNGs pre-rendered and shipped in `assets/thumbnails/`, lazily loaded on first hover via `GuiApp::getOrLoadTexture()` and cached as Vulkan descriptor sets | ~30–160 KB per PNG on disk, ~360 KB VRAM each, zero GPU dispatch on hover |
| **2 — Procedural schematics** | Algorithmic/mathematical tests: ray divergence cones, traversal orderings, wave ballot compaction, TLAS hierarchy, intersection primitives, ROP fill | Drawn on the fly with Dear ImGui `ImDrawList` primitives | 0 KB assets, infinitely crisp at any UI scale |

A third, *live post-run capture* tier was considered in the draft spec but
not implemented: the Ray Tracing Viewport already shows the user's actual
hardware output, and the cross-link button (below) jumps straight to it.

---

## 2. Asset Pipeline

### 2.1 Thumbnail generation

`scripts/make_thumbnails.py` downsamples the shipped full-resolution assets
to **400×225** (16:9, Lanczos) with Pillow:

```
python3 scripts/make_thumbnails.py            # regenerate all 36 thumbnails
python3 scripts/make_thumbnails.py --check    # CI-style staleness check
```

| Source | Output | Count |
|:-------|:-------|:------|
| `renders/render_<scene>_<stage>.png` (4K pipeline captures) | `assets/thumbnails/thumb_<scene>_<stage>.png` | 4 scenes × 8 stages = 32 |
| `docs/images/geometry_showroom_wireframe.png` | `thumb_blas_wireframe.png` | 1 |
| `docs/images/geometry_alpha_layers.png` | `thumb_alpha_layers.png` | 1 |
| `docs/images/material_lineup.png` | `thumb_material_lineup.png` | 1 |
| `docs/images/realistic_scene_material_range.png` | `thumb_material_range.png` | 1 |

Total payload ≈ **3 MB**. The generated files are committed and ship with the
package (see §7).

Scenes: `showroom`, `indoor`, `forest`, `outdoor`. Stages: `stage1_bvh`,
`stage2_primary`, `stage3_shadow`, `stage4_rtao`, `stage5_direct`,
`stage6_indirect`, `stage7_final`, `multilight_128_dgc`.

Note: `thumb_<scene>_stage4_rtao.png` and `thumb_material_range.png` are
generated for parity with the Ray Tracing Viewport's asset set but are not
referenced by any tooltip today (no RTAO-specific workload exists); they cost
~220 KB and make future additions trivial.

### 2.2 Scene-relative assets

Scene-dependent thumbnails carry a `{scene}` token in their metadata path,
e.g. `assets/thumbnails/thumb_{scene}_stage1_bvh.png`. At render time
`renderBenchmarkTooltip()` substitutes the active scene via
`currentSceneTag()` (or an explicit `sceneOverride` for scorecard rows that
belong to a specific scenario): the scene chosen in the Ray Tracing Scenario
selector, or the canonical Sponza/`indoor` scene when `all` is selected. This
keeps the tooltip in sync with what the test actually runs on.

### 2.3 Loading & caching

No new GPU code was required. The tooltip reuses the existing pipeline:

```
getOrLoadTexture(relPath)          // GuiApp.cpp — m_textureCache keyed by path
  └─ VulkanContext::loadTextureFromFile()   // stb_image decode, candidate paths
       └─ createTextureRgba()               // VkImage + sampler + descriptor set
            └─ drawList->AddImage(reinterpret_cast<ImTextureID>(tex.descriptorSet), ...)
```

Loading is lazy (first hover only) and cached for the session; the same cache
is shared with the Ray Tracing Viewport, so textures shown in both places
cost VRAM once.

---

## 3. Data Model

`cpp_src/gui/GuiApp.h`:

```cpp
struct BenchmarkVisualization {
    std::string assetPath;           // Tier 1, may contain "{scene}"
    std::string diagramId;           // Tier 2, e.g. "cone_divergence_45"
    std::string caption;             // one-line legend under the canvas
    float aspectRatio{16.0f / 9.0f};
    int linkedViewportMode{-1};      // -1 | 0 Scenes | 1 Passes | 2 Materials | 3 Geometry
    int linkedViewportSubIndex{0};   // pass / material / geometry index
    bool hasVisualization() const { return !assetPath.empty() || !diagramId.empty(); }
};

struct BenchmarkItem {
    // ... existing fields ...
    BenchmarkVisualization viz;      // optional, attached by metadata pass
};
```

Exactly one of `assetPath` / `diagramId` is set per workload.

---

## 4. Metadata Assignment

`GuiApp::applyVisualizationMetadata()` runs at the end of
`initializeBenchmarkCategories()` and attaches descriptors from a declarative
table keyed by **(engineId, configIndex)** — this keeps the large category
initializer lists untouched and mirrors the benchmark engine's config
numbering (`RaySchedulingBench::GetConfigName`, etc.):

```cpp
struct VizEntry { const char* engineId; int configIndex; BenchmarkVisualization viz; };
const VizEntry entries[] = {
    {"RayASBuild", 0, {"assets/thumbnails/thumb_blas_wireframe.png", "",
        "Vehicle mesh enclosed by hierarchical AABBs - ...", 16.0f/9.0f, 3, 0}},
    {"RayDivergence", 4, {"", "cone_divergence_0",
        "Perfectly parallel beam (mirror): ...", 16.0f/9.0f, -1, 0}},
    // ... 57 entries total
};
```

At startup the GUI logs coverage, and warns per unmatched entry:

```
[GPUBench GUI] Tooltip visualizations: 57/57 workloads mapped
```

### 4.1 Mapping table (57 workloads)

| Engine | Configs | Tier | Visualization |
|:-------|:--------|:-----|:--------------|
| `RayASBuild` | 0–4 (BLAS build/update) | 1 | `thumb_blas_wireframe.png` → Geometry & BVH view 0 |
| `RayASBuild` | 5 (TLAS: Corridor 20K) | 2 | `tlas_corridor_20k` → Geometry & BVH view 2 |
| `RayASBuild` | 6 (TLAS: Jungle 50K) | 2 | `tlas_jungle_50k` → Geometry & BVH view 2 |
| `RayASBuild` | 7 (TLAS: Open World 200K) | 2 | `tlas_openworld_200k` → Geometry & BVH view 2 |
| `RayScheduling` | 0–2 (Material) | 1 | `thumb_material_lineup.png` → PBR Materials view 0 |
| `RayScheduling` | 3–5, 23–24 (Path tracing) | 1 | `thumb_{scene}_stage6_indirect.png` → Pipeline Passes 5 |
| `RayScheduling` | 6–8 (Incoherent GI) | 2 | `incoherent_{naive,ser,dgc}` → Pipeline Passes 5 |
| `RayScheduling` | 9–11 (Full frame) | 1 | `thumb_{scene}_stage7_final.png` → Pipeline Passes 6 |
| `RayScheduling` | 12 (Scanline) | 2 | `traversal_scanline` → Pipeline Passes 0 |
| `RayScheduling` | 13, 26, 28 (Compaction) | 2 | `wave_ballot_compaction` |
| `RayScheduling` | 14 (Tiled 8×4) | 2 | `traversal_tiled_8x4` → Pipeline Passes 0 |
| `RayScheduling` | 15 (Morton 8×4) | 2 | `traversal_morton_8x4` → Pipeline Passes 0 |
| `RayScheduling` | 16 (Morton 4×8) | 2 | `traversal_morton_4x8` → Pipeline Passes 0 |
| `RayScheduling` | 17, 18, 29, 30 (Primary rays) | 1 | `thumb_{scene}_stage2_primary.png` → Pipeline Passes 1 |
| `RayScheduling` | 19–22 (Shadows) | 1 | `thumb_{scene}_stage3_shadow.png` → Pipeline Passes 2 |
| `RayScheduling` | 27 (Alpha cutout) | 1 | `thumb_alpha_layers.png` → Geometry & BVH view 1 |
| `RayScheduling` | 31, 32 (Multi-light, 1 light) | 1 | `thumb_{scene}_stage5_direct.png` → Pipeline Passes 4 |
| `RayScheduling` | 33, 34 (Multi-light, 128 lights) | 1 | `thumb_{scene}_multilight_128_dgc.png` → Pipeline Passes 4 |
| `RayRawTraversal` | 0 (Coherent tris) | 2 | `bvh_coherent_triangles` |
| `RayRawTraversal` | 1 (Deep boxes) | 2 | `bvh_nested_boxes` |
| `RayIntersect` | 0 (Ray-triangle) | 2 | `intersect_ray_triangle` |
| `RayIntersect` | 1 (Ray-box) | 2 | `intersect_ray_box` |
| `RayAnyHit` | 0 (100% solid baseline) | 2 | `anyhit_100_solid` |
| `RayAnyHit` | 1 (50% cutout stress) | 2 | `anyhit_50_cutout` |
| `RayProcedural` | 0 (AABB spheres) | 2 | `procedural_sphere` |
| `RayDivergence` | 0–4 (90°…0°) | 2 | `cone_divergence_{90,67.5,45,22.5,0}` |

> **Incoherent GI vs Ray Directional Coherence** — these two groups look
> similar but answer different questions, and the tooltips say so explicitly:
> `RayDivergence` is a *controlled cost curve* (synthetic chamber, one
> technique, cone angle swept 0°→90°) — "how much does divergence cost this
> hardware?" — while `Incoherent Ray Tracing` is a *technique comparison*
> (real scenes, full-hemisphere secondary rays, three scheduling techniques)
> — "how much do SER/DGC recover?". Their descriptions cross-reference each
> other, and the incoherent test gets its own mechanism diagrams
> (`incoherent_*`) rather than reusing the static cone, so the reordering
> being tested is actually visible.
| `Pixel Fill Rate` | 0–2 (RGBA8/HDR/blend) | 2 | `rop_fill_{rgba8,hdr,blend}` |

---

## 5. Tooltip Rendering

`GuiApp::renderBenchmarkTooltip(const BenchmarkItem&)` is the single tooltip
implementation, replacing the previous inline blocks in two places:

- **Benchmark Suite** workload rows (checkbox hover, `renderBenchmarkSuitePanel`)
- **Results Scorecard** benchmark column (`renderResultsScorecard`), which now
  matches the suite item by `(engine id, configIndex)` — the result's
  `benchmarkName` is `<id> (<config name>)` for multi-config engines — instead
  of the old fragile display-name scan.

Layout (all sizes go through `s()` so they track UI scale):

1. Workload name (cyan) + `Subcategory | Metric` line
2. Separator, wrapped description (wrap width `s(380)`)
3. Separator, **canvas** `s(340) × s(191.25)` (16:9):
   - Tier 1: letterboxed `AddImage` of the cached thumbnail (dark fill +
     border rect behind; fallback text if the file is missing)
   - Tier 2: `renderProceduralDiagram(diagramId, p0, p1)`
4. Gold caption line (wrapped to canvas width)
5. Optional `SmallButton("Open in Ray Tracing Viewport  ->")`

Unsupported workloads keep their dedicated red/amber tooltip (no viz).

### 5.1 Cross-linking

Clicking the button sets the Ray Tracing Viewport state and flips to its tab:

```cpp
m_rtViewportMode = item.viz.linkedViewportMode;
if (mode == 1) m_rtPassIndex       = item.viz.linkedViewportSubIndex;
if (mode == 2) m_rtMaterialIndex   = item.viz.linkedViewportSubIndex;
if (mode == 3) m_rtGeometryIndex   = item.viz.linkedViewportSubIndex;
m_switchToRtViewport = true;   // consumed in renderRightWorkspace() tab bar
```

`m_switchToRtViewport` mirrors the existing `m_switchToScorecard` mechanism:
the "Ray Tracing Viewport" tab item receives `ImGuiTabItemFlags_SetSelected`
for one frame.

---

## 6. Procedural Diagrams

`GuiApp::renderProceduralDiagram(diagramId, p0, p1)` — pure `ImDrawList`
vector drawing (lines, rects, circles, arcs, text), shared palette, title at
top, footnote at bottom. 21 diagram ids:

| Diagram id | Visual |
|:-----------|:-------|
| `cone_divergence_0` … `cone_divergence_90` | Surface line + normal + 7-ray fan inside the cone, amber angle arc, color shifts blue → amber → red as divergence grows; sub-label (mirror / glossy / semi-glossy / rough / diffuse) |
| `traversal_scanline` | 16×8 pixel grid, row-major path (green start, red end) |
| `traversal_tiled_8x4` | Same grid with 8×4 tile borders; path wraps at tile edges |
| `traversal_morton_8x4` / `_4x8` | 4×4 tile grid visited along a true Morton Z-curve (bit-interleaved codes), numbered dots, mini-raster lines inside tiles showing intra-tile orientation (wide vs tall) |
| `wave_ballot_compaction` | 32-lane wavefront (active/terminated), the 32-bit ballot mask, and the compacted queue packed left; footnote reports active-lane count |
| `incoherent_naive` / `incoherent_ser` / `incoherent_dgc` | Left: 12 color-coded bounce rays scattered from a column of scene hit points (4 direction groups). Right: the technique's result — one undivided dispatch (naive), 2×2 direction bins with parallel arrows (SER), or 4 compacted octant queue rows O0–O3 (DGC) |
| `tlas_hierarchy` | Root → 3 internal nodes → 6 `BLAS #i` leaves |
| `intersect_ray_triangle` | Triangle, ray with hit point, dashed barycentric spokes, `hit (t, u, v)` label |
| `intersect_ray_box` | 2.5D AABB wireframe, ray with `t near` / `t far` markers |
| `procedural_sphere` | Analytic sphere with rim highlight, ray, radial normal |
| `bvh_nested_boxes` | 3-level nested AABBs (1 → 2 → 4) with a ray and per-level step dots |
| `rop_fill_rgba8` / `rop_fill_hdr` / `rop_fill_blend` | Framebuffer with solid quad + scanline arrow / LDR→HDR gradient / two overlapping translucent quads with `src + dst` overlap marker |

Implementation notes:

- The project does **not** define `IMGUI_DEFINE_MATH_OPERATORS`, so the
  diagram code uses local `add`/`sub`/`mul` lambdas instead of ImVec2
  operators.
- All coordinates derive from the canvas rect and `s()`, so diagrams scale
  losslessly with the UI (verified at 1.5×).
- Unknown ids render a visible "Unknown diagram: <id>" fallback rather than
  failing silently.

---

## 7. Packaging

`CMakeLists.txt` installs the thumbnails for RPM/Flatpak/Debian builds:

```cmake
install(DIRECTORY assets/thumbnails/
        DESTINATION share/gpubench/assets/thumbnails
        FILES_MATCHING PATTERN "*.png")
```

`VulkanContext::loadTextureFromFile()` already probes
`/usr/share/gpubench/<path>` (and `/usr/local/share/gpubench/<path>`), so the
`assets/thumbnails/...` paths resolve unchanged in installed builds.

> Pre-existing gap (not introduced here): the Ray Tracing Viewport's full-res
> `renders/*.png` and `docs/images/*.png` assets are not installed either, so
> packaged builds show "image not found" placeholders in that tab. The
> tooltips are unaffected because they use the small shipped thumbnails.

---

## 8. Performance

- **Lazy load**: textures are decoded (stb_image) and uploaded only on first
  hover; `m_textureCache` deduplicates across tooltips and the RT viewport.
- **Bounded VRAM**: 36 thumbnails × 400×225×4 B ≈ **13 MB** worst case if
  every tooltip is hovered in one session; typical sessions touch a handful.
- **No GPU dispatch on hover**: Tier 1 is a texture blit, Tier 2 is CPU-side
  ImDrawList geometry (a few hundred primitives per diagram).
- **Decode cost**: 400×225 PNGs decode in well under a frame; the first hover
  on a given asset may cost a few ms (stb decode + VkImage upload) and is
  paid once.

---

## 9. Verification

1. **Build**: clean compile, zero warnings (`-Wformat-truncation` clean).
2. **Mapping coverage**: GUI startup logs `Tooltip visualizations: 57/57
   workloads mapped`; any unmatched table entry logs a named warning.
3. **Asset paths**: all 31 distinct metadata paths (28 scene-expanded + 4
   static, minus 1 shared) resolve to files on disk — checked across all four
   scene tags.
4. **Visual parity**: the drawing code was rendered headlessly (ImGui +
   software triangle rasterizer, no GPU/display) to PNGs and inspected:
   all 21 procedural diagrams plus full-tooltip composites (texture and
   procedural variants, cross-link button, 1.5× UI scale). One ImGui quirk
   surfaced and understood: a *newly created* tooltip window is hidden for
   one frame while it measures its size (16 ms at 60 Hz — imperceptible in
   real use; the harness must render two frames to capture a tooltip).
5. **Runtime**: GUI launches and runs stably on the target system
   (RADV STRIX_HALO, Vulkan 1.4).

---

## 10. Extending

- **New workload with a still image**: add the thumbnail to
  `scripts/make_thumbnails.py` (`source_list()`), regenerate, add one
  `VizEntry` row.
- **New workload with a schematic**: add a `diagramId` to the entry and a
  branch in `renderProceduralDiagram`.
- **New scene**: add its tag to `SCENES` in the script (the benchmark engine
  must already produce `renders/render_<tag>_<stage>.png` captures).
