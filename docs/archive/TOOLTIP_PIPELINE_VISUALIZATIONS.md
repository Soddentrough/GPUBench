# Graphics Pipeline Benchmark Visualizations in Tooltips
**Architecture & Implementation Specification**

---

## 1. Executive Summary & Purpose

GPUBench provides hover-over tooltips across its benchmark selection suite and results scorecard to explain what each benchmark evaluates and how it executes on GPU hardware. While compute arithmetic tests (e.g. FP32, INT8) can be described with concise textual formulas, benchmarks operating on **graphics and ray tracing pipelines** inherently manipulate spatial, geometric, and visual structures:
- Bounding Volume Hierarchies (BVH) and ray-box / ray-triangle traversal steps.
- G-Buffer surface properties, normals, and depth.
- Multi-bounce path tracing and diffuse global illumination.
- Directional shadow visibility and multi-light clusters.
- Physically Based Rendering (PBR) BSDF material archetypes.
- Ray traversal ordering (Morton Z-curves vs scanline) and angular ray divergence.
- Wavefront stream compaction and active thread voting.

Integrating rich, context-aware visual elements directly into these hover tooltips transforms abstract benchmark titles into clear, visually intuitive hardware demonstrations.

---

## 2. Three-Tier Visualization Architecture

To deliver fluid, responsive UI without hitching the render thread when a user hovers over workloads, GPUBench utilizes a **three-tier visualization model**:

```
                                  +------------------------------------+
                                  |     Benchmark Hover Tooltip        |
                                  +-----------------+------------------+
                                                    |
             +--------------------------------------+------------------------------------+
             |                                      |                                    |
             v                                      v                                    v
+---------------------------+        +------------------------------+        +---------------------------+
| Tier 1: Static Previews   |        | Tier 2: Vector Schematics    |        | Tier 3: Live Capture Hook |
| - BVH Traversal Heatmaps  |        | - Morton Z-Curves (8x4, 4x8) |        | - User GPU Framebuffer    |
| - PBR BSDF Swatches       |        | - Ray Divergence Cones       |        |   Dump (after test run)   |
| - Pipeline Stage G-Buffers|        | - Wave Ballot Compaction     |        | - Exact resolution output |
| - High-res asset cached   |        | - ImDrawList primitives      |        | - Instant parity feedback |
+---------------------------+        +------------------------------+        +---------------------------+
```

### Tier 1: Curated Texture Previews (Static / Pre-baked)
- **Target workloads**: Full-frame path tracing, BVH traversal heatmaps, pipeline stage G-buffers, multi-light evaluation, and PBR material swatches.
- **Mechanism**: Loaded lazily from disk on first hover via `GuiApp::getOrLoadTexture()` and retained in `m_textureCache` as Vulkan descriptor sets (`ImTextureID`).
- **Characteristics**: Instantaneous display, zero GPU dispatch overhead during GUI interaction, consistent authoring quality.

### Tier 2: Procedural Vector Schematics (`ImDrawList`)
- **Target workloads**: Algorithmic, spatial, and mathematical tests (e.g., Morton Z-order curve vs linear scanline, angular cone divergence $0^\circ \to 90^\circ$, wave ballot compaction bitmasks).
- **Mechanism**: Rendered dynamically using Dear ImGui `ImDrawList` vector primitives (lines, circles, bezier curves, rects).
- **Characteristics**: 0 KB disk assets, infinitely crisp across arbitrary DPI / UI scale factors (`m_uiScale`), mathematically exact.

### Tier 3: Dynamic Live Framebuffer Previews (Post-Execution)
- **Target workloads**: Executed ray tracing passes or image export dumps from current test runs.
- **Mechanism**: If a test has been executed in the current session and generated an offscreen framebuffer or PNG in `renders/`, the tooltip prioritizes showing the user's actual hardware output.
- **Characteristics**: Direct user feedback demonstrating real execution parity.

---

## 3. Data Structures & C++ Architecture

### 3.1 Metadata Definition (`cpp_src/gui/GuiApp.h`)

Extend `BenchmarkItem` with a dedicated `BenchmarkVisualization` descriptor:

```cpp
namespace gpubench::gui {

enum class TooltipVizType {
    None,
    TextureThumbnail,      // Static/pre-baked image asset
    ProceduralDiagram,     // Dynamic vector diagram drawn via ImDrawList
    PbrMaterialCard,       // Material preview swatch + BSDF breakdown
    LiveCapture            // Dynamic run output if available
};

struct BenchmarkVisualization {
    TooltipVizType type{TooltipVizType::None};
    
    // Tier 1 Asset Path (relative to application root)
    std::string assetPath; 
    
    // Tier 2 Procedural Diagram Identifier
    std::string diagramId; 
    
    // Descriptive caption and legend text
    std::string caption; 
    
    // Technical parameters for cards (e.g. BSDF parameters, wave width)
    std::string technicalDetails; 
    
    // Display aspect ratio (e.g. 16:9 for frames, 1:1 for material swatches)
    float aspectRatio{16.0f / 9.0f}; 
    
    // Cross-link to full Ray Tracing Viewport mode (-1: None, 0: Scenes, 1: Passes, 2: Materials, 3: Geometry)
    int linkedRtViewportMode{-1};
    int linkedRtSubIndex{0};
};

struct BenchmarkItem {
    std::string id;
    std::string subcategory;
    std::string name;
    std::string category;
    std::string metricType;
    std::string description;
    bool selected{true};
    bool isSupported{true};
    std::string supportReason;
    std::string limitationCategory;
    int configIndex{-1};
    
    // Attached pipeline visualization descriptor
    BenchmarkVisualization viz;
};

} // namespace gpubench::gui
```

---

## 4. Benchmark-to-Visualization Mapping Specification

The following table maps GPUBench graphics and ray tracing workloads to their visualization representations:

| Benchmark Subgroup | Workload Item | Viz Type | Asset Path / Diagram ID | Visual Explanation & Caption |
| :--- | :--- | :--- | :--- | :--- |
| **`RayASBuild`** | BLAS Build / Update (1M–10M Triangles) | `TextureThumbnail` | `docs/images/geometry_showroom_wireframe.png` | 3D wireframe mesh enclosed by hierarchical axis-aligned bounding boxes (AABBs). |
| **`RayASBuild`** | TLAS: Indoor Corridor (20K Instances) | `ProceduralDiagram` | `tlas_corridor_20k` | 100 discrete room cells (200 inst/room). Clean non-overlapping AABBs with 1:4 mesh uniqueness (~5,000 BLASes). |
| **`RayASBuild`** | TLAS: Dense Jungle (50K Instances) | `ProceduralDiagram` | `tlas_jungle_50k` | Continuous undulating terrain with heavy AABB overlap. 1:100 BLAS reuse stressing BVH splitting heuristics. |
| **`RayASBuild`** | TLAS: Massive Open World (200K Instances) | `ProceduralDiagram` | `tlas_openworld_200k` | 20 macro geographic sectors (10K inst/sector) spanning kilometers. Multi-tier hierarchy stressing instance table memory. |
| **`RayRawTraversal`** | Coherent Triangles / Deep Box Stress | `TextureThumbnail` | `renders/render_showroom_stage1_bvh.png` | BVH Traversal Cost Heatmap. Color ramp indicates ray-box/triangle test count per pixel. |
| **`RayIntersect`** | Ray-Box & Ray-Triangle Intersections | `ProceduralDiagram` | `intersect_primitives` | Ray vector intersecting ray-AABB slabs and Möller–Trumbore barycentric coordinates. |
| **`RayAnyHit`** | Alpha-Tested Geometry (Cutout Stress) | `TextureThumbnail` | `docs/images/geometry_alpha_layers.png` | Foliage alpha cutout layers requiring any-hit shader evaluation vs fully opaque geometry. |
| **`RayProcedural`** | AABB Spheres | `ProceduralDiagram` | `procedural_sphere` | Ray intersecting analytic quadric procedural surface within an intersection shader. |
| **`RayDivergence`** | $0^\circ$ Beam (Coherent) | `ProceduralDiagram` | `cone_divergence_0` | Perfectly parallel specular reflected rays ($0^\circ$ dispersion angle). |
| **`RayDivergence`** | $22.5^\circ$, $45^\circ$, $67.5^\circ$ Cone Spread | `ProceduralDiagram` | `cone_divergence_<deg>` | Jittered reflection rays within bounded angular cones representing glossy surface roughness. |
| **`RayDivergence`** | $90^\circ$ Hemispherical (Incoherent) | `ProceduralDiagram` | `cone_divergence_90` | Diffuse cosine-weighted hemispherical scattering creating SIMD warp divergence. |
| **`RayScheduling`** | Primary Rays (Compute / RTP / SER / DGC) | `TextureThumbnail` | `renders/render_showroom_stage2_primary.png` | Primary camera ray surface hits displaying G-Buffer world normals and depth. |
| **`RayScheduling`** | Directional Shadows (Single & Multi-Light) | `TextureThumbnail` | `renders/render_showroom_stage3_shadow.png` | Binary visibility and penumbra shadow mask traced to directional and point lights. |
| **`RayScheduling`** | Multi-Light Shading (128 Lights) | `TextureThumbnail` | `renders/render_showroom_multilight_128_dgc.png` | Dynamic GPU work queues grouping 128 light sources into coherent evaluation batches. |
| **`RayScheduling`** | Material (Megakernel / RTP+SER / DGC) | `PbrMaterialCard` | `docs/images/material_lineup.png` | 5 BSDF classes (Car Paint, Subsurface, Velvet, Glass, Rust) reordered via SER. |
| **`RayScheduling`** | Path Tracing (1 SPP & 16 SPP) | `TextureThumbnail` | `renders/render_showroom_stage6_indirect.png` | Multi-bounce indirect diffuse path tracing and color bleeding. |
| **`RayScheduling`** | Traversal: Linear 1D Scanline | `ProceduralDiagram` | `traversal_scanline` | Linear horizontal ray dispatch showing memory cache misses across vertical strides. |
| **`RayScheduling`** | Traversal: 2D Screen Tiled (8x4) | `ProceduralDiagram` | `traversal_tiled_8x4` | Rectangular tile grouping maintaining spatial L1/L2 cache locality. |
| **`RayScheduling`** | Traversal: 2D Morton Z-Order (8x4 & 4x8) | `ProceduralDiagram` | `traversal_morton_8x4` | Space-filling Morton Z-curve mapping 2D ray coordinates to maximize cache hit rates. |
| **`RayScheduling`** | Wavefront Ballot Compaction | `ProceduralDiagram` | `wave_ballot_compaction` | 32/64-thread SIMD lane bitmask compaction removing terminated rays. |
| **`Pixel Fill Rate`** | RGBA8 / RGBA16F / Alpha Blending | `ProceduralDiagram` | `rop_blend_gradient` | Overdraw and rasterizer alpha blending throughput simulation. |

---

## 5. UI/UX & Tooltip Presentation Design

### 5.1 Tooltip Layout Wireframe

```
+-------------------------------------------------------------+
| BLAS Build (1M Triangles)                         [Ray Tracing]
| Subcategory: BLAS Build & Update | Metric: MTris/s
|-------------------------------------------------------------|
| Measures how fast the GPU builds ray tracing geometry       |
| structures for 1 million triangles using hardware build     |
| pipelines.                                                  |
|-------------------------------------------------------------|
|  +-------------------------------------------------------+  |
|  |                                                       |  |
|  |            [ VISUALIZATION CANVAS: 340 x 191 ]        |  |
|  |               (Wireframe Mesh + BVH AABBs)            |  |
|  |                                                       |  |
|  +-------------------------------------------------------+  |
|  [Viz] Hardware Bounding Volume Hierarchy Geometry Structure |
|  * Hint: View full interactive model in RT Viewport tab      |
+-------------------------------------------------------------+
```

### 5.2 Implementation of `renderBenchmarkTooltip` in `GuiApp.cpp`

```cpp
void GuiApp::renderBenchmarkTooltip(const BenchmarkItem& item) {
    ImGui::BeginTooltip();

    // 1. Header & Badges
    ImGui::TextColored(ImVec4(0.40f, 0.80f, 1.00f, 1.0f), "%s", item.name.c_str());
    ImGui::TextDisabled("Category: %s | Subcategory: %s | Metric: %s",
                        item.category.c_str(), item.subcategory.c_str(), item.metricType.c_str());
    ImGui::Separator();

    // 2. Textual Explanation
    ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + s(380.0f));
    ImGui::TextUnformatted(item.description.c_str());
    ImGui::PopTextWrapPos();

    // 3. Visualization Canvas
    if (item.viz.type != TooltipVizType::None) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        float canvasWidth = s(340.0f);
        float canvasHeight = canvasWidth / (item.viz.aspectRatio > 0.0f ? item.viz.aspectRatio : (16.0f / 9.0f));

        if (item.viz.type == TooltipVizType::TextureThumbnail || item.viz.type == TooltipVizType::PbrMaterialCard) {
            auto tex = getOrLoadTexture(item.viz.assetPath);
            ImVec2 p0 = ImGui::GetCursorScreenPos();
            ImVec2 p1 = ImVec2(p0.x + canvasWidth, p0.y + canvasHeight);
            ImDrawList* drawList = ImGui::GetWindowDrawList();

            // Background & Border
            drawList->AddRectFilled(p0, p1, IM_COL32(12, 14, 20, 255), s(4.0f));
            if (tex.isValid()) {
                drawList->AddImage(reinterpret_cast<ImTextureID>(tex.descriptorSet), p0, p1);
            } else {
                std::string fallback = "Preview: " + item.viz.caption;
                ImVec2 sz = ImGui::CalcTextSize(fallback.c_str());
                drawList->AddText(ImVec2(p0.x + (canvasWidth - sz.x) * 0.5f, p0.y + (canvasHeight - sz.y) * 0.5f),
                                  IM_COL32(160, 170, 190, 255), fallback.c_str());
            }
            drawList->AddRect(p0, p1, IM_COL32(50, 60, 80, 255), s(4.0f));
            ImGui::Dummy(ImVec2(canvasWidth, canvasHeight));
        }
        else if (item.viz.type == TooltipVizType::ProceduralDiagram) {
            renderProceduralDiagram(item.viz.diagramId, canvasWidth, canvasHeight);
        }

        // Caption & Technical Legend
        if (!item.viz.caption.empty()) {
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.85f, 0.75f, 0.35f, 1.0f), "[Viz] %s", item.viz.caption.c_str());
        }
        if (!item.viz.technicalDetails.empty()) {
            ImGui::TextDisabled("%s", item.viz.technicalDetails.c_str());
        }

        // Navigation Hint
        if (item.viz.linkedRtViewportMode >= 0) {
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.35f, 0.75f, 0.95f, 0.85f), "-> Switch to 'Ray Tracing Viewport' tab to inspect");
        }
    }

    ImGui::EndTooltip();
}
```

---

## 6. Procedural Diagram Implementations

### 6.1 Ray Directional Divergence Cone (`cone_divergence_<deg>`)
Visualizes ray dispersion relative to surface normal vectors to explain memory and cache divergence:

```cpp
void GuiApp::renderRayConeDiagram(float angleDegrees, float width, float height) {
    ImVec2 p0 = ImGui::GetCursorScreenPos();
    ImDrawList* drawList = ImGui::GetWindowDrawList();

    // Canvas background
    drawList->AddRectFilled(p0, ImVec2(p0.x + width, p0.y + height), IM_COL32(14, 18, 26, 255), s(4.0f));
    drawList->AddRect(p0, ImVec2(p0.x + width, p0.y + height), IM_COL32(40, 50, 70, 255), s(4.0f));

    ImVec2 hitPoint = ImVec2(p0.x + width * 0.5f, p0.y + height * 0.82f);

    // Surface line
    drawList->AddLine(ImVec2(hitPoint.x - width * 0.40f, hitPoint.y),
                      ImVec2(hitPoint.x + width * 0.40f, hitPoint.y), IM_COL32(120, 130, 150, 255), 2.0f);

    // Surface Normal vector
    drawList->AddLine(hitPoint, ImVec2(hitPoint.x, hitPoint.y - height * 0.65f), IM_COL32(80, 180, 255, 220), 1.5f);

    // Scattered ray fan
    float rad = (angleDegrees * 0.5f) * (3.14159265f / 180.0f);
    const int numRays = 7;
    for (int i = 0; i < numRays; ++i) {
        float t = (numRays > 1) ? (static_cast<float>(i) / (numRays - 1) * 2.0f - 1.0f) : 0.0f;
        float rayAngle = t * rad;
        float rayLen = height * 0.58f;
        ImVec2 rayEnd = ImVec2(hitPoint.x + std::sin(rayAngle) * rayLen,
                               hitPoint.y - std::cos(rayAngle) * rayLen);

        ImU32 rayColor = (angleDegrees > 60.0f) ? IM_COL32(255, 100, 80, 220) : IM_COL32(255, 190, 60, 220);
        drawList->AddLine(hitPoint, rayEnd, rayColor, 1.5f);
    }

    // Angle label overlay
    char buf[32];
    std::snprintf(buf, sizeof(buf), "Cone: %.1f deg", angleDegrees);
    drawList->AddText(ImVec2(p0.x + s(10.0f), p0.y + s(8.0f)), IM_COL32(220, 220, 220, 255), buf);

    ImGui::Dummy(ImVec2(width, height));
}
```

### 6.2 2D Morton Z-Order Curve (`traversal_morton_8x4`)
Visualizes how Morton order groups rays into localized spatial tiles to preserve GPU L1/L2 cache lines:

```cpp
void GuiApp::renderMortonCurveDiagram(float width, float height) {
    ImVec2 p0 = ImGui::GetCursorScreenPos();
    ImDrawList* drawList = ImGui::GetWindowDrawList();

    drawList->AddRectFilled(p0, ImVec2(p0.x + width, p0.y + height), IM_COL32(14, 18, 26, 255), s(4.0f));
    drawList->AddRect(p0, ImVec2(p0.x + width, p0.y + height), IM_COL32(40, 50, 70, 255), s(4.0f));

    const int cols = 8;
    const int rows = 4;
    float cellW = (width - s(20.0f)) / cols;
    float cellH = (height - s(30.0f)) / rows;
    float startX = p0.x + s(10.0f);
    float startY = p0.y + s(20.0f);

    // Draw pixel grid
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            ImVec2 c0 = ImVec2(startX + c * cellW, startY + r * cellH);
            ImVec2 c1 = ImVec2(c0.x + cellW, c0.y + cellH);
            drawList->AddRect(c0, c1, IM_COL32(35, 45, 60, 180));
        }
    }

    // Interleave bits to compute Morton code
    auto morton2D = [](uint32_t x, uint32_t y) -> uint32_t {
        auto part1by1 = [](uint32_t n) -> uint32_t {
            n &= 0x0000ffff;
            n = (n | (n << 8)) & 0x00FF00FF;
            n = (n | (n << 4)) & 0x0F0F0F0F;
            n = (n | (n << 2)) & 0x33333333;
            n = (n | (n << 1)) & 0x55555555;
            return n;
        };
        return (part1by1(y) << 1) | part1by1(x);
    };

    // Sort coordinates by Morton code
    std::vector<std::pair<uint32_t, ImVec2>> points;
    for (uint32_t r = 0; r < static_cast<uint32_t>(rows); ++r) {
        for (uint32_t c = 0; c < static_cast<uint32_t>(cols); ++c) {
            uint32_t code = morton2D(c, r);
            ImVec2 center = ImVec2(startX + (c + 0.5f) * cellW, startY + (r + 0.5f) * cellH);
            points.push_back({code, center});
        }
    }
    std::sort(points.begin(), points.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

    // Draw continuous Z-curve path
    for (size_t i = 1; i < points.size(); ++i) {
        drawList->AddLine(points[i - 1].second, points[i].second, IM_COL32(0, 220, 180, 240), 1.8f);
        drawList->AddCircleFilled(points[i].second, s(2.5f), IM_COL32(255, 215, 0, 255));
    }

    drawList->AddText(ImVec2(p0.x + s(10.0f), p0.y + s(4.0f)), IM_COL32(200, 210, 230, 255), "8x4 Z-Order Spatial Locality Curve");
    ImGui::Dummy(ImVec2(width, height));
}
```

### 6.3 TLAS Structural Archetype Schematics

The three TLAS benchmarks in `RayASBuildBench.cpp` evaluate fundamentally different GPU acceleration structure building workloads and topological distributions. Grouping them under a single generic icon misrepresents their architectural purpose. Each must feature a dedicated visualization reflecting its spatial layout and hardware stress characteristics:

#### 1. `tlas_corridor_20k` (Indoor Corridor — 20,000 Instances)
- **Engine Logic**: 100 discrete rooms in a $10 \times 10$ floorplan grid ($40\text{m}$ room pitch). High geometric uniqueness (~1:4 ratio, 5,000 unique meshes).
- **Visualization**: Floorplan cell diagram showing clear corridor separations and non-overlapping blue bounding boxes enclosing tight furniture clusters.
- **Hardware Bottleneck Shown**: Rapid top-level node partitioning with disjoint bounding boxes; high BLAS address translation overhead.

```cpp
void GuiApp::renderTlasCorridorDiagram(float width, float height) {
    ImVec2 p0 = ImGui::GetCursorScreenPos();
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    drawList->AddRectFilled(p0, ImVec2(p0.x + width, p0.y + height), IM_COL32(14, 18, 26, 255), s(4.0f));
    drawList->AddRect(p0, ImVec2(p0.x + width, p0.y + height), IM_COL32(40, 50, 70, 255), s(4.0f));

    // Draw 4x4 sample room grid with distinct corridor gaps
    const int grid = 4;
    float pad = s(8.0f);
    float roomW = (width - s(20.0f) - pad * (grid - 1)) / grid;
    float roomH = (height - s(32.0f) - pad * (grid - 1)) / grid;
    float startX = p0.x + s(10.0f);
    float startY = p0.y + s(24.0f);

    for (int y = 0; y < grid; ++y) {
        for (int x = 0; x < grid; ++x) {
            ImVec2 r0(startX + x * (roomW + pad), startY + y * (roomH + pad));
            ImVec2 r1(r0.x + roomW, r0.y + roomH);
            // Room AABB (Disjoint / Non-overlapping)
            drawList->AddRectFilled(r0, r1, IM_COL32(20, 35, 55, 200), s(2.0f));
            drawList->AddRect(r0, r1, IM_COL32(40, 120, 200, 255), s(2.0f));
            // Clustered instance primitives
            for (int k = 0; k < 5; ++k) {
                float ox = r0.x + roomW * (0.25f + 0.12f * (k % 3));
                float oy = r0.y + roomH * (0.25f + 0.15f * (k / 2));
                drawList->AddCircleFilled(ImVec2(ox, oy), s(2.0f), IM_COL32(80, 200, 255, 255));
            }
        }
    }
    drawList->AddText(ImVec2(p0.x + s(10.0f), p0.y + s(4.0f)), IM_COL32(140, 210, 255, 255),
                      "100 Discrete Rooms: Clean Non-Overlapping AABBs");
    ImGui::Dummy(ImVec2(width, height));
}
```

#### 2. `tlas_jungle_50k` (Dense Jungle — 50,000 Instances)
- **Engine Logic**: Continuous undulating terrain ($z = \sin(0.05x) + \cos(0.05y)$) packed with 50,000 overlapping foliage instances reusing only 500 shared BLASes (~1:100 ratio).
- **Visualization**: Continuous sine wave elevation with heavily intersecting translucent green bounding boxes and foliage dots. Overlap zones highlighted in amber/red.
- **Hardware Bottleneck Shown**: High AABB bounding box collision and overlap; stresses BVH spatial splitting heuristics (e.g. SAH binning) because rays cannot easily discard candidate boxes.

```cpp
void GuiApp::renderTlasJungleDiagram(float width, float height) {
    ImVec2 p0 = ImGui::GetCursorScreenPos();
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    drawList->AddRectFilled(p0, ImVec2(p0.x + width, p0.y + height), IM_COL32(14, 18, 26, 255), s(4.0f));
    drawList->AddRect(p0, ImVec2(p0.x + width, p0.y + height), IM_COL32(40, 50, 70, 255), s(4.0f));

    // Draw undulating terrain curve
    const int segs = 30;
    for (int i = 0; i < segs; ++i) {
        float t0 = (float)i / segs;
        float t1 = (float)(i + 1) / segs;
        float x0 = p0.x + s(10.0f) + t0 * (width - s(20.0f));
        float x1 = p0.x + s(10.0f) + t1 * (width - s(20.0f));
        float y0 = p0.y + height * 0.70f + std::sin(t0 * 6.28f * 1.5f) * s(10.0f);
        float y1 = p0.y + height * 0.70f + std::sin(t1 * 6.28f * 1.5f) * s(10.0f);
        drawList->AddLine(ImVec2(x0, y0), ImVec2(x1, y1), IM_COL32(60, 110, 60, 255), 2.0f);
    }

    // Draw heavily overlapping foliage bounding boxes
    const int foliageCount = 14;
    for (int i = 0; i < foliageCount; ++i) {
        float fx = p0.x + s(15.0f) + (float)i * ((width - s(50.0f)) / foliageCount) + ((i % 3) - 1) * s(6.0f);
        float fy = p0.y + height * 0.55f + std::sin((float)i * 0.9f) * s(14.0f);
        float boxW = s(24.0f);
        float boxH = s(28.0f);
        // Semi-transparent overlapping box
        drawList->AddRectFilled(ImVec2(fx, fy), ImVec2(fx + boxW, fy + boxH), IM_COL32(35, 120, 50, 60), s(2.0f));
        drawList->AddRect(ImVec2(fx, fy), ImVec2(fx + boxW, fy + boxH), IM_COL32(80, 220, 100, 180), s(2.0f));
        // Foliage center
        drawList->AddCircleFilled(ImVec2(fx + boxW * 0.5f, fy + boxH * 0.4f), s(3.0f), IM_COL32(140, 255, 120, 240));
    }
    drawList->AddText(ImVec2(p0.x + s(10.0f), p0.y + s(4.0f)), IM_COL32(120, 240, 140, 255),
                      "Continuous Terrain: Heavy AABB Overlap (1:100 Reuse)");
    ImGui::Dummy(ImVec2(width, height));
}
```

#### 3. `tlas_openworld_200k` (Massive Open World — 200,000 Instances)
- **Engine Logic**: 20 massive geographic sectors ($5 \times 4$ sector grid, $500\text{m}$ pitch) spanning kilometers with 200,000 instances (~500M virtual triangles).
- **Visualization**: Multi-tier macro sector map showing large outer bounding boxes, intermediate local clusters, density gradients, and a hierarchical tree depth gauge.
- **Hardware Bottleneck Shown**: Multi-level tree depth, GPU memory bus throughput for large instance buffers, and top-level BVH node construction at scale.

```cpp
void GuiApp::renderTlasOpenWorldDiagram(float width, float height) {
    ImVec2 p0 = ImGui::GetCursorScreenPos();
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    drawList->AddRectFilled(p0, ImVec2(p0.x + width, p0.y + height), IM_COL32(14, 18, 26, 255), s(4.0f));
    drawList->AddRect(p0, ImVec2(p0.x + width, p0.y + height), IM_COL32(40, 50, 70, 255), s(4.0f));

    // Draw 3x2 macro sectors across vast expanse
    const int cols = 3;
    const int rows = 2;
    float pad = s(6.0f);
    float secW = (width - s(20.0f) - pad * (cols - 1)) / cols;
    float secH = (height - s(34.0f) - pad * (rows - 1)) / rows;
    float startX = p0.x + s(10.0f);
    float startY = p0.y + s(24.0f);

    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            ImVec2 s0(startX + x * (secW + pad), startY + y * (secH + pad));
            ImVec2 s1(s0.x + secW, s0.y + secH);
            // Macro Sector AABB (Dashed border appearance)
            drawList->AddRect(s0, s1, IM_COL32(180, 130, 40, 200), s(2.0f), 0, 1.5f);
            // High-density instance clusters
            for (int k = 0; k < 12; ++k) {
                float dx = s0.x + s(4.0f) + (float)(k % 4) * (secW - s(8.0f)) / 3.0f;
                float dy = s0.y + s(4.0f) + (float)(k / 4) * (secH - s(8.0f)) / 2.0f;
                drawList->AddCircleFilled(ImVec2(dx, dy), s(1.2f), IM_COL32(255, 200, 80, 220));
            }
        }
    }
    drawList->AddText(ImVec2(p0.x + s(10.0f), p0.y + s(4.0f)), IM_COL32(255, 190, 80, 255),
                      "200K Instances: 20 Macro Sectors & Multi-Tier Tree");
    ImGui::Dummy(ImVec2(width, height));
}
```

1. **Lazy Loading on Hover**:
   Never load textures during benchmark startup. Tooltip images are only fetched when `ImGui::IsItemHovered()` returns true. The existing `m_textureCache` handles descriptor reuse without redundant reloads.

2. **Downscaled Previews**:
   Original 4K output PNGs in `renders/` range between 15 MB and 25 MB each. Loading uncompressed 4K textures for a 340×191 preview card causes VRAM pressure and frame hitches. Previews must either:
   - Use dedicated lightweight thumbnails (e.g. 480×270 WebP or PNG, ~30–50 KB each) placed in `assets/thumbnails/`.
   - Or perform automatic bilinear downsampling during `loadTextureFromFile` if a thumbnail flag is present.

3. **Screen Boundary Clamping**:
   Dear ImGui automatically ensures that `BeginTooltip()` stays within viewport screen boundaries. Restricting the tooltip width to `s(340.0f) - s(380.0f)` guarantees that the tooltip remains visible across all monitor configurations without obscuring critical data tables.

---

## 8. Implementation Phases

```
+---------------------------------------------------------------------------------+
| Phase 1: Data Model & Tooltip Helper                                            |
| - Add BenchmarkVisualization to BenchmarkItem in GuiApp.h                       |
| - Replace plain ImGui::SetTooltip with GuiApp::renderBenchmarkTooltip           |
+---------------------------------------------------------------------------------+
                                        |
                                        v
+---------------------------------------------------------------------------------+
| Phase 2: Wire Static Assets (Renders & PBR Docs)                                |
| - Map RayASBuild, RayScheduling, and PBR Materials to existing PNG files        |
| - Verify lazy loading and VRAM caching in VulkanContext                         |
+---------------------------------------------------------------------------------+
                                        |
                                        v
+---------------------------------------------------------------------------------+
| Phase 3: Implement Procedural Schematics                                        |
| - Add renderRayConeDiagram (0°, 22.5°, 45°, 90°)                                |
| - Add renderMortonCurveDiagram (8x4, 4x8 vs Scanline)                           |
| - Add wave ballot compaction bitmask diagram                                    |
+---------------------------------------------------------------------------------+
                                        |
                                        v
+---------------------------------------------------------------------------------+
| Phase 4: Dynamic Result Hook & Viewport Cross-Linking                           |
| - If test has completed, show session output preview                            |
| - Add interactive jump button to open Ray Tracing Viewport directly             |
+---------------------------------------------------------------------------------+
```
