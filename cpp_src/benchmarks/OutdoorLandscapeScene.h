#pragma once

#include <array>
#include <cmath>
#include <vector>
#include <algorithm>

namespace OutdoorLandscapeScene {

// Open-World Outdoor Landscape Primitive Index Constants:
// 1. Mountain Valley Heightfield: 128x128 quads = 32,768 triangles (Material 0, 3, 4, 5 based on slope/altitude)
constexpr uint32_t PRIM_TERRAIN_END = 32768;

// 2. Alpine Lake / River Water Plane: 32x32 quads = 2,048 triangles (Material 2)
constexpr uint32_t PRIM_WATER_END   = 34816;

// 3. 100 Conifer Pine Trees: 100 * 224 = 22,400 triangles (Trunk: Material 6, Needles: Material 1)
constexpr uint32_t PRIM_TREES_END   = 57216;

// Procedural multi-frequency analytical heightfield
inline float terrainHeight(float x, float y) {
  // Central river valley carving down X = 0
  float valley = 1.0f - std::exp(-(x * x) / 50000.0f);

  // Broad mountain ridges on East and West
  float mountainR = std::abs(x) * 0.25f + 40.0f * std::cos(x * 0.008f) * std::sin(y * 0.006f);

  // Distant mountain massif to the North (y > 200)
  float distantPeaks = (y > 200.0f) ? (y - 200.0f) * 0.22f : 0.0f;

  // Natural southern slope rising toward foreground camera ridge (y < -50)
  float southRise = (y < -50.0f) ? (-50.0f - y) * 0.16f : 0.0f;

  // Foreground lake basin moraine enclosing the southern shoreline
  float basinRim = 14.0f * std::exp(-((y + 80.0f) * (y + 80.0f)) / 1600.0f);

  // Medium and high frequency terrain detail
  float fbm = 18.0f * std::sin(x * 0.025f + y * 0.015f) +
              9.0f  * std::cos(x * 0.055f - y * 0.045f) +
              4.0f  * std::sin(x * 0.12f  + y * 0.11f);

  float z = (mountainR + distantPeaks) * valley + fbm + southRise + basinRim;
  return std::max(z, -4.0f);
}

inline void appendTerrain(std::vector<float> &vertices) {
  const uint32_t grid_n = 128;
  const float x_min = -1000.0f, x_max = 1000.0f;
  const float y_min = -500.0f,  y_max = 1500.0f;

  auto getVertex = [&](uint32_t i, uint32_t j) -> std::array<float, 3> {
    float x = x_min + (x_max - x_min) * (float(i) / float(grid_n));
    float y = y_min + (y_max - y_min) * (float(j) / float(grid_n));
    float z = terrainHeight(x, y);
    return {x, y, z};
  };

  for (uint32_t i = 0; i < grid_n; ++i) {
    for (uint32_t j = 0; j < grid_n; ++j) {
      auto p00 = getVertex(i, j);
      auto p10 = getVertex(i + 1, j);
      auto p11 = getVertex(i + 1, j + 1);
      auto p01 = getVertex(i, j + 1);

      // Tri 1
      vertices.insert(vertices.end(), {p00[0], p00[1], p00[2], p10[0], p10[1], p10[2], p11[0], p11[1], p11[2]});
      // Tri 2
      vertices.insert(vertices.end(), {p00[0], p00[1], p00[2], p11[0], p11[1], p11[2], p01[0], p01[1], p01[2]});
    }
  }
}

inline void appendWaterPlane(std::vector<float> &vertices) {
  const uint32_t water_n = 32;
  const float x_min = -175.0f, x_max = 175.0f;
  const float y_min = -110.0f, y_max = 920.0f;
  const float z_water = 0.5f;

  for (uint32_t i = 0; i < water_n; ++i) {
    float x0 = x_min + (x_max - x_min) * (float(i) / float(water_n));
    float x1 = x_min + (x_max - x_min) * (float(i + 1) / float(water_n));
    for (uint32_t j = 0; j < water_n; ++j) {
      float y0 = y_min + (y_max - y_min) * (float(j) / float(water_n));
      float y1 = y_min + (y_max - y_min) * (float(j + 1) / float(water_n));

      vertices.insert(vertices.end(), {x0, y0, z_water, x1, y0, z_water, x1, y1, z_water});
      vertices.insert(vertices.end(), {x0, y0, z_water, x1, y1, z_water, x0, y1, z_water});
    }
  }
}

inline void appendPineTree(std::vector<float> &vertices, float root_x, float root_y, float scale) {
  float root_z = terrainHeight(root_x, root_y);
  const float pi2 = 6.283185307179586f;

  // 1. Trunk (Cylinder: 8 radial segs x 2 vertical = 16 quads = 32 tris)
  const uint32_t trunk_segs = 8;
  const float r_trunk = 0.45f * scale;
  const float h_trunk = 4.0f * scale;
  for (uint32_t i = 0; i < trunk_segs; ++i) {
    float a0 = (float(i) / float(trunk_segs)) * pi2;
    float a1 = (float(i + 1) / float(trunk_segs)) * pi2;
    float x0 = root_x + r_trunk * std::cos(a0);
    float y0 = root_y + r_trunk * std::sin(a0);
    float x1 = root_x + r_trunk * std::cos(a1);
    float y1 = root_y + r_trunk * std::sin(a1);

    vertices.insert(vertices.end(), {x0, y0, root_z, x1, y0, root_z, x1, y1, root_z + h_trunk});
    vertices.insert(vertices.end(), {x0, y0, root_z, x1, y1, root_z + h_trunk, x0, y0, root_z + h_trunk});
  }

  // 2. Three Conical Needle Tiers (3 cones: 16 radial segs x 2 = 32 quads / 64 tris each = 192 tris)
  // Total tree triangles = 32 + 192 = 224 triangles
  const uint32_t tier_segs = 16;
  for (uint32_t tier = 0; tier < 3; ++tier) {
    float tier_base_z = root_z + (2.5f + float(tier) * 3.2f) * scale;
    float tier_tip_z  = tier_base_z + 5.5f * scale;
    float r_tier = (3.5f - float(tier) * 0.9f) * scale;

    for (uint32_t i = 0; i < tier_segs; ++i) {
      float a0 = (float(i) / float(tier_segs)) * pi2;
      float a1 = (float(i + 1) / float(tier_segs)) * pi2;
      float x0 = root_x + r_tier * std::cos(a0);
      float y0 = root_y + r_tier * std::sin(a0);
      float x1 = root_x + r_tier * std::cos(a1);
      float y1 = root_y + r_tier * std::sin(a1);

      // Upper cone slant
      vertices.insert(vertices.end(), {x0, y0, tier_base_z, x1, y1, tier_base_z, root_x, root_y, tier_tip_z});
      // Lower underside skirt
      vertices.insert(vertices.end(), {x1, y1, tier_base_z, x0, y0, tier_base_z, root_x, root_y, tier_base_z - 0.5f * scale});
    }
  }
}

inline void appendFoliageCanopy(std::vector<float> &vertices) {
  // Deterministic PRNG for tree placement
  uint32_t seed = 421337;
  auto rand_f = [&]() -> float {
    seed = seed * 747796405u + 2891336453u;
    uint32_t word = ((seed >> ((seed >> 28u) + 4u)) ^ seed) * 277803737u;
    return float((word >> 22u) ^ word) / 4294967295.0f;
  };

  // Place 100 trees along the valley slopes and riverbanks
  for (uint32_t i = 0; i < 100; ++i) {
    float x = (rand_f() * 2.0f - 1.0f) * 320.0f;
    // Don't place in center of river (abs(x) < 28)
    if (std::abs(x) < 28.0f) {
      x = (x >= 0.0f) ? (x + 35.0f) : (x - 35.0f);
    }
    float y = -250.0f + rand_f() * 950.0f;
    float scale = 1.0f + rand_f() * 0.75f;
    for (int retry = 0; retry < 6 && terrainHeight(x, y) < 0.8f; ++retry) {
      x = (rand_f() * 2.0f - 1.0f) * 320.0f;
      if (std::abs(x) < 28.0f) {
        x = (x >= 0.0f) ? (x + 35.0f) : (x - 35.0f);
      }
      y = -250.0f + rand_f() * 950.0f;
    }
    appendPineTree(vertices, x, y, scale);
  }
}

inline std::vector<float> buildOutdoorLandscapeMesh() {
  std::vector<float> vertices;
  vertices.reserve(PRIM_TREES_END * 9);

  // 1. Terrain Heightfield (0 .. 32767)
  appendTerrain(vertices);

  // 2. Water Surface (32768 .. 34815)
  appendWaterPlane(vertices);

  // 3. Foliage Trees (34816 .. 57215)
  appendFoliageCanopy(vertices);

  return vertices;
}

} // namespace OutdoorLandscapeScene
