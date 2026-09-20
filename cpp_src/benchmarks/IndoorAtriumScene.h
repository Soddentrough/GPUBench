#pragma once

#include "ShowroomScene.h"
#include <array>
#include <cmath>
#include <vector>
#include <algorithm>

namespace IndoorAtriumScene {

using namespace ShowroomScene;

// Centerpiece and floor primitives inherit from ShowroomScene:
// Primitives 0 .. 27079: Knot, Sphere, Pedestal, Suzanne, Floor
// Additional Atrium Architectural Primitives:
// 6. Enclosing Walls (North, South, East, West): 1,024 triangles (Material 4)
constexpr uint32_t PRIM_WALLS_END   = 28104;

// 7. Vaulted Coffered Ceiling: 1,024 triangles (Material 6)
constexpr uint32_t PRIM_CEILING_END = 29128;

// 8. 8 Fluted Marble Architectural Columns: 6,144 triangles (Material 1)
constexpr uint32_t PRIM_COLUMNS_END = 35272;

inline void appendWalls(std::vector<float> &vertices) {
  // 4 walls bounding X in [-7.5, 7.5], Y in [-11.0, 9.0], Z in [-1.5, 6.0]
  const float x_min = -7.5f, x_max = 7.5f;
  const float y_min = -11.0f, y_max = 9.0f;
  const float z_min = -1.5f, z_max = 6.0f;

  const uint32_t segs_h = 16;
  const uint32_t segs_v = 8;

  // Helper quad append
  auto addQuad = [&](float x0, float y0, float z0,
                     float x1, float y1, float z1,
                     float x2, float y2, float z2,
                     float x3, float y3, float z3) {
    vertices.insert(vertices.end(), {x0, y0, z0, x1, y1, z1, x2, y2, z2});
    vertices.insert(vertices.end(), {x0, y0, z0, x2, y2, z2, x3, y3, z3});
  };

  // Back Wall (Y = y_max)
  for (uint32_t i = 0; i < segs_h; ++i) {
    float fx0 = x_min + (x_max - x_min) * (float(i) / float(segs_h));
    float fx1 = x_min + (x_max - x_min) * (float(i + 1) / float(segs_h));
    for (uint32_t j = 0; j < segs_v; ++j) {
      float fz0 = z_min + (z_max - z_min) * (float(j) / float(segs_v));
      float fz1 = z_min + (z_max - z_min) * (float(j + 1) / float(segs_v));
      addQuad(fx0, y_max, fz0, fx1, y_max, fz0, fx1, y_max, fz1, fx0, y_max, fz1);
    }
  }

  // Front Wall (Y = y_min, behind camera)
  for (uint32_t i = 0; i < segs_h; ++i) {
    float fx0 = x_min + (x_max - x_min) * (float(i) / float(segs_h));
    float fx1 = x_min + (x_max - x_min) * (float(i + 1) / float(segs_h));
    for (uint32_t j = 0; j < segs_v; ++j) {
      float fz0 = z_min + (z_max - z_min) * (float(j) / float(segs_v));
      float fz1 = z_min + (z_max - z_min) * (float(j + 1) / float(segs_v));
      addQuad(fx1, y_min, fz0, fx0, y_min, fz0, fx0, y_min, fz1, fx1, y_min, fz1);
    }
  }

  // Left Wall (X = x_min)
  for (uint32_t i = 0; i < segs_h; ++i) {
    float fy0 = y_min + (y_max - y_min) * (float(i) / float(segs_h));
    float fy1 = y_min + (y_max - y_min) * (float(i + 1) / float(segs_h));
    for (uint32_t j = 0; j < segs_v; ++j) {
      float fz0 = z_min + (z_max - z_min) * (float(j) / float(segs_v));
      float fz1 = z_min + (z_max - z_min) * (float(j + 1) / float(segs_v));
      addQuad(x_min, fy1, fz0, x_min, fy0, fz0, x_min, fy0, fz1, x_min, fy1, fz1);
    }
  }

  // Right Wall (X = x_max)
  for (uint32_t i = 0; i < segs_h; ++i) {
    float fy0 = y_min + (y_max - y_min) * (float(i) / float(segs_h));
    float fy1 = y_min + (y_max - y_min) * (float(i + 1) / float(segs_h));
    for (uint32_t j = 0; j < segs_v; ++j) {
      float fz0 = z_min + (z_max - z_min) * (float(j) / float(segs_v));
      float fz1 = z_min + (z_max - z_min) * (float(j + 1) / float(segs_v));
      addQuad(x_max, fy0, fz0, x_max, fy1, fz0, x_max, fy1, fz1, x_max, fy0, fz1);
    }
  }
}

inline void appendVaultedCeiling(std::vector<float> &vertices) {
  // Vaulted barrel arch ceiling: X in [-7.5, 7.5], Y in [-11.0, 9.0]
  // Height z = 6.0 + 2.0 * cos(pi * x / 15.0)
  const float x_min = -7.5f, x_max = 7.5f;
  const float y_min = -11.0f, y_max = 9.0f;
  const uint32_t segs_x = 32;
  const uint32_t segs_y = 16;
  const float pi = 3.141592653589793f;

  for (uint32_t i = 0; i < segs_x; ++i) {
    float u0 = float(i) / float(segs_x);
    float u1 = float(i + 1) / float(segs_x);
    float x0 = x_min + (x_max - x_min) * u0;
    float x1 = x_min + (x_max - x_min) * u1;
    float z0 = 6.0f + 2.0f * std::cos(pi * (x0 / 7.5f) * 0.5f);
    float z1 = 6.0f + 2.0f * std::cos(pi * (x1 / 7.5f) * 0.5f);

    for (uint32_t j = 0; j < segs_y; ++j) {
      float v0 = float(j) / float(segs_y);
      float v1 = float(j + 1) / float(segs_y);
      float y0 = y_min + (y_max - y_min) * v0;
      float y1 = y_min + (y_max - y_min) * v1;

      // Inward-facing ceiling quads
      vertices.insert(vertices.end(), {x0, y0, z0, x1, y0, z1, x1, y1, z1});
      vertices.insert(vertices.end(), {x0, y0, z0, x1, y1, z1, x0, y1, z0});
    }
  }
}

inline void appendColumns(std::vector<float> &vertices) {
  // 8 fluted Gothic columns: 4 on left (X = -5.0), 4 on right (X = 5.0)
  // Y positions: -7.0, -2.0, 3.0, 8.0
  const std::array<float, 2> col_x = {-5.0f, 5.0f};
  const std::array<float, 4> col_y = {-7.0f, -2.0f, 3.0f, 8.0f};

  const float r_col = 0.45f;
  const float z_base = -1.5f;
  const float z_top = 6.0f;
  const uint32_t radial_segs = 24;
  const uint32_t height_segs = 16;
  const float pi2 = 6.283185307179586f;

  for (float cx : col_x) {
    for (float cy : col_y) {
      for (uint32_t i = 0; i < radial_segs; ++i) {
        float a0 = (float(i) / float(radial_segs)) * pi2;
        float a1 = (float(i + 1) / float(radial_segs)) * pi2;

        // Subtle fluting groove modulation
        float r0 = r_col * (1.0f + 0.06f * std::cos(12.0f * a0));
        float r1 = r_col * (1.0f + 0.06f * std::cos(12.0f * a1));

        float x0 = cx + r0 * std::cos(a0);
        float y0 = cy + r0 * std::sin(a0);
        float x1 = cx + r1 * std::cos(a1);
        float y1 = cy + r1 * std::sin(a1);

        for (uint32_t j = 0; j < height_segs; ++j) {
          float z0 = z_base + (z_top - z_base) * (float(j) / float(height_segs));
          float z1 = z_base + (z_top - z_base) * (float(j + 1) / float(height_segs));

          // Cylinder Quad
          vertices.insert(vertices.end(), {x0, y0, z0, x1, y0, z1, x1, y1, z1});
          vertices.insert(vertices.end(), {x0, y0, z0, x1, y1, z1, x0, y1, z0});
        }
      }
    }
  }
}

inline std::vector<float> buildIndoorAtriumMesh() {
  std::vector<float> vertices;
  vertices.reserve(PRIM_COLUMNS_END * 9);

  // 1. Centerpiece & Floor (0 .. 27079)
  ShowroomScene::buildShowroomScene(vertices);

  // 2. Enclosing Walls (27080 .. 28103)
  appendWalls(vertices);

  // 3. Vaulted Ceiling (28104 .. 29127)
  appendVaultedCeiling(vertices);

  // 4. Columns (29128 .. 35271)
  appendColumns(vertices);

  return vertices;
}

} // namespace IndoorAtriumScene
