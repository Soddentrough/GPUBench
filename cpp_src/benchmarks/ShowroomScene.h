#pragma once

#include "SuzanneMesh.h"
#include <array>
#include <cmath>
#include <vector>
#include <algorithm>

namespace ShowroomScene {

// 1. Trefoil Knot: 160 u-segs x 24 v-segs = 3,840 quads = 7,680 triangles
//    Bands along curve (32 u-segs = 1,536 tris each):
//    - Band 0 [   0 .. 1535]: Ruby Car Paint (Material 0, Inner Arch)
//    - Band 1 [1536 .. 3071]: Jade Subsurface (Material 1, Left Loop)
//    - Band 2 [3072 .. 4607]: Chrome Mirror (Material 7, Top-Right Arch)
//    - Band 3 [4608 .. 6143]: Magenta Velvet (Material 3, Rear Loop)
//    - Band 4 [6144 .. 7679]: Weathered Rust (Material 4, Front Horizontal Tube)
constexpr uint32_t PRIM_KNOT_PAINT_END  = 1536;
constexpr uint32_t PRIM_KNOT_JADE_END   = 3072;
constexpr uint32_t PRIM_KNOT_CHROME_END = 4608;
constexpr uint32_t PRIM_KNOT_VELVET_END = 6144;
constexpr uint32_t PRIM_KNOT_RUST_END   = 7680;

// 2. Jade Subsurface Sphere (Left): 32x32 = 1,920 triangles (Material 1)
constexpr uint32_t PRIM_SPHERE_END      = 9600;

// 3. Gold Cylinder Pedestal (Center): 32 segs = 128 triangles (Material 6)
constexpr uint32_t PRIM_PEDESTAL_END    = 9728;

// 4. Suzanne Dispersive Glass Head (Right): 968 triangles (Material 2)
constexpr uint32_t PRIM_SUZANNE_END     = 10696;

// 5. Showroom Cyclorama Floor: 128x64 = 16,384 triangles (Material 20..31, Archetype 5)
constexpr uint32_t PRIM_FLOOR_END       = 27080;

inline void appendTrefoilKnot(std::vector<float> &vertices) {
  const uint32_t knot_u = 160;
  const uint32_t knot_v = 24;
  const float p_knot = 2.0f, q_knot = 3.0f;
  const float R_knot = 2.0f, r0_knot = 0.8f;
  const float scale_knot = 0.835f;
  const float r_tube = 0.45f * scale_knot;
  const float pi2 = 6.283185307179586f;

  std::vector<std::array<float, 3>> tangents(knot_u);
  std::vector<std::array<float, 3>> centers(knot_u);
  for (uint32_t i = 0; i < knot_u; ++i) {
    float u = (float(i) / float(knot_u)) * pi2;
    float r = (R_knot + r0_knot * std::cos(q_knot * u)) * scale_knot;
    float cx = r * std::cos(p_knot * u);
    float cy = r * std::sin(p_knot * u);
    float cz = -r0_knot * std::sin(q_knot * u) * scale_knot;

    float u_next = u + 0.0005f;
    float r_n = (R_knot + r0_knot * std::cos(q_knot * u_next)) * scale_knot;
    float tx = r_n * std::cos(p_knot * u_next) - cx;
    float ty = r_n * std::sin(p_knot * u_next) - cy;
    float tz = -r0_knot * std::sin(q_knot * u_next) * scale_knot - cz;
    float tlen = std::sqrt(tx * tx + ty * ty + tz * tz);
    tangents[i] = {tx / tlen, ty / tlen, tz / tlen};
    centers[i] = {cx, cy, cz};
  }

  // Parallel transport frame (Bishop frame)
  std::vector<std::array<float, 3>> normals(knot_u);
  std::vector<std::array<float, 3>> binormals(knot_u);

  // Initial normal perpendicular to tangents[0]
  std::array<float, 3> n0 = {0.0f, 0.0f, 1.0f};
  float d0 = n0[0] * tangents[0][0] + n0[1] * tangents[0][1] + n0[2] * tangents[0][2];
  n0[0] -= d0 * tangents[0][0]; n0[1] -= d0 * tangents[0][1]; n0[2] -= d0 * tangents[0][2];
  float n0_len = std::sqrt(n0[0]*n0[0] + n0[1]*n0[1] + n0[2]*n0[2]);
  normals[0] = {n0[0] / n0_len, n0[1] / n0_len, n0[2] / n0_len};
  binormals[0] = {
    tangents[0][1] * normals[0][2] - tangents[0][2] * normals[0][1],
    tangents[0][2] * normals[0][0] - tangents[0][0] * normals[0][2],
    tangents[0][0] * normals[0][1] - tangents[0][1] * normals[0][0]
  };

  for (uint32_t i = 0; i < knot_u - 1; ++i) {
    const auto &t1 = tangents[i];
    const auto &t2 = tangents[i + 1];
    float ax = t1[1] * t2[2] - t1[2] * t2[1];
    float ay = t1[2] * t2[0] - t1[0] * t2[2];
    float az = t1[0] * t2[1] - t1[1] * t2[0];
    float alen = std::sqrt(ax * ax + ay * ay + az * az);
    if (alen < 1e-6f) {
      normals[i + 1] = normals[i];
    } else {
      ax /= alen; ay /= alen; az /= alen;
      float cos_ang = std::clamp(t1[0]*t2[0] + t1[1]*t2[1] + t1[2]*t2[2], -1.0f, 1.0f);
      float ang = std::acos(cos_ang);
      float c = std::cos(ang), s = std::sin(ang);
      const auto &n1 = normals[i];
      float cross_x = ay * n1[2] - az * n1[1];
      float cross_y = az * n1[0] - ax * n1[2];
      float cross_z = ax * n1[1] - ay * n1[0];
      float dot_an = ax * n1[0] + ay * n1[1] + az * n1[2];
      float nx = n1[0] * c + cross_x * s + ax * dot_an * (1.0f - c);
      float ny = n1[1] * c + cross_y * s + ay * dot_an * (1.0f - c);
      float nz = n1[2] * c + cross_z * s + az * dot_an * (1.0f - c);
      float d = nx * t2[0] + ny * t2[1] + nz * t2[2];
      nx -= d * t2[0]; ny -= d * t2[1]; nz -= d * t2[2];
      float nlen = std::sqrt(nx * nx + ny * ny + nz * nz);
      normals[i + 1] = {nx / nlen, ny / nlen, nz / nlen};
    }
    binormals[i + 1] = {
      t2[1] * normals[i + 1][2] - t2[2] * normals[i + 1][1],
      t2[2] * normals[i + 1][0] - t2[0] * normals[i + 1][2],
      t2[0] * normals[i + 1][1] - t2[1] * normals[i + 1][0]
    };
  }

  // Loop closure holonomy correction:
  const auto &t_last = tangents[knot_u - 1];
  const auto &t_0 = tangents[0];
  float ax = t_last[1] * t_0[2] - t_last[2] * t_0[1];
  float ay = t_last[2] * t_0[0] - t_last[0] * t_0[2];
  float az = t_last[0] * t_0[1] - t_last[1] * t_0[0];
  float alen = std::sqrt(ax * ax + ay * ay + az * az);
  ax /= alen; ay /= alen; az /= alen;
  float cos_ang = std::clamp(t_last[0]*t_0[0] + t_last[1]*t_0[1] + t_last[2]*t_0[2], -1.0f, 1.0f);
  float ang = std::acos(cos_ang);
  float c = std::cos(ang), s = std::sin(ang);
  const auto &n_last = normals[knot_u - 1];
  float cross_x = ay * n_last[2] - az * n_last[1];
  float cross_y = az * n_last[0] - ax * n_last[2];
  float cross_z = ax * n_last[1] - ay * n_last[0];
  float dot_an = ax * n_last[0] + ay * n_last[1] + az * n_last[2];
  float n_cl_x = n_last[0] * c + cross_x * s + ax * dot_an * (1.0f - c);
  float n_cl_y = n_last[1] * c + cross_y * s + ay * dot_an * (1.0f - c);
  float n_cl_z = n_last[2] * c + cross_z * s + az * dot_an * (1.0f - c);
  float cos_holo = std::clamp(n_cl_x * normals[0][0] + n_cl_y * normals[0][1] + n_cl_z * normals[0][2], -1.0f, 1.0f);
  float cr_x = n_cl_y * normals[0][2] - n_cl_z * normals[0][1];
  float cr_y = n_cl_z * normals[0][0] - n_cl_x * normals[0][2];
  float cr_z = n_cl_x * normals[0][1] - n_cl_y * normals[0][0];
  float sin_holo = cr_x * t_0[0] + cr_y * t_0[1] + cr_z * t_0[2];
  float holo_angle = std::atan2(sin_holo, cos_holo);

  // Distribute twist evenly:
  for (uint32_t i = 0; i < knot_u; ++i) {
    float corr_ang = holo_angle * (float(i) / float(knot_u));
    float ca = std::cos(corr_ang), sa = std::sin(corr_ang);
    auto n = normals[i];
    auto b = binormals[i];
    normals[i] = {ca * n[0] + sa * b[0], ca * n[1] + sa * b[1], ca * n[2] + sa * b[2]};
    binormals[i] = {-sa * n[0] + ca * b[0], -sa * n[1] + ca * b[1], -sa * n[2] + ca * b[2]};
  }

  // Calibrated 3D Euler rotation matching Blender showroom scene
  const float R[3][3] = {
    { 0.929662f,  0.326751f,  0.170182f},
    {-0.239757f,  0.887321f, -0.393927f},
    {-0.279722f,  0.325417f,  0.903249f}
  };

  std::vector<std::vector<std::array<float, 3>>> knot_grid(knot_u, std::vector<std::array<float, 3>>(knot_v));
  for (uint32_t i = 0; i < knot_u; ++i) {
    const auto &c = centers[i];
    const auto &nx = normals[i];
    const auto &bx = binormals[i];
    for (uint32_t j = 0; j < knot_v; ++j) {
      float v = (float(j) / float(knot_v)) * pi2;
      float cv = std::cos(v), sv = std::sin(v);
      float px = c[0] + r_tube * (cv * nx[0] + sv * bx[0]);
      float py = c[1] + r_tube * (cv * nx[1] + sv * bx[1]);
      float pz = c[2] + r_tube * (cv * nx[2] + sv * bx[2]);
      float rx = R[0][0] * px + R[0][1] * py + R[0][2] * pz;
      float ry = R[1][0] * px + R[1][1] * py + R[1][2] * pz;
      float rz = R[2][0] * px + R[2][1] * py + R[2][2] * pz + 0.385f;
      knot_grid[i][j] = {rx, ry, rz};
    }
  }

  // Triangulate Knot with outward-facing normals
  for (uint32_t i = 0; i < knot_u; ++i) {
    uint32_t i_next = (i + 1) % knot_u;
    for (uint32_t j = 0; j < knot_v; ++j) {
      uint32_t j_next = (j + 1) % knot_v;
      const auto &p00 = knot_grid[i][j];
      const auto &p10 = knot_grid[i_next][j];
      const auto &p11 = knot_grid[i_next][j_next];
      const auto &p01 = knot_grid[i][j_next];

      // Outward normal winding
      vertices.insert(vertices.end(), {p00[0], p00[1], p00[2], p11[0], p11[1], p11[2], p10[0], p10[1], p10[2]});
      vertices.insert(vertices.end(), {p00[0], p00[1], p00[2], p01[0], p01[1], p01[2], p11[0], p11[1], p11[2]});
    }
  }
}

inline void appendSphere(std::vector<float> &vertices, float r, const std::array<float, 3> &center) {
  const uint32_t lat_segs = 32, lon_segs = 32;
  const float pi = 3.14159265f, pi2 = 6.283185307f;

  std::vector<std::vector<std::array<float, 3>>> grid(lat_segs, std::vector<std::array<float, 3>>(lon_segs));
  for (uint32_t i = 0; i < lat_segs; ++i) {
    float lat = (float(i) / float(lat_segs - 1)) * pi;
    for (uint32_t j = 0; j < lon_segs; ++j) {
      float lon = (float(j) / float(lon_segs)) * pi2;
      float x = r * std::sin(lat) * std::cos(lon) + center[0];
      float y = r * std::sin(lat) * std::sin(lon) + center[1];
      float z = r * std::cos(lat) + center[2];
      grid[i][j] = {x, y, z};
    }
  }

  for (uint32_t i = 0; i < lat_segs - 1; ++i) {
    for (uint32_t j = 0; j < lon_segs; ++j) {
      uint32_t j_next = (j + 1) % lon_segs;
      const auto &p00 = grid[i][j];
      const auto &p10 = grid[i + 1][j];
      const auto &p11 = grid[i + 1][j_next];
      const auto &p01 = grid[i][j_next];

      if (i > 0) {
        vertices.insert(vertices.end(), {p00[0], p00[1], p00[2], p10[0], p10[1], p10[2], p11[0], p11[1], p11[2]});
      }
      if (i < lat_segs - 2) {
        vertices.insert(vertices.end(), {p00[0], p00[1], p00[2], p11[0], p11[1], p11[2], p01[0], p01[1], p01[2]});
      }
    }
  }
}

inline void appendCylinder(std::vector<float> &vertices, float r, float z_bot, float z_top, const std::array<float, 2> &center) {
  const uint32_t segs = 32;
  const float pi2 = 6.283185307f;

  for (uint32_t i = 0; i < segs; ++i) {
    float a0 = (float(i) / float(segs)) * pi2;
    float a1 = (float(i + 1) / float(segs)) * pi2;
    float x0 = center[0] + r * std::cos(a0), y0 = center[1] + r * std::sin(a0);
    float x1 = center[0] + r * std::cos(a1), y1 = center[1] + r * std::sin(a1);

    vertices.insert(vertices.end(), {x0, y0, z_bot, x1, y1, z_bot, x1, y1, z_top});
    vertices.insert(vertices.end(), {x0, y0, z_bot, x1, y1, z_top, x0, y0, z_top});

    vertices.insert(vertices.end(), {center[0], center[1], z_top, x0, y0, z_top, x1, y1, z_top});
    vertices.insert(vertices.end(), {center[0], center[1], z_bot, x1, y1, z_bot, x0, y0, z_bot});
  }
}

inline void appendSuzanne(std::vector<float> &vertices) {
  const float R[3][3] = {
      { 0.929732f,  0.159532f, -0.331887f},
      {-0.343366f,  0.701199f, -0.624836f},
      { 0.133037f,  0.694888f,  0.706705f}
  };
  const float s = 1.05f;
  const float tx = 2.71f, ty = -0.57f, tz = -0.27f;

  for (uint32_t i = 0; i < SUZANNE_NUM_TRIANGLES; ++i) {
    for (int v = 0; v < 3; ++v) {
      float px = SUZANNE_VERTICES[i * 9 + v * 3 + 0] * s;
      float py = SUZANNE_VERTICES[i * 9 + v * 3 + 1] * s;
      float pz = SUZANNE_VERTICES[i * 9 + v * 3 + 2] * s;

      float x = R[0][0] * px + R[0][1] * py + R[0][2] * pz + tx;
      float y = R[1][0] * px + R[1][1] * py + R[1][2] * pz + ty;
      float z = R[2][0] * px + R[2][1] * py + R[2][2] * pz + tz;
      vertices.insert(vertices.end(), {x, y, z});
    }
  }
}

inline void appendFloor(std::vector<float> &vertices) {
  // Showroom Cyclorama Floor: 128 x 64 quads = 16,384 triangles
  // Extends far horizontally (-32 to +32) and curves up seamlessly in the back (v > 1.0)
  const uint32_t dish_u = 128;
  const uint32_t dish_v = 64;
  for (uint32_t i = 0; i < dish_u; ++i) {
    float u0 = -32.0f + (float(i) / float(dish_u)) * 64.0f;
    float u1 = -32.0f + (float(i + 1) / float(dish_u)) * 64.0f;
    for (uint32_t j = 0; j < dish_v; ++j) {
      float v0 = -16.0f + (float(j) / float(dish_v)) * 48.0f;
      float v1 = -16.0f + (float(j + 1) / float(dish_v)) * 48.0f;

      auto dish_z = [](float u, float v) -> float {
        float back_curve = (v > 1.0f) ? 0.022f * (v - 1.0f) * (v - 1.0f) : 0.0f;
        return -1.056f + back_curve;
      };

      float z00 = dish_z(u0, v0);
      float z10 = dish_z(u1, v0);
      float z11 = dish_z(u1, v1);
      float z01 = dish_z(u0, v1);

      vertices.insert(vertices.end(), {u0, v0, z00, u1, v0, z10, u1, v1, z11});
      vertices.insert(vertices.end(), {u0, v0, z00, u1, v1, z11, u0, v1, z01});
    }
  }
}

inline void buildShowroomScene(std::vector<float> &vertices) {
  vertices.clear();
  vertices.reserve(PRIM_FLOOR_END * 9);

  // 1. Trefoil Knot with 5 Material Bands: 7,680 tris (Primitives 0 .. 7679)
  appendTrefoilKnot(vertices);

  // 2. Jade Subsurface Sphere (Left): 1,920 tris (Primitives 7680 .. 9599)
  appendSphere(vertices, 0.548f, {-3.242f, 0.391f, -0.509f});

  // 3. Gold Cylinder Pedestal (Center): 128 tris (Primitives 9600 .. 9727)
  appendCylinder(vertices, 0.542f, -1.056f, -0.654f, {-0.460f, 0.060f});

  // 4. Suzanne Dispersive Glass Head (Right): 968 tris (Primitives 9728 .. 10695)
  appendSuzanne(vertices);

  // 5. Showroom Cyclorama Floor: 16,384 tris (Primitives 10696 .. 27079)
  appendFloor(vertices);
}

} // namespace ShowroomScene
