#pragma once

#include <array>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdint>

namespace AAAForestScene {

// Material Archetype IDs for Nature PBR
constexpr uint32_t MAT_LEAVES   = 0u; // Canopy Leaves & Needles (Two-sided transmission)
constexpr uint32_t MAT_BARK     = 1u; // Tree Bark & Roots (Vertical anisotropic GGX)
constexpr uint32_t MAT_ROCK     = 2u; // Granite Cliffs & Boulders (Tri-planar normal mapping)
constexpr uint32_t MAT_DIRT     = 3u; // Topsoil, Path Gravel & Wet Mud (Porous diffuse + wetness)
constexpr uint32_t MAT_GRASS    = 4u; // Alpine Meadow Grass & Ferns (Grazing Charlie sheen)
constexpr uint32_t MAT_WATER    = 5u; // River Water Surface & Bathymetry (Snell refraction + Beer-Lambert)
constexpr uint32_t MAT_SNOW     = 6u; // Alpine Snow & Glacial Frost (Micro-glint sparkle)
constexpr uint32_t MAT_TIMBER   = 7u; // Weathered Timber Bridge & Masonry Ruins

struct Vertex12 {
  float pos[3];
  float normal[3];
  float tangent[4];
  float uv[2];
};

// Analytical woodland terrain heightfield
inline float forestTerrainHeight(float x, float y) {
  // Gentle winding woodland trail along x = -1.5 + 2.5 * sin(y * 0.04)
  float trailCenter = -1.5f + 2.5f * std::sin(y * 0.04f);
  float distToTrail = std::abs(x - trailCenter);
  float trailDepression = -0.28f * std::exp(-(distToTrail * distToTrail) / 10.0f);

  // Winding woodland stream corridor along x = 20.0 + 4.5 * sin(y * 0.03)
  float streamCenter = 20.0f + 4.5f * std::sin(y * 0.03f);
  float distToStream = std::abs(x - streamCenter);
  float streamBed = -1.40f * std::exp(-(distToStream * distToStream) / 22.0f);

  // Organic woodland mounds, mossy knolls, and rolling forest floor
  float knolls = 1.10f * std::sin(x * 0.075f + y * 0.055f) +
                 0.65f * std::cos(x * 0.140f - y * 0.120f) +
                 0.30f * std::sin(x * 0.280f + y * 0.240f) +
                 0.12f * std::cos(x * 0.520f - y * 0.480f);

  // Gentle valley flanks rising on distant sides (|x| > 25)
  float flanks = 0.0f;
  if (std::abs(x) > 25.0f) {
    float dx = std::abs(x) - 25.0f;
    flanks = dx * 0.12f + 0.0015f * dx * dx;
  }

  // Slight rise into background grove (y > 40)
  float backgroundRise = (y > 40.0f) ? (y - 40.0f) * 0.035f : 0.0f;

  return trailDepression + streamBed + knolls + flanks + backgroundRise;
}

// Analytical gradient for smooth continuous vertex normals
inline std::array<float, 3> getTerrainNormal(float x, float y) {
  constexpr float eps = 0.25f;
  float hL = forestTerrainHeight(x - eps, y);
  float hR = forestTerrainHeight(x + eps, y);
  float hD = forestTerrainHeight(x, y - eps);
  float hU = forestTerrainHeight(x, y + eps);

  float dzdx = (hR - hL) / (2.0f * eps);
  float dzdy = (hU - hD) / (2.0f * eps);

  float nx = -dzdx;
  float ny = -dzdy;
  float nz = 1.0f;
  float invLen = 1.0f / std::sqrt(nx * nx + ny * ny + nz * nz);
  return {nx * invLen, ny * invLen, nz * invLen};
}

inline void appendTri(std::vector<float> &vertices,
                      std::vector<uint32_t> &triangleMats,
                      const Vertex12 &v0,
                      const Vertex12 &v1,
                      const Vertex12 &v2,
                      uint32_t matId) {
  auto pushV = [&](const Vertex12 &v) {
    vertices.push_back(v.pos[0]);
    vertices.push_back(v.pos[1]);
    vertices.push_back(v.pos[2]);
    vertices.push_back(v.normal[0]);
    vertices.push_back(v.normal[1]);
    vertices.push_back(v.normal[2]);
    vertices.push_back(v.tangent[0]);
    vertices.push_back(v.tangent[1]);
    vertices.push_back(v.tangent[2]);
    vertices.push_back(v.tangent[3]);
    vertices.push_back(v.uv[0]);
    vertices.push_back(v.uv[1]);
  };
  pushV(v0);
  pushV(v1);
  pushV(v2);
  triangleMats.push_back(matId);
}

inline void appendQuad(std::vector<float> &vertices,
                       std::vector<uint32_t> &triangleMats,
                       const Vertex12 &v0,
                       const Vertex12 &v1,
                       const Vertex12 &v2,
                       const Vertex12 &v3,
                       uint32_t matId) {
  appendTri(vertices, triangleMats, v0, v1, v2, matId);
  appendTri(vertices, triangleMats, v0, v2, v3, matId);
}

// 1. Woodland Floor & Stream Corridor Terrain (256x256 grid = 131,072 triangles)
inline void appendWoodlandTerrain(std::vector<float> &vertices,
                                  std::vector<uint32_t> &triangleMats) {
  const uint32_t grid_n = 256;
  const float x_min = -120.0f, x_max = 120.0f;
  const float y_min = -60.0f,  y_max = 180.0f;

  auto sampleGridVertex = [&](uint32_t i, uint32_t j) -> Vertex12 {
    float u = float(i) / float(grid_n);
    float v = float(j) / float(grid_n);
    float x = x_min + (x_max - x_min) * u;
    float y = y_min + (y_max - y_min) * v;
    float z = forestTerrainHeight(x, y);
    auto n = getTerrainNormal(x, y);

    float tx = 1.0f - n[0] * n[0];
    float ty = -n[0] * n[1];
    float tz = -n[0] * n[2];
    float tlen = std::max(0.001f, std::sqrt(tx * tx + ty * ty + tz * tz));

    Vertex12 vert;
    vert.pos[0] = x; vert.pos[1] = y; vert.pos[2] = z;
    vert.normal[0] = n[0]; vert.normal[1] = n[1]; vert.normal[2] = n[2];
    vert.tangent[0] = tx / tlen; vert.tangent[1] = ty / tlen; vert.tangent[2] = tz / tlen; vert.tangent[3] = 1.0f;
    vert.uv[0] = x * 0.15f; vert.uv[1] = y * 0.15f;
    return vert;
  };

  for (uint32_t i = 0; i < grid_n; ++i) {
    for (uint32_t j = 0; j < grid_n; ++j) {
      Vertex12 p00 = sampleGridVertex(i, j);
      Vertex12 p10 = sampleGridVertex(i + 1, j);
      Vertex12 p11 = sampleGridVertex(i + 1, j + 1);
      Vertex12 p01 = sampleGridVertex(i, j + 1);

      auto classifyMat = [&](float x, float y, float z, float nz) -> uint32_t {
        float trailCenter = -1.5f + 2.5f * std::sin(y * 0.04f);
        float distToTrail = std::abs(x - trailCenter);
        if (distToTrail < 2.0f) return MAT_DIRT; // Packed trail gravel/dirt

        float streamCenter = 20.0f + 4.5f * std::sin(y * 0.03f);
        float distToStream = std::abs(x - streamCenter);
        if (distToStream < 3.2f) return MAT_ROCK; // Stream bed pebbles & boulders

        if (nz < 0.68f) return MAT_ROCK; // Exposed granite knoll faces
        return MAT_GRASS; // Mossy forest duff & woodland grass
      };

      float cx0 = (p00.pos[0] + p10.pos[0] + p11.pos[0]) / 3.0f;
      float cy0 = (p00.pos[1] + p10.pos[1] + p11.pos[1]) / 3.0f;
      float cz0 = (p00.pos[2] + p10.pos[2] + p11.pos[2]) / 3.0f;
      float cnz0 = (p00.normal[2] + p10.normal[2] + p11.normal[2]) / 3.0f;
      uint32_t mat0 = classifyMat(cx0, cy0, cz0, cnz0);

      float cx1 = (p00.pos[0] + p11.pos[0] + p01.pos[0]) / 3.0f;
      float cy1 = (p00.pos[1] + p11.pos[1] + p01.pos[1]) / 3.0f;
      float cz1 = (p00.pos[2] + p11.pos[2] + p01.pos[2]) / 3.0f;
      float cnz1 = (p00.normal[2] + p11.normal[2] + p01.normal[2]) / 3.0f;
      uint32_t mat1 = classifyMat(cx1, cy1, cz1, cnz1);

      appendTri(vertices, triangleMats, p00, p10, p11, mat0);
      appendTri(vertices, triangleMats, p00, p11, p01, mat1);
    }
  }
}

// 2. Stream Water Ribbon (128x8 quads = 2,048 triangles)
inline void appendStreamWater(std::vector<float> &vertices,
                              std::vector<uint32_t> &triangleMats) {
  const uint32_t length_segs = 128;
  const uint32_t width_segs = 8;
  const float y_min = -60.0f, y_max = 180.0f;
  const float stream_half_w = 3.2f;

  for (uint32_t i = 0; i < length_segs; ++i) {
    float v0 = float(i) / float(length_segs);
    float v1 = float(i + 1) / float(length_segs);
    float y0 = y_min + (y_max - y_min) * v0;
    float y1 = y_min + (y_max - y_min) * v1;

    float cx0 = 20.0f + 4.5f * std::sin(y0 * 0.03f);
    float cx1 = 20.0f + 4.5f * std::sin(y1 * 0.03f);
    float z0 = -0.55f + 0.005f * y0;
    float z1 = -0.55f + 0.005f * y1;

    for (uint32_t j = 0; j < width_segs; ++j) {
      float u0 = float(j) / float(width_segs);
      float u1 = float(j + 1) / float(width_segs);
      float offset0 = (u0 * 2.0f - 1.0f) * stream_half_w;
      float offset1 = (u1 * 2.0f - 1.0f) * stream_half_w;

      Vertex12 p00{{cx0 + offset0, y0, z0}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {u0, v0 * 8.0f}};
      Vertex12 p10{{cx0 + offset1, y0, z0}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {u1, v0 * 8.0f}};
      Vertex12 p11{{cx1 + offset1, y1, z1}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {u1, v1 * 8.0f}};
      Vertex12 p01{{cx1 + offset0, y1, z1}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {u0, v1 * 8.0f}};

      appendQuad(vertices, triangleMats, p00, p10, p11, p01, MAT_WATER);
    }
  }
}

// 3. Mature Old-Growth Conifer / Pine Tree with Soaring Trunk & High Needle Canopy (~1,300 triangles per tree)
// Clear trunk extends up to 6.0m - 8.0m, branches arch overhead in a natural forest cathedral
inline void appendConiferTree(std::vector<float> &vertices,
                              std::vector<uint32_t> &triangleMats,
                              float root_x, float root_y, float scale, uint32_t seed) {
  float root_z = forestTerrainHeight(root_x, root_y);
  const float pi2 = 6.283185307179586f;

  auto prng = [&](uint32_t s) -> float {
    uint32_t state = s * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return float((word >> 22u) ^ word) / 4294967295.0f;
  };

  // Tall straight trunk: 8 radial segments x 7 slices = 56 quads = 112 triangles
  const uint32_t trunk_segs = 8;
  const uint32_t trunk_slices = 7;
  const float r_base = 0.55f * scale;
  const float h_tree = 28.0f * scale;

  for (uint32_t slice = 0; slice < trunk_slices; ++slice) {
    float t0 = float(slice) / float(trunk_slices);
    float t1 = float(slice + 1) / float(trunk_slices);
    float z0 = root_z + t0 * h_tree;
    float z1 = root_z + t1 * h_tree;
    float r0 = r_base * (1.0f - t0 * 0.88f);
    float r1 = r_base * (1.0f - t1 * 0.88f);

    for (uint32_t i = 0; i < trunk_segs; ++i) {
      float a0 = (float(i) / float(trunk_segs)) * pi2;
      float a1 = (float(i + 1) / float(trunk_segs)) * pi2;
      float cos0 = std::cos(a0), sin0 = std::sin(a0);
      float cos1 = std::cos(a1), sin1 = std::sin(a1);

      Vertex12 b0{{root_x + r0 * cos0, root_y + r0 * sin0, z0}, {cos0, sin0, 0.0f}, {-sin0, cos0, 0.0f, 1.0f}, {float(i), t0 * 6.0f}};
      Vertex12 b1{{root_x + r0 * cos1, root_y + r0 * sin1, z0}, {cos1, sin1, 0.0f}, {-sin1, cos1, 0.0f, 1.0f}, {float(i + 1), t0 * 6.0f}};
      Vertex12 top0{{root_x + r1 * cos0, root_y + r1 * sin0, z1}, {cos0, sin0, 0.0f}, {-sin0, cos0, 0.0f, 1.0f}, {float(i), t1 * 6.0f}};
      Vertex12 top1{{root_x + r1 * cos1, root_y + r1 * sin1, z1}, {cos1, sin1, 0.0f}, {-sin1, cos1, 0.0f, 1.0f}, {float(i + 1), t1 * 6.0f}};

      appendQuad(vertices, triangleMats, b0, b1, top1, top0, MAT_BARK);
    }
  }

  // 10 Branch Tiers starting at z = root_z + 6.5 * scale (clear cathedral understory)
  const uint32_t tiers = 10;
  const uint32_t branches_per_tier = 5;

  for (uint32_t tier = 0; tier < tiers; ++tier) {
    float tier_t = float(tier) / float(tiers);
    float tier_z = root_z + (6.5f + tier_t * 19.5f) * scale;
    float tier_radius = (6.0f * (1.0f - tier_t * 0.72f)) * scale;
    float tier_rot = tier_t * 2.7f + prng(seed + tier * 31u) * 0.35f;

    for (uint32_t b = 0; b < branches_per_tier; ++b) {
      float angle = (float(b) / float(branches_per_tier)) * pi2 + tier_rot;
      float b_cos = std::cos(angle);
      float b_sin = std::sin(angle);

      float tip_x = root_x + tier_radius * b_cos;
      float tip_y = root_y + tier_radius * b_sin;
      float tip_z = tier_z - tier_radius * 0.16f; // Natural downward bough droop

      // Branch spine quad (2 triangles, MAT_BARK)
      Vertex12 sp0{{root_x + b_cos * 0.25f * scale, root_y + b_sin * 0.25f * scale, tier_z}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}};
      Vertex12 sp1{{tip_x, tip_y, tip_z}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}};
      Vertex12 sp2{{tip_x, tip_y, tip_z - 0.12f * scale}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}};
      Vertex12 sp3{{root_x + b_cos * 0.25f * scale, root_y + b_sin * 0.25f * scale, tier_z - 0.15f * scale}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}};
      appendQuad(vertices, triangleMats, sp0, sp1, sp2, sp3, MAT_BARK);

      // 6 Intersecting Cross-Needle Spray Pairs along the branch arm (12 quads = 24 triangles)
      const uint32_t needle_pairs = 6;
      for (uint32_t np = 0; np < needle_pairs; ++np) {
        float np_t = float(np + 1) / float(needle_pairs + 1);
        float cx = root_x + (tip_x - root_x) * np_t;
        float cy = root_y + (tip_y - root_y) * np_t;
        float cz = tier_z + (tip_z - tier_z) * np_t;

        float card_w = (0.70f - np_t * 0.22f) * scale;
        float card_len = (0.85f - np_t * 0.18f) * scale;

        float perp_x = -b_sin;
        float perp_y = b_cos;

        // Plane A: horizontal/slight pitch
        float dxA = perp_x * card_w;
        float dyA = perp_y * card_w;
        float dzA = 0.12f * card_w;

        float fx = b_cos * card_len;
        float fy = b_sin * card_len;
        float fz = -0.10f * card_len;

        Vertex12 a0{{cx - dxA, cy - dyA, cz - dzA}, {-dyA, dxA, 0.6f}, {b_cos, b_sin, 0.0f, 1.0f}, {0.0f, 0.0f}};
        Vertex12 a1{{cx + dxA, cy + dyA, cz + dzA}, {-dyA, dxA, 0.6f}, {b_cos, b_sin, 0.0f, 1.0f}, {1.0f, 0.0f}};
        Vertex12 a2{{cx + dxA * 0.5f + fx, cy + dyA * 0.5f + fy, cz + dzA * 0.5f + fz}, {-dyA, dxA, 0.6f}, {b_cos, b_sin, 0.0f, 1.0f}, {1.0f, 1.0f}};
        Vertex12 a3{{cx - dxA * 0.5f + fx, cy - dyA * 0.5f + fy, cz - dzA * 0.5f + fz}, {-dyA, dxA, 0.6f}, {b_cos, b_sin, 0.0f, 1.0f}, {0.0f, 1.0f}};
        appendQuad(vertices, triangleMats, a0, a1, a2, a3, MAT_LEAVES);

        // Plane B: angled at 50 degrees for authentic volumetric needle thickness
        float dxB = perp_x * card_w * 0.68f;
        float dyB = perp_y * card_w * 0.68f;
        float dzB = card_w * 0.68f;

        Vertex12 b0_c{{cx - dxB, cy - dyB, cz - dzB}, {0.0f, 0.0f, 1.0f}, {b_cos, b_sin, 0.0f, 1.0f}, {0.0f, 0.0f}};
        Vertex12 b1_c{{cx + dxB, cy + dyB, cz + dzB}, {0.0f, 0.0f, 1.0f}, {b_cos, b_sin, 0.0f, 1.0f}, {1.0f, 0.0f}};
        Vertex12 b2_c{{cx + dxB * 0.5f + fx, cy + dyB * 0.5f + fy, cz + dzB * 0.5f + fz}, {0.0f, 0.0f, 1.0f}, {b_cos, b_sin, 0.0f, 1.0f}, {1.0f, 1.0f}};
        Vertex12 b3_c{{cx - dxB * 0.5f + fx, cy - dyB * 0.5f + fy, cz - dzB * 0.5f + fz}, {0.0f, 0.0f, 1.0f}, {b_cos, b_sin, 0.0f, 1.0f}, {0.0f, 1.0f}};
        appendQuad(vertices, triangleMats, b0_c, b1_c, b2_c, b3_c, MAT_LEAVES);
      }
    }
  }

  // Apex needle spray (8 quads = 16 triangles, MAT_LEAVES)
  float apex_z = root_z + h_tree;
  for (uint32_t a = 0; a < 4; ++a) {
    float ang = float(a) * (pi2 / 4.0f);
    float c = std::cos(ang) * 0.85f * scale;
    float s = std::sin(ang) * 0.85f * scale;
    Vertex12 a0{{root_x - c, root_y - s, apex_z - 1.2f * scale}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}};
    Vertex12 a1{{root_x + c, root_y + s, apex_z - 1.2f * scale}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}};
    Vertex12 a2{{root_x + c * 0.2f, root_y + s * 0.2f, apex_z + 0.6f * scale}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}};
    Vertex12 a3{{root_x - c * 0.2f, root_y - s * 0.2f, apex_z + 0.6f * scale}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}};
    appendQuad(vertices, triangleMats, a0, a1, a2, a3, MAT_LEAVES);
  }
}

// 4. Deciduous / Mountain Birch Tree with High Branching Canopy (~512 triangles per tree)
inline void appendDeciduousTree(std::vector<float> &vertices,
                                std::vector<uint32_t> &triangleMats,
                                float root_x, float root_y, float scale, uint32_t seed) {
  float root_z = forestTerrainHeight(root_x, root_y);
  const float pi2 = 6.283185307179586f;

  auto prng = [&](uint32_t s) -> float {
    uint32_t state = s * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return float((word >> 22u) ^ word) / 4294967295.0f;
  };

  // Trunk: 8 radial segments x 5 slices = 40 quads = 80 triangles (MAT_BARK)
  const uint32_t trunk_segs = 8;
  const uint32_t trunk_slices = 5;
  const float r_base = 0.40f * scale;
  const float h_trunk = 16.0f * scale;

  for (uint32_t slice = 0; slice < trunk_slices; ++slice) {
    float t0 = float(slice) / float(trunk_slices);
    float t1 = float(slice + 1) / float(trunk_slices);
    float z0 = root_z + t0 * h_trunk;
    float z1 = root_z + t1 * h_trunk;
    float r0 = r_base * (1.0f - t0 * 0.70f);
    float r1 = r_base * (1.0f - t1 * 0.70f);

    for (uint32_t i = 0; i < trunk_segs; ++i) {
      float a0 = (float(i) / float(trunk_segs)) * pi2;
      float a1 = (float(i + 1) / float(trunk_segs)) * pi2;
      float cos0 = std::cos(a0), sin0 = std::sin(a0);
      float cos1 = std::cos(a1), sin1 = std::sin(a1);

      Vertex12 b0{{root_x + r0 * cos0, root_y + r0 * sin0, z0}, {cos0, sin0, 0.0f}, {-sin0, cos0, 0.0f, 1.0f}, {float(i), t0 * 4.0f}};
      Vertex12 b1{{root_x + r0 * cos1, root_y + r0 * sin1, z0}, {cos1, sin1, 0.0f}, {-sin1, cos1, 0.0f, 1.0f}, {float(i + 1), t0 * 4.0f}};
      Vertex12 top0{{root_x + r1 * cos0, root_y + r1 * sin0, z1}, {cos0, sin0, 0.0f}, {-sin0, cos0, 0.0f, 1.0f}, {float(i), t1 * 4.0f}};
      Vertex12 top1{{root_x + r1 * cos1, root_y + r1 * sin1, z1}, {cos1, sin1, 0.0f}, {-sin1, cos1, 0.0f, 1.0f}, {float(i + 1), t1 * 4.0f}};

      appendQuad(vertices, triangleMats, b0, b1, top1, top0, MAT_BARK);
    }
  }

  // 3 High Branching Boughs (16 triangles each = 48 triangles, MAT_BARK) starting at z = 7.5m
  for (int limb = 0; limb < 3; ++limb) {
    float limb_ang = float(limb) * (pi2 / 3.0f) + 0.35f;
    float lx = root_x + std::cos(limb_ang) * 4.0f * scale;
    float ly = root_y + std::sin(limb_ang) * 4.0f * scale;
    float lz = root_z + (h_trunk * 0.65f + 3.5f) * scale;
    for (uint32_t i = 0; i < 4; ++i) {
      float a0 = (float(i) / 4.0f) * pi2;
      float a1 = (float(i + 1) / 4.0f) * pi2;
      Vertex12 b0{{root_x + std::cos(a0) * 0.20f * scale, root_y + std::sin(a0) * 0.20f * scale, root_z + h_trunk * 0.60f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}};
      Vertex12 b1{{root_x + std::cos(a1) * 0.20f * scale, root_y + std::sin(a1) * 0.20f * scale, root_z + h_trunk * 0.60f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}};
      Vertex12 t0{{lx + std::cos(a0) * 0.09f * scale, ly + std::sin(a0) * 0.09f * scale, lz}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}};
      Vertex12 t1{{lx + std::cos(a1) * 0.09f * scale, ly + std::sin(a1) * 0.09f * scale, lz}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}};
      appendQuad(vertices, triangleMats, b0, b1, t1, t0, MAT_BARK);
    }
  }

  // 24 Volumetric Leaf Card Clusters in the high canopy (384 triangles, MAT_LEAVES)
  const uint32_t num_clusters = 24;
  for (uint32_t c = 0; c < num_clusters; ++c) {
    float cu = prng(seed + c * 43u);
    float cv = prng(seed + c * 71u);
    float cw = prng(seed + c * 97u);

    float theta = cu * pi2;
    float phi = cv * 3.14159f * 0.70f;
    float rad = (2.2f + cw * 3.4f) * scale;

    float cx = root_x + rad * std::sin(phi) * std::cos(theta);
    float cy = root_y + rad * std::sin(phi) * std::sin(theta);
    float cz = root_z + (h_trunk * 0.62f + rad * std::cos(phi) + 3.0f) * scale;

    const uint32_t cards = 8;
    for (uint32_t k = 0; k < cards; ++k) {
      float card_ang = float(k) * (pi2 / float(cards)) + prng(seed + k * 13u) * 0.5f;
      float card_tilt = (float(k % 3) - 1.0f) * 0.55f;
      float cw_size = 0.95f * scale;

      float cosA = std::cos(card_ang) * cw_size;
      float sinA = std::sin(card_ang) * cw_size;
      float dz = std::sin(card_tilt) * cw_size;

      Vertex12 v0{{cx - cosA, cy - sinA, cz - dz}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}};
      Vertex12 v1{{cx + cosA, cy + sinA, cz + dz}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}};
      Vertex12 v2{{cx + cosA * 0.3f, cy + sinA * 0.3f, cz + cw_size + dz}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}};
      Vertex12 v3{{cx - cosA * 0.3f, cy - sinA * 0.3f, cz + cw_size - dz}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}};

      appendQuad(vertices, triangleMats, v0, v1, v2, v3, MAT_LEAVES);
    }
  }
}

// 5. Forest Floor Arching Fern Clumps (8 fronds x 4 segments = 32 quads = 64 triangles per clump)
inline void appendFernClump(std::vector<float> &vertices,
                            std::vector<uint32_t> &triangleMats,
                            float cx, float cy, float scale, uint32_t seed) {
  float cz = forestTerrainHeight(cx, cy);
  const float pi2 = 6.283185307179586f;

  auto prng = [&](uint32_t s) -> float {
    uint32_t state = s * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return float((word >> 22u) ^ word) / 4294967295.0f;
  };

  const uint32_t num_fronds = 8;
  for (uint32_t f = 0; f < num_fronds; ++f) {
    float base_angle = (float(f) / float(num_fronds)) * pi2 + prng(seed + f * 19u) * 0.3f;
    float cosF = std::cos(base_angle);
    float sinF = std::sin(base_angle);
    float frond_len = (1.5f + prng(seed + f * 41u) * 0.5f) * scale;

    float prev_x = cx;
    float prev_y = cy;
    float prev_z = cz;
    float prev_w = 0.06f * scale;

    for (int seg = 0; seg < 4; ++seg) {
      float t = float(seg + 1) / 4.0f;
      float seg_dist = frond_len * t;
      float seg_z = cz + (1.1f * t - 0.65f * t * t) * frond_len;
      float cur_x = cx + cosF * seg_dist;
      float cur_y = cy + sinF * seg_dist;
      float cur_z = seg_z;
      float cur_w = (0.22f * std::sin(t * 3.14159f) + 0.03f) * scale;

      float px = -sinF;
      float py = cosF;

      Vertex12 v0{{prev_x - px * prev_w, prev_y - py * prev_w, prev_z}, {0.0f, 0.0f, 1.0f}, {cosF, sinF, 0.0f, 1.0f}, {0.0f, float(seg) * 0.25f}};
      Vertex12 v1{{prev_x + px * prev_w, prev_y + py * prev_w, prev_z}, {0.0f, 0.0f, 1.0f}, {cosF, sinF, 0.0f, 1.0f}, {1.0f, float(seg) * 0.25f}};
      Vertex12 v2{{cur_x + px * cur_w, cur_y + py * cur_w, cur_z}, {0.0f, 0.0f, 1.0f}, {cosF, sinF, 0.0f, 1.0f}, {1.0f, float(seg + 1) * 0.25f}};
      Vertex12 v3{{cur_x - px * cur_w, cur_y - py * cur_w, cur_z}, {0.0f, 0.0f, 1.0f}, {cosF, sinF, 0.0f, 1.0f}, {0.0f, float(seg + 1) * 0.25f}};

      appendQuad(vertices, triangleMats, v0, v1, v2, v3, MAT_GRASS);

      prev_x = cur_x;
      prev_y = cur_y;
      prev_z = cur_z;
      prev_w = cur_w;
    }
  }
}

// 6. Woodland Shrubs & Broadleaf Ground Bushes (8 cards = 16 triangles per bush)
inline void appendWoodlandShrub(std::vector<float> &vertices,
                               std::vector<uint32_t> &triangleMats,
                               float cx, float cy, float scale, uint32_t seed) {
  float cz = forestTerrainHeight(cx, cy);
  const float pi2 = 6.283185307179586f;

  for (uint32_t i = 0; i < 8; ++i) {
    float a = (float(i) / 8.0f) * pi2;
    float ca = std::cos(a) * 0.65f * scale;
    float sa = std::sin(a) * 0.65f * scale;
    float h = (0.55f + float(i % 3) * 0.18f) * scale;

    Vertex12 v0{{cx - ca, cy - sa, cz}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}};
    Vertex12 v1{{cx + ca, cy + sa, cz}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}};
    Vertex12 v2{{cx + ca * 0.4f, cy + sa * 0.4f, cz + h}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}};
    Vertex12 v3{{cx - ca * 0.4f, cy - sa * 0.4f, cz + h}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}};

    appendQuad(vertices, triangleMats, v0, v1, v2, v3, MAT_LEAVES);
  }
}

// 7. Fallen Nurse Logs with Moss & Bark (8 radial x 6 length slices = 96 tris + 16 end cap tris = 112 triangles)
inline void appendFallenLog(std::vector<float> &vertices,
                            std::vector<uint32_t> &triangleMats,
                            float x0, float y0, float length, float angle, float radius) {
  float z0 = forestTerrainHeight(x0, y0) + radius * 0.35f;
  float x1 = x0 + std::cos(angle) * length;
  float y1 = y0 + std::sin(angle) * length;
  float z1 = forestTerrainHeight(x1, y1) + radius * 0.35f;

  float cosA = std::cos(angle);
  float sinA = std::sin(angle);
  const float pi2 = 6.283185307179586f;

  const uint32_t radial_segs = 8;
  const uint32_t len_slices = 6;

  for (uint32_t s = 0; s < len_slices; ++s) {
    float t0 = float(s) / float(len_slices);
    float t1 = float(s + 1) / float(len_slices);
    float lx0 = x0 + (x1 - x0) * t0;
    float ly0 = y0 + (y1 - y0) * t0;
    float lz0 = z0 + (z1 - z0) * t0;
    float lx1 = x0 + (x1 - x0) * t1;
    float ly1 = y0 + (y1 - y0) * t1;
    float lz1 = z0 + (z1 - z0) * t1;

    for (uint32_t i = 0; i < radial_segs; ++i) {
      float phi0 = (float(i) / float(radial_segs)) * pi2;
      float phi1 = (float(i + 1) / float(radial_segs)) * pi2;

      float nx0 = -sinA * std::cos(phi0);
      float ny0 =  cosA * std::cos(phi0);
      float nz0 =  std::sin(phi0);

      float nx1 = -sinA * std::cos(phi1);
      float ny1 =  cosA * std::cos(phi1);
      float nz1 =  std::sin(phi1);

      Vertex12 b0{{lx0 + nx0 * radius, ly0 + ny0 * radius, lz0 + nz0 * radius}, {nx0, ny0, nz0}, {cosA, sinA, 0.0f, 1.0f}, {float(i), t0 * 3.0f}};
      Vertex12 b1{{lx0 + nx1 * radius, ly0 + ny1 * radius, lz0 + nz1 * radius}, {nx1, ny1, nz1}, {cosA, sinA, 0.0f, 1.0f}, {float(i + 1), t0 * 3.0f}};
      Vertex12 top0{{lx1 + nx0 * radius, ly1 + ny0 * radius, lz1 + nz0 * radius}, {nx0, ny0, nz0}, {cosA, sinA, 0.0f, 1.0f}, {float(i), t1 * 3.0f}};
      Vertex12 top1{{lx1 + nx1 * radius, ly1 + ny1 * radius, lz1 + nz1 * radius}, {nx1, ny1, nz1}, {cosA, sinA, 0.0f, 1.0f}, {float(i + 1), t1 * 3.0f}};

      appendQuad(vertices, triangleMats, b0, b1, top1, top0, MAT_TIMBER);
    }
  }
}

// 8. Geological Boulders & River Stones (80 triangles per boulder)
inline void appendBoulder(std::vector<float> &vertices,
                          std::vector<uint32_t> &triangleMats,
                          float cx, float cy, float radius, uint32_t seed) {
  float cz = forestTerrainHeight(cx, cy) + radius * 0.40f;
  const float pi = 3.141592653589793f;
  const float pi2 = 6.283185307179586f;

  auto prng = [&](uint32_t s) -> float {
    uint32_t state = s * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return float((word >> 22u) ^ word) / 4294967295.0f;
  };

  const uint32_t rings = 5;
  const uint32_t segs = 8;
  for (uint32_t r = 0; r < rings; ++r) {
    float p0 = float(r) / float(rings) * pi;
    float p1 = float(r + 1) / float(rings) * pi;
    for (uint32_t s = 0; s < segs; ++s) {
      float t0 = float(s) / float(segs) * pi2;
      float t1 = float(s + 1) / float(segs) * pi2;

      auto makeBV = [&](float p, float t, uint32_t vidx) -> Vertex12 {
        float jitter = 0.80f + 0.38f * prng(seed + vidx * 7919u);
        float nx = std::sin(p) * std::cos(t);
        float ny = std::sin(p) * std::sin(t);
        float nz = std::cos(p);
        float r_eff = radius * jitter;
        return Vertex12{{cx + r_eff * nx, cy + r_eff * ny, cz + r_eff * nz},
                        {nx, ny, nz}, {-ny, nx, 0.0f, 1.0f}, {t * 0.25f, p * 0.25f}};
      };

      Vertex12 v00 = makeBV(p0, t0, r * segs + s);
      Vertex12 v10 = makeBV(p1, t0, (r + 1) * segs + s);
      Vertex12 v11 = makeBV(p1, t1, (r + 1) * segs + (s + 1));
      Vertex12 v01 = makeBV(p0, t1, r * segs + (s + 1));

      appendTri(vertices, triangleMats, v00, v10, v11, MAT_ROCK);
      appendTri(vertices, triangleMats, v00, v11, v01, MAT_ROCK);
    }
  }
}

// Master assembly function producing exactly ~1,000,000 triangles with rich multi-layered foliage
inline void buildForestMesh(std::vector<float> &vertices,
                            std::vector<uint32_t> &triangleMats) {
  vertices.reserve(1010000 * 36);
  triangleMats.reserve(1010000);

  // 1. Detailed Woodland Floor Terrain (131,072 triangles)
  appendWoodlandTerrain(vertices, triangleMats);

  // 2. Stream Water Plane (2,048 triangles)
  appendStreamWater(vertices, triangleMats);

  // Deterministic PRNG
  uint32_t seed = 918273645;
  auto rand_f = [&]() -> float {
    seed = seed * 747796405u + 2891336453u;
    uint32_t word = ((seed >> ((seed >> 28u) + 4u)) ^ seed) * 277803737u;
    return float((word >> 22u) ^ word) / 4294967295.0f;
  };

  // Camera is at (0.0, -18.0, 0.55), looking along +Y trail corridor.
  // Curated Foreground Framing Elements on the verges (|x| > 3.0):
  // Left majestic framing pine tree
  appendConiferTree(vertices, triangleMats, -6.5f, -12.0f, 1.15f, seed + 1);
  // Right majestic framing pine tree
  appendConiferTree(vertices, triangleMats, 7.0f, -10.0f, 1.20f, seed + 2);
  // Elegant birch tree on left
  appendDeciduousTree(vertices, triangleMats, -5.0f, -3.0f, 1.10f, seed + 3);
  // Elegant birch tree on right
  appendDeciduousTree(vertices, triangleMats, 6.5f, 4.0f, 1.15f, seed + 4);

  // Foreground mossy nurse log on right verge
  appendFallenLog(vertices, triangleMats, 4.5f, -12.0f, 8.0f, 0.55f, 0.40f);
  // Small mossy rocks on the path margins
  appendBoulder(vertices, triangleMats, -3.2f, -13.5f, 0.55f, seed + 5);
  appendBoulder(vertices, triangleMats, 3.2f, -14.0f, 0.45f, seed + 6);

  // Delicate ferns lining the path edges
  appendFernClump(vertices, triangleMats, -2.2f, -15.0f, 0.95f, seed + 7);
  appendFernClump(vertices, triangleMats, 2.2f, -15.5f, 0.90f, seed + 8);
  appendFernClump(vertices, triangleMats, -2.0f, -12.0f, 1.05f, seed + 9);
  appendFernClump(vertices, triangleMats, 2.4f, -11.0f, 1.00f, seed + 10);
  appendFernClump(vertices, triangleMats, -2.5f, -8.0f, 1.10f, seed + 11);
  appendFernClump(vertices, triangleMats, 2.6f, -7.0f, 1.15f, seed + 12);

  // 3. 350 Mature Conifer / Pine Trees (~450,000 triangles)
  for (uint32_t i = 0; i < 350; ++i) {
    float x = (rand_f() * 2.0f - 1.0f) * 85.0f;
    float y = -45.0f + rand_f() * 195.0f;

    // Keep trail corridor open in front of camera
    float trailCenter = -1.5f + 2.5f * std::sin(y * 0.04f);
    if (std::abs(x - trailCenter) < 3.2f) {
      x += (x >= trailCenter) ? 3.8f : -3.8f;
    }
    // Camera clear zone
    if (y > -22.0f && y < -10.0f && std::abs(x) < 4.5f) {
      x += (x >= 0.0f) ? 5.0f : -5.0f;
    }
    // Keep out of stream bed
    float streamCenter = 20.0f + 4.5f * std::sin(y * 0.03f);
    if (std::abs(x - streamCenter) < 3.2f) {
      x += (x >= streamCenter) ? 3.8f : -3.8f;
    }

    float scale = 0.85f + rand_f() * 0.55f;
    appendConiferTree(vertices, triangleMats, x, y, scale, seed + i * 13u);
  }

  // 4. 180 Deciduous Birch Trees (~92,160 triangles)
  for (uint32_t i = 0; i < 180; ++i) {
    float x = (rand_f() * 2.0f - 1.0f) * 75.0f;
    float y = -35.0f + rand_f() * 180.0f;

    float trailCenter = -1.5f + 2.5f * std::sin(y * 0.04f);
    if (std::abs(x - trailCenter) < 2.8f) {
      x += (x >= trailCenter) ? 3.5f : -3.5f;
    }
    if (y > -22.0f && y < -10.0f && std::abs(x) < 4.0f) {
      x += (x >= 0.0f) ? 4.5f : -4.5f;
    }
    float streamCenter = 20.0f + 4.5f * std::sin(y * 0.03f);
    if (std::abs(x - streamCenter) < 2.8f) {
      x += (x >= streamCenter) ? 3.5f : -3.5f;
    }

    float scale = 0.80f + rand_f() * 0.45f;
    appendDeciduousTree(vertices, triangleMats, x, y, scale, seed + i * 29u);
  }

  // 5. 3,500 Understory Fern Clumps (~224,000 triangles)
  for (uint32_t i = 0; i < 3500; ++i) {
    float x = (rand_f() * 2.0f - 1.0f) * 65.0f;
    float y = -40.0f + rand_f() * 185.0f;
    float scale = 0.70f + rand_f() * 0.60f;
    appendFernClump(vertices, triangleMats, x, y, scale, seed + i * 37u);
  }

  // 6. 2,500 Woodland Shrubs & Bushes (~40,000 triangles)
  for (uint32_t i = 0; i < 2500; ++i) {
    float x = (rand_f() * 2.0f - 1.0f) * 70.0f;
    float y = -40.0f + rand_f() * 180.0f;
    float scale = 0.65f + rand_f() * 0.55f;
    appendWoodlandShrub(vertices, triangleMats, x, y, scale, seed + i * 47u);
  }

  // 7. 180 Fallen Nurse Logs (~20,000 triangles)
  for (uint32_t i = 0; i < 180; ++i) {
    float x = (rand_f() * 2.0f - 1.0f) * 75.0f;
    float y = -40.0f + rand_f() * 180.0f;
    float trailCenter = -1.5f + 2.5f * std::sin(y * 0.04f);
    if (std::abs(x - trailCenter) < 2.8f) {
      x += (x >= trailCenter) ? 3.5f : -3.5f;
    }
    if (y > -22.0f && y < -8.0f && std::abs(x) < 4.0f) {
      x += (x >= 0.0f) ? 4.5f : -4.5f;
    }
    float len = 5.0f + rand_f() * 7.0f;
    float ang = rand_f() * 6.283185f;
    float rad = 0.35f + rand_f() * 0.35f;
    appendFallenLog(vertices, triangleMats, x, y, len, ang, rad);
  }

  // 8. 600 Geological Boulders & River Stones (~48,000 triangles)
  for (uint32_t i = 0; i < 600; ++i) {
    float x = (rand_f() * 2.0f - 1.0f) * 80.0f;
    float y = -45.0f + rand_f() * 190.0f;

    // Strict trail and camera clearance
    float trailCenter = -1.5f + 2.5f * std::sin(y * 0.04f);
    if (std::abs(x - trailCenter) < 2.5f) {
      x += (x >= trailCenter) ? 3.2f : -3.2f;
    }
    if (y > -22.0f && y < -6.0f && std::abs(x) < 4.5f) {
      x += (x >= 0.0f) ? 5.0f : -5.0f;
    }

    float r = 0.35f + rand_f() * 1.1f;
    appendBoulder(vertices, triangleMats, x, y, r, seed + i * 53u);
  }
}

} // namespace AAAForestScene
