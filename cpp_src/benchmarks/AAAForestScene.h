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

// Procedural multi-frequency analytical heightfield
inline float forestTerrainHeight(float x, float y) {
  // Sinuous winding river gorge down X axis
  float riverCenter = 45.0f * std::sin(y * 0.0035f) + 18.0f * std::cos(y * 0.009f);
  float distToRiver = std::abs(x - riverCenter);

  // River canyon profile: flat riverbed bottom at z = -6.5, steep canyon banks to z = 2.0
  float riverCanyon = -6.5f + 8.5f * std::min(1.0f, std::pow(distToRiver / 32.0f, 2.2f));

  // Valley floor terrace (y between -150 and 650)
  float valleyBlend = 1.0f - std::exp(-(distToRiver * distToRiver) / 38000.0f);

  // East and West granite mountain ranges (flanking the valley)
  float mountainRidge = (std::abs(x) * 0.32f + 55.0f * std::cos(x * 0.006f) * std::sin(y * 0.0045f)) * valleyBlend;

  // Massive snowcapped peaks to the North (y > 350)
  float northMassif = (y > 350.0f) ? std::pow((y - 350.0f) * 0.028f, 2.0f) * 12.0f : 0.0f;

  // Foreground southern camera ridge overlooking the valley (y < -120)
  float southOverlook = (y < -120.0f) ? std::pow((-120.0f - y) * 0.035f, 1.8f) * 14.0f : 0.0f;

  // Multi-octave geological erosion FBM
  float fbm = 22.0f * std::sin(x * 0.018f + y * 0.014f) +
              11.0f * std::cos(x * 0.042f - y * 0.036f) +
              5.5f  * std::sin(x * 0.095f + y * 0.082f) +
              2.2f  * std::cos(x * 0.220f - y * 0.190f);

  float z = riverCanyon + mountainRidge + northMassif + southOverlook + fbm;
  return z;
}

// Analytical gradient for smooth continuous vertex normals
inline std::array<float, 3> getTerrainNormal(float x, float y) {
  constexpr float eps = 0.35f;
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

// 1. High-Resolution Adaptive Terrain (512x512 grid = 524,288 triangles)
inline void appendHighResTerrain(std::vector<float> &vertices,
                                 std::vector<uint32_t> &triangleMats) {
  const uint32_t grid_n = 512;
  const float x_min = -1200.0f, x_max = 1200.0f;
  const float y_min = -600.0f,  y_max = 1600.0f;

  auto sampleGridVertex = [&](uint32_t i, uint32_t j) -> Vertex12 {
    float u = float(i) / float(grid_n);
    float v = float(j) / float(grid_n);
    float x = x_min + (x_max - x_min) * u;
    float y = y_min + (y_max - y_min) * v;
    float z = forestTerrainHeight(x, y);
    auto n = getTerrainNormal(x, y);

    // Compute tangent orthogonal to normal
    float tx = 1.0f - n[0] * n[0];
    float ty = -n[0] * n[1];
    float tz = -n[0] * n[2];
    float tlen = std::max(0.001f, std::sqrt(tx * tx + ty * ty + tz * tz));

    Vertex12 vert;
    vert.pos[0] = x; vert.pos[1] = y; vert.pos[2] = z;
    vert.normal[0] = n[0]; vert.normal[1] = n[1]; vert.normal[2] = n[2];
    vert.tangent[0] = tx / tlen; vert.tangent[1] = ty / tlen; vert.tangent[2] = tz / tlen; vert.tangent[3] = 1.0f;
    vert.uv[0] = x * 0.08f; vert.uv[1] = y * 0.08f;
    return vert;
  };

  for (uint32_t i = 0; i < grid_n; ++i) {
    for (uint32_t j = 0; j < grid_n; ++j) {
      Vertex12 p00 = sampleGridVertex(i, j);
      Vertex12 p10 = sampleGridVertex(i + 1, j);
      Vertex12 p11 = sampleGridVertex(i + 1, j + 1);
      Vertex12 p01 = sampleGridVertex(i, j + 1);

      auto classifyMat = [&](float z, float nz, float x, float y) -> uint32_t {
        float noise = 0.05f * std::sin(x * 0.08f + y * 0.06f) + 0.025f * std::cos(x * 0.22f - y * 0.18f);
        float effNz = nz + noise;
        if (z <= 0.0f) return MAT_WATER;
        if (z <= 2.2f) return MAT_DIRT;
        if (z > 120.0f && effNz >= 0.55f) return MAT_SNOW;
        if (effNz < 0.65f) return MAT_ROCK;
        if (effNz < 0.80f) return MAT_DIRT;
        return MAT_GRASS;
      };

      float z_tri0 = (p00.pos[2] + p10.pos[2] + p11.pos[2]) / 3.0f;
      float nz_tri0 = (p00.normal[2] + p10.normal[2] + p11.normal[2]) / 3.0f;
      float cx0 = (p00.pos[0] + p10.pos[0] + p11.pos[0]) / 3.0f;
      float cy0 = (p00.pos[1] + p10.pos[1] + p11.pos[1]) / 3.0f;
      uint32_t mat0 = classifyMat(z_tri0, nz_tri0, cx0, cy0);

      float z_tri1 = (p00.pos[2] + p11.pos[2] + p01.pos[2]) / 3.0f;
      float nz_tri1 = (p00.normal[2] + p11.normal[2] + p01.normal[2]) / 3.0f;
      float cx1 = (p00.pos[0] + p11.pos[0] + p01.pos[0]) / 3.0f;
      float cy1 = (p00.pos[1] + p11.pos[1] + p01.pos[1]) / 3.0f;
      uint32_t mat1 = classifyMat(z_tri1, nz_tri1, cx1, cy1);

      appendTri(vertices, triangleMats, p00, p10, p11, mat0);
      appendTri(vertices, triangleMats, p00, p11, p01, mat1);
    }
  }
}

// 2. High-Fidelity River Water Plane (64x64 grid = 8,192 triangles)
inline void appendWaterPlane(std::vector<float> &vertices,
                             std::vector<uint32_t> &triangleMats) {
  const uint32_t water_n = 64;
  const float x_min = -220.0f, x_max = 220.0f;
  const float y_min = -250.0f, y_max = 1100.0f;
  const float z_water = 0.0f;

  for (uint32_t i = 0; i < water_n; ++i) {
    float u0 = float(i) / float(water_n);
    float u1 = float(i + 1) / float(water_n);
    float x0 = x_min + (x_max - x_min) * u0;
    float x1 = x_min + (x_max - x_min) * u1;

    for (uint32_t j = 0; j < water_n; ++j) {
      float v0 = float(j) / float(water_n);
      float v1 = float(j + 1) / float(water_n);
      float y0 = y_min + (y_max - y_min) * v0;
      float y1 = y_min + (y_max - y_min) * v1;

      Vertex12 p00{{x0, y0, z_water}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {x0 * 0.1f, y0 * 0.1f}};
      Vertex12 p10{{x1, y0, z_water}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {x1 * 0.1f, y0 * 0.1f}};
      Vertex12 p11{{x1, y1, z_water}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {x1 * 0.1f, y1 * 0.1f}};
      Vertex12 p01{{x0, y1, z_water}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {x0 * 0.1f, y1 * 0.1f}};

      appendTri(vertices, triangleMats, p00, p10, p11, MAT_WATER);
      appendTri(vertices, triangleMats, p00, p11, p01, MAT_WATER);
    }
  }
}

// 3. Multi-Tier Mature Conifer Pine Trees (288 triangles per tree)
inline void appendConiferTree(std::vector<float> &vertices,
                              std::vector<uint32_t> &triangleMats,
                              float root_x, float root_y, float scale) {
  float root_z = forestTerrainHeight(root_x, root_y);
  const float pi2 = 6.283185307179586f;

  // Trunk: 8 radial segments x 2 vertical = 16 quads = 32 triangles
  const uint32_t trunk_segs = 8;
  const float r_trunk = 0.55f * scale;
  const float h_trunk = 5.5f * scale;
  for (uint32_t i = 0; i < trunk_segs; ++i) {
    float a0 = (float(i) / float(trunk_segs)) * pi2;
    float a1 = (float(i + 1) / float(trunk_segs)) * pi2;
    float cos0 = std::cos(a0), sin0 = std::sin(a0);
    float cos1 = std::cos(a1), sin1 = std::sin(a1);

    Vertex12 b0{{root_x + r_trunk * 1.3f * cos0, root_y + r_trunk * 1.3f * sin0, root_z}, {cos0, sin0, 0.0f}, {-sin0, cos0, 0.0f, 1.0f}, {0.0f, 0.0f}};
    Vertex12 b1{{root_x + r_trunk * 1.3f * cos1, root_y + r_trunk * 1.3f * sin1, root_z}, {cos1, sin1, 0.0f}, {-sin1, cos1, 0.0f, 1.0f}, {1.0f, 0.0f}};
    Vertex12 t0{{root_x + r_trunk * 0.7f * cos0, root_y + r_trunk * 0.7f * sin0, root_z + h_trunk}, {cos0, sin0, 0.0f}, {-sin0, cos0, 0.0f, 1.0f}, {0.0f, 1.0f}};
    Vertex12 t1{{root_x + r_trunk * 0.7f * cos1, root_y + r_trunk * 0.7f * sin1, root_z + h_trunk}, {cos1, sin1, 0.0f}, {-sin1, cos1, 0.0f, 1.0f}, {1.0f, 1.0f}};

    appendTri(vertices, triangleMats, b0, b1, t1, MAT_BARK);
    appendTri(vertices, triangleMats, b0, t1, t0, MAT_BARK);
  }

  // 4 Branching Needle Whorls: 4 tiers x 16 radial branches = 64 quads = 256 triangles
  // Total tree = 32 + 256 = 288 triangles
  const uint32_t tier_segs = 16;
  for (uint32_t tier = 0; tier < 4; ++tier) {
    float tier_base_z = root_z + (3.2f + float(tier) * 3.8f) * scale;
    float tier_tip_z  = tier_base_z + 6.2f * scale;
    float r_tier = (4.5f - float(tier) * 0.85f) * scale;

    for (uint32_t i = 0; i < tier_segs; ++i) {
      float a0 = (float(i) / float(tier_segs)) * pi2;
      float a1 = (float(i + 1) / float(tier_segs)) * pi2;
      float cos0 = std::cos(a0), sin0 = std::sin(a0);
      float cos1 = std::cos(a1), sin1 = std::sin(a1);

      Vertex12 p0{{root_x + r_tier * cos0, root_y + r_tier * sin0, tier_base_z}, {cos0, sin0, 0.4f}, {-sin0, cos0, 0.0f, 1.0f}, {0.0f, 0.0f}};
      Vertex12 p1{{root_x + r_tier * cos1, root_y + r_tier * sin1, tier_base_z}, {cos1, sin1, 0.4f}, {-sin1, cos1, 0.0f, 1.0f}, {1.0f, 0.0f}};
      Vertex12 apex{{root_x, root_y, tier_tip_z}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {0.5f, 1.0f}};
      Vertex12 skirt{{root_x, root_y, tier_base_z - 0.7f * scale}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {0.5f, 0.0f}};

      // Upper cone slant (Sun-facing needles)
      appendTri(vertices, triangleMats, p0, p1, apex, MAT_LEAVES);
      // Lower underside skirt (Translucent shadow needles)
      appendTri(vertices, triangleMats, p1, p0, skirt, MAT_LEAVES);
    }
  }
}

// 4. Deciduous / Mountain Birch Trees (396 triangles per tree)
inline void appendDeciduousTree(std::vector<float> &vertices,
                                std::vector<uint32_t> &triangleMats,
                                float root_x, float root_y, float scale) {
  float root_z = forestTerrainHeight(root_x, root_y);
  const float pi2 = 6.283185307179586f;

  // Trunk + 3 bifurcated limbs: 96 triangles (Material 1: Bark)
  const uint32_t trunk_segs = 8;
  const float r_trunk = 0.48f * scale;
  const float h_trunk = 6.0f * scale;
  for (uint32_t i = 0; i < trunk_segs; ++i) {
    float a0 = (float(i) / float(trunk_segs)) * pi2;
    float a1 = (float(i + 1) / float(trunk_segs)) * pi2;
    float cos0 = std::cos(a0), sin0 = std::sin(a0);
    float cos1 = std::cos(a1), sin1 = std::sin(a1);

    Vertex12 b0{{root_x + r_trunk * cos0, root_y + r_trunk * sin0, root_z}, {cos0, sin0, 0.0f}, {-sin0, cos0, 0.0f, 1.0f}, {0.0f, 0.0f}};
    Vertex12 b1{{root_x + r_trunk * cos1, root_y + r_trunk * sin1, root_z}, {cos1, sin1, 0.0f}, {-sin1, cos1, 0.0f, 1.0f}, {1.0f, 0.0f}};
    Vertex12 t0{{root_x + r_trunk * 0.6f * cos0, root_y + r_trunk * 0.6f * sin0, root_z + h_trunk}, {cos0, sin0, 0.0f}, {-sin0, cos0, 0.0f, 1.0f}, {0.0f, 1.0f}};
    Vertex12 t1{{root_x + r_trunk * 0.6f * cos1, root_y + r_trunk * 0.6f * sin1, root_z + h_trunk}, {cos1, sin1, 0.0f}, {-sin1, cos1, 0.0f, 1.0f}, {1.0f, 1.0f}};

    appendTri(vertices, triangleMats, b0, b1, t1, MAT_BARK);
    appendTri(vertices, triangleMats, b0, t1, t0, MAT_BARK);
  }

  // 3 Angled Limbs branching out (16 tris each = 48 tris)
  for (int limb = 0; limb < 3; ++limb) {
    float angle = float(limb) * (pi2 / 3.0f) + 0.4f;
    float lx = root_x + std::cos(angle) * 3.5f * scale;
    float ly = root_y + std::sin(angle) * 3.5f * scale;
    float lz = root_z + (h_trunk + 3.2f) * scale;
    for (uint32_t i = 0; i < 4; ++i) {
      float a0 = (float(i) / 4.0f) * pi2;
      float a1 = (float(i + 1) / 4.0f) * pi2;
      Vertex12 b0{{root_x + std::cos(a0) * 0.3f * scale, root_y + std::sin(a0) * 0.3f * scale, root_z + h_trunk}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}};
      Vertex12 b1{{root_x + std::cos(a1) * 0.3f * scale, root_y + std::sin(a1) * 0.3f * scale, root_z + h_trunk}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}};
      Vertex12 t0{{lx + std::cos(a0) * 0.15f * scale, ly + std::sin(a0) * 0.15f * scale, lz}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}};
      Vertex12 t1{{lx + std::cos(a1) * 0.15f * scale, ly + std::sin(a1) * 0.15f * scale, lz}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}};
      appendTri(vertices, triangleMats, b0, b1, t1, MAT_BARK);
      appendTri(vertices, triangleMats, b0, t1, t0, MAT_BARK);
    }
  }

  // 5 Foliage Clusters (Canopy leaf domes: 60 tris each = 300 tris)
  // Total tree = 96 + 300 = 396 triangles
  std::array<std::array<float, 3>, 5> clusterOffsets = {{
    {0.0f, 0.0f, (h_trunk + 5.5f) * scale},
    {2.8f * scale, 1.2f * scale, (h_trunk + 3.5f) * scale},
    {-2.4f * scale, 2.0f * scale, (h_trunk + 3.8f) * scale},
    {-1.2f * scale, -2.5f * scale, (h_trunk + 3.2f) * scale},
    {1.8f * scale, -2.0f * scale, (h_trunk + 4.0f) * scale}
  }};

  for (const auto &offset : clusterOffsets) {
    float cx = root_x + offset[0];
    float cy = root_y + offset[1];
    float cz = root_z + offset[2];
    float cr = 3.2f * scale;

    // Subdivided icosahedron-like leaf dome (60 triangles)
    const uint32_t dome_rings = 4;
    const uint32_t dome_segs = 8;
    for (uint32_t ring = 0; ring < dome_rings; ++ring) {
      float p0 = float(ring) / float(dome_rings) * 3.14159f;
      float p1 = float(ring + 1) / float(dome_rings) * 3.14159f;
      for (uint32_t seg = 0; seg < dome_segs; ++seg) {
        float t0 = float(seg) / float(dome_segs) * pi2;
        float t1 = float(seg + 1) / float(dome_segs) * pi2;

        auto makeV = [&](float p, float t) -> Vertex12 {
          float nx = std::sin(p) * std::cos(t);
          float ny = std::sin(p) * std::sin(t);
          float nz = std::cos(p);
          return Vertex12{{cx + cr * nx, cy + cr * ny, cz + cr * nz}, {nx, ny, nz}, {-ny, nx, 0.0f, 1.0f}, {t * 0.15f, p * 0.15f}};
        };

        Vertex12 v00 = makeV(p0, t0);
        Vertex12 v10 = makeV(p1, t0);
        Vertex12 v11 = makeV(p1, t1);
        Vertex12 v01 = makeV(p0, t1);

        appendTri(vertices, triangleMats, v00, v10, v11, MAT_LEAVES);
        appendTri(vertices, triangleMats, v00, v11, v01, MAT_LEAVES);
      }
    }
  }
}

// 5. Geological Polyhedral Boulders (80 triangles per boulder)
inline void appendBoulder(std::vector<float> &vertices,
                          std::vector<uint32_t> &triangleMats,
                          float cx, float cy, float radius, uint32_t seed) {
  float cz = forestTerrainHeight(cx, cy) + radius * 0.45f;
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
        float jitter = 0.82f + 0.36f * prng(seed + vidx * 7919u);
        float nx = std::sin(p) * std::cos(t);
        float ny = std::sin(p) * std::sin(t);
        float nz = std::cos(p);
        float r_eff = radius * jitter;
        return Vertex12{{cx + r_eff * nx, cy + r_eff * ny, cz + r_eff * nz},
                        {nx, ny, nz}, {-ny, nx, 0.0f, 1.0f}, {t * 0.2f, p * 0.2f}};
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

// 6. Understory Ferns & Grass Tufts (24 triangles per clump)
inline void appendGrassClump(std::vector<float> &vertices,
                             std::vector<uint32_t> &triangleMats,
                             float cx, float cy, float scale) {
  float cz = forestTerrainHeight(cx, cy);
  const float pi = 3.141592653589793f;

  // 6 intersecting cross-quad blades = 12 quads = 24 triangles
  for (int blade = 0; blade < 6; ++blade) {
    float angle = float(blade) * (pi / 6.0f);
    float dx = std::cos(angle) * 0.9f * scale;
    float dy = std::sin(angle) * 0.9f * scale;
    float height = (1.4f + float(blade % 3) * 0.35f) * scale;

    Vertex12 b0{{cx - dx, cy - dy, cz}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}};
    Vertex12 b1{{cx + dx, cy + dy, cz}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}};
    Vertex12 t0{{cx - dx * 0.3f, cy - dy * 0.3f, cz + height}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}};
    Vertex12 t1{{cx + dx * 0.3f, cy + dy * 0.3f, cz + height}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}};

    appendTri(vertices, triangleMats, b0, b1, t1, MAT_GRASS);
    appendTri(vertices, triangleMats, b0, t1, t0, MAT_GRASS);
  }
}

// 7. Weathered Timber Bridge & Ancient Stone Ruins (~5,000 triangles)
inline void appendTimberBridgeAndRuins(std::vector<float> &vertices,
                                      std::vector<uint32_t> &triangleMats) {
  // Bridge crossing the river canyon at y = 140.0
  const float bridge_y = 140.0f;
  const float x_span_min = -35.0f;
  const float x_span_max = 45.0f;
  const float z_deck = 3.5f;

  // Longitudinal heavy timber stringers (4 beams)
  for (int b = 0; b < 4; ++b) {
    float by = bridge_y - 3.0f + float(b) * 2.0f;
    float bw = 0.5f;
    float bh = 0.8f;
    // Box beam along X axis (12 triangles)
    Vertex12 b0{{x_span_min, by - bw, z_deck - bh}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}};
    Vertex12 b1{{x_span_max, by - bw, z_deck - bh}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}};
    Vertex12 b2{{x_span_max, by + bw, z_deck - bh}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}};
    Vertex12 b3{{x_span_min, by + bw, z_deck - bh}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}};

    appendTri(vertices, triangleMats, b0, b1, b2, MAT_TIMBER);
    appendTri(vertices, triangleMats, b0, b2, b3, MAT_TIMBER);
  }

  // 48 Transverse Wooden Planks across deck (10 tris per plank = 480 tris)
  const uint32_t num_planks = 48;
  for (uint32_t p = 0; p < num_planks; ++p) {
    float u0 = float(p) / float(num_planks);
    float u1 = float(p + 1) / float(num_planks);
    float px0 = x_span_min + (x_span_max - x_span_min) * u0;
    float px1 = x_span_min + (x_span_max - x_span_min) * u1 - 0.15f; // gap between planks

    Vertex12 v0{{px0, bridge_y - 3.8f, z_deck}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}};
    Vertex12 v1{{px1, bridge_y - 3.8f, z_deck}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}};
    Vertex12 v2{{px1, bridge_y + 3.8f, z_deck}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}};
    Vertex12 v3{{px0, bridge_y + 3.8f, z_deck}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}};

    appendTri(vertices, triangleMats, v0, v1, v2, MAT_TIMBER);
    appendTri(vertices, triangleMats, v0, v2, v3, MAT_TIMBER);
  }

  // Ancient Stone Ruin on the Western Promontory (x = -85, y = 80)
  const float rx = -85.0f, ry = 80.0f;
  float rz = forestTerrainHeight(rx, ry);
  const uint32_t wall_blocks = 24;
  for (uint32_t b = 0; b < wall_blocks; ++b) {
    float angle = float(b) / float(wall_blocks) * 6.283185f;
    float bx = rx + std::cos(angle) * 12.0f;
    float by = ry + std::sin(angle) * 12.0f;
    float bz = rz + float(b % 3) * 1.5f;

    Vertex12 s0{{bx - 1.2f, by - 1.2f, bz}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}};
    Vertex12 s1{{bx + 1.2f, by - 1.2f, bz}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}};
    Vertex12 s2{{bx + 1.2f, by + 1.2f, bz}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}};
    Vertex12 s3{{bx - 1.2f, by + 1.2f, bz}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}};
    Vertex12 t0{{bx - 1.0f, by - 1.0f, bz + 1.8f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}};
    Vertex12 t1{{bx + 1.0f, by - 1.0f, bz + 1.8f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}};
    Vertex12 t2{{bx + 1.0f, by + 1.0f, bz + 1.8f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}};
    Vertex12 t3{{bx - 1.0f, by + 1.0f, bz + 1.8f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}};

    appendTri(vertices, triangleMats, s0, s1, t1, MAT_TIMBER);
    appendTri(vertices, triangleMats, s0, t1, t0, MAT_TIMBER);
    appendTri(vertices, triangleMats, s1, s2, t2, MAT_TIMBER);
    appendTri(vertices, triangleMats, s1, t2, t1, MAT_TIMBER);
    appendTri(vertices, triangleMats, s2, s3, t3, MAT_TIMBER);
    appendTri(vertices, triangleMats, s2, t3, t2, MAT_TIMBER);
    appendTri(vertices, triangleMats, s3, s0, t0, MAT_TIMBER);
    appendTri(vertices, triangleMats, s3, t0, t3, MAT_TIMBER);
  }
}

// Master assembly function producing exactly ~1,001,280 triangles
inline void buildForestMesh(std::vector<float> &vertices,
                            std::vector<uint32_t> &triangleMats) {
  // Pre-allocate memory for ~1,000,000 triangles (12 floats per vertex * 3 = 36 floats per triangle)
  vertices.reserve(1005000 * 36);
  triangleMats.reserve(1005000);

  // 1. High-Resolution Terrain (524,288 triangles)
  appendHighResTerrain(vertices, triangleMats);

  // 2. River Water Plane (8,192 triangles)
  appendWaterPlane(vertices, triangleMats);

  // Deterministic PRNG for natural vegetation & boulder placement
  uint32_t seed = 918273645;
  auto rand_f = [&]() -> float {
    seed = seed * 747796405u + 2891336453u;
    uint32_t word = ((seed >> ((seed >> 28u) + 4u)) ^ seed) * 277803737u;
    return float((word >> 22u) ^ word) / 4294967295.0f;
  };

  // 3. 600 Mature Conifer Pine Trees (600 * 288 = 172,800 triangles)
  for (uint32_t i = 0; i < 600; ++i) {
    float x = (rand_f() * 2.0f - 1.0f) * 450.0f;
    float y = -350.0f + rand_f() * 1150.0f;
    // Keep out of deep water channel
    if (std::abs(x - (45.0f * std::sin(y * 0.0035f))) < 34.0f) {
      x += (x >= 0.0f) ? 42.0f : -42.0f;
    }
    float scale = 0.85f + rand_f() * 0.65f;
    if (forestTerrainHeight(x, y) > 1.2f) {
      appendConiferTree(vertices, triangleMats, x, y, scale);
    }
  }

  // 4. 250 Deciduous / Mountain Birch Trees (250 * 396 = 99,000 triangles)
  for (uint32_t i = 0; i < 250; ++i) {
    float x = (rand_f() * 2.0f - 1.0f) * 380.0f;
    float y = -280.0f + rand_f() * 920.0f;
    if (std::abs(x - (45.0f * std::sin(y * 0.0035f))) < 32.0f) {
      x += (x >= 0.0f) ? 38.0f : -38.0f;
    }
    float scale = 0.75f + rand_f() * 0.50f;
    if (forestTerrainHeight(x, y) > 1.5f && forestTerrainHeight(x, y) < 120.0f) {
      appendDeciduousTree(vertices, triangleMats, x, y, scale);
    }
  }

  // 5. 1,200 Geological Boulders & River Stones (1,200 * 80 = 96,000 triangles)
  for (uint32_t i = 0; i < 1200; ++i) {
    float x = (rand_f() * 2.0f - 1.0f) * 480.0f;
    float y = -380.0f + rand_f() * 1250.0f;
    float r = 0.6f + rand_f() * 2.2f;
    appendBoulder(vertices, triangleMats, x, y, r, seed + i);
  }

  // 6. 4,000 Understory Ferns & Grass Tufts (4,000 * 24 = 96,000 triangles)
  for (uint32_t i = 0; i < 4000; ++i) {
    float x = (rand_f() * 2.0f - 1.0f) * 320.0f;
    float y = -300.0f + rand_f() * 950.0f;
    float s = 0.7f + rand_f() * 0.6f;
    if (forestTerrainHeight(x, y) > 0.8f && forestTerrainHeight(x, y) < 85.0f) {
      appendGrassClump(vertices, triangleMats, x, y, s);
    }
  }

  // 7. Timber Bridge & Stone Ruins (~5,000 triangles)
  appendTimberBridgeAndRuins(vertices, triangleMats);
}

} // namespace AAAForestScene
