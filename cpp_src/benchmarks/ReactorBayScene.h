#pragma once

#include "scene/GltfScene.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace ReactorBayScene {

// 8 Production Industrial Sci-Fi PBR Material IDs (1:1 mapped to BSDF Archetypes 0..7)
constexpr uint32_t MAT_TITANIUM   = 0u; // Archetype 0: Anisotropic Brushed Titanium Girders, Railings & Pipe Racks
constexpr uint32_t MAT_CERAMIC    = 1u; // Archetype 1: Ribbed Ceramic Reactor Insulators & Braided Cryo Conduits
constexpr uint32_t MAT_GLASS      = 2u; // Archetype 2: Refractive Armored Control-Booth Glass & Coolant Trench Covers
constexpr uint32_t MAT_MLI_FOIL   = 3u; // Archetype 3: Gold/Silver Quilted MLI Thermal Foil Blankets & Baffles
constexpr uint32_t MAT_FERROCRETE = 4u; // Archetype 4: Weathered Ferrocrete Hangar Deck & Blast Bulkheads
constexpr uint32_t MAT_COOLANT    = 5u; // Archetype 5: Coolant-Slicked Channels & Iridescent Tokamak Coils
constexpr uint32_t MAT_CLEARCOAT  = 6u; // Archetype 6: Safety-Orange Clearcoat Robotic Arms & Reactor Shrouds
constexpr uint32_t MAT_HEX_GRATE  = 7u; // Archetype 7: Perforated Alpha-Cutout Hex-Grate Catwalks, Cages & Cable Trays

struct Vertex12 {
  float pos[3];
  float normal[3];
  float tangent[4];
  float uv[2];
};

namespace detail {

struct Vec3 {
  float x, y, z;
};

inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 operator*(Vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }

inline float vdot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 vcross(Vec3 a, Vec3 b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline Vec3 vnorm(Vec3 a) {
  float l2 = vdot(a, a);
  if (l2 < 1e-12f) return {0.0f, 0.0f, 1.0f};
  float inv = 1.0f / std::sqrt(l2);
  return {a.x * inv, a.y * inv, a.z * inv};
}

inline Vec3 makeTangent(Vec3 n) {
  Vec3 up = (std::fabs(n.z) < 0.95f) ? Vec3{0.0f, 0.0f, 1.0f} : Vec3{1.0f, 0.0f, 0.0f};
  return vnorm(vcross(up, n));
}

inline float smoothstep(float edge0, float edge1, float x) {
  float t = std::clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
  return t * t * (3.0f - 2.0f * t);
}

inline void pushVert(std::vector<float> &vertices, const Vertex12 &v) {
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
}

inline void appendTri(std::vector<float> &vertices,
                      std::vector<uint32_t> &triangleMats,
                      const Vertex12 &v0,
                      const Vertex12 &v1,
                      const Vertex12 &v2,
                      uint32_t matId) {
  Vec3 p0{v0.pos[0], v0.pos[1], v0.pos[2]};
  Vec3 p1{v1.pos[0], v1.pos[1], v1.pos[2]};
  Vec3 p2{v2.pos[0], v2.pos[1], v2.pos[2]};
  Vec3 e1 = p1 - p0;
  Vec3 e2 = p2 - p0;
  Vec3 cr = vcross(e1, e2);
  if (vdot(cr, cr) < 1e-12f) return; // Skip degenerate triangles cleanly
  pushVert(vertices, v0);
  pushVert(vertices, v1);
  pushVert(vertices, v2);
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

inline void appendBox(std::vector<float> &vertices,
                      std::vector<uint32_t> &triangleMats,
                      Vec3 c, Vec3 halfExt,
                      uint32_t matId, float uvScale = 0.5f,
                      uint32_t nxSub = 1u, uint32_t nySub = 1u, uint32_t nzSub = 1u) {
  auto emitFace = [&](Vec3 origin, Vec3 axisU, Vec3 axisV, Vec3 n, uint32_t su, uint32_t sv, float uLen, float vLen) {
    Vec3 t = vnorm(axisU);
    for (uint32_t iu = 0; iu < su; ++iu) {
      float u0 = float(iu) / float(su);
      float u1 = float(iu + 1u) / float(su);
      for (uint32_t iv = 0; iv < sv; ++iv) {
        float v0 = float(iv) / float(sv);
        float v1 = float(iv + 1u) / float(sv);
        Vec3 p00 = origin + axisU * u0 + axisV * v0;
        Vec3 p10 = origin + axisU * u1 + axisV * v0;
        Vec3 p11 = origin + axisU * u1 + axisV * v1;
        Vec3 p01 = origin + axisU * u0 + axisV * v1;
        Vertex12 a{{p00.x, p00.y, p00.z}, {n.x, n.y, n.z}, {t.x, t.y, t.z, 1.0f}, {u0 * uLen * uvScale, v0 * vLen * uvScale}};
        Vertex12 b{{p10.x, p10.y, p10.z}, {n.x, n.y, n.z}, {t.x, t.y, t.z, 1.0f}, {u1 * uLen * uvScale, v0 * vLen * uvScale}};
        Vertex12 d{{p11.x, p11.y, p11.z}, {n.x, n.y, n.z}, {t.x, t.y, t.z, 1.0f}, {u1 * uLen * uvScale, v1 * vLen * uvScale}};
        Vertex12 e{{p01.x, p01.y, p01.z}, {n.x, n.y, n.z}, {t.x, t.y, t.z, 1.0f}, {u0 * uLen * uvScale, v1 * vLen * uvScale}};
        appendQuad(vertices, triangleMats, a, b, d, e, matId);
      }
    }
  };
  float dx = halfExt.x * 2.0f, dy = halfExt.y * 2.0f, dz = halfExt.z * 2.0f;
  Vec3 p000{c.x - halfExt.x, c.y - halfExt.y, c.z - halfExt.z};
  // +Z, -Z
  emitFace({p000.x, p000.y, c.z + halfExt.z}, {dx, 0, 0}, {0, dy, 0}, {0, 0, 1}, nxSub, nySub, dx, dy);
  emitFace({p000.x, c.y + halfExt.y, p000.z}, {dx, 0, 0}, {0, -dy, 0}, {0, 0, -1}, nxSub, nySub, dx, dy);
  // +Y, -Y
  emitFace({c.x + halfExt.x, c.y + halfExt.y, p000.z}, {-dx, 0, 0}, {0, 0, dz}, {0, 1, 0}, nxSub, nzSub, dx, dz);
  emitFace({p000.x, p000.y, p000.z}, {dx, 0, 0}, {0, 0, dz}, {0, -1, 0}, nxSub, nzSub, dx, dz);
  // +X, -X
  emitFace({c.x + halfExt.x, p000.y, p000.z}, {0, dy, 0}, {0, 0, dz}, {1, 0, 0}, nySub, nzSub, dy, dz);
  emitFace({p000.x, c.y + halfExt.y, p000.z}, {0, -dy, 0}, {0, 0, dz}, {-1, 0, 0}, nySub, nzSub, dy, dz);
}

inline void appendOrientedBeam(std::vector<float> &vertices,
                               std::vector<uint32_t> &triangleMats,
                               Vec3 pA, Vec3 pB,
                               float halfW, float halfH,
                               uint32_t matId, float uvScale = 0.6f,
                               uint32_t lengthSub = 1u) {
  Vec3 dir = pB - pA;
  float len = std::sqrt(vdot(dir, dir));
  if (len < 1e-4f) return;
  Vec3 fwd = dir * (1.0f / len);
  Vec3 refUp = (std::fabs(fwd.z) < 0.95f) ? Vec3{0.0f, 0.0f, 1.0f} : Vec3{0.0f, 1.0f, 0.0f};
  Vec3 right = vnorm(vcross(fwd, refUp));
  Vec3 up = vnorm(vcross(right, fwd));

  Vec3 offsets[4] = {
    right * (-halfW) + up * (-halfH),
    right * ( halfW) + up * (-halfH),
    right * ( halfW) + up * ( halfH),
    right * (-halfW) + up * ( halfH)
  };
  Vec3 norms[4] = { up * -1.0f, right, up, right * -1.0f };

  for (uint32_t s = 0; s < lengthSub; ++s) {
    float t0 = float(s) / float(lengthSub);
    float t1 = float(s + 1u) / float(lengthSub);
    Vec3 a = pA + dir * t0;
    Vec3 b = pA + dir * t1;
    for (int f = 0; f < 4; ++f) {
      int fNext = (f + 1) & 3;
      Vec3 q00 = a + offsets[f];
      Vec3 q10 = a + offsets[fNext];
      Vec3 q11 = b + offsets[fNext];
      Vec3 q01 = b + offsets[f];
      Vec3 n = norms[f];
      Vec3 tg = fwd;
      Vertex12 v0{{q00.x, q00.y, q00.z}, {n.x, n.y, n.z}, {tg.x, tg.y, tg.z, 1.0f}, {0.0f, t0 * len * uvScale}};
      Vertex12 v1{{q10.x, q10.y, q10.z}, {n.x, n.y, n.z}, {tg.x, tg.y, tg.z, 1.0f}, {1.0f, t0 * len * uvScale}};
      Vertex12 v2{{q11.x, q11.y, q11.z}, {n.x, n.y, n.z}, {tg.x, tg.y, tg.z, 1.0f}, {1.0f, t1 * len * uvScale}};
      Vertex12 v3{{q01.x, q01.y, q01.z}, {n.x, n.y, n.z}, {tg.x, tg.y, tg.z, 1.0f}, {0.0f, t1 * len * uvScale}};
      appendQuad(vertices, triangleMats, v0, v1, v2, v3, matId);
    }
  }
  // End caps
  for (int endIdx = 0; endIdx < 2; ++endIdx) {
    Vec3 base = (endIdx == 0) ? pA : pB;
    Vec3 n = (endIdx == 0) ? (fwd * -1.0f) : fwd;
    Vec3 tg = right;
    Vec3 q0 = base + offsets[0];
    Vec3 q1 = base + offsets[1];
    Vec3 q2 = base + offsets[2];
    Vec3 q3 = base + offsets[3];
    if (endIdx == 0) std::swap(q1, q3);
    Vertex12 v0{{q0.x, q0.y, q0.z}, {n.x, n.y, n.z}, {tg.x, tg.y, tg.z, 1.0f}, {0.0f, 0.0f}};
    Vertex12 v1{{q1.x, q1.y, q1.z}, {n.x, n.y, n.z}, {tg.x, tg.y, tg.z, 1.0f}, {1.0f, 0.0f}};
    Vertex12 v2{{q2.x, q2.y, q2.z}, {n.x, n.y, n.z}, {tg.x, tg.y, tg.z, 1.0f}, {1.0f, 1.0f}};
    Vertex12 v3{{q3.x, q3.y, q3.z}, {n.x, n.y, n.z}, {tg.x, tg.y, tg.z, 1.0f}, {0.0f, 1.0f}};
    appendQuad(vertices, triangleMats, v0, v1, v2, v3, matId);
  }
}

inline void appendCylinder(std::vector<float> &vertices,
                           std::vector<uint32_t> &triangleMats,
                           Vec3 pA, Vec3 pB,
                           float rA, float rB,
                           uint32_t segs, uint32_t stacks,
                           uint32_t matId, float uvScale = 1.0f,
                           bool capEnds = true, float crinkleAmp = 0.0f) {
  Vec3 axis = pB - pA;
  float len = std::sqrt(vdot(axis, axis));
  if (len < 1e-4f) return;
  Vec3 fwd = axis * (1.0f / len);
  Vec3 ref = (std::fabs(fwd.z) < 0.95f) ? Vec3{0.0f, 0.0f, 1.0f} : Vec3{0.0f, 1.0f, 0.0f};
  Vec3 uAxis = vnorm(vcross(fwd, ref));
  Vec3 vAxis = vnorm(vcross(uAxis, fwd));

  constexpr float kTwoPi = 6.283185307179586f;
  for (uint32_t st = 0; st < stacks; ++st) {
    float t0 = float(st) / float(stacks);
    float t1 = float(st + 1u) / float(stacks);
    float rad0 = rA + (rB - rA) * t0;
    float rad1 = rA + (rB - rA) * t1;
    Vec3 c0 = pA + axis * t0;
    Vec3 c1 = pA + axis * t1;

    for (uint32_t s = 0; s < segs; ++s) {
      float a0 = (float(s) / float(segs)) * kTwoPi;
      float a1 = (float(s + 1u) / float(segs)) * kTwoPi;

      auto radialPt = [&](Vec3 center, float rad, float ang, float tz, Vec3 &outNorm, Vec3 &outTan) -> Vec3 {
        float ca = std::cos(ang), sa = std::sin(ang);
        float cMod = (crinkleAmp > 0.0f)
            ? crinkleAmp * (std::sin(ang * 10.0f + tz * 22.0f) * 0.6f + std::cos(ang * 18.0f - tz * 15.0f) * 0.4f)
            : 0.0f;
        Vec3 nDir = uAxis * ca + vAxis * sa;
        outNorm = nDir;
        outTan = uAxis * (-sa) + vAxis * ca;
        return center + nDir * (rad + cMod);
      };

      Vec3 n00, t00, n10, t10, n11, t11, n01, t01;
      Vec3 p00 = radialPt(c0, rad0, a0, t0, n00, t00);
      Vec3 p10 = radialPt(c0, rad0, a1, t0, n10, t10);
      Vec3 p11 = radialPt(c1, rad1, a1, t1, n11, t11);
      Vec3 p01 = radialPt(c1, rad1, a0, t1, n01, t01);

      float u0 = (float(s) / float(segs)) * uvScale;
      float u1 = (float(s + 1u) / float(segs)) * uvScale;
      float v0 = t0 * len * 0.35f * uvScale;
      float v1 = t1 * len * 0.35f * uvScale;

      Vertex12 q00{{p00.x, p00.y, p00.z}, {n00.x, n00.y, n00.z}, {t00.x, t00.y, t00.z, 1.0f}, {u0, v0}};
      Vertex12 q10{{p10.x, p10.y, p10.z}, {n10.x, n10.y, n10.z}, {t10.x, t10.y, t10.z, 1.0f}, {u1, v0}};
      Vertex12 q11{{p11.x, p11.y, p11.z}, {n11.x, n11.y, n11.z}, {t11.x, t11.y, t11.z, 1.0f}, {u1, v1}};
      Vertex12 q01{{p01.x, p01.y, p01.z}, {n01.x, n01.y, n01.z}, {t01.x, t01.y, t01.z, 1.0f}, {u0, v1}};
      appendQuad(vertices, triangleMats, q00, q10, q11, q01, matId);
    }
  }

  if (capEnds) {
    for (int endIdx = 0; endIdx < 2; ++endIdx) {
      Vec3 center = (endIdx == 0) ? pA : pB;
      float rad = (endIdx == 0) ? rA : rB;
      if (rad < 1e-4f) continue;
      Vec3 capN = (endIdx == 0) ? (fwd * -1.0f) : fwd;
      Vertex12 vc{{center.x, center.y, center.z}, {capN.x, capN.y, capN.z}, {uAxis.x, uAxis.y, uAxis.z, 1.0f}, {0.5f, 0.5f}};
      for (uint32_t s = 0; s < segs; ++s) {
        float a0 = (float(s) / float(segs)) * kTwoPi;
        float a1 = (float(s + 1u) / float(segs)) * kTwoPi;
        Vec3 rp0 = center + (uAxis * std::cos(a0) + vAxis * std::sin(a0)) * rad;
        Vec3 rp1 = center + (uAxis * std::cos(a1) + vAxis * std::sin(a1)) * rad;
        Vertex12 vr0{{rp0.x, rp0.y, rp0.z}, {capN.x, capN.y, capN.z}, {uAxis.x, uAxis.y, uAxis.z, 1.0f}, {0.5f + 0.5f * std::cos(a0), 0.5f + 0.5f * std::sin(a0)}};
        Vertex12 vr1{{rp1.x, rp1.y, rp1.z}, {capN.x, capN.y, capN.z}, {uAxis.x, uAxis.y, uAxis.z, 1.0f}, {0.5f + 0.5f * std::cos(a1), 0.5f + 0.5f * std::sin(a1)}};
        if (endIdx == 0) {
          appendTri(vertices, triangleMats, vc, vr1, vr0, matId);
        } else {
          appendTri(vertices, triangleMats, vc, vr0, vr1, matId);
        }
      }
    }
  }
}

inline void appendTorus(std::vector<float> &vertices,
                        std::vector<uint32_t> &triangleMats,
                        Vec3 center, Vec3 axis,
                        float majorR, float minorR,
                        uint32_t majorSegs, uint32_t minorSegs,
                        uint32_t matId, float uvScale = 2.0f,
                        float ridgeAmp = 0.0f, float ridgeFreq = 16.0f) {
  Vec3 nAxis = vnorm(axis);
  Vec3 ref = (std::fabs(nAxis.z) < 0.95f) ? Vec3{0.0f, 0.0f, 1.0f} : Vec3{1.0f, 0.0f, 0.0f};
  Vec3 uDir = vnorm(vcross(nAxis, ref));
  Vec3 vDir = vnorm(vcross(nAxis, uDir));

  constexpr float kTwoPi = 6.283185307179586f;
  auto evalPt = [&](uint32_t i, uint32_t j) -> Vertex12 {
    float u = (float(i % majorSegs) / float(majorSegs)) * kTwoPi;
    float v = (float(j % minorSegs) / float(minorSegs)) * kTwoPi;
    float cu = std::cos(u), su = std::sin(u);
    float cv = std::cos(v), sv = std::sin(v);
    Vec3 radial = uDir * cu + vDir * su;
    Vec3 tangU = uDir * (-su) + vDir * cu;
    float rMod = minorR + ridgeAmp * std::cos(u * ridgeFreq);
    Vec3 n = vnorm(radial * cv + nAxis * sv);
    Vec3 p = center + radial * (majorR + rMod * cv) + nAxis * (rMod * sv);
    return Vertex12{
        {p.x, p.y, p.z},
        {n.x, n.y, n.z},
        {tangU.x, tangU.y, tangU.z, 1.0f},
        {(float(i) / float(majorSegs)) * uvScale * 4.0f, (float(j) / float(minorSegs)) * uvScale}
    };
  };

  for (uint32_t i = 0; i < majorSegs; ++i) {
    for (uint32_t j = 0; j < minorSegs; ++j) {
      Vertex12 p00 = evalPt(i, j);
      Vertex12 p10 = evalPt(i + 1u, j);
      Vertex12 p11 = evalPt(i + 1u, j + 1u);
      Vertex12 p01 = evalPt(i, j + 1u);
      appendQuad(vertices, triangleMats, p00, p10, p11, p01, matId);
    }
  }
}

inline void appendCurvedConduit(std::vector<float> &vertices,
                                std::vector<uint32_t> &triangleMats,
                                Vec3 p0, Vec3 p1, Vec3 p2, Vec3 p3,
                                float radius, uint32_t pathSegs, uint32_t ringSegs,
                                uint32_t matId, bool addCollars = true) {
  auto bezierPos = [&](float t) -> Vec3 {
    float it = 1.0f - t;
    float b0 = it * it * it;
    float b1 = 3.0f * it * it * t;
    float b2 = 3.0f * it * t * t;
    float b3 = t * t * t;
    return p0 * b0 + p1 * b1 + p2 * b2 + p3 * b3;
  };
  constexpr float kTwoPi = 6.283185307179586f;
  std::vector<Vec3> pts(pathSegs + 1u), fwds(pathSegs + 1u);
  for (uint32_t i = 0; i <= pathSegs; ++i) {
    pts[i] = bezierPos(float(i) / float(pathSegs));
  }
  for (uint32_t i = 0; i <= pathSegs; ++i) {
    uint32_t a = (i > 0u) ? (i - 1u) : 0u;
    uint32_t b = (i < pathSegs) ? (i + 1u) : pathSegs;
    fwds[i] = vnorm(pts[b] - pts[a]);
  }

  for (uint32_t i = 0; i < pathSegs; ++i) {
    Vec3 f0 = fwds[i];
    Vec3 f1 = fwds[i + 1u];
    Vec3 r0U = vnorm(vcross(f0, (std::fabs(f0.z) < 0.95f) ? Vec3{0, 0, 1} : Vec3{1, 0, 0}));
    Vec3 r0V = vnorm(vcross(r0U, f0));
    Vec3 r1U = vnorm(vcross(f1, (std::fabs(f1.z) < 0.95f) ? Vec3{0, 0, 1} : Vec3{1, 0, 0}));
    Vec3 r1V = vnorm(vcross(r1U, f1));

    // Ribbed corrugation for braided cryo conduits (MAT_CERAMIC)
    float rib0 = 1.0f + 0.07f * std::sin(float(i) * 1.4f);
    float rib1 = 1.0f + 0.07f * std::sin(float(i + 1u) * 1.4f);

    for (uint32_t j = 0; j < ringSegs; ++j) {
      float a0 = (float(j) / float(ringSegs)) * kTwoPi;
      float a1 = (float(j + 1u) / float(ringSegs)) * kTwoPi;
      float ca0 = std::cos(a0), sa0 = std::sin(a0);
      float ca1 = std::cos(a1), sa1 = std::sin(a1);

      Vec3 n00 = vnorm(r0U * ca0 + r0V * sa0);
      Vec3 n10 = vnorm(r0U * ca1 + r0V * sa1);
      Vec3 n11 = vnorm(r1U * ca1 + r1V * sa1);
      Vec3 n01 = vnorm(r1U * ca0 + r1V * sa0);

      Vec3 q00 = pts[i] + n00 * (radius * rib0);
      Vec3 q10 = pts[i] + n10 * (radius * rib0);
      Vec3 q11 = pts[i + 1u] + n11 * (radius * rib1);
      Vec3 q01 = pts[i + 1u] + n01 * (radius * rib1);

      float vCoord0 = float(i) * 0.25f;
      float vCoord1 = float(i + 1u) * 0.25f;
      Vertex12 vtx0{{q00.x, q00.y, q00.z}, {n00.x, n00.y, n00.z}, {f0.x, f0.y, f0.z, 1.0f}, {float(j) / ringSegs, vCoord0}};
      Vertex12 vtx1{{q10.x, q10.y, q10.z}, {n10.x, n10.y, n10.z}, {f0.x, f0.y, f0.z, 1.0f}, {float(j + 1u) / ringSegs, vCoord0}};
      Vertex12 vtx2{{q11.x, q11.y, q11.z}, {n11.x, n11.y, n11.z}, {f1.x, f1.y, f1.z, 1.0f}, {float(j + 1u) / ringSegs, vCoord1}};
      Vertex12 vtx3{{q01.x, q01.y, q01.z}, {n01.x, n01.y, n01.z}, {f1.x, f1.y, f1.z, 1.0f}, {float(j) / ringSegs, vCoord1}};
      appendQuad(vertices, triangleMats, vtx0, vtx1, vtx2, vtx3, matId);
    }

    if (addCollars && (i % 10u == 5u) && i + 1u < pathSegs) {
      appendCylinder(vertices, triangleMats,
                     pts[i] - f0 * (radius * 0.45f),
                     pts[i] + f0 * (radius * 0.45f),
                     radius * 1.28f, radius * 1.28f,
                     ringSegs, 1u, MAT_TITANIUM, 1.0f, true);
    }
  }
}

// Physical 3D open-grid catwalk deck + perforated safety screen panels (MAT_TITANIUM frame + MAT_HEX_GRATE infill)
inline void appendCatwalkSegment(std::vector<float> &vertices,
                                 std::vector<uint32_t> &triangleMats,
                                 float xMin, float xMax,
                                 float yMin, float yMax,
                                 float deckZ,
                                 bool railFront, bool railBack,
                                 bool railLeft, bool railRight) {
  float cx = 0.5f * (xMin + xMax);
  float cy = 0.5f * (yMin + yMax);
  float wx = std::max(0.2f, xMax - xMin);
  float wy = std::max(0.2f, yMax - yMin);

  // Heavy I-beam stringer rails along catwalk edges
  if (wx >= wy) {
    appendBox(vertices, triangleMats, {cx, yMin + 0.08f, deckZ - 0.14f}, {wx * 0.5f, 0.08f, 0.14f}, MAT_TITANIUM, 0.5f, 4u, 1u, 1u);
    appendBox(vertices, triangleMats, {cx, yMax - 0.08f, deckZ - 0.14f}, {wx * 0.5f, 0.08f, 0.14f}, MAT_TITANIUM, 0.5f, 4u, 1u, 1u);
  } else {
    appendBox(vertices, triangleMats, {xMin + 0.08f, cy, deckZ - 0.14f}, {0.08f, wy * 0.5f, 0.14f}, MAT_TITANIUM, 0.5f, 1u, 4u, 1u);
    appendBox(vertices, triangleMats, {xMax - 0.08f, cy, deckZ - 0.14f}, {0.08f, wy * 0.5f, 0.14f}, MAT_TITANIUM, 0.5f, 1u, 4u, 1u);
  }

  const bool isForegroundCatwalk = (deckZ < 4.0f);

  // Cross support ribs + open-slat tread strips (physical 3D open aperture between grates!)
  uint32_t numSlats = static_cast<uint32_t>(std::max(4.0f, std::floor(std::max(wx, wy) * (isForegroundCatwalk ? 2.6f : 1.4f))));
  for (uint32_t i = 0; i < numSlats; ++i) {
    float frac = (float(i) + 0.5f) / float(numSlats);
    uint32_t slatMat = (isForegroundCatwalk && (i & 1u)) ? MAT_TITANIUM : MAT_HEX_GRATE;
    float barHalfW = isForegroundCatwalk ? 0.028f : 0.06f;
    float barHalfH = isForegroundCatwalk ? 0.022f : 0.03f;
    if (wx >= wy) {
      float x = xMin + wx * frac;
      appendBox(vertices, triangleMats, {x, cy, deckZ - 0.025f}, {barHalfW, wy * 0.46f, barHalfH}, slatMat, 1.5f, 1u, 2u, 1u);
    } else {
      float y = yMin + wy * frac;
      appendBox(vertices, triangleMats, {cx, y, deckZ - 0.025f}, {wx * 0.46f, barHalfW, barHalfH}, slatMat, 1.5f, 2u, 1u, 1u);
    }
  }

  if (isForegroundCatwalk) {
    // Open 3D bar grating: perpendicular runner rods let opaque camera rays interleave deck bars & trenches/floor below
    const float inset = 0.18f;
    float spanCross = std::max(0.2f, ((wx >= wy) ? wy : wx) - 2.0f * inset);
    uint32_t numRunners = static_cast<uint32_t>(std::max(3.0f, std::floor(spanCross * 2.4f)));
    for (uint32_t r = 0; r < numRunners; ++r) {
      float frac = (float(r) + 0.5f) / float(numRunners);
      uint32_t runMat = (r % 3u == 0u) ? MAT_CLEARCOAT : ((r & 1u) ? MAT_TITANIUM : MAT_HEX_GRATE);
      if (wx >= wy) {
        float y = (yMin + inset) + spanCross * frac;
        appendBox(vertices, triangleMats, {cx, y, deckZ - 0.015f}, {wx * 0.48f, 0.020f, 0.018f}, runMat, 1.5f, 6u, 1u, 1u);
      } else {
        float x = (xMin + inset) + spanCross * frac;
        appendBox(vertices, triangleMats, {x, cy, deckZ - 0.015f}, {0.020f, wy * 0.48f, 0.018f}, runMat, 1.5f, 1u, 6u, 1u);
      }
    }
  } else {
    // Perforated thin-sheet hex-grate walkway deck strip centered inside frame (for elevated catwalks & Alpha Cutout mode)
    const float inset = 0.14f;
    float gx0 = xMin + inset, gx1 = xMax - inset;
    float gy0 = yMin + inset, gy1 = yMax - inset;
    if (gx1 > gx0 && gy1 > gy0) {
      float uvX = (gx1 - gx0) * 1.8f;
      float uvY = (gy1 - gy0) * 1.8f;
      Vertex12 g0{{gx0, gy0, deckZ + 0.005f}, {0, 0, 1}, {1, 0, 0, 1}, {0.0f, 0.0f}};
      Vertex12 g1{{gx1, gy0, deckZ + 0.005f}, {0, 0, 1}, {1, 0, 0, 1}, {uvX, 0.0f}};
      Vertex12 g2{{gx1, gy1, deckZ + 0.005f}, {0, 0, 1}, {1, 0, 0, 1}, {uvX, uvY}};
      Vertex12 g3{{gx0, gy1, deckZ + 0.005f}, {0, 0, 1}, {1, 0, 0, 1}, {0.0f, uvY}};
      appendQuad(vertices, triangleMats, g0, g1, g2, g3, MAT_HEX_GRATE);
    }
  }

  auto buildRailingRun = [&](Vec3 a, Vec3 b, bool addPerfGuard) {
    Vec3 d = b - a;
    float rlen = std::sqrt(vdot(d, d));
    if (rlen < 0.2f) return;
    // Top handrail + mid bar + toe kickboard
    appendCylinder(vertices, triangleMats, a + Vec3{0, 0, 1.05f}, b + Vec3{0, 0, 1.05f}, 0.045f, 0.045f, 8u, 1u, MAT_TITANIUM, 1.0f, true);
    appendCylinder(vertices, triangleMats, a + Vec3{0, 0, 0.56f}, b + Vec3{0, 0, 0.56f}, 0.032f, 0.032f, 6u, 1u, MAT_TITANIUM, 1.0f, true);
    appendOrientedBeam(vertices, triangleMats, a + Vec3{0, 0, 0.08f}, b + Vec3{0, 0, 0.08f}, 0.025f, 0.07f, MAT_CLEARCOAT, 0.8f, 1u);

    uint32_t numPosts = static_cast<uint32_t>(std::max(2.0f, std::ceil(rlen / 1.6f))) + 1u;
    for (uint32_t p = 0; p < numPosts; ++p) {
      float f = float(p) / float(numPosts - 1u);
      Vec3 pt = a + d * f;
      appendCylinder(vertices, triangleMats, pt, pt + Vec3{0, 0, 1.08f}, 0.045f, 0.045f, 8u, 1u, MAT_TITANIUM, 1.0f, true);
    }

    if (addPerfGuard) {
      if (isForegroundCatwalk) {
        // Physical vertical 3D safety pickets for foreground railing (open sightlines + crisp sub-tile material mixing)
        uint32_t numPickets = static_cast<uint32_t>(std::max(4.0f, std::floor(rlen / 0.28f)));
        for (uint32_t pk = 1u; pk < numPickets; ++pk) {
          float f = float(pk) / float(numPickets);
          Vec3 pt = a + d * f;
          uint32_t pMat = (pk & 1u) ? MAT_HEX_GRATE : MAT_CERAMIC;
          appendCylinder(vertices, triangleMats, pt + Vec3{0, 0, 0.16f}, pt + Vec3{0, 0, 0.54f},
                         0.015f, 0.015f, 6u, 1u, pMat, 1.0f, false);
        }
      } else {
        // Lower safety cage perforated mesh panel (Z = 0.16 .. 0.54)
        Vec3 tang = vnorm(d);
        Vec3 n = vnorm(vcross(tang, {0.0f, 0.0f, 1.0f}));
        float uMax = rlen * 1.6f;
        Vec3 p0 = a + Vec3{0, 0, 0.16f}, p1 = b + Vec3{0, 0, 0.16f};
        Vec3 p2 = b + Vec3{0, 0, 0.53f}, p3 = a + Vec3{0, 0, 0.53f};
        Vertex12 q0{{p0.x, p0.y, p0.z}, {n.x, n.y, n.z}, {tang.x, tang.y, tang.z, 1.0f}, {0.0f, 0.0f}};
        Vertex12 q1{{p1.x, p1.y, p1.z}, {n.x, n.y, n.z}, {tang.x, tang.y, tang.z, 1.0f}, {uMax, 0.0f}};
        Vertex12 q2{{p2.x, p2.y, p2.z}, {n.x, n.y, n.z}, {tang.x, tang.y, tang.z, 1.0f}, {uMax, 0.65f}};
        Vertex12 q3{{p3.x, p3.y, p3.z}, {n.x, n.y, n.z}, {tang.x, tang.y, tang.z, 1.0f}, {0.0f, 0.65f}};
        appendQuad(vertices, triangleMats, q0, q1, q2, q3, MAT_HEX_GRATE);
      }
    }
  };

  if (railFront) buildRailingRun({xMin, yMin, deckZ}, {xMax, yMin, deckZ}, true);
  if (railBack)  buildRailingRun({xMin, yMax, deckZ}, {xMax, yMax, deckZ}, true);
  if (railLeft)  buildRailingRun({xMin, yMin, deckZ}, {xMin, yMax, deckZ}, true);
  if (railRight) buildRailingRun({xMax, yMin, deckZ}, {xMax, yMax, deckZ}, true);
}

} // namespace detail

inline void buildReactorBayMesh(std::vector<float> &vertices,
                                std::vector<uint32_t> &triangleMats) {
  using namespace detail;
  vertices.clear();
  triangleMats.clear();
  vertices.reserve(680000u * 36u);
  triangleMats.reserve(680000u);

  // =========================================================================
  // 1. HANGAR DECK, RECESSED COOLANT TRENCHES & ARMORED GLASS WALK COVERS
  //    X in [-28, 28], Y in [-28, 34] (~100,000 triangles)
  // =========================================================================
  {
    const uint32_t nx = 160u, ny = 192u;
    const float x0 = -28.0f, x1 = 28.0f;
    const float y0 = -28.0f, y1 = 34.0f;

    auto deckHeight = [](float x, float y) -> float {
      // Twin recessed glowing coolant trenches flanking the central aisle (X in [-5.8..-2.8] & [2.8..5.8])
      float ax = std::fabs(x);
      if (ax > 2.6f && ax < 6.0f && y > -20.0f && y < 24.0f) {
        float bevel = std::clamp(std::min(ax - 2.6f, 6.0f - ax) * 3.0f, 0.0f, 1.0f);
        float yBevel = std::clamp(std::min(y + 20.0f, 24.0f - y) * 1.5f, 0.0f, 1.0f);
        return -0.95f * bevel * yBevel;
      }
      // Shallow expansion-joint grooves across ferrocrete slabs
      float joint = (std::fabs(std::fmod(x + 28.0f, 4.0f) - 2.0f) > 1.92f ||
                     std::fabs(std::fmod(y + 28.0f, 4.0f) - 2.0f) > 1.92f) ? -0.025f : 0.0f;
      return joint;
    };

    for (uint32_t j = 0; j < ny; ++j) {
      float ty0 = float(j) / float(ny);
      float ty1 = float(j + 1u) / float(ny);
      float py0 = y0 + (y1 - y0) * ty0;
      float py1 = y0 + (y1 - y0) * ty1;
      for (uint32_t i = 0; i < nx; ++i) {
        float tx0 = float(i) / float(nx);
        float tx1 = float(i + 1u) / float(nx);
        float px0 = x0 + (x1 - x0) * tx0;
        float px1 = x0 + (x1 - x0) * tx1;

        float z00 = deckHeight(px0, py0);
        float z10 = deckHeight(px1, py0);
        float z11 = deckHeight(px1, py1);
        float z01 = deckHeight(px0, py1);

        auto gradNorm = [&](float x, float y) -> Vec3 {
          float eps = 0.08f;
          float dx = (deckHeight(x + eps, y) - deckHeight(x - eps, y)) / (2.0f * eps);
          float dy = (deckHeight(x, y + eps) - deckHeight(x, y - eps)) / (2.0f * eps);
          return vnorm({-dx, -dy, 1.0f});
        };
        Vec3 n00 = gradNorm(px0, py0), n10 = gradNorm(px1, py0);
        Vec3 n11 = gradNorm(px1, py1), n01 = gradNorm(px0, py1);

        float cx = 0.5f * (px0 + px1);
        float cy = 0.5f * (py0 + py1);
        float cz = 0.25f * (z00 + z10 + z11 + z01);
        uint32_t matId = MAT_FERROCRETE;
        if (cz < -0.55f) {
          matId = MAT_COOLANT;
        } else if (cz < -0.08f) {
          matId = MAT_TITANIUM;
        } else if (std::fabs(cx) < 2.55f && std::fabs(cx) > 2.25f && cy > -20.0f && cy < 24.0f) {
          matId = MAT_CLEARCOAT; // Hazard trim flanking trench
        }

        Vertex12 a{{px0, py0, z00}, {n00.x, n00.y, n00.z}, {1, 0, 0, 1}, {px0 * 0.25f, py0 * 0.25f}};
        Vertex12 b{{px1, py0, z10}, {n10.x, n10.y, n10.z}, {1, 0, 0, 1}, {px1 * 0.25f, py0 * 0.25f}};
        Vertex12 c{{px1, py1, z11}, {n11.x, n11.y, n11.z}, {1, 0, 0, 1}, {px1 * 0.25f, py1 * 0.25f}};
        Vertex12 d{{px0, py1, z01}, {n01.x, n01.y, n01.z}, {1, 0, 0, 1}, {px0 * 0.25f, py1 * 0.25f}};
        appendQuad(vertices, triangleMats, a, b, c, d, matId);
      }
    }

    // Armored-glass coolant trench covers + submerged glowing plasma conduit pipes inside the trenches!
    for (int side : {-1, 1}) {
      float cxTrench = float(side) * 4.3f;
      // Glowing energy busbars submerged under glass covers (Z = -0.55m)
      appendCylinder(vertices, triangleMats,
                     {cxTrench - 0.55f, -19.5f, -0.55f}, {cxTrench - 0.55f, 23.5f, -0.55f},
                     0.22f, 0.22f, 16u, 32u, MAT_COOLANT, 1.5f, true);
      appendCylinder(vertices, triangleMats,
                     {cxTrench + 0.55f, -19.5f, -0.55f}, {cxTrench + 0.55f, 23.5f, -0.55f},
                     0.22f, 0.22f, 16u, 32u, MAT_CERAMIC, 1.5f, true);
      // Heavy submerged support cradles every 3m
      for (float yC = -17.0f; yC <= 21.0f; yC += 3.0f) {
        appendBox(vertices, triangleMats, {cxTrench, yC, -0.68f}, {1.25f, 0.16f, 0.18f}, MAT_TITANIUM, 0.8f, 2u, 1u, 1u);
      }
      // Segmented Armored-Glass Viewing Panels + Titanium crossbars + Hex Grate panels over the trenches
      for (uint32_t k = 0; k < 14u; ++k) {
        float yMinP = -18.8f + float(k) * 3.0f;
        float yMaxP = yMinP + 2.45f;
        appendBox(vertices, triangleMats, {cxTrench, 0.5f * (yMinP + yMaxP), -0.04f},
                  {1.32f, 0.5f * (yMaxP - yMinP), 0.03f}, MAT_GLASS, 0.5f, 4u, 6u, 1u);
        appendBox(vertices, triangleMats, {cxTrench, yMaxP + 0.275f, -0.02f},
                  {1.55f, 0.275f, 0.06f}, (k % 2u == 0u) ? MAT_HEX_GRATE : MAT_TITANIUM, 1.2f, 4u, 1u, 1u);
      }
    }
  }

  // =========================================================================
  // 2. ENCLOSING HANGAR BULKHEADS, SIDE ALCOVES & OVERHEAD SKYLIGHT LOUVERS
  //    (~95,000 triangles)
  // =========================================================================
  {
    // Left & Right Structural Side Bulkheads + Deep Maintenance Alcoves
    for (int side : {-1, 1}) {
      float wallX = float(side) * 23.5f;
      // Outer ferrocrete blast wall
      appendBox(vertices, triangleMats, {wallX + float(side) * 1.2f, 3.0f, 10.5f},
                {0.8f, 32.0f, 11.5f}, MAT_FERROCRETE, 0.25f, 2u, 32u, 14u);

      // 8 Massive Angled Architectural Structural Ribs per side (classic heavy brutalist sci-fi hangar)
      for (uint32_t bay = 0; bay < 9u; ++bay) {
        float yRib = -22.0f + float(bay) * 6.8f;
        // Vertical lower column
        appendBox(vertices, triangleMats, {wallX - float(side) * 1.1f, yRib, 5.2f},
                  {0.95f, 0.55f, 5.2f}, MAT_TITANIUM, 0.4f, 2u, 2u, 8u);
        // Haunched diagonal upper rib arching toward roof truss
        appendOrientedBeam(vertices, triangleMats,
                           {wallX - float(side) * 1.0f, yRib, 10.0f},
                           {float(side) * 12.0f, yRib, 19.6f},
                           0.48f, 0.65f, MAT_TITANIUM, 0.4f, 6u);
        // Clearcoat hazard base plinth & ceramic conduit junction box inside each alcove bay
        appendBox(vertices, triangleMats, {wallX - float(side) * 1.2f, yRib, 0.85f},
                  {1.15f, 0.72f, 0.85f}, MAT_CLEARCOAT, 0.6f, 2u, 2u, 2u);
        if (bay < 8u) {
          float yMid = yRib + 3.4f;
          // Deep alcove equipment rack / thermal-blanketed acoustic baffle panels
          appendBox(vertices, triangleMats, {wallX - float(side) * 0.2f, yMid, 3.2f},
                    {0.65f, 2.1f, 2.2f}, (bay & 1u) ? MAT_MLI_FOIL : MAT_CLEARCOAT, 0.8f, 2u, 6u, 6u);
          // Multi-material upper wall conduit bundles + ribbed ceramic/clearcoat acoustic baffle cassettes
          for (uint32_t tier = 0; tier < 3u; ++tier) {
            float zTier = 7.4f + float(tier) * 3.1f;
            float pipeX = wallX - float(side) * 0.45f;
            uint32_t pMat = (tier == 0u) ? MAT_CERAMIC : ((tier == 1u) ? MAT_CLEARCOAT : MAT_MLI_FOIL);
            appendCylinder(vertices, triangleMats, {pipeX, yRib + 0.5f, zTier}, {pipeX, yRib + 6.3f, zTier},
                           0.20f, 0.20f, 10u, 2u, pMat, 1.2f, false);
            appendBox(vertices, triangleMats, {wallX - float(side) * 0.15f, yMid + float(tier - 1) * 0.6f, zTier + 0.85f},
                      {0.25f, 1.85f, 0.42f}, (tier & 1u) ? MAT_TITANIUM : MAT_CERAMIC, 1.0f, 1u, 3u, 1u);
          }
          // Vertical wall coolant riser + diagnostic status strip
          appendCylinder(vertices, triangleMats, {wallX - float(side) * 0.65f, yMid - 1.4f, 1.2f},
                         {wallX - float(side) * 0.65f, yMid - 1.4f, 17.5f}, 0.16f, 0.16f, 10u, 3u, MAT_COOLANT, 1.2f, false);
          // Perforated alcove maintenance grille screen
          appendBox(vertices, triangleMats, {wallX - float(side) * 1.4f, yMid, 12.2f},
                    {0.18f, 2.3f, 1.4f}, MAT_HEX_GRATE, 1.6f, 1u, 4u, 4u);
        }
      }
    }

    // Far Rear Hangar Blast Bulkhead (Y = 30.5m) with Giant Recessed Containment Portal
    appendBox(vertices, triangleMats, {0.0f, 32.0f, 10.5f}, {26.0f, 1.0f, 11.5f}, MAT_FERROCRETE, 0.22f, 32u, 2u, 16u);
    appendTorus(vertices, triangleMats, {0.0f, 29.5f, 7.6f}, {0.0f, 1.0f, 0.0f},
                6.8f, 0.85f, 48u, 16u, MAT_TITANIUM, 1.5f, 0.08f, 24.0f);
    appendTorus(vertices, triangleMats, {0.0f, 29.2f, 7.6f}, {0.0f, 1.0f, 0.0f},
                5.4f, 0.45f, 48u, 14u, MAT_CLEARCOAT, 1.5f);
    // Segmented blast door leaves with chevron ribs
    for (int s : {-1, 1}) {
      appendBox(vertices, triangleMats, {float(s) * 2.55f, 30.2f, 6.2f},
                {2.45f, 0.45f, 6.0f}, MAT_TITANIUM, 0.45f, 6u, 1u, 10u);
    }

    // Front Behind-Camera Bulkhead (Y = -28.0m) so bounce rays stay 100% enclosed in hangar!
    appendBox(vertices, triangleMats, {0.0f, -28.5f, 10.5f}, {26.0f, 0.8f, 11.5f}, MAT_FERROCRETE, 0.25f, 20u, 1u, 10u);

    // Overhead Hangar Roof Girders & Angled Skylight Louver Bays (Z = 19.8m .. 22.5m)
    // Open gaps between louvers let crisp overhead sunlight shafts cut through into the bay!
    for (float yBeam = -24.0f; yBeam <= 30.0f; yBeam += 4.5f) {
      appendBox(vertices, triangleMats, {0.0f, yBeam, 20.2f}, {23.0f, 0.55f, 0.70f}, MAT_TITANIUM, 0.35f, 16u, 2u, 2u);
    }
    // Side ceiling soffits leaving central X in [-9.5, 9.5] skylight atrium partially louvered
    for (int side : {-1, 1}) {
      float sx = float(side) * 16.5f;
      appendBox(vertices, triangleMats, {sx, 3.0f, 20.8f},
                {7.2f, 31.0f, 0.45f}, MAT_FERROCRETE, 0.3f, 8u, 24u, 1u);
      // Longitudinal overhead gantry crane rails & insulator/MLI coffer cassette panels beneath soffits
      appendBox(vertices, triangleMats, {float(side) * 11.4f, 3.0f, 19.35f},
                {0.42f, 29.0f, 0.42f}, MAT_TITANIUM, 0.5f, 1u, 18u, 1u);
      appendBox(vertices, triangleMats, {float(side) * 11.4f, 3.0f, 18.85f},
                {0.24f, 29.0f, 0.16f}, MAT_CLEARCOAT, 0.7f, 1u, 18u, 1u);
      for (int by = -20; by <= 24; by += 6) {
        uint32_t cofferMat = ((by / 6) & 1) ? MAT_MLI_FOIL : MAT_CERAMIC;
        appendBox(vertices, triangleMats, {float(side) * 16.8f, float(by), 20.22f},
                  {3.4f, 2.1f, 0.14f}, cofferMat, 1.0f, 2u, 3u, 1u);
        appendBox(vertices, triangleMats, {float(side) * 14.5f, float(by) + 2.8f, 19.65f},
                  {1.15f, 0.24f, 0.16f}, MAT_HEX_GRATE, 1.5f, 2u, 1u, 1u);
        appendCylinder(vertices, triangleMats, {float(side) * 12.8f, float(by) + 2.8f, 19.55f},
                       {float(side) * 16.2f, float(by) + 2.8f, 19.55f}, 0.10f, 0.10f, 8u, 1u, MAT_COOLANT, 1.0f, false);
      }
    }
    // Angled overhead skylight louvers across X in [-9.2, 9.2]
    for (float yLouver = -21.0f; yLouver <= 27.0f; yLouver += 3.0f) {
      appendOrientedBeam(vertices, triangleMats,
                         {-9.2f, yLouver, 21.4f}, {9.2f, yLouver, 22.1f},
                         0.42f, 0.14f, MAT_TITANIUM, 0.5f, 8u);
    }
  }

  // =========================================================================
  // 3. CENTRAL MAGNETIC CONFINEMENT REACTOR CORE ("ZERO-DAY" HERO PIECE)
  //    Center = (0.0, 5.2, 7.6), ~450,000 triangles across all 8 materials!
  // =========================================================================
  {
    const Vec3 rc{0.0f, 5.2f, 7.6f};

    // Octagonal ferrocrete + titanium reactor pedestal cradle seated on deck
    appendCylinder(vertices, triangleMats, {0.0f, 5.2f, 0.0f}, {0.0f, 5.2f, 1.4f}, 5.8f, 5.2f, 48u, 4u, MAT_FERROCRETE, 1.0f, true);
    appendCylinder(vertices, triangleMats, {0.0f, 5.2f, 1.4f}, {0.0f, 5.2f, 2.2f}, 5.0f, 4.4f, 64u, 4u, MAT_TITANIUM, 1.2f, true);

    // 4 Massive Slanted Titanium A-Frame Core Cradle Struts anchoring Torus at Z=7.6m
    for (int sx : {-1, 1}) {
      for (int sy : {-1, 1}) {
        Vec3 basePt{float(sx) * 6.8f, rc.y + float(sy) * 4.2f, 0.2f};
        Vec3 topPt{float(sx) * 3.8f, rc.y + float(sy) * 2.2f, 6.2f};
        appendOrientedBeam(vertices, triangleMats, basePt, topPt, 0.42f, 0.48f, MAT_TITANIUM, 0.7f, 6u);
        appendCylinder(vertices, triangleMats, basePt, basePt + Vec3{0, 0, 1.1f}, 0.62f, 0.56f, 24u, 3u, MAT_CLEARCOAT, 1.0f, true);
      }
    }

    // Inner Glowing Iridescent Plasma Containment Vessel (Primary Torus tilted slightly toward camera)
    const Vec3 coreAxis = vnorm({0.0f, 0.92f, 0.39f});
    appendTorus(vertices, triangleMats, rc, coreAxis, 3.85f, 1.05f, 192u, 64u, MAT_COOLANT, 2.5f, 0.04f, 24.0f);

    // Armored Quartz Plasma Containment Shroud Rings (Refractive Glass Shells!)
    appendTorus(vertices, triangleMats, rc, coreAxis, 3.85f, 1.24f, 160u, 48u, MAT_GLASS, 2.0f);

    // Heavy Clearcoat Safety-Orange Containment Collar Plates wrapping outer arcs of the Torus
    for (uint32_t seg = 0; seg < 8u; ++seg) {
      float offsetY = (float(seg) - 3.5f) * 1.15f;
      Vec3 ringCenter = rc + Vec3{0.0f, offsetY * 0.45f, -offsetY * 0.18f};
      float rad = 3.2f + 1.4f * std::cos(offsetY * 0.45f);
      appendTorus(vertices, triangleMats, ringCenter, coreAxis, rad, 0.34f, 128u, 36u,
                  (seg % 2u == 0u) ? MAT_CLEARCOAT : MAT_TITANIUM, 1.8f, 0.03f, 16.0f);
    }

    // 28 Radial D-Shaped Tokamak Poloidal Field Magnet Coils (combining Iridescent Coil, Ceramic Insulator & Titanium Clamps)
    const uint32_t kNumCoils = 28u;
    constexpr float kTwoPi = 6.283185307179586f;
    Vec3 cU = vnorm(vcross(coreAxis, {1, 0, 0}));
    Vec3 cV = vnorm(vcross(coreAxis, cU));
    for (uint32_t c = 0; c < kNumCoils; ++c) {
      float phi = (float(c) / float(kNumCoils)) * kTwoPi;
      Vec3 radialOut = cU * std::cos(phi) + cV * std::sin(phi);
      Vec3 coilAxis = vnorm(vcross(coreAxis, radialOut));
      Vec3 coilCenter = rc + radialOut * 3.85f;
      // Alternating iridescent superconducting winding & clearcoat casing & ribbed ceramic collar
      appendTorus(vertices, triangleMats, coilCenter, coilAxis, 1.42f, 0.22f, 88u, 32u, MAT_COOLANT, 1.5f);
      appendTorus(vertices, triangleMats, coilCenter, coilAxis, 1.52f, 0.16f, 72u, 28u,
                  (c % 3u == 0u) ? MAT_CLEARCOAT : ((c % 3u == 1u) ? MAT_CERAMIC : MAT_TITANIUM), 1.4f, 0.025f, 12.0f);
    }

    // Central Vertical Tokamak Solenoid Column + Ribbed Ceramic Insulator Discs + Upper Cooling Manifold
    appendCylinder(vertices, triangleMats, {rc.x, rc.y, 1.8f}, {rc.x, rc.y, 16.5f},
                   1.45f, 1.30f, 64u, 24u, MAT_TITANIUM, 1.5f, true);
    for (uint32_t d = 0; d < 18u; ++d) {
      float zd = 2.8f + float(d) * 0.68f;
      appendCylinder(vertices, triangleMats, {rc.x, rc.y, zd}, {rc.x, rc.y, zd + 0.30f},
                     1.78f, 1.62f, 48u, 4u, (d % 2u == 0u) ? MAT_CERAMIC : MAT_CLEARCOAT, 1.6f, true);
    }

    // Quilted MLI Gold Thermal-Insulation Core Injector Pods (2 large drum shrouds behind/beside reactor)
    for (int side : {-1, 1}) {
      Vec3 drumBase{float(side) * 7.8f, 11.5f, 1.0f};
      Vec3 drumTop{float(side) * 7.8f, 11.5f, 9.8f};
      appendCylinder(vertices, triangleMats, drumBase, drumTop, 1.95f, 1.90f, 48u, 28u, MAT_MLI_FOIL, 2.0f, true, 0.075f);
      // Restraining titanium strap bands around MLI tanks
      for (float zBand : {2.5f, 5.2f, 8.0f}) {
        appendCylinder(vertices, triangleMats,
                       {float(side) * 7.8f, 11.5f, zBand}, {float(side) * 7.8f, 11.5f, zBand + 0.28f},
                       2.05f, 2.05f, 40u, 2u, MAT_TITANIUM, 1.0f, true);
      }
      // Curved cryogenic braided umbilicals connecting MLI tanks to central reactor solenoid & floor deck
      appendCurvedConduit(vertices, triangleMats,
                          {float(side) * 6.0f, 10.5f, 8.5f},
                          {float(side) * 4.2f, 8.2f, 11.8f},
                          {float(side) * 2.2f, 6.2f, 10.6f},
                          {float(side) * 1.3f, 5.2f, 8.8f},
                          0.26f, 40u, 16u, MAT_CERAMIC, true);
      appendCurvedConduit(vertices, triangleMats,
                          {float(side) * 6.2f, 9.5f, 4.2f},
                          {float(side) * 4.8f, 6.8f, 3.0f},
                          {float(side) * 3.8f, 3.2f, 2.2f},
                          {float(side) * 4.3f, -2.5f, 0.2f},
                          0.22f, 44u, 14u, MAT_CERAMIC, true);
    }
  }

  // =========================================================================
  // 4. ARMORED-GLASS ELEVATED CONTROL BOOTH (+X RIGHT MEZZANINE WITH DETAILED INTERIOR!)
  //    X in [10.0, 17.5], Y in [-4.5, 7.5], Z in [4.8, 9.8] (~45,000 triangles)
  // =========================================================================
  {
    const float bx0 = 10.2f, bx1 = 17.6f;
    const float by0 = -4.2f, by1 = 7.4f;
    const float bz0 = 4.8f,  bz1 = 9.4f;

    // Twin cantilevered structural steel support pylons under control booth
    for (float yP : {-3.2f, 6.4f}) {
      appendBox(vertices, triangleMats, {14.2f, yP, 2.4f}, {0.75f, 0.55f, 2.4f}, MAT_FERROCRETE, 0.5f, 2u, 2u, 6u);
      appendOrientedBeam(vertices, triangleMats, {15.5f, yP, 0.8f}, {10.4f, yP, 4.6f}, 0.32f, 0.42f, MAT_TITANIUM, 0.6f, 4u);
    }
    // Control booth floor, recessed roof cap, back/side walls, and Mullion frame
    appendBox(vertices, triangleMats, {0.5f * (bx0 + bx1), 0.5f * (by0 + by1), bz0 - 0.15f}, {3.8f, 5.9f, 0.18f}, MAT_TITANIUM, 0.6f, 6u, 8u, 1u);
    appendBox(vertices, triangleMats, {0.5f * (bx0 + bx1), 0.5f * (by0 + by1), bz1 + 0.25f}, {4.0f, 6.1f, 0.28f}, MAT_CLEARCOAT, 0.6f, 6u, 8u, 1u);
    appendBox(vertices, triangleMats, {bx1 - 0.15f, 0.5f * (by0 + by1), 0.5f * (bz0 + bz1)}, {0.20f, 5.8f, 2.3f}, MAT_FERROCRETE, 0.4f, 1u, 8u, 4u);
    appendBox(vertices, triangleMats, {0.5f * (bx0 + bx1), by1 - 0.15f, 0.5f * (bz0 + bz1)}, {3.7f, 0.20f, 2.3f}, MAT_FERROCRETE, 0.4f, 6u, 1u, 4u);
    // Low knee wall along front-left viewing edge + horizontal armor transom bar + louver vents
    appendBox(vertices, triangleMats, {bx0 + 0.12f, 0.5f * (by0 + by1), bz0 + 0.55f}, {0.16f, 5.8f, 0.55f}, MAT_CLEARCOAT, 0.7f, 1u, 8u, 2u);
    appendBox(vertices, triangleMats, {bx0 + 0.08f, 0.5f * (by0 + by1), 7.55f}, {0.12f, 5.8f, 0.11f}, MAT_TITANIUM, 0.8f, 1u, 8u, 1u);
    appendBox(vertices, triangleMats, {0.5f * (bx0 + bx1), by0 + 0.08f, 7.55f}, {3.6f, 0.12f, 0.11f}, MAT_TITANIUM, 0.8f, 6u, 1u, 1u);
    // Exterior blast-apron louver grilles & diagnostic gauge cluster pipes
    for (uint32_t k = 0; k < 5u; ++k) {
      float yk = by0 + 1.15f + float(k) * 2.3f;
      appendBox(vertices, triangleMats, {bx0 + 0.01f, yk, bz0 + 0.55f}, {0.06f, 0.72f, 0.28f}, MAT_HEX_GRATE, 1.8f, 1u, 2u, 1u);
      appendBox(vertices, triangleMats, {bx0 + 0.02f, yk, bz1 + 0.25f}, {0.10f, 0.55f, 0.12f}, (k & 1u) ? MAT_COOLANT : MAT_CERAMIC, 1.0f, 1u, 2u, 1u);
    }
    // Vertical steel window mullions (denser multi-material blast framing)
    for (uint32_t m = 0; m <= 6u; ++m) {
      float ym = by0 + 0.12f + (float(m) / 6.0f) * (by1 - by0 - 0.24f);
      uint32_t mMat = (m == 2u || m == 4u) ? MAT_CLEARCOAT : MAT_TITANIUM;
      appendBox(vertices, triangleMats, {bx0 + 0.12f, ym, 0.5f * (bz0 + bz1)}, {0.15f, 0.13f, 2.3f}, mMat, 0.8f, 1u, 1u, 4u);
    }
    for (float xm : {bx0 + 0.2f, 12.15f, 13.9f, 15.65f, bx1 - 0.2f}) {
      appendBox(vertices, triangleMats, {xm, by0 + 0.12f, 0.5f * (bz0 + bz1)}, {0.13f, 0.15f, 2.3f}, MAT_TITANIUM, 0.8f, 1u, 1u, 4u);
    }

    // Subdivided Refractive Armored Glass Panes (MAT_GLASS - rays refract into visible consoles inside!)
    for (uint32_t p = 0; p < 6u; ++p) {
      float py0 = by0 + 0.26f + float(p) * ((by1 - by0 - 0.52f) / 6.0f);
      float py1 = by0 + 0.26f + float(p + 1u) * ((by1 - by0 - 0.52f) / 6.0f);
      appendBox(vertices, triangleMats, {bx0 + 0.12f, 0.5f * (py0 + py1), 7.55f},
                {0.05f, 0.44f * (py1 - py0), 1.66f}, MAT_GLASS, 0.6f, 1u, 2u, 4u);
    }
    for (uint32_t px = 0; px < 4u; ++px) {
      float xL = (bx0 + 0.2f) + float(px) * 1.75f;
      appendBox(vertices, triangleMats, {xL + 0.875f, by0 + 0.12f, 7.55f}, {0.76f, 0.05f, 1.66f}, MAT_GLASS, 0.6f, 2u, 1u, 4u);
    }

    // Interior Operator Consoles, Glowing Displays & Server Cabinets visible *through* the armored glass!
    appendBox(vertices, triangleMats, {12.2f, 1.4f, bz0 + 0.52f}, {0.85f, 3.8f, 0.52f}, MAT_TITANIUM, 1.0f, 3u, 8u, 2u);
    for (float yScr : {-1.4f, 0.8f, 3.0f}) {
      appendBox(vertices, triangleMats, {12.0f, yScr, bz0 + 1.45f}, {0.08f, 0.82f, 0.45f}, MAT_COOLANT, 1.0f, 1u, 3u, 2u);
      appendBox(vertices, triangleMats, {16.2f, yScr, bz0 + 1.35f}, {0.55f, 0.75f, 1.35f}, MAT_CLEARCOAT, 1.0f, 2u, 2u, 4u);
    }
  }

  // =========================================================================
  // 5. DUAL ARTICULATED CLEARCOAT ROBOTIC MANIPULATOR ARMS
  //    Left & Right flanking foreground-midground (~75,000 triangles)
  // =========================================================================
  for (int side : {-1, 1}) {
    float rx = float(side) * 7.6f;
    float ry = -7.2f;
    // Heavy anchored turret base
    appendCylinder(vertices, triangleMats, {rx, ry, 0.0f}, {rx, ry, 1.1f}, 2.05f, 1.85f, 32u, 4u, MAT_FERROCRETE, 1.2f, true);
    appendCylinder(vertices, triangleMats, {rx, ry, 1.1f}, {rx, ry, 2.2f}, 1.55f, 1.45f, 36u, 5u, MAT_TITANIUM, 1.4f, true);
    appendTorus(vertices, triangleMats, {rx, ry, 2.25f}, {0.0f, 0.0f, 1.0f}, 1.35f, 0.25f, 32u, 14u, MAT_CLEARCOAT, 1.4f);

    Vec3 shoulder{rx, ry, 2.8f};
    Vec3 elbow{float(side) * 5.8f, -3.5f, 7.8f};
    Vec3 wrist{float(side) * 4.2f, 0.6f, 6.2f};
    Vec3 gripper{float(side) * 3.1f, 2.4f, 6.6f};

    // Shoulder / Elbow / Wrist Joint Cylinders
    appendCylinder(vertices, triangleMats, shoulder - Vec3{0.9f, 0, 0}, shoulder + Vec3{0.9f, 0, 0}, 0.78f, 0.78f, 28u, 3u, MAT_TITANIUM, 1.2f, true);
    appendCylinder(vertices, triangleMats, elbow - Vec3{0.7f, 0, 0}, elbow + Vec3{0.7f, 0, 0}, 0.62f, 0.62f, 28u, 3u, MAT_TITANIUM, 1.2f, true);
    appendCylinder(vertices, triangleMats, wrist - Vec3{0.45f, 0, 0}, wrist + Vec3{0.45f, 0, 0}, 0.42f, 0.42f, 24u, 2u, MAT_CERAMIC, 1.2f, true);

    // Clearcoat sculpted armor boom shells
    appendOrientedBeam(vertices, triangleMats, shoulder, elbow, 0.48f, 0.56f, MAT_CLEARCOAT, 0.9f, 6u);
    appendOrientedBeam(vertices, triangleMats, elbow, wrist, 0.34f, 0.42f, MAT_CLEARCOAT, 0.9f, 6u);
    appendCylinder(vertices, triangleMats, wrist, gripper, 0.24f, 0.32f, 20u, 4u, MAT_TITANIUM, 1.2f, true);

    // Parallel hydraulic actuators & MLI flex sleeve
    appendCylinder(vertices, triangleMats, shoulder + Vec3{float(side) * 0.4f, -0.5f, -0.2f},
                   elbow + Vec3{float(side) * 0.3f, -0.4f, -0.5f}, 0.16f, 0.13f, 14u, 6u, MAT_TITANIUM, 1.0f, true);
    appendCylinder(vertices, triangleMats, shoulder + Vec3{0, 0.25f, 0.3f},
                   shoulder + (elbow - shoulder) * 0.42f + Vec3{0, 0.25f, 0.3f},
                   0.56f, 0.52f, 24u, 10u, MAT_MLI_FOIL, 1.8f, true, 0.045f);

    // Curved braided hydraulic umbilicals trailing from robot base to elbow & wrist
    appendCurvedConduit(vertices, triangleMats,
                        {rx, ry - 0.8f, 1.8f},
                        {rx + float(side) * 1.4f, -5.8f, 5.2f},
                        elbow + Vec3{float(side) * 0.6f, -0.5f, 0.8f},
                        wrist + Vec3{0, -0.2f, 0.3f},
                        0.15f, 36u, 12u, MAT_CERAMIC, true);

    // Precision magnetic clamp jaws + optical laser glass emitter
    appendBox(vertices, triangleMats, gripper + Vec3{0, 0.2f, 0.25f}, {0.26f, 0.45f, 0.12f}, MAT_TITANIUM, 1.0f, 2u, 2u, 1u);
    appendBox(vertices, triangleMats, gripper + Vec3{0, 0.2f, -0.25f}, {0.26f, 0.45f, 0.12f}, MAT_TITANIUM, 1.0f, 2u, 2u, 1u);
    appendCylinder(vertices, triangleMats, wrist, gripper + Vec3{0, 0.1f, 0.0f}, 0.14f, 0.18f, 16u, 2u, MAT_GLASS, 1.0f, true);
  }

  // =========================================================================
  // 6. MULTI-TIER PERFORATED CATWALKS, SAFETY CAGES & OVERHEAD CABLE TRAYS
  //    Foreground + Mid-Ground + Mezzanines (~95,000 triangles)
  // =========================================================================
  {
    // Foreground Observation Catwalk where the camera stands (Y in [-23.5, -16.2], Z = 2.6m)
    // Open railing in front of camera frames the lower coolant trenches without blocking the reactor view!
    appendCatwalkSegment(vertices, triangleMats, -16.0f, 16.0f, -23.5f, -16.8f, 2.4f, false, true, true, true);

    // Foreground Flanking Diagonal Perforated Cable-Tray & Catwalk Gantry Bridges (left/right X edges!)
    // Placed so every 8x4 edge tile interleaves perf-grate, titanium rails, robot arms, cryo tanks & glass booth
    appendCatwalkSegment(vertices, triangleMats, -18.5f, -9.2f, -16.8f, 22.0f, 5.2f, true, true, false, true);
    appendCatwalkSegment(vertices, triangleMats, 9.2f, 18.5f, -16.8f, -4.5f, 4.8f, true, true, true, false);
    appendCatwalkSegment(vertices, triangleMats, 9.2f, 18.5f, 7.8f, 22.0f, 4.8f, true, true, true, false);

    // High Overhead Inspection Crosswalk Span above Reactor Core (Z = 14.4m, Y in [1.0, 3.8])
    // Gives dramatic silhouette framing against overhead skylight shafts & heavy alpha-cutout depth layering!
    appendCatwalkSegment(vertices, triangleMats, -19.0f, 19.0f, 1.0f, 3.6f, 14.4f, true, true, false, false);

    // Vertical Structural Catwalk Columns anchoring walkways to hangar floor
    for (float xc : {-15.8f, -9.4f, 9.4f, 15.8f}) {
      for (float yc : {-16.5f, -5.0f, 8.5f, 21.5f}) {
        appendBox(vertices, triangleMats, {xc, yc, 2.5f}, {0.22f, 0.22f, 2.5f}, MAT_TITANIUM, 0.8f, 1u, 1u, 4u);
      }
    }

    // Dual Overhead Perforated Cable Trays + Overhead Coolant Manifold Pipe Racks (Z = 15.8 .. 17.6m)
    for (int side : {-1, 1}) {
      float cxTray = float(side) * 7.5f;
      appendBox(vertices, triangleMats, {cxTray, 2.0f, 15.8f}, {1.35f, 26.0f, 0.04f}, MAT_HEX_GRATE, 1.8f, 4u, 48u, 1u);
      appendBox(vertices, triangleMats, {cxTray - 1.38f, 2.0f, 15.95f}, {0.06f, 26.0f, 0.22f}, MAT_TITANIUM, 0.6f, 1u, 24u, 1u);
      appendBox(vertices, triangleMats, {cxTray + 1.38f, 2.0f, 15.95f}, {0.06f, 26.0f, 0.22f}, MAT_TITANIUM, 0.6f, 1u, 24u, 1u);

      // 4 High-Pressure Parallel Coolant/Cryo Overhead Pipes per side
      for (uint32_t p = 0; p < 4u; ++p) {
        float pxPipe = cxTray - 0.9f + float(p) * 0.6f;
        uint32_t pMat = (p == 0u) ? MAT_CLEARCOAT : ((p == 2u) ? MAT_CERAMIC : MAT_TITANIUM);
        appendCylinder(vertices, triangleMats,
                       {pxPipe, -24.0f, 16.55f}, {pxPipe, 29.0f, 16.55f},
                       0.20f, 0.20f, 14u, 48u, pMat, 1.5f, true);
      }
      // Hanger yoke brackets every 4.5m
      for (float yH = -20.0f; yH <= 26.0f; yH += 4.5f) {
        appendBox(vertices, triangleMats, {cxTray, yH, 16.25f}, {1.55f, 0.14f, 0.12f}, MAT_TITANIUM, 0.8f, 2u, 1u, 1u);
        appendCylinder(vertices, triangleMats, {cxTray - 1.4f, yH, 16.2f}, {cxTray - 1.4f, yH, 20.2f}, 0.06f, 0.06f, 6u, 1u, MAT_TITANIUM);
        appendCylinder(vertices, triangleMats, {cxTray + 1.4f, yH, 16.2f}, {cxTray + 1.4f, yH, 20.2f}, 0.06f, 0.06f, 6u, 1u, MAT_TITANIUM);
      }
    }

    // Foreground & Mid-Ground Free-Standing Perforated Safety Screens & Diagnostic Terminal Stands
    // Deliberately placed on lower-left/lower-right flanks to boost primary-ray alpha-cutout + multi-material wave mixing
    for (int side : {-1, 1}) {
      float fx = float(side) * 6.4f;
      float fy = -12.8f;
      // Equipment frame + MLI thermal blanket wrap + armored glass gauge + hex grate vent door
      appendBox(vertices, triangleMats, {fx, fy, 1.2f}, {1.05f, 0.85f, 1.2f}, MAT_TITANIUM, 0.9f, 2u, 2u, 3u);
      appendBox(vertices, triangleMats, {fx, fy - 0.88f, 1.4f}, {0.82f, 0.04f, 0.82f}, MAT_HEX_GRATE, 2.0f, 4u, 1u, 4u);
      appendBox(vertices, triangleMats, {fx, fy - 0.89f, 2.25f}, {0.65f, 0.08f, 0.35f}, MAT_GLASS, 1.0f, 2u, 1u, 2u);
      appendCylinder(vertices, triangleMats,
                     {fx - float(side) * 0.95f, fy, 0.2f}, {fx - float(side) * 0.95f, fy, 2.8f},
                     0.42f, 0.40f, 24u, 12u, MAT_MLI_FOIL, 1.8f, true, 0.035f);
    }
  }

  // =========================================================================
  // 7. HANGING BRAIDED CONDUIT BUNDLES, DEBRIS/CARGO PALLETS & COOLANT MANIFOLDS
  //    Adds high-frequency triangle density & octant normal distribution (~180,000 tris)
  // =========================================================================
  {
    // 16 Draped Braided Umbilical Conduits curving symmetrically from Reactor Core out to Mezzanines
    for (uint32_t k = 0; k < 16u; ++k) {
      float sideSign = (k & 1u) ? 1.0f : -1.0f;
      float group = float(k >> 1u);
      Vec3 a{sideSign * (2.6f + 0.30f * group), 3.0f + group * 1.1f, 11.2f + (k % 3u) * 0.6f};
      Vec3 b{sideSign * (6.5f + 0.55f * group), 0.5f + group * 0.9f, 8.8f - (k % 2u) * 1.2f};
      Vec3 c{sideSign * (9.0f + 0.45f * group), -3.5f + group * 1.4f, 6.2f + (k % 3u) * 0.5f};
      Vec3 d{sideSign * 11.2f, -6.0f + group * 2.0f, 5.6f};
      appendCurvedConduit(vertices, triangleMats, a, b, c, d, 0.16f + (k % 3u) * 0.03f,
                          48u, 20u, (k % 3u == 0u) ? MAT_CLEARCOAT : MAT_CERAMIC, true);
    }

    // 10 High-Detail Cryo Dewar Canister Pods & Valve Assemblies along left/right lower aisles
    for (int side : {-1, 1}) {
      for (uint32_t row = 0; row < 5u; ++row) {
        float px = float(side) * (13.2f + float(row & 1u) * 2.2f);
        float py = -12.0f + float(row) * 7.2f;
        appendCylinder(vertices, triangleMats, {px, py, 0.0f}, {px, py, 3.6f}, 1.05f, 1.00f, 56u, 24u,
                       (row % 2u == 0u) ? MAT_MLI_FOIL : MAT_CLEARCOAT, 1.8f, true,
                       (row % 2u == 0u) ? 0.04f : 0.0f);
        appendTorus(vertices, triangleMats, {px, py, 3.65f}, {0.0f, 0.0f, 1.0f},
                    0.82f, 0.16f, 44u, 18u, MAT_TITANIUM, 1.4f);
        appendCylinder(vertices, triangleMats, {px, py, 3.6f}, {px, py, 4.3f},
                       0.38f, 0.28f, 28u, 4u, MAT_CERAMIC, 1.2f, true);
      }
    }

    // High-density Reactor Core Protective Baffle Louver Grid (finely tessellated rings + bolts for BVH stress)
    for (uint32_t ring = 0; ring < 32u; ++ring) {
      float ang = (float(ring) / 32.0f) * 6.2831853f;
      float gx = std::cos(ang) * 5.35f;
      float gz = 7.6f + std::sin(ang) * 5.35f;
      if (gz > 1.8f) {
        appendCylinder(vertices, triangleMats,
                       {gx, 1.2f, gz}, {gx, 9.2f, gz},
                       0.09f, 0.09f, 20u, 32u,
                       (ring & 1u) ? MAT_TITANIUM : MAT_COOLANT, 1.2f, true);
      }
    }
  }
}

inline void generateReactorBayPbrTextures(
    std::vector<GltfMaterial> &outMats,
    std::vector<GltfTextureHeader> &outHeaders,
    std::vector<uint32_t> &outPixels) {
  constexpr uint32_t kSize = 256u;
  constexpr uint32_t kTexelsPerTex = kSize * kSize;
  constexpr uint32_t kNumMats = 8u;
  constexpr uint32_t kNumTex = kNumMats * 3u;

  outMats.assign(kNumMats, GltfMaterial{});

  // Mat 0: Brushed Titanium Girders, Railings & Pipe Racks
  outMats[0].baseColorFactor[0] = 0.64f; outMats[0].baseColorFactor[1] = 0.68f; outMats[0].baseColorFactor[2] = 0.74f; outMats[0].baseColorFactor[3] = 1.0f;
  outMats[0].metallicFactor = 0.90f; outMats[0].roughnessFactor = 0.30f; outMats[0].archetype = 0u;

  // Mat 1: Ribbed Ceramic Reactor Insulators & Braided Cryo Conduits
  outMats[1].baseColorFactor[0] = 0.80f; outMats[1].baseColorFactor[1] = 0.56f; outMats[1].baseColorFactor[2] = 0.42f; outMats[1].baseColorFactor[3] = 1.0f;
  outMats[1].metallicFactor = 0.18f; outMats[1].roughnessFactor = 0.46f; outMats[1].archetype = 1u;

  // Mat 2: Refractive Armored Control-Booth Glass & Coolant Trench Covers
  outMats[2].baseColorFactor[0] = 0.84f; outMats[2].baseColorFactor[1] = 0.96f; outMats[2].baseColorFactor[2] = 1.00f; outMats[2].baseColorFactor[3] = 1.0f;
  outMats[2].metallicFactor = 0.0f; outMats[2].roughnessFactor = 0.05f;
  outMats[2].transmissionFactor = 0.90f; outMats[2].ior = 1.52f; outMats[2].archetype = 2u;

  // Mat 3: Gold/Silver Quilted MLI Thermal Foil Blankets & Baffles
  outMats[3].baseColorFactor[0] = 0.94f; outMats[3].baseColorFactor[1] = 0.74f; outMats[3].baseColorFactor[2] = 0.32f; outMats[3].baseColorFactor[3] = 1.0f;
  outMats[3].metallicFactor = 0.38f; outMats[3].roughnessFactor = 0.52f; outMats[3].archetype = 3u;

  // Mat 4: Weathered Ferrocrete Hangar Deck & Blast Bulkheads
  outMats[4].baseColorFactor[0] = 0.48f; outMats[4].baseColorFactor[1] = 0.50f; outMats[4].baseColorFactor[2] = 0.54f; outMats[4].baseColorFactor[3] = 1.0f;
  outMats[4].metallicFactor = 0.08f; outMats[4].roughnessFactor = 0.84f; outMats[4].archetype = 4u;

  // Mat 5: Coolant-Slicked Trench & Iridescent Tokamak Containment Coils
  outMats[5].baseColorFactor[0] = 0.24f; outMats[5].baseColorFactor[1] = 0.78f; outMats[5].baseColorFactor[2] = 0.94f; outMats[5].baseColorFactor[3] = 1.0f;
  outMats[5].metallicFactor = 0.72f; outMats[5].roughnessFactor = 0.18f;
  outMats[5].emissiveFactor[0] = 0.08f; outMats[5].emissiveFactor[1] = 0.36f; outMats[5].emissiveFactor[2] = 0.52f; outMats[5].emissiveFactor[3] = 1.0f;
  outMats[5].archetype = 5u;

  // Mat 6: Clearcoat Safety-Orange Robotic Arms & Reactor Containment Shrouds
  outMats[6].baseColorFactor[0] = 0.92f; outMats[6].baseColorFactor[1] = 0.34f; outMats[6].baseColorFactor[2] = 0.08f; outMats[6].baseColorFactor[3] = 1.0f;
  outMats[6].metallicFactor = 0.42f; outMats[6].roughnessFactor = 0.32f; outMats[6].archetype = 6u;

  // Mat 7: Perforated Alpha-Cutout Hex-Grate Catwalks, Safety Cages & Cable Trays
  outMats[7].baseColorFactor[0] = 0.60f; outMats[7].baseColorFactor[1] = 0.64f; outMats[7].baseColorFactor[2] = 0.68f; outMats[7].baseColorFactor[3] = 1.0f;
  outMats[7].metallicFactor = 0.78f; outMats[7].roughnessFactor = 0.40f;
  outMats[7].alphaMode = 1u; outMats[7].alphaCutoff = 0.5f; outMats[7].archetype = 7u;

  outHeaders.resize(kNumTex);
  outPixels.assign(static_cast<size_t>(kNumTex) * kTexelsPerTex, 0xFFFFFFFFu);

  auto hash2 = [](uint32_t x, uint32_t y, uint32_t seed) -> float {
    uint32_t h = (x & 255u) * 374761393u + (y & 255u) * 668265263u + seed * 1442695041u;
    h = (h ^ (h >> 13u)) * 1274126177u;
    h ^= (h >> 16u);
    return static_cast<float>(h & 0x00FFFFFFu) / 16777216.0f;
  };

  auto tileNoise = [&](float u, float v, float freq, uint32_t seed) -> float {
    const float fx = std::fmod(u * freq + 256.0f, freq);
    const float fy = std::fmod(v * freq + 256.0f, freq);
    const uint32_t ix = static_cast<uint32_t>(fx);
    const uint32_t iy = static_cast<uint32_t>(fy);
    const uint32_t period = std::max(1u, static_cast<uint32_t>(freq));
    const uint32_t ix1 = (ix + 1u) % period;
    const uint32_t iy1 = (iy + 1u) % period;
    const float tx = fx - static_cast<float>(ix);
    const float ty = fy - static_cast<float>(iy);
    const float sx = tx * tx * (3.0f - 2.0f * tx);
    const float sy = ty * ty * (3.0f - 2.0f * ty);
    const float n00 = hash2(ix % period, iy % period, seed);
    const float n10 = hash2(ix1, iy % period, seed);
    const float n01 = hash2(ix % period, iy1, seed);
    const float n11 = hash2(ix1, iy1, seed);
    const float nx0 = n00 + (n10 - n00) * sx;
    const float nx1 = n01 + (n11 - n01) * sx;
    return nx0 + (nx1 - nx0) * sy;
  };

  auto packRgba8 = [](float r, float g, float b, float a) -> uint32_t {
    auto clampU8 = [](float v) -> uint32_t {
      return static_cast<uint32_t>(std::clamp(v * 255.0f + 0.5f, 0.0f, 255.0f));
    };
    return clampU8(r) | (clampU8(g) << 8u) | (clampU8(b) << 16u) | (clampU8(a) << 24u);
  };

  std::vector<float> heightField(kTexelsPerTex, 0.0f);

  for (uint32_t m = 0; m < kNumMats; ++m) {
    const uint32_t albedoIdx = m * 3u + 0u;
    const uint32_t normalIdx = m * 3u + 1u;
    const uint32_t mrIdx = m * 3u + 2u;

    for (uint32_t t = 0; t < 3u; ++t) {
      const uint32_t tid = m * 3u + t;
      outHeaders[tid].width = kSize;
      outHeaders[tid].height = kSize;
      outHeaders[tid].offset = tid * kTexelsPerTex;
      outHeaders[tid].pad = 0u;
    }

    outMats[m].baseColorTexIdx = static_cast<int32_t>(albedoIdx);
    outMats[m].normalTexIdx = static_cast<int32_t>(normalIdx);
    outMats[m].metallicRoughnessTexIdx = static_cast<int32_t>(mrIdx);
    if (m == MAT_COOLANT) {
      outMats[m].emissiveTexIdx = static_cast<int32_t>(albedoIdx);
    }
    outMats[m].normalScale = (m == MAT_GLASS) ? 0.14f : 0.42f;

    const uint32_t seed = 409u + m * 1013u;
    for (uint32_t y = 0; y < kSize; ++y) {
      const float v = static_cast<float>(y) / static_cast<float>(kSize);
      for (uint32_t x = 0; x < kSize; ++x) {
        const float u = static_cast<float>(x) / static_cast<float>(kSize);
        float n0 = tileNoise(u, v, 8.0f, seed);
        float n1 = tileNoise(u, v, 24.0f, seed + 13u);
        float n2 = tileNoise(u, v, 64.0f, seed + 29u);
        float h = 0.5f * n0 + 0.32f * n1 + 0.18f * n2;
        float r = 0.88f, g = 0.88f, b = 0.88f, a = 1.0f;
        float roughMul = 0.85f + 0.22f * (n1 - 0.5f);
        float metalMul = 1.0f;

        if (m == MAT_TITANIUM) {
          // Brushed anisotropic metal grain + beveled plate joints
          float brush = tileNoise(u * 0.25f, v * 2.0f, 32.0f, seed + 5u);
          float seamX = std::min(std::fmod(u * 2.0f, 1.0f), 1.0f - std::fmod(u * 2.0f, 1.0f));
          float seamY = std::min(std::fmod(v * 2.0f, 1.0f), 1.0f - std::fmod(v * 2.0f, 1.0f));
          float seam = (std::min(seamX, seamY) < 0.025f) ? 0.0f : 1.0f;
          h = (0.55f * brush + 0.25f * n1 + 0.20f * n2) * seam;
          r = 0.80f + 0.22f * h;
          g = 0.82f + 0.20f * h;
          b = 0.86f + 0.18f * h;
          roughMul = (seam < 0.5f) ? 1.25f : (0.78f + 0.25f * brush);
        } else if (m == MAT_CERAMIC) {
          // Diamond-braided cryo sheath & ceramic insulator ribs
          float braid = 0.5f + 0.25f * std::sin((u + v) * 6.2831853f * 8.0f)
                             + 0.25f * std::sin((u - v) * 6.2831853f * 8.0f);
          h = 0.65f * braid + 0.20f * n1 + 0.15f * n2;
          r = 0.82f + 0.22f * braid;
          g = 0.78f + 0.18f * braid;
          b = 0.74f + 0.16f * n2;
          roughMul = 0.76f + 0.28f * (1.0f - braid);
        } else if (m == MAT_GLASS) {
          // Subtle etched armored-glass security reticle
          float grid = (std::fmod(u * 8.0f, 1.0f) < 0.03f || std::fmod(v * 8.0f, 1.0f) < 0.03f) ? 0.88f : 1.0f;
          h = 0.5f * grid + 0.05f * n2;
          r = 0.90f * grid;
          g = 0.96f * grid;
          b = 1.00f * grid;
          roughMul = (grid < 0.95f) ? 1.4f : 0.75f;
        } else if (m == MAT_MLI_FOIL) {
          // Quilted crinkled MLI thermal insulation foil
          float diagA = std::fabs(std::sin((u + v) * 6.2831853f * 4.0f));
          float diagB = std::fabs(std::sin((u - v) * 6.2831853f * 4.0f));
          float quilt = std::min(diagA, diagB);
          h = 0.55f * quilt + 0.30f * n1 + 0.15f * n2;
          r = 0.86f + 0.20f * h;
          g = 0.82f + 0.18f * h;
          b = 0.72f + 0.16f * h;
          roughMul = 0.70f + 0.35f * (1.0f - quilt);
        } else if (m == MAT_FERROCRETE) {
          // Pitted industrial ferrocrete slabs with crisp expansion seams
          float panel = (std::fmod(u * 2.0f, 1.0f) < 0.025f || std::fmod(v * 2.0f, 1.0f) < 0.025f) ? 0.35f : 1.0f;
          float pit = (n2 > 0.78f) ? 0.80f : (0.92f + 0.14f * n1);
          h = (0.45f * n0 + 0.35f * n1 + 0.20f * n2) * panel;
          r = 0.86f * pit * (0.6f + 0.4f * panel);
          g = 0.86f * pit * (0.6f + 0.4f * panel);
          b = 0.87f * pit * (0.6f + 0.4f * panel);
          roughMul = 0.88f + 0.22f * (1.0f - n1);
        } else if (m == MAT_COOLANT) {
          // Flowing plasma/coolant strips alternating with dark containment bands
          float pulse = 0.5f + 0.5f * std::sin(v * 6.2831853f * 6.0f + n0 * 0.8f);
          float band = detail::smoothstep(0.30f, 0.72f, pulse);
          h = 0.55f * band + 0.25f * n1 + 0.20f * n2;
          r = 0.18f + 0.72f * band;
          g = 0.30f + 0.72f * band;
          b = 0.38f + 0.62f * band;
          metalMul = 1.0f - 0.5f * band;
          roughMul = 0.65f + 0.30f * (1.0f - band);
        } else if (m == MAT_CLEARCOAT) {
          // Industrial safety-orange clearcoat panels with dark charcoal trim stripes
          float stripe = (std::fmod(u + v, 0.5f) < 0.07f) ? 0.22f : 1.0f;
          h = (0.65f + 0.25f * n1) * (0.5f + 0.5f * stripe);
          r = (0.94f + 0.08f * n2) * (0.28f + 0.72f * stripe);
          g = (0.86f + 0.10f * n2) * (0.28f + 0.72f * stripe);
          b = (0.82f + 0.10f * n2) * (0.35f + 0.65f * stripe);
          roughMul = 0.75f + 0.22f * n2;
        } else {
          // MAT_HEX_GRATE: Perforated hexagonal/diamond steel grating (~52% open apertures a < 0.5!)
          float gu = std::fmod(u * 10.0f, 1.0f) - 0.5f;
          float gv = std::fmod(v * 10.0f, 1.0f) - 0.5f;
          float dHex = std::max(std::fabs(gu), std::fabs(gu) * 0.5f + std::fabs(gv) * 0.866f);
          float holeMask = (dHex < 0.36f) ? 0.0f : 1.0f;
          h = holeMask * (0.75f + 0.20f * n1);
          r = 0.78f + 0.20f * n1;
          g = 0.80f + 0.20f * n1;
          b = 0.84f + 0.18f * n2;
          a = holeMask;
          roughMul = 0.80f + 0.25f * n2;
        }

        const uint32_t pIdx = y * kSize + x;
        heightField[pIdx] = h;
        outPixels[static_cast<size_t>(outHeaders[albedoIdx].offset) + pIdx] = packRgba8(r, g, b, a);
        outPixels[static_cast<size_t>(outHeaders[mrIdx].offset) + pIdx] = packRgba8(0.88f + 0.12f * h, roughMul, metalMul, 1.0f);
      }
    }

    for (uint32_t y = 0; y < kSize; ++y) {
      const uint32_t ym = (y + kSize - 1u) & (kSize - 1u);
      const uint32_t yp = (y + 1u) & (kSize - 1u);
      for (uint32_t x = 0; x < kSize; ++x) {
        const uint32_t xm = (x + kSize - 1u) & (kSize - 1u);
        const uint32_t xp = (x + 1u) & (kSize - 1u);
        const float dhdx = (heightField[y * kSize + xp] - heightField[y * kSize + xm]) * 2.4f;
        const float dhdy = (heightField[yp * kSize + x] - heightField[ym * kSize + x]) * 2.4f;
        const float invL = 1.0f / std::sqrt(dhdx * dhdx + dhdy * dhdy + 1.0f);
        const float nx = -dhdx * invL * 0.5f + 0.5f;
        const float ny = -dhdy * invL * 0.5f + 0.5f;
        const float nz = invL * 0.5f + 0.5f;
        outPixels[static_cast<size_t>(outHeaders[normalIdx].offset) + y * kSize + x] = packRgba8(nx, ny, nz, 1.0f);
      }
    }
  }
}

} // namespace ReactorBayScene
