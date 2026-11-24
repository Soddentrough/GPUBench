#include "showcase/ShowcaseScene.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <unordered_map>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace gpubench::showcase {

namespace {

std::filesystem::path resolveExecutableDir() {
#ifdef _WIN32
  std::wstring buf(MAX_PATH, L'\0');
  DWORD len = GetModuleFileNameW(nullptr, buf.data(), static_cast<DWORD>(buf.size()));
  if (len > 0 && len < buf.size()) {
    buf.resize(len);
    return std::filesystem::path(buf).parent_path();
  }
#else
  std::array<char, 4096> buf{};
  ssize_t len = readlink("/proc/self/exe", buf.data(), buf.size() - 1);
  if (len > 0) {
    buf[static_cast<size_t>(len)] = '\0';
    return std::filesystem::path(buf.data()).parent_path();
  }
#endif
  return std::filesystem::current_path();
}

std::string findAssetModel(const std::string &modelName) {
  std::filesystem::path exeDir = resolveExecutableDir();
  std::vector<std::filesystem::path> candidates = {
      std::filesystem::path("assets/models") / modelName,
      std::filesystem::path("../assets/models") / modelName,
      std::filesystem::path("../../assets/models") / modelName,
      exeDir / "assets" / "models" / modelName,
      exeDir / ".." / "assets" / "models" / modelName,
      exeDir / ".." / ".." / "assets" / "models" / modelName,
      exeDir / ".." / "share" / "gpubench" / "models" / modelName,
      exeDir / "share" / "gpubench" / "models" / modelName,
      std::filesystem::path("share/gpubench/models") / modelName,
      std::filesystem::path("/usr/share/gpubench/models") / modelName,
      std::filesystem::path("/usr/local/share/gpubench/models") / modelName,
  };
  std::error_code ec;
  for (const auto &p : candidates) {
    if (std::filesystem::exists(p, ec) && std::filesystem::is_regular_file(p, ec)) {
      return p.string();
    }
  }
  return "";
}

Vec3 computeDisplacementNormal(const Vec3 &pos, const Vec3 &nrm, const Vec3 &tanU,
                               const Vec3 &bitanV, const Vec2 &uv) {
  constexpr float kStep = 0.25f;
  Vec3 t = normalize(tanU);
  Vec3 b = normalize(bitanV);
  Vec3 pC = evaluateSculptedDisplacement(pos, nrm, uv);
  Vec3 pU = evaluateSculptedDisplacement(pos + t * kStep, nrm, uv);
  Vec3 pV = evaluateSculptedDisplacement(pos + b * kStep, nrm, uv);
  Vec3 du = pU - pC;
  Vec3 dv = pV - pC;
  Vec3 crossN = cross(du, dv);
  if (dot(crossN, nrm) < 0.0f) {
    crossN = -crossN;
  }
  float len = length(crossN);
  return (len > 1e-6f) ? (crossN / len) : normalize(nrm);
}

// Builds meshlets (<=64 vertices, <=124 tris) from a local triangle list referencing global vertices[].
void buildMeshletsForTriangles(
    const std::vector<uint32_t> &triVertexIndices, // 3*T global vertex indices
    uint32_t materialId,
    uint32_t flags,
    ShowcaseSceneData &scene,
    uint32_t &outFirstMeshlet,
    uint32_t &outMeshletCount) {
  outFirstMeshlet = static_cast<uint32_t>(scene.meshlets.size());
  outMeshletCount = 0;

  const uint32_t numTris = static_cast<uint32_t>(triVertexIndices.size() / 3u);
  if (numTris == 0u) {
    return;
  }

  // Sort triangles along longest spatial axis of centroids so clusters stay spatially tight
  struct TriRef {
    uint32_t idx;
    float key;
  };
  Vec3 cMin{1e9f, 1e9f, 1e9f};
  Vec3 cMax{-1e9f, -1e9f, -1e9f};
  for (uint32_t t = 0; t < numTris; ++t) {
    const auto &v0 = scene.vertices[triVertexIndices[t * 3u + 0u]];
    const auto &v1 = scene.vertices[triVertexIndices[t * 3u + 1u]];
    const auto &v2 = scene.vertices[triVertexIndices[t * 3u + 2u]];
    Vec3 c{(v0.pos[0] + v1.pos[0] + v2.pos[0]) / 3.0f,
           (v0.pos[1] + v1.pos[1] + v2.pos[1]) / 3.0f,
           (v0.pos[2] + v1.pos[2] + v2.pos[2]) / 3.0f};
    cMin.x = std::min(cMin.x, c.x); cMin.y = std::min(cMin.y, c.y); cMin.z = std::min(cMin.z, c.z);
    cMax.x = std::max(cMax.x, c.x); cMax.y = std::max(cMax.y, c.y); cMax.z = std::max(cMax.z, c.z);
  }
  Vec3 span{cMax.x - cMin.x, cMax.y - cMin.y, cMax.z - cMin.z};
  int sortAxis = (span.x >= span.y && span.x >= span.z) ? 0 : ((span.y >= span.z) ? 1 : 2);

  std::vector<TriRef> order(numTris);
  for (uint32_t t = 0; t < numTris; ++t) {
    const auto &v0 = scene.vertices[triVertexIndices[t * 3u + 0u]];
    const auto &v1 = scene.vertices[triVertexIndices[t * 3u + 1u]];
    const auto &v2 = scene.vertices[triVertexIndices[t * 3u + 2u]];
    float c = (v0.pos[sortAxis] + v1.pos[sortAxis] + v2.pos[sortAxis]) / 3.0f;
    order[t] = {t, c};
  }
  std::sort(order.begin(), order.end(), [](const TriRef &a, const TriRef &b) {
    return a.key < b.key;
  });

  // Pack meshlets greedily; map global index via linear scan of <=64 used vertices
  uint32_t localVerts[kMaxMeshletVertices];
  uint32_t packedTris[kMaxMeshletTriangles];
  std::vector<uint32_t> linearTriIndices;
  linearTriIndices.reserve(kMaxMeshletTriangles * 3u);

  auto flushMeshlet = [&](uint32_t vCount, uint32_t tCount) {
    if (vCount == 0u || tCount == 0u) return;
    ShowcaseMeshlet m{};
    m.vertexOffset = static_cast<uint32_t>(scene.meshletVertices.size());
    m.triangleOffset = static_cast<uint32_t>(scene.meshletTriangles.size());
    m.vertexCount = vCount;
    m.triangleCount = tCount;
    m.materialId = materialId;
    m.flags = flags;
    m.globalIndexBase = static_cast<uint32_t>(scene.indices.size());

    Vec3 bMin{1e9f, 1e9f, 1e9f};
    Vec3 bMax{-1e9f, -1e9f, -1e9f};
    for (uint32_t i = 0; i < vCount; ++i) {
      uint32_t gv = localVerts[i];
      scene.meshletVertices.push_back(gv);
      const auto &p = scene.vertices[gv].pos;
      bMin.x = std::min(bMin.x, p[0]); bMin.y = std::min(bMin.y, p[1]); bMin.z = std::min(bMin.z, p[2]);
      bMax.x = std::max(bMax.x, p[0]); bMax.y = std::max(bMax.y, p[1]); bMax.z = std::max(bMax.z, p[2]);
    }
    Vec3 center{(bMin.x + bMax.x) * 0.5f, (bMin.y + bMax.y) * 0.5f, (bMin.z + bMax.z) * 0.5f};
    float radius = 0.01f;
    for (uint32_t i = 0; i < vCount; ++i) {
      const auto &p = scene.vertices[localVerts[i]].pos;
      Vec3 d{p[0] - center.x, p[1] - center.y, p[2] - center.z};
      radius = std::max(radius, length(d));
    }
    m.boundSphere[0] = center.x;
    m.boundSphere[1] = center.y;
    m.boundSphere[2] = center.z;
    m.boundSphere[3] = radius;

    Vec3 normalSum{0.0f, 0.0f, 0.0f};
    std::vector<Vec3> triNormals;
    triNormals.reserve(tCount);
    for (uint32_t t = 0; t < tCount; ++t) {
      scene.meshletTriangles.push_back(packedTris[t]);
      uint32_t gi0 = localVerts[(packedTris[t] >> 0u) & 0xFFu];
      uint32_t gi1 = localVerts[(packedTris[t] >> 8u) & 0xFFu];
      uint32_t gi2 = localVerts[(packedTris[t] >> 16u) & 0xFFu];
      scene.indices.push_back(gi0);
      scene.indices.push_back(gi1);
      scene.indices.push_back(gi2);

      const auto &vp0 = scene.vertices[gi0].pos;
      const auto &vp1 = scene.vertices[gi1].pos;
      const auto &vp2 = scene.vertices[gi2].pos;
      Vec3 e1{vp1[0] - vp0[0], vp1[1] - vp0[1], vp1[2] - vp0[2]};
      Vec3 e2{vp2[0] - vp0[0], vp2[1] - vp0[1], vp2[2] - vp0[2]};
      Vec3 fn = cross(e1, e2);
      float fnLen = length(fn);
      if (fnLen > 1e-7f) {
        fn = fn / fnLen;
        normalSum += fn;
        triNormals.push_back(fn);
      }
    }

    if ((flags & SHOWCASE_FLAG_DOUBLE_SIDED) || triNormals.empty() || length(normalSum) < 1e-4f) {
      m.coneAxis[0] = 0.0f;
      m.coneAxis[1] = 1.0f;
      m.coneAxis[2] = 0.0f;
      m.coneCutoff = 1.0f; // Never cone cull
    } else {
      Vec3 axis = normalize(normalSum);
      float minDot = 1.0f;
      for (const Vec3 &tn : triNormals) {
        minDot = std::min(minDot, dot(axis, tn));
      }
      m.coneAxis[0] = axis.x;
      m.coneAxis[1] = axis.y;
      m.coneAxis[2] = axis.z;
      if (minDot <= 0.12f) {
        m.coneCutoff = 1.0f;
      } else {
        m.coneCutoff = std::sqrt(std::max(0.0f, 1.0f - minDot * minDot));
      }
    }

    scene.meshlets.push_back(m);
    outMeshletCount++;
  };

  uint32_t curV = 0u;
  uint32_t curT = 0u;
  auto findOrAddLocal = [&](uint32_t gv, uint32_t &tempV) -> uint32_t {
    for (uint32_t i = 0; i < curV; ++i) {
      if (localVerts[i] == gv) return i;
    }
    for (uint32_t i = curV; i < tempV; ++i) {
      if (localVerts[i] == gv) return i;
    }
    if (tempV >= kMaxMeshletVertices) {
      return UINT32_MAX;
    }
    localVerts[tempV] = gv;
    return tempV++;
  };

  for (uint32_t idx = 0; idx < numTris; ++idx) {
    uint32_t t = order[idx].idx;
    uint32_t g0 = triVertexIndices[t * 3u + 0u];
    uint32_t g1 = triVertexIndices[t * 3u + 1u];
    uint32_t g2 = triVertexIndices[t * 3u + 2u];

    uint32_t tempV = curV;
    uint32_t l0 = findOrAddLocal(g0, tempV);
    uint32_t l1 = (l0 != UINT32_MAX) ? findOrAddLocal(g1, tempV) : UINT32_MAX;
    uint32_t l2 = (l1 != UINT32_MAX) ? findOrAddLocal(g2, tempV) : UINT32_MAX;

    if (curT >= kMaxMeshletTriangles || l2 == UINT32_MAX) {
      flushMeshlet(curV, curT);
      curV = 0u;
      curT = 0u;
      tempV = 0u;
      l0 = findOrAddLocal(g0, tempV);
      l1 = findOrAddLocal(g1, tempV);
      l2 = findOrAddLocal(g2, tempV);
    }
    curV = tempV;
    packedTris[curT++] = (l0 & 0xFFu) | ((l1 & 0xFFu) << 8u) | ((l2 & 0xFFu) << 16u);
  }
  flushMeshlet(curV, curT);
}

// Appends an instance's task packets split into chunks of <= 32 meshlets.
void appendTaskPacketsForInstance(uint32_t instanceIdx, const ShowcaseInstance &inst,
                                  uint32_t phaseMask, ShowcaseSceneData &scene) {
  uint32_t maxMeshlets = std::max({inst.lodMeshletCount[0], inst.lodMeshletCount[1], inst.lodMeshletCount[2]});
  if (maxMeshlets == 0u) return;
  auto &dst = (phaseMask == 1u) ? scene.phase0Packets : scene.phase1Packets;
  for (uint32_t start = 0; start < maxMeshlets; start += 32u) {
    ShowcaseTaskPacket pkt{};
    pkt.instanceIdx = instanceIdx;
    pkt.localStart = start;
    pkt.count = std::min(32u, maxMeshlets - start);
    pkt.phaseMask = phaseMask;
    dst.push_back(pkt);
  }
}

struct ProtoGeometry {
  uint32_t baseFirstIndex{0};
  uint32_t baseIndexCount{0};
  int32_t baseVertexOffset{0};
  uint32_t lodFirstMeshlet[3]{0, 0, 0};
  uint32_t lodMeshletCount[3]{0, 0, 0};
  Vec4 localSphere{0.0f, 0.0f, 0.0f, 1.0f};
  uint32_t materialId{0};
  uint32_t flags{0};
  uint32_t lod0TriCount{0};
};

// Generates a sculpted amphora/urn: coarse patch mesh (for Naive/Legacy HW tess) + 3 pre-displaced LODs
ProtoGeometry createSculptedUrnPrototype(ShowcaseSceneData &scene, uint32_t materialId) {
  ProtoGeometry proto{};
  proto.materialId = materialId;
  proto.flags = SHOWCASE_FLAG_TESSELLATED;

  auto latheRadius = [](float v) -> float {
    float r = 18.0f;
    if (v < 0.14f) {
      r = 16.0f + 5.0f * (1.0f - v / 0.14f);
    } else if (v < 0.72f) {
      float t = (v - 0.14f) / 0.58f;
      r = 14.0f + 18.0f * std::sin(t * kPi);
    } else if (v < 0.90f) {
      float t = (v - 0.72f) / 0.18f;
      r = 14.0f - 2.0f * std::sin(t * kPi);
    } else {
      float t = (v - 0.90f) / 0.10f;
      r = 14.0f + 5.0f * t;
    }
    return r;
  };
  constexpr float kHeight = 96.0f;

  auto buildGrid = [&](uint32_t segU, uint32_t segV, bool displaced,
                       std::vector<ShowcaseVertex> &outVerts,
                       std::vector<uint32_t> &outLocalIndices) {
    const uint32_t stride = segU + 1u;
    outVerts.reserve(static_cast<size_t>(stride) * (segV + 1u));
    for (uint32_t iv = 0; iv <= segV; ++iv) {
      float v = static_cast<float>(iv) / static_cast<float>(segV);
      float vPrev = std::max(0.0f, v - 0.01f);
      float vNext = std::min(1.0f, v + 0.01f);
      float r = latheRadius(v);
      float drdv = (latheRadius(vNext) - latheRadius(vPrev)) / std::max(0.005f, vNext - vPrev);
      float y = v * kHeight;

      for (uint32_t iu = 0; iu <= segU; ++iu) {
        float u = static_cast<float>(iu) / static_cast<float>(segU);
        float theta = u * 2.0f * kPi;
        float ct = std::cos(theta);
        float st = std::sin(theta);

        Vec3 basePos{r * ct, y, r * st};
        Vec3 tanU{-st, 0.0f, ct};
        Vec3 tanV{drdv * ct, kHeight, drdv * st};
        // Cross(tanV, tanU) points outward radially for CCW outward triangles
        Vec3 baseNrm = normalize(cross(tanV, tanU));

        Vec3 finalPos = displaced ? evaluateSculptedDisplacement(basePos, baseNrm, {u, v}) : basePos;
        Vec3 finalNrm = displaced ? computeDisplacementNormal(basePos, baseNrm, tanU, tanV, {u, v}) : baseNrm;

        ShowcaseVertex sv{};
        sv.pos[0] = finalPos.x; sv.pos[1] = finalPos.y; sv.pos[2] = finalPos.z;
        sv.matId = materialId;
        sv.normal[0] = finalNrm.x; sv.normal[1] = finalNrm.y; sv.normal[2] = finalNrm.z;
        sv.flags = SHOWCASE_FLAG_TESSELLATED;
        sv.tangent[0] = tanU.x; sv.tangent[1] = tanU.y; sv.tangent[2] = tanU.z; sv.tangent[3] = 1.0f;
        sv.uv[0] = u; sv.uv[1] = v;
        outVerts.push_back(sv);
      }
    }

    for (uint32_t iv = 0; iv < segV; ++iv) {
      for (uint32_t iu = 0; iu < segU; ++iu) {
        uint32_t i00 = iv * stride + iu;
        uint32_t i10 = iv * stride + (iu + 1u);
        uint32_t i01 = (iv + 1u) * stride + iu;
        uint32_t i11 = (iv + 1u) * stride + (iu + 1u);
        // CCW outward winding
        outLocalIndices.push_back(i00);
        outLocalIndices.push_back(i01);
        outLocalIndices.push_back(i10);
        outLocalIndices.push_back(i10);
        outLocalIndices.push_back(i01);
        outLocalIndices.push_back(i11);
      }
    }
  };

  // Coarse base mesh for Naive 64x / Legacy Tessellation (10x6 = 120 patches)
  {
    std::vector<ShowcaseVertex> cv;
    std::vector<uint32_t> ci;
    buildGrid(10u, 6u, false, cv, ci);
    proto.baseVertexOffset = static_cast<int32_t>(scene.vertices.size());
    proto.baseFirstIndex = static_cast<uint32_t>(scene.indices.size());
    proto.baseIndexCount = static_cast<uint32_t>(ci.size());
    scene.vertices.insert(scene.vertices.end(), cv.begin(), cv.end());
    scene.indices.insert(scene.indices.end(), ci.begin(), ci.end());
  }

  // Modern 3 pre-displaced LODs matching evaluateSculptedDisplacement
  const uint32_t uSegs[3] = {44u, 18u, 8u};
  const uint32_t vSegs[3] = {26u, 10u, 5u};
  for (uint32_t lod = 0; lod < 3u; ++lod) {
    std::vector<ShowcaseVertex> lv;
    std::vector<uint32_t> li;
    buildGrid(uSegs[lod], vSegs[lod], true, lv, li);
    uint32_t gOff = static_cast<uint32_t>(scene.vertices.size());
    scene.vertices.insert(scene.vertices.end(), lv.begin(), lv.end());
    std::vector<uint32_t> globalTriIdx(li.size());
    for (size_t i = 0; i < li.size(); ++i) {
      globalTriIdx[i] = gOff + li[i];
    }
    buildMeshletsForTriangles(globalTriIdx, materialId, proto.flags, scene,
                              proto.lodFirstMeshlet[lod], proto.lodMeshletCount[lod]);
    if (lod == 0u) {
      proto.lod0TriCount = static_cast<uint32_t>(li.size() / 3u);
    }
  }

  proto.localSphere = {0.0f, kHeight * 0.5f, 0.0f, 62.0f};
  return proto;
}

// Generates a carved cobblestone / relief plinth slab (tessellatable)
ProtoGeometry createReliefPlinthPrototype(ShowcaseSceneData &scene, uint32_t materialId) {
  ProtoGeometry proto{};
  proto.materialId = materialId;
  proto.flags = SHOWCASE_FLAG_TESSELLATED;
  constexpr float kHalfW = 46.0f;
  constexpr float kTopY = 12.0f;

  auto buildSlab = [&](uint32_t seg, bool displaced,
                       std::vector<ShowcaseVertex> &outVerts,
                       std::vector<uint32_t> &outLocalIndices) {
    const uint32_t stride = seg + 1u;
    for (uint32_t iz = 0; iz <= seg; ++iz) {
      float v = static_cast<float>(iz) / static_cast<float>(seg);
      float z = (v * 2.0f - 1.0f) * kHalfW;
      for (uint32_t ix = 0; ix <= seg; ++ix) {
        float u = static_cast<float>(ix) / static_cast<float>(seg);
        float x = (u * 2.0f - 1.0f) * kHalfW;
        // Subtle domed relief cushion
        float crown = 4.0f * (1.0f - 0.5f * ((2.0f * u - 1.0f) * (2.0f * u - 1.0f) +
                                             (2.0f * v - 1.0f) * (2.0f * v - 1.0f)));
        Vec3 basePos{x, kTopY + std::max(0.0f, crown), z};
        Vec3 baseNrm{0.0f, 1.0f, 0.0f};
        Vec3 tanU{1.0f, 0.0f, 0.0f};
        Vec3 bitanV{0.0f, 0.0f, -1.0f};

        Vec3 finalPos = displaced ? evaluateSculptedDisplacement(basePos, baseNrm, {u, v}) : basePos;
        Vec3 finalNrm = displaced ? computeDisplacementNormal(basePos, baseNrm, tanU, bitanV, {u, v}) : baseNrm;

        ShowcaseVertex sv{};
        sv.pos[0] = finalPos.x; sv.pos[1] = finalPos.y; sv.pos[2] = finalPos.z;
        sv.matId = materialId;
        sv.normal[0] = finalNrm.x; sv.normal[1] = finalNrm.y; sv.normal[2] = finalNrm.z;
        sv.flags = SHOWCASE_FLAG_TESSELLATED;
        sv.tangent[0] = 1.0f; sv.tangent[1] = 0.0f; sv.tangent[2] = 0.0f; sv.tangent[3] = 1.0f;
        sv.uv[0] = u; sv.uv[1] = v;
        outVerts.push_back(sv);
      }
    }
    for (uint32_t iz = 0; iz < seg; ++iz) {
      for (uint32_t ix = 0; ix < seg; ++ix) {
        uint32_t i00 = iz * stride + ix;
        uint32_t i10 = iz * stride + (ix + 1u);
        uint32_t i01 = (iz + 1u) * stride + ix;
        uint32_t i11 = (iz + 1u) * stride + (ix + 1u);
        // Upward (+Y) CCW winding in right-handed XZ plane (+Z x +X = +Y)
        outLocalIndices.push_back(i00);
        outLocalIndices.push_back(i01);
        outLocalIndices.push_back(i10);
        outLocalIndices.push_back(i10);
        outLocalIndices.push_back(i01);
        outLocalIndices.push_back(i11);
      }
    }
  };

  {
    std::vector<ShowcaseVertex> cv;
    std::vector<uint32_t> ci;
    buildSlab(6u, false, cv, ci);
    proto.baseVertexOffset = static_cast<int32_t>(scene.vertices.size());
    proto.baseFirstIndex = static_cast<uint32_t>(scene.indices.size());
    proto.baseIndexCount = static_cast<uint32_t>(ci.size());
    scene.vertices.insert(scene.vertices.end(), cv.begin(), cv.end());
    scene.indices.insert(scene.indices.end(), ci.begin(), ci.end());
  }

  const uint32_t segs[3] = {24u, 10u, 4u};
  for (uint32_t lod = 0; lod < 3u; ++lod) {
    std::vector<ShowcaseVertex> lv;
    std::vector<uint32_t> li;
    buildSlab(segs[lod], true, lv, li);
    uint32_t gOff = static_cast<uint32_t>(scene.vertices.size());
    scene.vertices.insert(scene.vertices.end(), lv.begin(), lv.end());
    std::vector<uint32_t> globalTriIdx(li.size());
    for (size_t i = 0; i < li.size(); ++i) globalTriIdx[i] = gOff + li[i];
    buildMeshletsForTriangles(globalTriIdx, materialId, proto.flags, scene,
                              proto.lodFirstMeshlet[lod], proto.lodMeshletCount[lod]);
    if (lod == 0u) proto.lod0TriCount = static_cast<uint32_t>(li.size() / 3u);
  }
  proto.localSphere = {0.0f, kTopY, 0.0f, 68.0f};
  return proto;
}

// High-detail sculpted fluted gallery column + finial (3 LOD meshlets; non-tessellated base)
// Used densely behind opaque Sponza walls so Legacy CPU frustum culling suffers realistic AAA vertex/G-Buffer cost
// while Modern Hi-Z culls occluded behind-wall clusters cleanly before vertices run.
ProtoGeometry createCloisterRelicPrototype(ShowcaseSceneData &scene, uint32_t materialId) {
  ProtoGeometry proto{};
  proto.materialId = materialId;
  proto.flags = 0u;

  auto buildRelic = [&](uint32_t segU, uint32_t segV,
                        std::vector<ShowcaseVertex> &outVerts,
                        std::vector<uint32_t> &outLocalIndices) {
    const uint32_t stride = segU + 1u;
    for (uint32_t iv = 0; iv <= segV; ++iv) {
      float v = static_cast<float>(iv) / static_cast<float>(segV);
      float y = v * 135.0f;
      float rBase = (v < 0.12f || v > 0.88f) ? 24.0f : (15.5f + 2.5f * std::sin(v * kPi * 3.0f));
      for (uint32_t iu = 0; iu <= segU; ++iu) {
        float u = static_cast<float>(iu) / static_cast<float>(segU);
        float theta = u * 2.0f * kPi;
        float flute = 1.6f * std::cos(theta * 12.0f) * std::sin(v * kPi);
        float r = rBase + flute;
        float ct = std::cos(theta), st = std::sin(theta);
        Vec3 nrm = normalize({ct, 0.15f * std::cos(v * kPi * 4.0f), st});
        ShowcaseVertex sv{};
        sv.pos[0] = r * ct; sv.pos[1] = y; sv.pos[2] = r * st;
        sv.matId = materialId;
        sv.normal[0] = nrm.x; sv.normal[1] = nrm.y; sv.normal[2] = nrm.z;
        sv.flags = 0u;
        sv.tangent[0] = -st; sv.tangent[1] = 0.0f; sv.tangent[2] = ct; sv.tangent[3] = 1.0f;
        sv.uv[0] = u * 2.0f; sv.uv[1] = v * 3.0f;
        outVerts.push_back(sv);
      }
    }
    for (uint32_t iv = 0; iv < segV; ++iv) {
      for (uint32_t iu = 0; iu < segU; ++iu) {
        uint32_t i00 = iv * stride + iu;
        uint32_t i10 = iv * stride + (iu + 1u);
        uint32_t i01 = (iv + 1u) * stride + iu;
        uint32_t i11 = (iv + 1u) * stride + (iu + 1u);
        outLocalIndices.insert(outLocalIndices.end(), {i00, i01, i10, i10, i01, i11});
      }
    }
  };

  // Base coarse & LOD0 (~2,160 tris -> LOD1 ~512 tris -> LOD2 ~128 tris)
  const uint32_t uSegs[3] = {45u, 22u, 10u};
  const uint32_t vSegs[3] = {24u, 12u, 6u};
  for (uint32_t lod = 0; lod < 3u; ++lod) {
    std::vector<ShowcaseVertex> lv;
    std::vector<uint32_t> li;
    buildRelic(uSegs[lod], vSegs[lod], lv, li);
    uint32_t gOff = static_cast<uint32_t>(scene.vertices.size());
    if (lod == 0u) {
      proto.baseVertexOffset = static_cast<int32_t>(gOff);
      proto.baseFirstIndex = static_cast<uint32_t>(scene.indices.size());
      proto.baseIndexCount = static_cast<uint32_t>(li.size());
      scene.indices.insert(scene.indices.end(), li.begin(), li.end());
      proto.lod0TriCount = static_cast<uint32_t>(li.size() / 3u);
    }
    scene.vertices.insert(scene.vertices.end(), lv.begin(), lv.end());
    std::vector<uint32_t> globalTriIdx(li.size());
    for (size_t i = 0; i < li.size(); ++i) globalTriIdx[i] = gOff + li[i];
    buildMeshletsForTriangles(globalTriIdx, materialId, proto.flags, scene,
                              proto.lodFirstMeshlet[lod], proto.lodMeshletCount[lod]);
  }
  proto.localSphere = {0.0f, 67.5f, 0.0f, 74.0f};
  return proto;
}

// Generates a dense multi-layered foliage shrub cluster (triggers alpha cutout + heavy forward overdraw)
ProtoGeometry createFoliageShrubPrototype(ShowcaseSceneData &scene, uint32_t materialId) {
  ProtoGeometry proto{};
  proto.materialId = materialId;
  proto.flags = SHOWCASE_FLAG_ALPHA_CUTOUT | SHOWCASE_FLAG_DOUBLE_SIDED;

  auto buildShrub = [&](uint32_t fronds, uint32_t subPerFrond,
                        std::vector<ShowcaseVertex> &outVerts,
                        std::vector<uint32_t> &outLocalIndices) {
    for (uint32_t f = 0; f < fronds; ++f) {
      float angle = (static_cast<float>(f) / static_cast<float>(fronds)) * 2.0f * kPi;
      float tilt = 0.35f + 0.45f * std::sin(static_cast<float>(f) * 2.3f);
      float ca = std::cos(angle), sa = std::sin(angle);
      Vec3 dir{ca, tilt, sa};
      dir = normalize(dir);
      Vec3 side = normalize(cross(dir, Vec3{0.0f, 1.0f, 0.0f}));
      Vec3 nrm = normalize(cross(side, dir));

      float len = 46.0f + 14.0f * std::cos(static_cast<float>(f) * 1.7f);
      float halfW = 18.0f;

      uint32_t baseV = static_cast<uint32_t>(outVerts.size());
      for (uint32_t s = 0; s <= subPerFrond; ++s) {
        float t = static_cast<float>(s) / static_cast<float>(subPerFrond);
        float wScale = std::sin(t * kPi) * 0.9f + 0.15f;
        Vec3 spine = dir * (t * len) + Vec3{0.0f, 12.0f + t * t * -10.0f, 0.0f};
        Vec3 leftP = spine - side * (halfW * wScale);
        Vec3 rightP = spine + side * (halfW * wScale);

        for (int lr = 0; lr < 2; ++lr) {
          Vec3 p = (lr == 0) ? leftP : rightP;
          ShowcaseVertex sv{};
          sv.pos[0] = p.x; sv.pos[1] = p.y; sv.pos[2] = p.z;
          sv.matId = materialId;
          sv.normal[0] = nrm.x; sv.normal[1] = nrm.y; sv.normal[2] = nrm.z;
          sv.flags = proto.flags;
          sv.tangent[0] = side.x; sv.tangent[1] = side.y; sv.tangent[2] = side.z; sv.tangent[3] = 1.0f;
          sv.uv[0] = (lr == 0) ? 0.0f : 1.0f;
          sv.uv[1] = t;
          outVerts.push_back(sv);
        }
      }
      for (uint32_t s = 0; s < subPerFrond; ++s) {
        uint32_t i0 = baseV + s * 2u;
        uint32_t i1 = i0 + 1u;
        uint32_t i2 = i0 + 2u;
        uint32_t i3 = i0 + 3u;
        // Front face + Back face (double sided geometry -> strict CCW hardware backface cull stays valid)
        outLocalIndices.insert(outLocalIndices.end(), {i0, i1, i2, i1, i3, i2});
        outLocalIndices.insert(outLocalIndices.end(), {i0, i2, i1, i1, i2, i3});
      }
    }
  };

  {
    std::vector<ShowcaseVertex> lv;
    std::vector<uint32_t> li;
    buildShrub(8u, 3u, lv, li); // High detail LOD0 & Naive base
    proto.baseVertexOffset = static_cast<int32_t>(scene.vertices.size());
    proto.baseFirstIndex = static_cast<uint32_t>(scene.indices.size());
    proto.baseIndexCount = static_cast<uint32_t>(li.size());
    uint32_t gOff = static_cast<uint32_t>(scene.vertices.size());
    scene.vertices.insert(scene.vertices.end(), lv.begin(), lv.end());
    std::vector<uint32_t> globalTriIdx(li.size());
    for (size_t i = 0; i < li.size(); ++i) globalTriIdx[i] = gOff + li[i];
    buildMeshletsForTriangles(globalTriIdx, materialId, proto.flags, scene,
                              proto.lodFirstMeshlet[0], proto.lodMeshletCount[0]);
    proto.lod0TriCount = static_cast<uint32_t>(li.size() / 3u);
  }
  const uint32_t frondLods[2] = {5u, 3u};
  const uint32_t subLods[2] = {2u, 1u};
  for (uint32_t lod = 1u; lod <= 2u; ++lod) {
    std::vector<ShowcaseVertex> lv;
    std::vector<uint32_t> li;
    buildShrub(frondLods[lod - 1u], subLods[lod - 1u], lv, li);
    uint32_t gOff = static_cast<uint32_t>(scene.vertices.size());
    scene.vertices.insert(scene.vertices.end(), lv.begin(), lv.end());
    std::vector<uint32_t> globalTriIdx(li.size());
    for (size_t i = 0; i < li.size(); ++i) globalTriIdx[i] = gOff + li[i];
    buildMeshletsForTriangles(globalTriIdx, materialId, proto.flags, scene,
                              proto.lodFirstMeshlet[lod], proto.lodMeshletCount[lod]);
  }
  proto.localSphere = {0.0f, 32.0f, 0.0f, 65.0f};
  return proto;
}

GltfTexture makeProceduralLeafTexture() {
  GltfTexture tex{};
  tex.width = 128;
  tex.height = 128;
  tex.channels = 4;
  tex.pixels.resize(128 * 128 * 4);
  for (int y = 0; y < 128; ++y) {
    float v = static_cast<float>(y) / 127.0f;
    for (int x = 0; x < 128; ++x) {
      float u = static_cast<float>(x) / 127.0f;
      float cx = std::abs(u - 0.5f) * 2.0f;
      // Botanical pointed leaf silhouette with serrated lobes so alpha-cutout overdraw is realistic
      float maxW = std::sin(v * kPi) * (0.85f + 0.12f * std::sin(v * 26.0f));
      float distFromVein = std::abs(u - 0.5f - 0.04f * std::sin(v * 8.0f));
      bool inside = (cx <= maxW) && (v > 0.02f) && (v < 0.98f);
      float vein = std::exp(-distFromVein * 45.0f) * 0.25f;
      float latVein = std::exp(-std::abs(std::sin((v * 9.0f - cx * 1.4f) * kPi)) * 8.0f) * 0.14f;
      uint8_t r = static_cast<uint8_t>(std::clamp(48.0f + (vein + latVein) * 140.0f + v * 32.0f, 0.0f, 255.0f));
      uint8_t g = static_cast<uint8_t>(std::clamp(112.0f + (vein + latVein) * 130.0f + v * 48.0f, 0.0f, 255.0f));
      uint8_t b = static_cast<uint8_t>(std::clamp(34.0f + (vein + latVein) * 60.0f, 0.0f, 255.0f));
      size_t idx = (static_cast<size_t>(y) * 128u + static_cast<size_t>(x)) * 4u;
      tex.pixels[idx + 0] = r;
      tex.pixels[idx + 1] = g;
      tex.pixels[idx + 2] = b;
      tex.pixels[idx + 3] = inside ? 255u : 0u;
    }
  }
  return tex;
}

GltfTexture makeProceduralMarbleTexture() {
  GltfTexture tex{};
  tex.width = 128;
  tex.height = 128;
  tex.channels = 4;
  tex.pixels.resize(128 * 128 * 4);
  for (int y = 0; y < 128; ++y) {
    float v = static_cast<float>(y) / 127.0f;
    for (int x = 0; x < 128; ++x) {
      float u = static_cast<float>(x) / 127.0f;
      float vein = 0.5f + 0.5f * std::sin((u * 3.0f + v * 5.0f + 0.3f * std::sin(u * 12.0f)) * kPi);
      float speck = 0.5f + 0.5f * std::cos(u * 37.0f) * std::sin(v * 31.0f);
      float shade = 0.55f + 0.30f * vein + 0.15f * speck;
      uint8_t r = static_cast<uint8_t>(std::clamp(shade * 215.0f, 0.0f, 255.0f));
      uint8_t g = static_cast<uint8_t>(std::clamp(shade * 188.0f, 0.0f, 255.0f));
      uint8_t b = static_cast<uint8_t>(std::clamp(shade * 155.0f, 0.0f, 255.0f));
      size_t idx = (static_cast<size_t>(y) * 128u + static_cast<size_t>(x)) * 4u;
      tex.pixels[idx + 0] = r;
      tex.pixels[idx + 1] = g;
      tex.pixels[idx + 2] = b;
      tex.pixels[idx + 3] = 255u;
    }
  }
  return tex;
}

void buildProceduralFallbackAtrium(ShowcaseSceneData &scene) {
  scene.textures.clear();
  GltfTexture whiteTex{1, 1, 4, {230, 224, 212, 255}};
  GltfTexture flatNorm{1, 1, 4, {128, 128, 255, 255}};
  GltfTexture mrTex{1, 1, 4, {0, 192, 0, 255}};
  scene.textures.push_back(whiteTex);
  scene.textures.push_back(flatNorm);
  scene.textures.push_back(mrTex);

  ShowcaseMaterial stoneMat{};
  stoneMat.baseColorFactor[0] = 0.85f;
  stoneMat.baseColorFactor[1] = 0.80f;
  stoneMat.baseColorFactor[2] = 0.72f;
  stoneMat.roughnessFactor = 0.65f;
  stoneMat.archetype = 1u;
  scene.materials.push_back(stoneMat);

  auto addBox = [&](Vec3 bMin, Vec3 bMax, uint32_t matId, bool occluder) {
    const uint32_t flags = occluder ? SHOWCASE_FLAG_OCCLUDER : 0u;
    struct Face { Vec3 p0, p1, p2, p3, n, t; };
    Face faces[6] = {
        {{bMin.x, bMax.y, bMax.z}, {bMax.x, bMax.y, bMax.z}, {bMax.x, bMax.y, bMin.z}, {bMin.x, bMax.y, bMin.z}, {0, 1, 0}, {1, 0, 0}},
        {{bMin.x, bMin.y, bMin.z}, {bMax.x, bMin.y, bMin.z}, {bMax.x, bMin.y, bMax.z}, {bMin.x, bMin.y, bMax.z}, {0, -1, 0}, {1, 0, 0}},
        {{bMin.x, bMin.y, bMax.z}, {bMax.x, bMin.y, bMax.z}, {bMax.x, bMax.y, bMax.z}, {bMin.x, bMax.y, bMax.z}, {0, 0, 1}, {1, 0, 0}},
        {{bMax.x, bMin.y, bMin.z}, {bMin.x, bMin.y, bMin.z}, {bMin.x, bMax.y, bMin.z}, {bMax.x, bMax.y, bMin.z}, {0, 0, -1}, {-1, 0, 0}},
        {{bMax.x, bMin.y, bMax.z}, {bMax.x, bMin.y, bMin.z}, {bMax.x, bMax.y, bMin.z}, {bMax.x, bMax.y, bMax.z}, {1, 0, 0}, {0, 0, -1}},
        {{bMin.x, bMin.y, bMin.z}, {bMin.x, bMin.y, bMax.z}, {bMin.x, bMax.y, bMax.z}, {bMin.x, bMax.y, bMin.z}, {-1, 0, 0}, {0, 0, 1}},
    };
    uint32_t gOff = static_cast<uint32_t>(scene.vertices.size());
    std::vector<uint32_t> triVerts;
    for (int f = 0; f < 6; ++f) {
      Vec3 pts[4] = {faces[f].p0, faces[f].p1, faces[f].p2, faces[f].p3};
      for (int k = 0; k < 4; ++k) {
        ShowcaseVertex sv{};
        sv.pos[0] = pts[k].x; sv.pos[1] = pts[k].y; sv.pos[2] = pts[k].z;
        sv.matId = matId;
        sv.normal[0] = faces[f].n.x; sv.normal[1] = faces[f].n.y; sv.normal[2] = faces[f].n.z;
        sv.flags = flags;
        sv.tangent[0] = faces[f].t.x; sv.tangent[1] = faces[f].t.y; sv.tangent[2] = faces[f].t.z; sv.tangent[3] = 1.0f;
        sv.uv[0] = (k == 1 || k == 2) ? 1.0f : 0.0f;
        sv.uv[1] = (k >= 2) ? 1.0f : 0.0f;
        scene.vertices.push_back(sv);
      }
      uint32_t b = gOff + static_cast<uint32_t>(f) * 4u;
      triVerts.insert(triVerts.end(), {b, b + 1u, b + 2u, b, b + 2u, b + 3u});
    }
    ShowcaseInstance inst{};
    Mat4 I = Mat4::identity();
    std::memcpy(inst.model, I.m, sizeof(I.m));
    inst.baseFirstIndex = static_cast<uint32_t>(scene.indices.size());
    buildMeshletsForTriangles(triVerts, matId, flags, scene,
                              inst.lodFirstMeshlet[0], inst.lodMeshletCount[0]);
    inst.baseIndexCount = static_cast<uint32_t>(scene.indices.size()) - inst.baseFirstIndex;
    inst.vertexOffset = 0;
    inst.materialId = matId;
    inst.flags = flags;
    Vec3 center = (bMin + bMax) * 0.5f;
    inst.boundSphereWorld[0] = center.x;
    inst.boundSphereWorld[1] = center.y;
    inst.boundSphereWorld[2] = center.z;
    inst.boundSphereWorld[3] = length(bMax - bMin) * 0.5f;
    uint32_t instIdx = static_cast<uint32_t>(scene.instances.size());
    scene.instances.push_back(inst);
    appendTaskPacketsForInstance(instIdx, inst, 1u, scene);
  };

  addBox({-1400.0f, -20.0f, -800.0f}, {1400.0f, 0.0f, 800.0f}, 0u, true);
  addBox({-1400.0f, 0.0f, -800.0f}, {1400.0f, 900.0f, -740.0f}, 0u, true);
  addBox({-1400.0f, 0.0f, 740.0f}, {1400.0f, 900.0f, 800.0f}, 0u, true);
  for (int c = -5; c <= 5; ++c) {
    float x = static_cast<float>(c) * 230.0f;
    addBox({x - 32.0f, 0.0f, -430.0f}, {x + 32.0f, 520.0f, -360.0f}, 0u, true);
    addBox({x - 32.0f, 0.0f, 360.0f}, {x + 32.0f, 520.0f, 430.0f}, 0u, true);
  }
  scene.sceneSourceLabel = "Procedural Atrium Fallback";
}

} // namespace

Vec3 evaluateSculptedDisplacement(const Vec3 &localPos, const Vec3 &localNormal, const Vec2 & /*uv*/) {
  float radial = std::sqrt(localPos.x * localPos.x + localPos.z * localPos.z);
  float poleAtten = std::clamp(radial * 0.125f, 0.0f, 1.0f);
  float ang = std::atan2(localPos.z, localPos.x);
  float flutes = std::cos(16.0f * ang) * poleAtten;
  float rings = std::sin(localPos.y * 0.28f + radial * 0.18f);
  float micro = std::sin(localPos.x * 0.24f) * std::cos(localPos.z * 0.24f);
  float disp = (0.60f * flutes * rings + 0.40f * micro) * 2.2f;
  return localPos + normalize(localNormal) * disp;
}

CameraState ShowcaseSceneData::evalTourCamera(float timeSec) const {
  constexpr float kLoopDuration = 18.0f;
  float u = std::fmod(std::max(0.0f, timeSec), kLoopDuration) / kLoopDuration;

  const Vec3 wayPos[6] = {
      {-1060.0f, 200.0f, 0.0f},
      {-560.0f, 185.0f, -150.0f},
      {40.0f, 210.0f, -240.0f},
      {640.0f, 260.0f, 60.0f},
      {120.0f, 430.0f, 250.0f},
      {-650.0f, 270.0f, 180.0f},
  };
  const Vec3 wayLook[6] = {
      {200.0f, 150.0f, 0.0f},
      {260.0f, 145.0f, -190.0f},
      {600.0f, 170.0f, 40.0f},
      {-200.0f, 140.0f, 0.0f},
      {-500.0f, 140.0f, -40.0f},
      {100.0f, 160.0f, 0.0f},
  };

  float scaled = u * 6.0f;
  int seg = static_cast<int>(std::floor(scaled)) % 6;
  float frac = scaled - std::floor(scaled);

  auto wrap = [](int i) { return (i % 6 + 6) % 6; };
  Vec3 pos = evalCatmullRom(wayPos[wrap(seg - 1)], wayPos[wrap(seg)],
                            wayPos[wrap(seg + 1)], wayPos[wrap(seg + 2)], frac);
  Vec3 look = evalCatmullRom(wayLook[wrap(seg - 1)], wayLook[wrap(seg)],
                             wayLook[wrap(seg + 1)], wayLook[wrap(seg + 2)], frac);
  Vec3 dir = normalize(look - pos);

  CameraState cam{};
  cam.pos = pos;
  cam.yaw = std::atan2(dir.z, dir.x);
  cam.pitch = std::asin(std::clamp(dir.y, -0.95f, 0.95f));
  cam.fovY = 0.78f;
  return cam;
}

bool buildShowcaseScene(ShowcaseSceneData &scene, std::string &outError) {
  scene = ShowcaseSceneData{};
  scene.sunDir = normalize({0.26f, 0.93f, 0.25f});

  std::string sponzaPath = findAssetModel("sponza.glb");
  GltfScene gltf;
  std::string gltfErr;
  bool loadedGltf = (!sponzaPath.empty() && gltf.loadFromFile(sponzaPath, gltfErr));

  if (loadedGltf) {
    scene.sceneSourceLabel = "Khronos Sponza Atrium (sponza.glb + AAA Enrichment)";

    // 1. Copy decoded textures (keeping valid indices <= kMaxShowcaseTextures - 4)
    const size_t maxSrcTex = std::max<size_t>(3u, kMaxShowcaseTextures - 4u);
    const auto &srcTex = gltf.getTextures();
    size_t copyTexCount = std::min(srcTex.size(), maxSrcTex);
    scene.textures.reserve(copyTexCount + 3u);
    for (size_t i = 0; i < copyTexCount; ++i) {
      scene.textures.push_back(srcTex[i]);
    }
    auto mapTex = [&](int32_t idx, int32_t fallbackIdx) -> int32_t {
      if (idx < 0 || static_cast<size_t>(idx) >= copyTexCount) return fallbackIdx;
      return idx;
    };

    // 2. Copy materials
    const auto &srcMats = gltf.getMaterials();
    scene.materials.reserve(srcMats.size() + 4u);
    for (const auto &gm : srcMats) {
      ShowcaseMaterial sm{};
      std::memcpy(sm.baseColorFactor, gm.baseColorFactor, sizeof(sm.baseColorFactor));
      std::memcpy(sm.emissiveFactor, gm.emissiveFactor, sizeof(sm.emissiveFactor));
      sm.metallicFactor = gm.metallicFactor;
      sm.roughnessFactor = std::clamp(gm.roughnessFactor, 0.08f, 1.0f);
      sm.normalScale = gm.normalScale;
      sm.alphaCutoff = gm.alphaCutoff;
      sm.baseColorTexIdx = mapTex(gm.baseColorTexIdx, 0);
      sm.normalTexIdx = mapTex(gm.normalTexIdx, 1);
      sm.metalRoughTexIdx = mapTex(gm.metallicRoughnessTexIdx, 2);
      sm.occlusionTexIdx = mapTex(gm.occlusionTexIdx, 0);
      sm.alphaMode = gm.alphaMode;
      sm.archetype = gm.archetype;
      sm.flags = (gm.alphaMode == 1u || gm.archetype == 7u)
                     ? (SHOWCASE_FLAG_ALPHA_CUTOUT | SHOWCASE_FLAG_DOUBLE_SIDED)
                     : ((gm.archetype == 3u) ? SHOWCASE_FLAG_DOUBLE_SIDED : 0u);
      scene.materials.push_back(sm);
    }

    // 3. Copy Sponza vertices + partition triangles by (materialId, spatialCell)
    float minB[3], maxB[3];
    gltf.getBounds(minB, maxB);
    scene.boundsMin = {minB[0], minB[1], minB[2]};
    scene.boundsMax = {maxB[0], maxB[1], maxB[2]};

    const auto &srcVerts = gltf.getVertices();
    const auto &srcIdx = gltf.getIndices();
    const auto &srcTriMats = gltf.getTriangleMaterials();
    scene.sponzaSourceTriangles = static_cast<uint32_t>(srcIdx.size() / 3u);

    scene.vertices.resize(srcVerts.size());
    for (size_t i = 0; i < srcVerts.size(); ++i) {
      ShowcaseVertex &dst = scene.vertices[i];
      std::memcpy(dst.pos, srcVerts[i].pos, sizeof(dst.pos));
      std::memcpy(dst.normal, srcVerts[i].normal, sizeof(dst.normal));
      std::memcpy(dst.tangent, srcVerts[i].tangent, sizeof(dst.tangent));
      std::memcpy(dst.uv, srcVerts[i].uv, sizeof(dst.uv));
      dst.matId = 0u;
      dst.flags = SHOWCASE_FLAG_OCCLUDER;
    }

    // Spatial cells: 6 along X, 2 along Y, 4 along Z -> ~200-350 cohesive chunks
    constexpr int kGx = 6, kGy = 2, kGz = 4;
    Vec3 span{
        std::max(1.0f, maxB[0] - minB[0]),
        std::max(1.0f, maxB[1] - minB[1]),
        std::max(1.0f, maxB[2] - minB[2])};

    std::unordered_map<uint64_t, std::vector<uint32_t>> chunkTris;
    chunkTris.reserve(512u);

    for (uint32_t t = 0; t < scene.sponzaSourceTriangles; ++t) {
      uint32_t i0 = srcIdx[t * 3u + 0u];
      uint32_t i1 = srcIdx[t * 3u + 1u];
      uint32_t i2 = srcIdx[t * 3u + 2u];
      uint32_t matId = (t < srcTriMats.size()) ? srcTriMats[t] : 0u;
      if (matId >= scene.materials.size()) matId = 0u;

      Vec3 c{
          (scene.vertices[i0].pos[0] + scene.vertices[i1].pos[0] + scene.vertices[i2].pos[0]) / 3.0f,
          (scene.vertices[i0].pos[1] + scene.vertices[i1].pos[1] + scene.vertices[i2].pos[1]) / 3.0f,
          (scene.vertices[i0].pos[2] + scene.vertices[i1].pos[2] + scene.vertices[i2].pos[2]) / 3.0f};
      int cx = std::clamp(static_cast<int>(((c.x - minB[0]) / span.x) * kGx), 0, kGx - 1);
      int cy = std::clamp(static_cast<int>(((c.y - minB[1]) / span.y) * kGy), 0, kGy - 1);
      int cz = std::clamp(static_cast<int>(((c.z - minB[2]) / span.z) * kGz), 0, kGz - 1);
      uint32_t cellId = static_cast<uint32_t>((cx * kGy + cy) * kGz + cz);
      uint64_t key = (static_cast<uint64_t>(matId) << 16u) | cellId;

      auto &list = chunkTris[key];
      list.push_back(i0);
      list.push_back(i1);
      list.push_back(i2);

      // Update vertex material ID
      scene.vertices[i0].matId = matId;
      scene.vertices[i1].matId = matId;
      scene.vertices[i2].matId = matId;

      const uint32_t mFlags = scene.materials[matId].flags;
      if (mFlags & SHOWCASE_FLAG_DOUBLE_SIDED) {
        // Append reverse-winding duplicate so banners/leaves stay double-sided under BACK_BIT culling
        list.push_back(i0);
        list.push_back(i2);
        list.push_back(i1);
      }
    }

    // Convert each chunk into a ShowcaseInstance with meshlets
    std::vector<uint64_t> sortedKeys;
    sortedKeys.reserve(chunkTris.size());
    for (const auto &kv : chunkTris) sortedKeys.push_back(kv.first);
    std::sort(sortedKeys.begin(), sortedKeys.end());

    for (uint64_t key : sortedKeys) {
      uint32_t matId = static_cast<uint32_t>(key >> 16u);
      const auto &tris = chunkTris[key];
      const uint32_t mFlags = scene.materials[matId].flags;
      const bool isCutout = (mFlags & SHOWCASE_FLAG_ALPHA_CUTOUT) != 0u;
      const uint32_t instFlags = isCutout ? mFlags : (SHOWCASE_FLAG_OCCLUDER | mFlags);

      ShowcaseInstance inst{};
      Mat4 I = Mat4::identity();
      std::memcpy(inst.model, I.m, sizeof(I.m));
      inst.baseFirstIndex = static_cast<uint32_t>(scene.indices.size());
      buildMeshletsForTriangles(tris, matId, instFlags, scene,
                                inst.lodFirstMeshlet[0], inst.lodMeshletCount[0]);
      inst.baseIndexCount = static_cast<uint32_t>(scene.indices.size()) - inst.baseFirstIndex;
      inst.vertexOffset = 0; // meshlet indices are global
      inst.materialId = matId;
      inst.flags = instFlags;

      // Compute tight world sphere for the chunk
      Vec3 bMin{1e9f, 1e9f, 1e9f}, bMax{-1e9f, -1e9f, -1e9f};
      for (uint32_t vIdx : tris) {
        const auto &p = scene.vertices[vIdx].pos;
        bMin.x = std::min(bMin.x, p[0]); bMin.y = std::min(bMin.y, p[1]); bMin.z = std::min(bMin.z, p[2]);
        bMax.x = std::max(bMax.x, p[0]); bMax.y = std::max(bMax.y, p[1]); bMax.z = std::max(bMax.z, p[2]);
      }
      Vec3 center{(bMin.x + bMax.x) * 0.5f, (bMin.y + bMax.y) * 0.5f, (bMin.z + bMax.z) * 0.5f};
      float radius = 1.0f;
      for (uint32_t vIdx : tris) {
        const auto &p = scene.vertices[vIdx].pos;
        radius = std::max(radius, length({p[0] - center.x, p[1] - center.y, p[2] - center.z}));
      }
      inst.boundSphereWorld[0] = center.x;
      inst.boundSphereWorld[1] = center.y;
      inst.boundSphereWorld[2] = center.z;
      inst.boundSphereWorld[3] = radius;
      inst.drawOrderScore = static_cast<uint32_t>(std::max(0.0f, (center.x + 2000.0f) + std::abs(center.z) * 0.25f));

      uint32_t instIdx = static_cast<uint32_t>(scene.instances.size());
      scene.instances.push_back(inst);
      appendTaskPacketsForInstance(instIdx, inst, isCutout ? 2u : 1u, scene);
    }
  } else {
    if (!sponzaPath.empty()) {
      std::cerr << "[ShowcaseScene] Failed to load " << sponzaPath << ": " << gltfErr << "\n";
    }
    buildProceduralFallbackAtrium(scene);
  }

  // 4. Add AAA Enrichment Textures & Materials (Urns, Relief Slabs, Atrium Ivy)
  const int32_t leafTexIdx = static_cast<int32_t>(scene.textures.size());
  scene.textures.push_back(makeProceduralLeafTexture());
  const int32_t marbleTexIdx = static_cast<int32_t>(scene.textures.size());
  scene.textures.push_back(makeProceduralMarbleTexture());

  const uint32_t urnMatId = static_cast<uint32_t>(scene.materials.size());
  {
    ShowcaseMaterial mat{};
    mat.baseColorFactor[0] = 0.88f;
    mat.baseColorFactor[1] = 0.68f;
    mat.baseColorFactor[2] = 0.42f;
    mat.metallicFactor = 0.78f;
    mat.roughnessFactor = 0.28f;
    mat.baseColorTexIdx = marbleTexIdx;
    mat.archetype = 4u; // Ornate bronze/weathered sculpted relief
    scene.materials.push_back(mat);
  }

  const uint32_t plinthMatId = static_cast<uint32_t>(scene.materials.size());
  {
    ShowcaseMaterial mat{};
    mat.baseColorFactor[0] = 0.72f;
    mat.baseColorFactor[1] = 0.69f;
    mat.baseColorFactor[2] = 0.64f;
    mat.metallicFactor = 0.08f;
    mat.roughnessFactor = 0.46f;
    mat.baseColorTexIdx = marbleTexIdx;
    mat.archetype = 5u; // Carved architectural stone
    scene.materials.push_back(mat);
  }

  const uint32_t ivyMatId = static_cast<uint32_t>(scene.materials.size());
  {
    ShowcaseMaterial mat{};
    mat.baseColorFactor[0] = 0.85f;
    mat.baseColorFactor[1] = 1.0f;
    mat.baseColorFactor[2] = 0.78f;
    mat.metallicFactor = 0.0f;
    mat.roughnessFactor = 0.58f;
    mat.alphaCutoff = 0.45f;
    mat.baseColorTexIdx = leafTexIdx;
    mat.alphaMode = 1u;
    mat.archetype = 7u;
    mat.flags = SHOWCASE_FLAG_ALPHA_CUTOUT | SHOWCASE_FLAG_DOUBLE_SIDED;
    scene.materials.push_back(mat);
  }

  // Build shared prototypes
  const ProtoGeometry urnProto = createSculptedUrnPrototype(scene, urnMatId);
  const ProtoGeometry plinthProto = createReliefPlinthPrototype(scene, plinthMatId);
  const ProtoGeometry shrubProto = createFoliageShrubPrototype(scene, ivyMatId);
  const ProtoGeometry relicProto = createCloisterRelicPrototype(scene, plinthMatId);

  // Sponza's courtyard nave floor sits near y = 0 (minBound.y may dip to ~ -60 for steps/trim).
  const float floorY = 0.0f;

  auto spawnPrototypeInstance = [&](const ProtoGeometry &proto, const Vec3 &pos, float rotY,
                                    const Vec3 &scale, uint32_t score) {
    ShowcaseInstance inst{};
    Mat4 M = matTRS(pos, rotY, scale);
    std::memcpy(inst.model, M.m, sizeof(M.m));
    Vec4 ws = transformBoundingSphere(M, proto.localSphere);
    inst.boundSphereWorld[0] = ws.x;
    inst.boundSphereWorld[1] = ws.y;
    inst.boundSphereWorld[2] = ws.z;
    inst.boundSphereWorld[3] = ws.w;
    for (uint32_t lod = 0; lod < kShowcaseNumLods; ++lod) {
      inst.lodFirstMeshlet[lod] = proto.lodFirstMeshlet[lod];
      inst.lodMeshletCount[lod] = proto.lodMeshletCount[lod];
    }
    inst.baseFirstIndex = proto.baseFirstIndex;
    inst.baseIndexCount = proto.baseIndexCount;
    inst.vertexOffset = proto.baseVertexOffset;
    inst.materialId = proto.materialId;
    inst.drawOrderScore = score;
    inst.flags = proto.flags;

    if (proto.flags & SHOWCASE_FLAG_TESSELLATED) {
      scene.tessellatedBasePatches += proto.baseIndexCount / 3u;
    }

    uint32_t instIdx = static_cast<uint32_t>(scene.instances.size());
    scene.instances.push_back(inst);
    appendTaskPacketsForInstance(instIdx, inst, 2u, scene);
  };

  // 5. Place Tessellated Urns & Carved Medallion Slabs (nave visible + heavy occluded cloisters/end-halls)
  uint32_t seed = 421337u;
  auto rand01 = [&]() -> float {
    seed = seed * 1664525u + 1013904223u;
    return static_cast<float>((seed >> 8u) & 0xFFFFFFu) / static_cast<float>(0x1000000u);
  };

  // Nave aisle rows (x in [-1020, +1020], z = +-205) + 4 occluded side-cloister rows behind walls/columns
  const float zUrnRows[6] = {-205.0f, 205.0f, -520.0f, 520.0f, -660.0f, 660.0f};
  for (int r = 0; r < 6; ++r) {
    const int rowCount = (r < 2) ? 16 : 24;
    for (int i = 0; i < rowCount; ++i) {
      float x = -1040.0f + (static_cast<float>(i) / static_cast<float>(rowCount - 1)) * 2080.0f;
      float rot = rand01() * 2.0f * kPi;
      float s = 0.92f + 0.16f * rand01();
      uint32_t score = static_cast<uint32_t>(std::max(0.0f, (x + 2000.0f) + std::abs(zUrnRows[r]) * 0.25f));
      spawnPrototypeInstance(plinthProto, {x, floorY, zUrnRows[r]}, rot * 0.25f, {s, 1.0f, s}, score + 5u);
      spawnPrototypeInstance(urnProto, {x, floorY + 14.0f, zUrnRows[r]}, rot, {s, s, s}, score);
    }
  }

  // Central promenade carved relief cobblestones + behind-wall East/West occluded hall vaults
  for (int i = 0; i < 20; ++i) {
    float x = -1000.0f + static_cast<float>(i) * 105.0f;
    spawnPrototypeInstance(plinthProto, {x, floorY - 2.0f, 0.0f}, 0.0f, {0.9f, 0.65f, 0.9f},
                           static_cast<uint32_t>(x + 2000.0f));
  }
  for (int side = 0; side < 2; ++side) {
    float xWall = (side == 0) ? 1310.0f : -1360.0f; // Inside view frustum behind East Sponza facade
    for (int r = 0; r < 6; ++r) {
      float z = -540.0f + static_cast<float>(r) * 216.0f;
      uint32_t score = static_cast<uint32_t>(std::max(0.0f, (xWall + 2000.0f) + std::abs(z) * 0.25f));
      spawnPrototypeInstance(urnProto, {xWall, floorY + 14.0f, z}, rand01() * 6.28f, {0.95f, 0.95f, 0.95f}, score);
      spawnPrototypeInstance(plinthProto, {xWall, floorY, z}, 0.0f, {1.0f, 1.0f, 1.0f}, score + 5u);
    }
  }
  // Clustered ornate urns in upper colonnade galleries (y = 428)
  const float upperZ[4] = {-490.0f, 490.0f, -630.0f, 630.0f};
  for (int side = 0; side < 4; ++side) {
    for (int i = 0; i < 18; ++i) {
      float x = -1020.0f + static_cast<float>(i) * 120.0f;
      uint32_t score = static_cast<uint32_t>(std::max(0.0f, (x + 2000.0f) + std::abs(upperZ[side]) * 0.25f));
      spawnPrototypeInstance(urnProto, {x, 428.0f, upperZ[side]}, rand01() * 6.28f, {0.82f, 0.82f, 0.82f}, score);
    }
  }

  // Dense high-poly sculpted cloister statuary behind Sponza's opaque walls & East gallery
  // (~2,400 instances * 2,160 tris = 5.18M extra frustum tris that Legacy draws, while Modern Hi-Z culls!)
  const float relicZ[6] = {-500.0f, 500.0f, -610.0f, 610.0f, -720.0f, 720.0f};
  for (int rz = 0; rz < 6; ++rz) {
    for (int i = 0; i < 160; ++i) {
      float rx = -1150.0f + (static_cast<float>(i) / 159.0f) * 2350.0f;
      float yLvl = (i % 3 == 2) ? 405.0f : floorY;
      float sc = 0.85f + 0.25f * rand01();
      uint32_t score = static_cast<uint32_t>(std::max(0.0f, (rx + 2000.0f) + std::abs(relicZ[rz]) * 0.25f));
      spawnPrototypeInstance(relicProto, {rx, yLvl, relicZ[rz] + (rand01() - 0.5f) * 28.0f},
                             rand01() * 6.283f, {sc, sc, sc}, score);
    }
  }
  // Stacked behind-East-wall cathedral vault rows (x in [+1320, +1680], completely behind East Sponza facade)
  for (int col = 0; col < 12; ++col) {
    float rx = 1320.0f + static_cast<float>(col) * 32.0f;
    for (int row = 0; row < 48; ++row) {
      float rz = -620.0f + (static_cast<float>(row) / 47.0f) * 1240.0f;
      float ry = (row % 2 == 0) ? floorY : 260.0f;
      uint32_t score = static_cast<uint32_t>(std::max(0.0f, (rx + 2000.0f) + std::abs(rz) * 0.25f));
      spawnPrototypeInstance(relicProto, {rx, ry, rz}, rand01() * 6.283f, {0.9f, 0.9f, 0.9f}, score);
    }
  }

  // 6. Place Dense Layered Courtyard Foliage & Behind-Wall Cloister Shrubs (~3,600 instances)
  // Overlapping layers along colonnades -> high Naive forward overdraw + massive Modern Hi-Z culling behind walls
  const float foliageZonesZ[8] = {-295.0f, 295.0f, -490.0f, 490.0f, -620.0f, 620.0f, -740.0f, 740.0f};
  for (int zIdx = 0; zIdx < 8; ++zIdx) {
    const bool isSideCloister = (zIdx >= 2);
    const int countAlongX = isSideCloister ? 88 : 56;
    for (int ix = 0; ix < countAlongX; ++ix) {
      float fx = -1280.0f + (static_cast<float>(ix) / static_cast<float>(countAlongX - 1)) * 2560.0f;
      const int layers = isSideCloister ? 5 : 4;
      for (int layer = 0; layer < layers; ++layer) {
        float jitterX = (rand01() - 0.5f) * 34.0f;
        float jitterZ = (rand01() - 0.5f) * 68.0f;
        float yBase = (layer == layers - 1 && !isSideCloister) ? 390.0f : floorY;
        float scale = 0.72f + 0.45f * rand01();
        float wx = fx + jitterX;
        float wz = foliageZonesZ[zIdx] + jitterZ;
        uint32_t drawScore = static_cast<uint32_t>(std::max(0.0f, (wx + 2000.0f) + std::abs(wz) * 0.25f));
        spawnPrototypeInstance(shrubProto, {wx, yBase, wz},
                               rand01() * 6.283f, {scale, scale, scale}, drawScore);
      }
    }
  }

  // Dense multi-layered central courtyard planters around urns for heavy alpha-cutout depth overlap
  for (int i = 0; i < 64; ++i) {
    float x = -1080.0f + (static_cast<float>(i) / 63.0f) * 2160.0f;
    float sideSign = (i % 2 == 0) ? 1.0f : -1.0f;
    for (int sub = 0; sub < 6; ++sub) {
      float sx = x + (rand01() - 0.5f) * 42.0f;
      float sz = sideSign * (150.0f + rand01() * 64.0f);
      float sc = 0.56f + 0.36f * rand01();
      uint32_t drawScore = static_cast<uint32_t>(std::max(0.0f, (sx + 2000.0f) + std::abs(sz) * 0.25f));
      spawnPrototypeInstance(shrubProto, {sx, floorY, sz}, rand01() * 6.283f, {sc, sc * 0.9f, sc}, drawScore);
    }
  }

  // Upper arch hanging ivy clusters
  for (int i = 0; i < 128; ++i) {
    float x = -1200.0f + (static_cast<float>(i) / 127.0f) * 2400.0f;
    float zSign = (i % 2 == 0) ? -1.0f : 1.0f;
    float wz = zSign * 360.0f;
    uint32_t drawScore = static_cast<uint32_t>(std::max(0.0f, (x + 2000.0f) + std::abs(wz) * 0.25f));
    spawnPrototypeInstance(shrubProto, {x, 485.0f + (rand01() - 0.5f) * 40.0f, wz},
                           rand01() * 6.283f, {0.75f, 0.75f, 0.75f}, drawScore);
  }

  // Accumulate LOD0 meshlet triangle count metric
  uint64_t totalLod0Tris = 0;
  for (const auto &inst : scene.instances) {
    for (uint32_t m = 0; m < inst.lodMeshletCount[0]; ++m) {
      totalLod0Tris += scene.meshlets[inst.lodFirstMeshlet[0] + m].triangleCount;
    }
  }
  scene.totalMeshletTrianglesLod0 = static_cast<uint32_t>(std::min<uint64_t>(totalLod0Tris, 0xFFFFFFFFull));

  // 7. Build Naive back-to-front draw order (maximizing overdraw & state/instance thrash)
  //    vs Legacy CPU front-to-back opaque-then-cutout draw lists
  scene.naiveDrawOrder.resize(scene.instances.size());
  for (uint32_t i = 0; i < static_cast<uint32_t>(scene.instances.size()); ++i) {
    scene.naiveDrawOrder[i] = i;
  }
  // Interleave far/foliage first then opaque walls/props to simulate unsorted forward draw-call storm
  std::sort(scene.naiveDrawOrder.begin(), scene.naiveDrawOrder.end(), [&](uint32_t a, uint32_t b) {
    const auto &ia = scene.instances[a];
    const auto &ib = scene.instances[b];
    if (ia.drawOrderScore != ib.drawOrderScore) {
      return ia.drawOrderScore > ib.drawOrderScore;
    }
    return a < b;
  });

  // Legacy splits opaque (sorted roughly front-to-back / by material) vs alpha cutout
  for (uint32_t i = 0; i < static_cast<uint32_t>(scene.instances.size()); ++i) {
    if (scene.instances[i].flags & SHOWCASE_FLAG_ALPHA_CUTOUT) {
      scene.legacyCutoutDrawList.push_back(i);
    } else {
      scene.legacyOpaqueDrawList.push_back(i);
    }
  }
  std::sort(scene.legacyOpaqueDrawList.begin(), scene.legacyOpaqueDrawList.end(), [&](uint32_t a, uint32_t b) {
    return scene.instances[a].materialId < scene.instances[b].materialId;
  });

  // 8. Place 16 warm Atrium Courtyard Torch Lights
  scene.lights.reserve(16u);
  for (int i = 0; i < 8; ++i) {
    float x = -980.0f + static_cast<float>(i) * 280.0f;
    ShowcaseLight l0{};
    l0.posRadius[0] = x; l0.posRadius[1] = 240.0f; l0.posRadius[2] = -335.0f; l0.posRadius[3] = 520.0f;
    l0.colorIntensity[0] = 1.0f; l0.colorIntensity[1] = 0.68f; l0.colorIntensity[2] = 0.32f; l0.colorIntensity[3] = 4.2f;
    scene.lights.push_back(l0);

    ShowcaseLight l1 = l0;
    l1.posRadius[2] = 335.0f;
    scene.lights.push_back(l1);
  }

  // 9. Deterministic Camera Presets (tuned to Sponza coordinates)
  scene.presets = {
      {"Central Nave",
       "Balanced view down Sponza's courtyard showing sculpted urns, cobblestone medallions, hanging ivy, and multi-bounce torchlight.",
       CameraState{{-1060.0f, 200.0f, 0.0f}, 0.02f, -0.05f, 0.78f, 10.0f, 12000.0f}},
      {"Occluded Colonnade",
       "Angled through heavy stone pillars toward dense hidden side-cloister props to spotlight Two-Phase Hi-Z Occlusion Culling.",
       CameraState{{-680.0f, 185.0f, -110.0f}, -0.58f, -0.04f, 0.78f, 10.0f, 12000.0f}},
      {"Tessellation Close-Up",
       "Faces foreground sculpted urns and relief slabs while row after row recedes down the hall—contrasting 64x micro-triangles vs. Adaptive Meshlet LOD.",
       CameraState{{-740.0f, 115.0f, -105.0f}, -0.36f, -0.14f, 0.72f, 10.0f, 12000.0f}},
      {"Atrium Overview",
       "Elevated wide balcony overlook ideal for Freeze Culling X-Ray inspection.",
       CameraState{{-320.0f, 920.0f, 240.0f}, -0.22f, -0.68f, 0.85f, 10.0f, 12000.0f}},
  };

  if (scene.instances.empty() || scene.meshlets.empty() || scene.textures.empty()) {
    outError = "Failed to build valid ShowcaseScene geometry";
    return false;
  }
  return true;
}

} // namespace gpubench::showcase
