#pragma once

#include "scene/GltfScene.h"
#include "showcase/ShowcaseMath.h"

#include <cstdint>
#include <string>
#include <vector>

namespace gpubench::showcase {

constexpr uint32_t kMaxMeshletVertices = 64u;
constexpr uint32_t kMaxMeshletTriangles = 124u;
constexpr uint32_t kMaxShowcaseTextures = 192u;
constexpr uint32_t kShowcaseNumLods = 3u;

// Instance & vertex flags (must match shaders/showcase_common.glsl)
constexpr uint32_t SHOWCASE_FLAG_TESSELLATED  = 1u << 0; // Sculpted prop: HW tess in Naive/Legacy, LOD Meshlets in Modern
constexpr uint32_t SHOWCASE_FLAG_ALPHA_CUTOUT = 1u << 1; // Alpha-tested foliage / ivy / leaves
constexpr uint32_t SHOWCASE_FLAG_OCCLUDER     = 1u << 2; // Large structural Sponza architecture rendered in Hi-Z Phase 0
constexpr uint32_t SHOWCASE_FLAG_DOUBLE_SIDED = 1u << 3; // No backface cone culling

struct alignas(16) ShowcaseVertex {
  float pos[3]{0.0f, 0.0f, 0.0f};
  uint32_t matId{0};
  float normal[3]{0.0f, 1.0f, 0.0f};
  uint32_t flags{0};
  float tangent[4]{1.0f, 0.0f, 0.0f, 1.0f};
  float uv[2]{0.0f, 0.0f};
  float pad[2]{0.0f, 0.0f};
};
static_assert(sizeof(ShowcaseVertex) == 64, "ShowcaseVertex must be 64 bytes std430");

struct alignas(16) ShowcaseMeshlet {
  uint32_t vertexOffset{0};   // Offset into meshletVertices[]
  uint32_t triangleOffset{0}; // Offset into meshletTriangles[] (packed u32: i0|(i1<<8)|(i2<<16))
  uint32_t vertexCount{0};
  uint32_t triangleCount{0};
  float boundSphere[4]{0.0f, 0.0f, 0.0f, 1.0f}; // local xyz + radius
  float coneAxis[3]{0.0f, 1.0f, 0.0f};
  float coneCutoff{1.0f};
  uint32_t materialId{0};
  uint32_t flags{0};
  uint32_t globalIndexBase{0}; // Offset in global indices[] for compute-indirect fallback
  uint32_t pad0{0};
};
static_assert(sizeof(ShowcaseMeshlet) == 64, "ShowcaseMeshlet must be 64 bytes std430");

struct alignas(16) ShowcaseInstance {
  float model[16]{};
  float boundSphereWorld[4]{0.0f, 0.0f, 0.0f, 1.0f};
  // Meshlet index ranges in global meshlets[] per LOD [0, 1, 2], plus coarse index offset for Naive/Legacy
  uint32_t lodFirstMeshlet[3]{0, 0, 0};
  uint32_t baseFirstIndex{0};
  uint32_t lodMeshletCount[3]{0, 0, 0};
  uint32_t baseIndexCount{0};
  int32_t vertexOffset{0};   // Base vertex offset for coarse index buffer (Naive/Legacy)
  uint32_t materialId{0};
  uint32_t drawOrderScore{0};
  uint32_t flags{0};
};
static_assert(sizeof(ShowcaseInstance) == 128, "ShowcaseInstance must be 128 bytes std430");

struct alignas(16) ShowcaseMaterial {
  float baseColorFactor[4]{1.0f, 1.0f, 1.0f, 1.0f};
  float emissiveFactor[4]{0.0f, 0.0f, 0.0f, 0.0f};
  float metallicFactor{0.0f};
  float roughnessFactor{0.75f};
  float normalScale{1.0f};
  float alphaCutoff{0.5f};
  int32_t baseColorTexIdx{0};
  int32_t normalTexIdx{1};
  int32_t metalRoughTexIdx{2};
  int32_t occlusionTexIdx{0};
  uint32_t alphaMode{0};
  uint32_t archetype{0};
  uint32_t flags{0};
  uint32_t pad0{0};
};
static_assert(sizeof(ShowcaseMaterial) == 80, "ShowcaseMaterial must be 80 bytes std430");

struct alignas(16) ShowcaseLight {
  float posRadius[4]{0.0f, 0.0f, 0.0f, 500.0f};
  float colorIntensity[4]{1.0f, 0.7f, 0.35f, 1.0f};
};
static_assert(sizeof(ShowcaseLight) == 32, "ShowcaseLight must be 32 bytes std430");

struct alignas(16) ShowcaseTaskPacket {
  uint32_t instanceIdx{0};
  uint32_t localStart{0};
  uint32_t count{0};
  uint32_t phaseMask{0}; // bit 0 = occluder (Phase 0), bit 1 = occludee/detail (Phase 1)
};
static_assert(sizeof(ShowcaseTaskPacket) == 16, "ShowcaseTaskPacket must be 16 bytes std430");

struct CameraPreset {
  const char *name;
  const char *description;
  CameraState camera;
};

struct ShowcaseSceneData {
  std::vector<ShowcaseVertex> vertices;
  std::vector<uint32_t> indices; // Coarse index buffer used by Naive (with tess patches) & Legacy
  std::vector<ShowcaseMeshlet> meshlets;
  std::vector<uint32_t> meshletVertices;  // Global vertex indices referenced by meshlets
  std::vector<uint32_t> meshletTriangles; // Packed local tri indices: i0 | (i1 << 8) | (i2 << 16)
  std::vector<ShowcaseInstance> instances;
  std::vector<ShowcaseMaterial> materials;
  std::vector<ShowcaseLight> lights;
  std::vector<GltfTexture> textures;

  // Ordered task packets for Task+Mesh Pass 0 (Occluders) and Pass 1 (Occludees)
  std::vector<ShowcaseTaskPacket> phase0Packets;
  std::vector<ShowcaseTaskPacket> phase1Packets;

  // Naive draw order (back-to-front / unsorted worst-case overdraw order) & Legacy CPU-sorted opaque/cutout list
  std::vector<uint32_t> naiveDrawOrder;
  std::vector<uint32_t> legacyOpaqueDrawList;
  std::vector<uint32_t> legacyCutoutDrawList;

  Vec3 boundsMin{-1500.0f, -50.0f, -1000.0f};
  Vec3 boundsMax{1500.0f, 1300.0f, 1000.0f};
  Vec3 sunDir{0.28f, 0.92f, 0.26f}; // Normalized toward sun
  uint32_t sponzaSourceTriangles{0};
  uint32_t tessellatedBasePatches{0};
  uint32_t totalMeshletTrianglesLod0{0};
  std::string sceneSourceLabel;

  std::vector<CameraPreset> presets;
  CameraState evalTourCamera(float timeSec) const;
};

// Shared displacement function matching GLSL showcaseDisplacement() in shaders/showcase_common.glsl
Vec3 evaluateSculptedDisplacement(const Vec3 &localPos, const Vec3 &localNormal, const Vec2 &uv);

// Loads assets/models/sponza.glb (with procedural Atrium fallback if absent) and enriches with
// sculpted tessellatable relief props & dense courtyard/occluded instanced foliage/props.
bool buildShowcaseScene(ShowcaseSceneData &outScene, std::string &outError);

} // namespace gpubench::showcase
