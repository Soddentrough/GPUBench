#pragma once

#include <cstdint>

struct GltfVertex {
  float pos[3];
  float normal[3];
  float tangent[4];
  float uv[2];
};

struct alignas(16) GltfMaterial {
  float baseColorFactor[4]{1.0f, 1.0f, 1.0f, 1.0f}; // 16B (offset 0)
  float metallicFactor{1.0f};                        // 4B  (offset 16)
  float roughnessFactor{1.0f};                       // 4B  (offset 20)
  float transmissionFactor{0.0f};                    // 4B  (offset 24)
  float ior{1.5f};                                   // 4B  (offset 28)
  float emissiveFactor[4]{0.0f, 0.0f, 0.0f, 0.0f};   // 16B (offset 32)
  int32_t baseColorTexIdx{-1};                       // 4B  (offset 48)
  int32_t metallicRoughnessTexIdx{-1};               // 4B  (offset 52)
  int32_t normalTexIdx{-1};                          // 4B  (offset 56)
  int32_t occlusionTexIdx{-1};                       // 4B  (offset 60)
  int32_t emissiveTexIdx{-1};                        // 4B  (offset 64)
  float normalScale{1.0f};                           // 4B  (offset 68)
  float alphaCutoff{0.5f};                           // 4B  (offset 72)
  uint32_t alphaMode{0};                             // 4B  (offset 76)
  uint32_t archetype{0};                             // 4B  (offset 80)
  uint32_t extraParam1{0};                           // 4B  (offset 84)
  uint32_t extraParam2{0};                           // 4B  (offset 88)
  uint32_t pad1{0};                                  // 4B  (offset 92)
};

struct GltfTextureHeader {
  uint32_t width{0};
  uint32_t height{0};
  uint32_t offset{0}; // Element offset in packedPixels
  uint32_t pad{0};
};
