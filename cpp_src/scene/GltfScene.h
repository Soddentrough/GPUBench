#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <array>

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

struct GltfTexture {
  int width{0};
  int height{0};
  int channels{4};
  std::vector<uint8_t> pixels; // Raw 32-bit RGBA8 pixels
};

struct GltfTextureHeader {
  uint32_t width{0};
  uint32_t height{0};
  uint32_t offset{0}; // Element offset in packedPixels
  uint32_t pad{0};
};

struct GltfCameraParams {
  float pos[3]{0.0f, 0.0f, 0.0f};
  float forward[3]{0.0f, 1.0f, 0.0f};
  float right[3]{1.0f, 0.0f, 0.0f};
  float up[3]{0.0f, 0.0f, 1.0f};
  float fov{0.65f};
};

class GltfScene {
public:
  GltfScene() = default;
  ~GltfScene() = default;

  bool loadFromFile(const std::string& filepath, std::string& outError);

  const std::vector<GltfVertex>& getVertices() const { return vertices_; }
  const std::vector<uint32_t>& getIndices() const { return indices_; }
  std::vector<GltfVertex> getUnrolledVertices() const;
  const std::vector<uint32_t>& getTriangleMaterials() const { return triangleMaterials_; }
  const std::vector<GltfMaterial>& getMaterials() const { return materials_; }
  const std::vector<GltfTexture>& getTextures() const { return textures_; }
  const std::vector<GltfTextureHeader>& getTextureHeaders() const { return textureHeaders_; }
  const std::vector<uint32_t>& getPackedPixels() const { return packedPixels_; }

  uint32_t getTriangleCount() const { return static_cast<uint32_t>(indices_.size() / 3); }
  uint32_t getVertexCount() const { return static_cast<uint32_t>(vertices_.size()); }

  void getBounds(float outMin[3], float outMax[3]) const {
    for (int i = 0; i < 3; ++i) {
      outMin[i] = minBound_[i];
      outMax[i] = maxBound_[i];
    }
  }

  GltfCameraParams getRecommendedCamera() const;

private:
  void initDefaultTextures();

  std::vector<GltfVertex> vertices_;
  std::vector<uint32_t> indices_;
  std::vector<uint32_t> triangleMaterials_; // Per-triangle material index
  std::vector<GltfMaterial> materials_;
  std::vector<GltfTexture> textures_;
  std::vector<GltfTextureHeader> textureHeaders_;
  std::vector<uint32_t> packedPixels_;

  float minBound_[3]{1e9f, 1e9f, 1e9f};
  float maxBound_[3]{-1e9f, -1e9f, -1e9f};
  std::string filename_;
};
