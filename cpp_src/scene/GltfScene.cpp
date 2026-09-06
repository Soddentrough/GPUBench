#define CGLTF_IMPLEMENTATION
#include "third_party/cgltf.h"

#define STB_IMAGE_IMPLEMENTATION
#include "third_party/stb_image.h"

#include "scene/GltfScene.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cstring>

void GltfScene::initDefaultTextures() {
  textures_.clear();

  // Texture 0: 1x1 RGBA8 White (BaseColor / AO / Emissive fallback)
  GltfTexture whiteTex;
  whiteTex.width = 1;
  whiteTex.height = 1;
  whiteTex.channels = 4;
  whiteTex.pixels = {255, 255, 255, 255};
  textures_.push_back(std::move(whiteTex));

  // Texture 1: 1x1 RGBA8 Flat Normal (Tangent space (0, 0, 1) -> (128, 128, 255))
  GltfTexture normalTex;
  normalTex.width = 1;
  normalTex.height = 1;
  normalTex.channels = 4;
  normalTex.pixels = {128, 128, 255, 255};
  textures_.push_back(std::move(normalTex));

  // Texture 2: 1x1 RGBA8 Metal/Rough (Roughness 1.0, Metallic 0.0 -> G=255, B=0)
  GltfTexture mrTex;
  mrTex.width = 1;
  mrTex.height = 1;
  mrTex.channels = 4;
  mrTex.pixels = {0, 255, 0, 255};
  textures_.push_back(std::move(mrTex));
}

bool GltfScene::loadFromFile(const std::string& filepath, std::string& outError) {
  filename_ = filepath;
  vertices_.clear();
  indices_.clear();
  triangleMaterials_.clear();
  materials_.clear();

  initDefaultTextures();

  minBound_[0] = minBound_[1] = minBound_[2] = 1e9f;
  maxBound_[0] = maxBound_[1] = maxBound_[2] = -1e9f;

  cgltf_options options{};
  cgltf_data* data = nullptr;

  cgltf_result result = cgltf_parse_file(&options, filepath.c_str(), &data);
  if (result != cgltf_result_success) {
    outError = "Failed to parse glTF file: " + filepath + " (code: " + std::to_string(result) + ")";
    return false;
  }

  result = cgltf_load_buffers(&options, data, filepath.c_str());
  if (result != cgltf_result_success) {
    outError = "Failed to load glTF buffers: " + filepath + " (code: " + std::to_string(result) + ")";
    cgltf_free(data);
    return false;
  }

  // 1. Decode Textures
  std::vector<int32_t> imageToTextureMap(data->images_count, -1);
  for (size_t i = 0; i < data->images_count; ++i) {
    const auto& img = data->images[i];
    int w = 0, h = 0, comp = 0;
    uint8_t* pixels = nullptr;

    if (img.buffer_view && img.buffer_view->buffer && img.buffer_view->buffer->data) {
      const uint8_t* buf = static_cast<const uint8_t*>(img.buffer_view->buffer->data);
      const uint8_t* srcBytes = buf + img.buffer_view->offset;
      size_t srcLen = img.buffer_view->size;
      pixels = stbi_load_from_memory(srcBytes, static_cast<int>(srcLen), &w, &h, &comp, 4);
    } else if (img.uri) {
      // Find relative to glTF directory
      std::string dir;
      size_t lastSlash = filepath.find_last_of("/\\");
      if (lastSlash != std::string::npos) {
        dir = filepath.substr(0, lastSlash + 1);
      }
      std::string fullUri = dir + img.uri;
      pixels = stbi_load(fullUri.c_str(), &w, &h, &comp, 4);
    }

    if (pixels) {
      GltfTexture tex;
      tex.width = w;
      tex.height = h;
      tex.channels = 4;
      tex.pixels.assign(pixels, pixels + (w * h * 4));
      stbi_image_free(pixels);

      imageToTextureMap[i] = static_cast<int32_t>(textures_.size());
      textures_.push_back(std::move(tex));
    }
  }

  // 1b. Build texture headers and packed pixels
  textureHeaders_.clear();
  packedPixels_.clear();
  for (size_t t = 0; t < textures_.size(); ++t) {
    const auto& tex = textures_[t];
    GltfTextureHeader hdr;
    hdr.width = static_cast<uint32_t>(tex.width);
    hdr.height = static_cast<uint32_t>(tex.height);
    hdr.offset = static_cast<uint32_t>(packedPixels_.size());
    hdr.pad = 0;
    textureHeaders_.push_back(hdr);

    const uint32_t* p32 = reinterpret_cast<const uint32_t*>(tex.pixels.data());
    size_t pixelCount = static_cast<size_t>(tex.width * tex.height);
    packedPixels_.insert(packedPixels_.end(), p32, p32 + pixelCount);
  }

  // Helper lambda to find texture index from a cgltf_texture_view
  auto getTextureIndex = [&](const cgltf_texture_view& view, int32_t defaultIdx) -> int32_t {
    if (view.texture && view.texture->image) {
      size_t imgIdx = view.texture->image - data->images;
      if (imgIdx < imageToTextureMap.size() && imageToTextureMap[imgIdx] != -1) {
        return imageToTextureMap[imgIdx];
      }
    }
    return defaultIdx;
  };

  // 2. Decode Materials
  if (data->materials_count == 0) {
    // Default material
    GltfMaterial defaultMat;
    materials_.push_back(defaultMat);
  } else {
    for (size_t m = 0; m < data->materials_count; ++m) {
      const auto& srcMat = data->materials[m];
      GltfMaterial mat;

      if (srcMat.has_pbr_metallic_roughness) {
        const auto& pbr = srcMat.pbr_metallic_roughness;
        for (int c = 0; c < 4; ++c) mat.baseColorFactor[c] = pbr.base_color_factor[c];
        mat.metallicFactor = pbr.metallic_factor;
        mat.roughnessFactor = pbr.roughness_factor;
        mat.baseColorTexIdx = getTextureIndex(pbr.base_color_texture, -1);
        mat.metallicRoughnessTexIdx = getTextureIndex(pbr.metallic_roughness_texture, -1);
      }

      mat.normalTexIdx = getTextureIndex(srcMat.normal_texture, -1);
      mat.normalScale = srcMat.normal_texture.scale;

      mat.occlusionTexIdx = getTextureIndex(srcMat.occlusion_texture, -1);

      if (srcMat.has_emissive_strength) {
        for (int c = 0; c < 3; ++c) {
          mat.emissiveFactor[c] = srcMat.emissive_factor[c] * srcMat.emissive_strength.emissive_strength;
        }
      } else {
        for (int c = 0; c < 3; ++c) mat.emissiveFactor[c] = srcMat.emissive_factor[c];
      }
      mat.emissiveTexIdx = getTextureIndex(srcMat.emissive_texture, -1);

      mat.alphaMode = static_cast<uint32_t>(srcMat.alpha_mode);
      mat.alphaCutoff = srcMat.alpha_cutoff;

      if (srcMat.has_transmission) {
        mat.transmissionFactor = srcMat.transmission.transmission_factor;
      }
      if (srcMat.has_ior) {
        mat.ior = srcMat.ior.ior;
      }

      // Map glTF materials to the 8 production AAA BSDF archetypes
      std::string matName = srcMat.name ? srcMat.name : "";
      std::transform(matName.begin(), matName.end(), matName.begin(), ::tolower);

      if (mat.transmissionFactor > 0.05f || matName.find("glass") != std::string::npos || matName.find("window") != std::string::npos) {
        mat.archetype = 2; // Dielectric Transmission / Glass
      } else if (matName.find("fabric") != std::string::npos || matName.find("curtain") != std::string::npos || matName.find("banner") != std::string::npos || matName.find("cloth") != std::string::npos) {
        mat.archetype = 3; // Anisotropic Velvet / Fabric Sheen
      } else if (mat.alphaMode == 1 || matName.find("leaf") != std::string::npos || matName.find("leaves") != std::string::npos || matName.find("plant") != std::string::npos || matName.find("ivy") != std::string::npos || matName.find("flora") != std::string::npos) {
        mat.archetype = 7; // Alpha-Tested Foliage / Cutouts
      } else if (matName.find("lion") != std::string::npos || matName.find("statue") != std::string::npos || matName.find("column") != std::string::npos || matName.find("pillar") != std::string::npos || matName.find("arch") != std::string::npos) {
        mat.archetype = 1; // Subsurface Scattering (Marble / Stone)
      } else if (matName.find("paint") != std::string::npos || matName.find("body") != std::string::npos || matName.find("hood") != std::string::npos || matName.find("car") != std::string::npos) {
        mat.archetype = 6; // Clearcoat Automotive Paint
      } else if (matName.find("floor") != std::string::npos || matName.find("terrazzo") != std::string::npos || matName.find("ground") != std::string::npos || matName.find("pavement") != std::string::npos) {
        mat.archetype = 5; // Polished Architectural Stone / Terrazzo
      } else if (matName.find("wall") != std::string::npos || matName.find("rust") != std::string::npos || matName.find("roof") != std::string::npos || matName.find("brick") != std::string::npos || matName.find("wood") != std::string::npos) {
        mat.archetype = 4; // Weathered Multi-Layered Rust / Stone
      } else if (mat.metallicFactor > 0.5f || matName.find("metal") != std::string::npos || matName.find("brass") != std::string::npos || matName.find("gold") != std::string::npos || matName.find("chain") != std::string::npos) {
        mat.archetype = 0; // Standard Conductor PBR
      } else {
        mat.archetype = static_cast<uint32_t>(m % 8); // Spread unclassified materials across 0..7
      }

      materials_.push_back(mat);
    }
  }

  // 3. Extract Mesh Primitives & Geometry
  for (size_t m = 0; m < data->meshes_count; ++m) {
    const auto& mesh = data->meshes[m];
    for (size_t p = 0; p < mesh.primitives_count; ++p) {
      const auto& prim = mesh.primitives[p];
      if (prim.type != cgltf_primitive_type_triangles) continue;

      const cgltf_accessor* posAcc = nullptr;
      const cgltf_accessor* normAcc = nullptr;
      const cgltf_accessor* tanAcc = nullptr;
      const cgltf_accessor* uvAcc = nullptr;

      for (size_t a = 0; a < prim.attributes_count; ++a) {
        const auto& attr = prim.attributes[a];
        if (attr.type == cgltf_attribute_type_position) posAcc = attr.data;
        else if (attr.type == cgltf_attribute_type_normal) normAcc = attr.data;
        else if (attr.type == cgltf_attribute_type_tangent) tanAcc = attr.data;
        else if (attr.type == cgltf_attribute_type_texcoord) uvAcc = attr.data;
      }

      if (!posAcc || posAcc->count == 0) continue;

      uint32_t vertexOffset = static_cast<uint32_t>(vertices_.size());
      size_t vCount = posAcc->count;
      vertices_.resize(vertexOffset + vCount);

      for (size_t i = 0; i < vCount; ++i) {
        GltfVertex& v = vertices_[vertexOffset + i];
        cgltf_accessor_read_float(posAcc, i, v.pos, 3);
        for (int c = 0; c < 3; ++c) {
          minBound_[c] = std::min(minBound_[c], v.pos[c]);
          maxBound_[c] = std::max(maxBound_[c], v.pos[c]);
        }

        if (normAcc) {
          cgltf_accessor_read_float(normAcc, i, v.normal, 3);
        } else {
          v.normal[0] = 0.0f; v.normal[1] = 1.0f; v.normal[2] = 0.0f;
        }

        if (tanAcc) {
          cgltf_accessor_read_float(tanAcc, i, v.tangent, 4);
        } else {
          v.tangent[0] = 1.0f; v.tangent[1] = 0.0f; v.tangent[2] = 0.0f; v.tangent[3] = 1.0f;
        }

        if (uvAcc) {
          cgltf_accessor_read_float(uvAcc, i, v.uv, 2);
        } else {
          v.uv[0] = 0.0f; v.uv[1] = 0.0f;
        }
      }

      uint32_t matId = 0;
      if (prim.material) {
        matId = static_cast<uint32_t>(prim.material - data->materials);
      }

      if (prim.indices) {
        size_t idxCount = prim.indices->count;
        for (size_t i = 0; i < idxCount; ++i) {
          uint32_t idx = static_cast<uint32_t>(cgltf_accessor_read_index(prim.indices, i));
          indices_.push_back(vertexOffset + idx);
        }
        for (size_t t = 0; t < idxCount / 3; ++t) {
          triangleMaterials_.push_back(matId);
        }
      } else {
        for (size_t i = 0; i < vCount; ++i) {
          indices_.push_back(vertexOffset + static_cast<uint32_t>(i));
        }
        for (size_t t = 0; t < vCount / 3; ++t) {
          triangleMaterials_.push_back(matId);
        }
      }
    }
  }

  cgltf_free(data);
  return true;
}

std::vector<GltfVertex> GltfScene::getUnrolledVertices() const {
  std::vector<GltfVertex> unrolled;
  unrolled.reserve(indices_.size());
  for (uint32_t idx : indices_) {
    unrolled.push_back(vertices_[idx]);
  }
  return unrolled;
}

GltfCameraParams GltfScene::getRecommendedCamera() const {
  GltfCameraParams cam;

  std::string lowerName = filename_;
  std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), ::tolower);

  if (lowerName.find("sponza") != std::string::npos) {
    // Sponza: Colonnade longitudinal perspective
    // Looking down the center nave towards the arches
    cam.pos[0] = -1150.0f;
    cam.pos[1] = 220.0f;
    cam.pos[2] = -40.0f;

    cam.forward[0] = 1.0f;
    cam.forward[1] = 0.02f;
    cam.forward[2] = 0.0f;
    float flen = std::sqrt(cam.forward[0]*cam.forward[0] + cam.forward[1]*cam.forward[1] + cam.forward[2]*cam.forward[2]);
    cam.forward[0] /= flen; cam.forward[1] /= flen; cam.forward[2] /= flen;

    cam.right[0] = 0.0f;
    cam.right[1] = 0.0f;
    cam.right[2] = -1.0f;

    cam.up[0] = 0.0f;
    cam.up[1] = 1.0f;
    cam.up[2] = 0.0f;

    cam.fov = 0.55f;
  } else if (lowerName.find("car") != std::string::npos || lowerName.find("toy") != std::string::npos) {
    // Automotive Beauty 3/4 Turntable Angle
    cam.pos[0] = 480.0f;
    cam.pos[1] = 420.0f;
    cam.pos[2] = 260.0f;

    float target[3] = {0.0f, 0.0f, 40.0f};
    cam.forward[0] = target[0] - cam.pos[0];
    cam.forward[1] = target[1] - cam.pos[1];
    cam.forward[2] = target[2] - cam.pos[2];
    float flen = std::sqrt(cam.forward[0]*cam.forward[0] + cam.forward[1]*cam.forward[1] + cam.forward[2]*cam.forward[2]);
    cam.forward[0] /= flen; cam.forward[1] /= flen; cam.forward[2] /= flen;

    // Up is (0, 0, 1) in ToyCar coordinates
    cam.right[0] = -cam.forward[1];
    cam.right[1] = cam.forward[0];
    cam.right[2] = 0.0f;
    float rlen = std::sqrt(cam.right[0]*cam.right[0] + cam.right[1]*cam.right[1]);
    cam.right[0] /= rlen; cam.right[1] /= rlen;

    cam.up[0] = cam.right[1]*cam.forward[2] - cam.right[2]*cam.forward[1];
    cam.up[1] = cam.right[2]*cam.forward[0] - cam.right[0]*cam.forward[2];
    cam.up[2] = cam.right[0]*cam.forward[1] - cam.right[1]*cam.forward[0];

    cam.fov = 0.45f;
  } else {
    // Generic bounding-box framing
    float center[3] = {
      (minBound_[0] + maxBound_[0]) * 0.5f,
      (minBound_[1] + maxBound_[1]) * 0.5f,
      (minBound_[2] + maxBound_[2]) * 0.5f
    };
    float dim[3] = {
      maxBound_[0] - minBound_[0],
      maxBound_[1] - minBound_[1],
      maxBound_[2] - minBound_[2]
    };
    float maxDim = std::max(dim[0], std::max(dim[1], dim[2]));

    cam.pos[0] = center[0];
    cam.pos[1] = center[1] + maxDim * 1.5f;
    cam.pos[2] = center[2] + maxDim * 0.5f;

    cam.forward[0] = 0.0f;
    cam.forward[1] = -0.9f;
    cam.forward[2] = -0.4f;
    float flen = std::sqrt(cam.forward[1]*cam.forward[1] + cam.forward[2]*cam.forward[2]);
    cam.forward[1] /= flen; cam.forward[2] /= flen;

    cam.right[0] = 1.0f; cam.right[1] = 0.0f; cam.right[2] = 0.0f;
    cam.up[0] = 0.0f; cam.up[1] = 0.4f; cam.up[2] = 0.9f;
    cam.fov = 0.60f;
  }

  return cam;
}
