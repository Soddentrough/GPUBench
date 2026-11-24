#version 460
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_nonuniform_qualifier : require

#include "showcase_common.glsl"

SHOWCASE_DECLARE_SCENE_SET(0);
SHOWCASE_DECLARE_FRAME_SET(1);

layout(location = 3) in vec2 vUV;
layout(location = 4) flat in uint vMatId;
layout(location = 5) flat in uint vFlags;
layout(location = 6) perprimitiveEXT flat in uvec2 vVisData;

layout(location = 0) out uvec2 outVisData;

void main() {
  if ((vFlags & SHOWCASE_FLAG_ALPHA_CUTOUT) != 0u) {
    ShowcaseMaterial mat = materials[vMatId];
    float alpha = mat.baseColorFactor.a;
    if (mat.texIndices.x >= 0) {
      alpha *= texture(uTextures[nonuniformEXT(mat.texIndices.x)], vUV).a;
    }
    if (alpha < mat.pbrParams.w) {
      discard;
    }
  }

  outVisData = vVisData;
}
