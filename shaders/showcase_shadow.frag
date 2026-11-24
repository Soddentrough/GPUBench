#version 460
#extension GL_EXT_nonuniform_qualifier : require

#include "showcase_common.glsl"

SHOWCASE_DECLARE_SCENE_SET(0);
SHOWCASE_DECLARE_FRAME_SET(1);

layout(location = 0) in vec2 vUV;
layout(location = 1) flat in uint vMatId;
layout(location = 2) flat in uint vFlags;

void main() {
  if ((vFlags & SHOWCASE_FLAG_ALPHA_CUTOUT) != 0u) {
    uint matIdx = vMatId;
    ShowcaseMaterial mat = materials[matIdx];
    int texIdx = mat.texIndices.x;
    float alpha = mat.baseColorFactor.a;
    if (texIdx >= 0) {
      alpha *= texture(uTextures[nonuniformEXT(texIdx)], vUV).a;
    }
    if (alpha < mat.pbrParams.w) {
      discard;
    }
  }
}
