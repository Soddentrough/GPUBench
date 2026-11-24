#version 460
#extension GL_EXT_nonuniform_qualifier : require

#include "showcase_common.glsl"

SHOWCASE_DECLARE_SCENE_SET(0);
SHOWCASE_DECLARE_FRAME_SET(1);

layout(location = 0) in vec2 vUV;
layout(location = 1) flat in uint vMatId;
layout(location = 2) flat in uint vFlags;
layout(location = 3) flat in uvec2 vMetaIds;

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
  ivec2 px = ivec2(gl_FragCoord.xy);
  imageAtomicAdd(uOverdrawImg, px, 1u);
  outVisData = uvec2((vMetaIds.x << 8u) | (uint(gl_PrimitiveID) & 0xFFu), vMetaIds.y);
}
