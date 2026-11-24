#version 460
#extension GL_EXT_mesh_shader : require

#include "showcase_common.glsl"

layout(early_fragment_tests) in;

SHOWCASE_DECLARE_SCENE_SET(0);
SHOWCASE_DECLARE_FRAME_SET(1);

layout(location = 3) in vec2 vUV;
layout(location = 4) flat in uint vMatId;
layout(location = 5) flat in uint vFlags;
layout(location = 6) perprimitiveEXT flat in uvec2 vVisData;

layout(location = 0) out uvec2 outVisData;

void main() {
  outVisData = vVisData;
}
