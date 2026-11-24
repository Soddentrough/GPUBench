#version 460

#include "showcase_common.glsl"

layout(vertices = 3) out;

SHOWCASE_DECLARE_SCENE_SET(0);
SHOWCASE_DECLARE_FRAME_SET(1);

layout(location = 0) in vec3 vLocalPos[];
layout(location = 1) in vec3 vLocalNormal[];
layout(location = 2) in vec4 vLocalTangent[];
layout(location = 3) in vec2 vUV[];
layout(location = 4) flat in uint vMatId[];
layout(location = 5) flat in uint vFlags[];
layout(location = 6) flat in uint vInstanceIdx[];

layout(location = 0) out vec3 tcLocalPos[3];
layout(location = 1) out vec3 tcLocalNormal[3];
layout(location = 2) out vec4 tcLocalTangent[3];
layout(location = 3) out vec2 tcUV[3];
layout(location = 4) flat out uint tcMatId[3];
layout(location = 5) flat out uint tcFlags[3];
layout(location = 6) flat out uint tcInstanceIdx[3];

void main() {
  tcLocalPos[gl_InvocationID] = vLocalPos[gl_InvocationID];
  tcLocalNormal[gl_InvocationID] = vLocalNormal[gl_InvocationID];
  tcLocalTangent[gl_InvocationID] = vLocalTangent[gl_InvocationID];
  tcUV[gl_InvocationID] = vUV[gl_InvocationID];
  tcMatId[gl_InvocationID] = vMatId[gl_InvocationID];
  tcFlags[gl_InvocationID] = vFlags[gl_InvocationID];
  tcInstanceIdx[gl_InvocationID] = vInstanceIdx[gl_InvocationID];
  gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;

  if (gl_InvocationID == 0) {
    float level = clamp(ubo.miscParams.z, 1.0, 64.0);
    if (ubo.screenInfo.z != SHOWCASE_MODE_NAIVE) {
      // Legacy distance-clamped tessellation (scales down gracefully with distance)
      ShowcaseInstance inst = instances[vInstanceIdx[0]];
      vec3 localCtr = (vLocalPos[0] + vLocalPos[1] + vLocalPos[2]) * (1.0 / 3.0);
      vec3 worldCtr = (inst.model * vec4(localCtr, 1.0)).xyz;
      float dist = length(ubo.camPosFov.xyz - worldCtr);
      level = clamp(8500.0 / max(dist, 80.0), 4.0, 24.0);
    }
    gl_TessLevelOuter[0] = level;
    gl_TessLevelOuter[1] = level;
    gl_TessLevelOuter[2] = level;
    gl_TessLevelInner[0] = level;
  }
}
