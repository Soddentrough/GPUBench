#version 460
#extension GL_EXT_nonuniform_qualifier : require

#include "showcase_common.glsl"

layout(triangles, fractional_even_spacing, ccw) in;

SHOWCASE_DECLARE_SCENE_SET(0);
SHOWCASE_DECLARE_FRAME_SET(1);

layout(location = 0) in vec3 tcLocalPos[];
layout(location = 1) in vec3 tcLocalNormal[];
layout(location = 2) in vec4 tcLocalTangent[];
layout(location = 3) in vec2 tcUV[];
layout(location = 4) flat in uint tcMatId[];
layout(location = 5) flat in uint tcFlags[];
layout(location = 6) flat in uint tcInstanceIdx[];

layout(location = 0) out vec3 vWorldPos;
layout(location = 1) out vec3 vWorldNormal;
layout(location = 2) out vec4 vWorldTangent;
layout(location = 3) out vec2 vUV;
layout(location = 4) flat out uint vMatId;
layout(location = 5) flat out uint vFlags;

void main() {
  vec3 b = gl_TessCoord;
  vec3 localPos = tcLocalPos[0] * b.x + tcLocalPos[1] * b.y + tcLocalPos[2] * b.z;
  vec3 localNrm = normalize(tcLocalNormal[0] * b.x + tcLocalNormal[1] * b.y + tcLocalNormal[2] * b.z);
  vec4 localTan = tcLocalTangent[0] * b.x + tcLocalTangent[1] * b.y + tcLocalTangent[2] * b.z;
  vec2 uv = tcUV[0] * b.x + tcUV[1] * b.y + tcUV[2] * b.z;

  vec3 dispPos = showcaseDisplacement(localPos, localNrm);
  vec3 dispNrm = showcaseDisplacementNormal(localPos, localNrm, localTan.xyz);

  ShowcaseInstance inst = instances[tcInstanceIdx[0]];
  mat3 nrmMat = mat3(inst.model);
  vec4 wp = inst.model * vec4(dispPos, 1.0);

  vWorldPos = wp.xyz;
  vWorldNormal = normalize(nrmMat * dispNrm);
  vWorldTangent = vec4(normalize(nrmMat * localTan.xyz), localTan.w >= 0.0 ? 1.0 : -1.0);
  vUV = uv;
  vMatId = tcMatId[0];
  vFlags = tcFlags[0];

  gl_Position = ubo.viewProj * wp;
}
