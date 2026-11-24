#version 460

#include "showcase_common.glsl"

SHOWCASE_DECLARE_SCENE_SET(0);
SHOWCASE_DECLARE_FRAME_SET(1);

layout(location = 0) in vec2 vNdc;
layout(location = 0) out vec4 outColor;

void main() {
  mat4 invVP = inverse(ubo.viewProj);
  vec4 wpFar = invVP * vec4(vNdc, 0.001, 1.0);
  vec3 rayDir = normalize(wpFar.xyz / wpFar.w - ubo.camPosFov.xyz);
  vec3 sky = showcaseLinearToSrgb(showcaseAcesTonemap(showcaseSkyColor(rayDir, ubo.sunDirIntensity.xyz)));
  outColor = vec4(sky, 1.0);
}
