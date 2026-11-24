#version 460

#include "showcase_common.glsl"

SHOWCASE_DECLARE_SCENE_SET(0);
SHOWCASE_DECLARE_FRAME_SET(1);

void main() {
  // 12 edges = 24 line vertices of NDC box [-1,1] x [-1,1] x [far=0.002, near=0.998] in Reversed-Z
  const vec3 corners[8] = vec3[8](
      vec3(-1.0, -1.0, 0.998), vec3(1.0, -1.0, 0.998),
      vec3( 1.0,  1.0, 0.998), vec3(-1.0, 1.0, 0.998),
      vec3(-1.0, -1.0, 0.002), vec3(1.0, -1.0, 0.002),
      vec3( 1.0,  1.0, 0.002), vec3(-1.0, 1.0, 0.002));
  const int edges[24] = int[24](
      0, 1, 1, 2, 2, 3, 3, 0, // Near rect
      4, 5, 5, 6, 6, 7, 7, 4, // Far rect
      0, 4, 1, 5, 2, 6, 3, 7  // Sides
  );

  vec3 ndcCorner = corners[edges[gl_VertexIndex % 24]];
  mat4 invCullVP = inverse(ubo.cullViewProj);
  vec4 wp = invCullVP * vec4(ndcCorner, 1.0);
  wp /= wp.w;
  gl_Position = ubo.viewProj * wp;
}
