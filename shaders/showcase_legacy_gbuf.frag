#version 460
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_fragment_shader_barycentric : require

#include "showcase_common.glsl"

SHOWCASE_DECLARE_SCENE_SET(0);
SHOWCASE_DECLARE_FRAME_SET(1);

layout(location = 0) in vec3 vWorldPos;
layout(location = 1) in vec3 vWorldNormal;
layout(location = 2) in vec4 vWorldTangent;
layout(location = 3) in vec2 vUV;
layout(location = 4) flat in uint vMatId;
layout(location = 5) flat in uint vFlags;

layout(location = 0) out vec4 outGBufAlbedoAo;   // RGBA16F: rgb=albedo, a=ao
layout(location = 1) out vec4 outGBufNormalRough; // RGBA16F: rgb=worldN, a=roughness
layout(location = 2) out vec4 outGBufWorldMeta;   // RGBA32F: rgb=worldPos, a=metallic (+2 if doubleSided)

void main() {
  ShowcaseMaterial mat = materials[vMatId];
  vec4 baseSample = vec4(1.0);
  if (mat.texIndices.x >= 0) {
    baseSample = texture(uTextures[nonuniformEXT(mat.texIndices.x)], vUV);
  }
  vec4 baseCol = baseSample * mat.baseColorFactor;
  if ((vFlags & SHOWCASE_FLAG_ALPHA_CUTOUT) != 0u && baseCol.a < mat.pbrParams.w) {
    discard;
  }

  ivec2 px = ivec2(gl_FragCoord.xy);
  imageAtomicAdd(uOverdrawImg, px, 1u);

  vec2 dbx = dFdx(gl_BaryCoordEXT.xy);
  vec2 dby = dFdy(gl_BaryCoordEXT.xy);
  float det = abs(dbx.x * dby.y - dbx.y * dby.x);
  float triAreaPx = 0.5 / max(det, 1e-6);
  imageStore(uTriAreaImg, px, vec4(triAreaPx, 0.0, 0.0, 0.0));

  vec3 bary = gl_BaryCoordEXT;
  vec3 fw = max(fwidth(bary), vec3(1e-5));
  float minEdge = min(min(bary.x / fw.x, bary.y / fw.y), bary.z / fw.z);
  uint edgeByte = clamp(uint((1.0 - smoothstep(0.0, 1.35, minEdge)) * 255.0), 0u, 255u);
  uint tag = ((vMatId * 131u + uint(gl_PrimitiveID)) << 8u) | edgeByte;
  imageStore(uDebugTagImg, px, uvec4(tag, 0u, 0u, 0u));

  float metallic = mat.pbrParams.x;
  float roughness = mat.pbrParams.y;
  if (mat.texIndices.z >= 0 && mat.texIndices.z != 2) {
    vec4 mr = texture(uTextures[nonuniformEXT(mat.texIndices.z)], vUV);
    roughness *= mr.g;
    metallic *= mr.b;
  }
  roughness = clamp(roughness, 0.08, 1.0);

  float ao = 1.0;
  if (mat.texIndices.w >= 0 && mat.texIndices.w != 0) {
    ao = texture(uTextures[nonuniformEXT(mat.texIndices.w)], vUV).r;
  }

  vec3 N = normalize(vWorldNormal);
  if (mat.texIndices.y >= 0 && mat.texIndices.y != 1) {
    vec3 mapN = texture(uTextures[nonuniformEXT(mat.texIndices.y)], vUV).xyz * 2.0 - 1.0;
    mapN.xy *= mat.pbrParams.z;
    vec3 T = normalize(vWorldTangent.xyz - N * dot(vWorldTangent.xyz, N));
    vec3 B = cross(N, T) * vWorldTangent.w;
    N = normalize(mat3(T, B, N) * mapN);
  }

  outGBufAlbedoAo = vec4(baseCol.rgb, ao);
  outGBufNormalRough = vec4(N, roughness);
  float packedMetal = ((vFlags & SHOWCASE_FLAG_DOUBLE_SIDED) != 0u) ? (metallic + 2.0) : metallic;
  outGBufWorldMeta = vec4(vWorldPos, packedMetal);
}
