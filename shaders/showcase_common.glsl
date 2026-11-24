#ifndef SHOWCASE_COMMON_GLSL
#define SHOWCASE_COMMON_GLSL

#define SHOWCASE_PI 3.14159265359

#define SHOWCASE_FLAG_TESSELLATED  (1u << 0)
#define SHOWCASE_FLAG_ALPHA_CUTOUT (1u << 1)
#define SHOWCASE_FLAG_OCCLUDER     (1u << 2)
#define SHOWCASE_FLAG_DOUBLE_SIDED (1u << 3)

#define SHOWCASE_MODE_NAIVE  0u
#define SHOWCASE_MODE_LEGACY 1u
#define SHOWCASE_MODE_MODERN 2u

#define SHOWCASE_OVERLAY_BEAUTY   0u
#define SHOWCASE_OVERLAY_OVERDRAW 1u
#define SHOWCASE_OVERLAY_MICROTRI 2u
#define SHOWCASE_OVERLAY_MESHLETS 3u

#define SHOWCASE_OPT_ADAPTIVE_LOD  (1u << 0)
#define SHOWCASE_OPT_FRUSTUM_CONE  (1u << 1)
#define SHOWCASE_OPT_HIZ_OCCLUSION (1u << 2)
#define SHOWCASE_OPT_VIS_BUFFER    (1u << 3)
#define SHOWCASE_OPT_DGC_BINNING   (1u << 4)

struct ShowcaseVertex {
  vec4 posMat;      // xyz = pos, floatBitsToUint(w) = matId
  vec4 normalFlags; // xyz = normal, floatBitsToUint(w) = flags
  vec4 tangent;     // xyz = tangent, w = sign
  vec4 uvPad;       // xy = uv
};

struct ShowcaseMeshlet {
  uint vertexOffset;
  uint triangleOffset;
  uint vertexCount;
  uint triangleCount;
  vec4 boundSphere;
  vec4 coneAxisCutoff;
  uvec4 meta; // x=materialId, y=flags, z=globalIndexBase
};

struct ShowcaseInstance {
  mat4 model;
  vec4 boundSphereWorld;
  uvec4 lodFirstBase; // x=lod0, y=lod1, z=lod2, w=baseFirstIndex
  uvec4 lodCountBase; // x=lod0Count, y=lod1Count, z=lod2Count, w=baseIndexCount
  uvec4 meta;         // x=vertexOffset(int), y=materialId, z=drawOrderScore, w=flags
};

struct ShowcaseMaterial {
  vec4 baseColorFactor;
  vec4 emissiveFactor;
  vec4 pbrParams;   // x=metallic, y=roughness, z=normalScale, w=alphaCutoff
  ivec4 texIndices; // x=baseColor, y=normal, z=metalRough, w=occlusion
  uvec4 flags;      // x=alphaMode, y=archetype, z=flags
};

struct ShowcaseLight {
  vec4 posRadius;
  vec4 colorIntensity;
};

struct ShowcaseTaskPacket {
  uint instanceIdx;
  uint localStart;
  uint count;
  uint phaseMask;
};

struct ShowcaseDgcSeq {
  uint pipelineIndex;
  uint pc0_arch;
  uint pc1_queueCapacity;
  uint pc2_width;
  uint pc3_height;
  uint pc4_optFlags;
  uint pc5_overlayMode;
  uint pc6_mode;
  uint pc7_pad;
  uint dispatchX;
  uint dispatchY;
  uint dispatchZ;
};

struct ShowcaseUniforms {
  mat4 viewProj;
  mat4 prevViewProj;
  mat4 cullViewProj;
  mat4 shadowViewProj;
  vec4 cullFrustumPlanes[6];
  vec4 camPosFov;       // xyz = camPos, w = tanHalfFovY
  vec4 cullCamPos;      // xyz = cullPos, w = cullTanHalfFovY
  vec4 sunDirIntensity; // xyz = sunDir, w = intensity
  uvec4 screenInfo;     // x=width, y=height, z=renderMode, w=overlayMode
  uvec4 countsInfo;     // x=phase0Packets, y=phase1Packets, z=numLights, w=optFlags
  vec4 miscParams;      // x=splitWipeX, y=splitWipeActive, z=tessLevel, w=timeSec
  uvec4 frameMeta;      // x=frameIdx, y=hizMips, z=queueCapacity, w=freezeCull
};

struct ShowcaseGpuStats {
  uint totalMeshlets;
  uint renderedMeshlets;
  uint lodCulled;
  uint frustumCulled;
  uint coneCulled;
  uint hizCulled;
  uint renderedTriangles;
  uint activeDgcBins;
  uint overdrawSum;
  uint subPixelTriCount;
  uint totalShadedPixels;
  uint totalFragments;
  uint paritySqErrQuant;
  uint parityExactPixels;
  uint parityNearPixels;
  uint parityTotalPixels;
};

// Deterministic sculpted relief matching CPU evaluateSculptedDisplacement() in ShowcaseScene.cpp
vec3 showcaseDisplacement(vec3 localPos, vec3 localNormal) {
  float radial = length(localPos.xz);
  float poleAtten = clamp(radial * 0.125, 0.0, 1.0);
  float ang = (radial > 1e-4) ? atan(localPos.z, localPos.x) : 0.0;
  float flutes = cos(16.0 * ang) * poleAtten;
  float rings = sin(localPos.y * 0.28 + radial * 0.18);
  float micro = sin(localPos.x * 0.24) * cos(localPos.z * 0.24);
  float disp = (0.60 * flutes * rings + 0.40 * micro) * 2.2;
  return localPos + normalize(localNormal) * disp;
}

vec3 showcaseDisplacementNormal(vec3 localPos, vec3 localNormal, vec3 localTangent) {
  vec3 N = normalize(localNormal);
  vec3 T = localTangent - N * dot(localTangent, N);
  float tLen = length(T);
  T = (tLen > 1e-4) ? (T / tLen) : normalize(abs(N.y) < 0.9 ? cross(N, vec3(0.0, 1.0, 0.0)) : cross(N, vec3(1.0, 0.0, 0.0)));
  vec3 B = normalize(cross(N, T));
  const float kStep = 0.25;
  vec3 pC = showcaseDisplacement(localPos, N);
  vec3 pU = showcaseDisplacement(localPos + T * kStep, N);
  vec3 pV = showcaseDisplacement(localPos + B * kStep, N);
  vec3 outN = cross(pU - pC, pV - pC);
  if (dot(outN, N) < 0.0) outN = -outN;
  float len = length(outN);
  return (len > 1e-5) ? (outN / len) : N;
}

vec3 showcaseTurboColormap(float t) {
  t = clamp(t, 0.0, 1.0);
  const vec3 c0 = vec3(0.11, 0.22, 0.65);
  const vec3 c1 = vec3(0.08, 0.72, 0.78);
  const vec3 c2 = vec3(0.55, 0.88, 0.20);
  const vec3 c3 = vec3(0.98, 0.56, 0.10);
  const vec3 c4 = vec3(0.96, 0.12, 0.28);
  if (t < 0.25) return mix(c0, c1, t * 4.0);
  if (t < 0.50) return mix(c1, c2, (t - 0.25) * 4.0);
  if (t < 0.75) return mix(c2, c3, (t - 0.50) * 4.0);
  return mix(c3, c4, (t - 0.75) * 4.0);
}

vec3 showcaseHashColor(uint id) {
  uint h = id * 747796405u + 2891336453u;
  h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
  h = (h >> 22u) ^ h;
  return vec3(float((h >> 0u) & 255u), float((h >> 8u) & 255u), float((h >> 16u) & 255u)) / 255.0;
}

vec3 showcaseSkyColor(vec3 rayDir, vec3 sunDir) {
  float horizon = clamp(rayDir.y * 0.5 + 0.5, 0.0, 1.0);
  vec3 zen = vec3(0.18, 0.36, 0.68);
  vec3 hor = vec3(0.76, 0.72, 0.64);
  vec3 sky = mix(hor, zen, pow(clamp(rayDir.y, 0.0, 1.0), 0.55));
  float sunAmt = max(dot(normalize(rayDir), normalize(sunDir)), 0.0);
  sky += vec3(1.0, 0.88, 0.66) * (0.28 * pow(sunAmt, 14.0) + 1.4 * pow(sunAmt, 220.0));
  return sky;
}

vec3 showcaseAcesTonemap(vec3 x) {
  const float a = 2.51;
  const float b = 0.03;
  const float c = 2.43;
  const float d = 0.59;
  const float e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

vec3 showcaseLinearToSrgb(vec3 lin) {
  lin = clamp(lin, 0.0, 1.0);
  vec3 lo = lin * 12.92;
  vec3 hi = 1.055 * pow(max(lin, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
  return mix(hi, lo, lessThan(lin, vec3(0.0031308)));
}

float showcaseDistributionGGX(vec3 N, vec3 H, float roughness) {
  float a = roughness * roughness;
  float a2 = a * a;
  float NdotH = max(dot(N, H), 0.0);
  float NdotH2 = NdotH * NdotH;
  float denom = NdotH2 * (a2 - 1.0) + 1.0;
  return a2 / max(SHOWCASE_PI * denom * denom, 1e-6);
}

float showcaseGeometrySchlickGGX(float NdotV, float roughness) {
  float r = roughness + 1.0;
  float k = (r * r) * 0.125;
  return NdotV / max(NdotV * (1.0 - k) + k, 1e-6);
}

float showcaseGeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
  return showcaseGeometrySchlickGGX(max(dot(N, V), 0.0), roughness) *
         showcaseGeometrySchlickGGX(max(dot(N, L), 0.0), roughness);
}

vec3 showcaseFresnelSchlick(float cosTheta, vec3 F0) {
  return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

#define SHOWCASE_DECLARE_SCENE_SET(SET_IDX) \
  layout(std430, set = SET_IDX, binding = 0) readonly buffer ShowcaseVerticesBuf { ShowcaseVertex vertices[]; }; \
  layout(std430, set = SET_IDX, binding = 1) readonly buffer ShowcaseMeshletsBuf { ShowcaseMeshlet meshlets[]; }; \
  layout(std430, set = SET_IDX, binding = 2) readonly buffer ShowcaseMeshletVertsBuf { uint meshletVertices[]; }; \
  layout(std430, set = SET_IDX, binding = 3) readonly buffer ShowcaseMeshletTrisBuf { uint meshletTriangles[]; }; \
  layout(std430, set = SET_IDX, binding = 4) readonly buffer ShowcaseInstancesBuf { ShowcaseInstance instances[]; }; \
  layout(std430, set = SET_IDX, binding = 5) readonly buffer ShowcaseMaterialsBuf { ShowcaseMaterial materials[]; }; \
  layout(std430, set = SET_IDX, binding = 6) readonly buffer ShowcaseLightsBuf { ShowcaseLight lights[]; }; \
  layout(std430, set = SET_IDX, binding = 7) readonly buffer ShowcaseTaskPackets0Buf { ShowcaseTaskPacket taskPackets0[]; }; \
  layout(std430, set = SET_IDX, binding = 8) readonly buffer ShowcaseTaskPackets1Buf { ShowcaseTaskPacket taskPackets1[]; }; \
  layout(set = SET_IDX, binding = 9) uniform sampler2D uTextures[192]

#define SHOWCASE_DECLARE_FRAME_SET(SET_IDX) \
  layout(std140, set = SET_IDX, binding = 0) uniform ShowcaseUbo { ShowcaseUniforms ubo; }; \
  layout(std430, set = SET_IDX, binding = 1) buffer ShowcaseStatsBuf { ShowcaseGpuStats stats; }; \
  layout(set = SET_IDX, binding = 2) uniform sampler2D uShadowMap; \
  layout(set = SET_IDX, binding = 3) uniform sampler2D uHiZTex; \
  layout(set = SET_IDX, binding = 4, r32ui) uniform uimage2D uOverdrawImg; \
  layout(set = SET_IDX, binding = 5, r32f) uniform image2D uTriAreaImg; \
  layout(set = SET_IDX, binding = 6, rg32ui) uniform uimage2D uVisBufferImg; \
  layout(set = SET_IDX, binding = 7) uniform sampler2D uDepthTex; \
  layout(set = SET_IDX, binding = 8) uniform sampler2D uGBufAlbedo; \
  layout(set = SET_IDX, binding = 9) uniform sampler2D uGBufNormal; \
  layout(set = SET_IDX, binding = 10) uniform sampler2D uGBufWorldPos; \
  layout(set = SET_IDX, binding = 11, rgba8) uniform image2D uSceneColorImg; \
  layout(set = SET_IDX, binding = 12) uniform sampler2D uSceneColorTex; \
  layout(set = SET_IDX, binding = 13, rgba8) uniform image2D uDisplayImg; \
  layout(set = SET_IDX, binding = 14) uniform sampler2D uAltColorTex; \
  layout(std430, set = SET_IDX, binding = 15) buffer ShowcaseWorklistBuf { uint worklistCounters[16]; uint worklistData[]; }; \
  layout(std430, set = SET_IDX, binding = 16) buffer ShowcaseDgcBuf { uint dgcCount; uint dgcPad[3]; ShowcaseDgcSeq dgcSeqs[8]; }; \
  layout(set = SET_IDX, binding = 17, r32ui) uniform uimage2D uDebugTagImg; \
  layout(std430, set = SET_IDX, binding = 18) buffer ShowcaseFallbackBuf { uint fallbackCount; uint fallbackPad[3]; uvec4 fallbackCmds[]; }

#endif
