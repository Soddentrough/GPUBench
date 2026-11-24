#ifndef SHOWCASE_PBR_GLSL
#define SHOWCASE_PBR_GLSL

#include "showcase_common.glsl"

float sampleShowcaseShadowPcf(vec3 worldPos, vec3 geomNormal, vec3 sunDir) {
  float nDotL = clamp(dot(geomNormal, sunDir), 0.0, 1.0);
  float bias = mix(0.0028, 0.0007, nDotL);
  vec4 sc = ubo.shadowViewProj * vec4(worldPos + geomNormal * 2.5, 1.0);
  vec3 ndc = sc.xyz / max(sc.w, 1e-5);
  vec2 uv = ndc.xy * 0.5 + 0.5;
  if (uv.x < 0.002 || uv.x > 0.998 || uv.y < 0.002 || uv.y > 0.998 || ndc.z <= 0.0 || ndc.z >= 1.0) {
    return 1.0;
  }
  vec2 texel = vec2(1.0 / 2048.0);
  float vis = 0.0;
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      float mapDepth = textureLod(uShadowMap, uv + vec2(dx, dy) * texel, 0.0).r;
      vis += ((ndc.z - bias) <= mapDepth) ? 1.0 : 0.0;
    }
  }
  return vis * (1.0 / 9.0);
}

vec3 evaluateShowcasePbr(vec3 worldPos, vec3 N, vec3 albedo, float metallic, float roughness,
                         float ao, vec3 emissive, uint flags) {
  vec3 V = normalize(ubo.camPosFov.xyz - worldPos);
  if ((flags & SHOWCASE_FLAG_DOUBLE_SIDED) != 0u && dot(N, V) < 0.0) {
    N = -N;
  }
  float NdotV = max(dot(N, V), 0.001);
  vec3 F0 = mix(vec3(0.04), albedo, metallic);

  vec3 Lo = vec3(0.0);

  // 1. Directional Courtyard Sun + 3x3 PCF Shadows
  vec3 sunL = normalize(ubo.sunDirIntensity.xyz);
  float sunNdotL = max(dot(N, sunL), 0.0);
  if (sunNdotL > 0.0) {
    float shadow = sampleShowcaseShadowPcf(worldPos, N, sunL);
    vec3 H = normalize(V + sunL);
    float D = showcaseDistributionGGX(N, H, roughness);
    float G = showcaseGeometrySmith(N, V, sunL, roughness);
    vec3 F = showcaseFresnelSchlick(max(dot(H, V), 0.0), F0);
    vec3 spec = (D * G * F) / max(4.0 * NdotV * sunNdotL, 0.001);
    vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);
    vec3 sunRadiance = vec3(1.0, 0.93, 0.80) * ubo.sunDirIntensity.w;
    Lo += (kD * albedo / SHOWCASE_PI + spec) * sunRadiance * sunNdotL * shadow;
  }

  // 2. Warm Atrium Brazier / Torch Point Lights
  uint numLights = min(ubo.countsInfo.z, 16u);
  for (uint i = 0u; i < numLights; ++i) {
    ShowcaseLight light = lights[i];
    vec3 lVec = light.posRadius.xyz - worldPos;
    float dist = length(lVec);
    float rad = max(light.posRadius.w, 1.0);
    if (dist < rad) {
      vec3 L = lVec / max(dist, 0.001);
      float NdotL = max(dot(N, L), 0.0);
      if (NdotL > 0.0) {
        float t = clamp(1.0 - dist / rad, 0.0, 1.0);
        float atten = (t * t) / (1.0 + 0.00018 * dist * dist);
        vec3 H = normalize(V + L);
        float D = showcaseDistributionGGX(N, H, roughness);
        float G = showcaseGeometrySmith(N, V, L, roughness);
        vec3 F = showcaseFresnelSchlick(max(dot(H, V), 0.0), F0);
        vec3 spec = (D * G * F) / max(4.0 * NdotV * NdotL, 0.001);
        vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);
        vec3 radiance = light.colorIntensity.rgb * light.colorIntensity.w * atten;
        Lo += (kD * albedo / SHOWCASE_PI + spec) * radiance * NdotL;
      }
    }
  }

  // 3. Courtyard Sky + Warm Sandstone Bounce Ambient
  vec3 skyCol = vec3(0.24, 0.34, 0.50);
  vec3 groundCol = vec3(0.28, 0.21, 0.14);
  vec3 hemi = mix(groundCol, skyCol, N.y * 0.5 + 0.5);
  vec3 Famb = showcaseFresnelSchlick(NdotV, F0);
  vec3 kDamb = (1.0 - Famb) * (1.0 - metallic);
  vec3 ambient = (kDamb * albedo * hemi * 0.42 + Famb * hemi * (1.0 - roughness * 0.65) * 0.25) * ao;

  vec3 hdr = Lo + ambient + emissive;

  // Subtle golden courtyard aerial scattering
  float viewDist = length(ubo.camPosFov.xyz - worldPos);
  float fog = 1.0 - exp(-viewDist * 0.00016);
  vec3 fogColor = vec3(0.46, 0.44, 0.40);
  hdr = mix(hdr, fogColor, clamp(fog, 0.0, 0.38));

  vec3 ldr = showcaseLinearToSrgb(showcaseAcesTonemap(hdr * 1.15));
  return ldr;
}

#endif
