module rt_pbr;

import rt_types;
import rt_math;
import rt_scene;

static const uint SHADOW_RAY_FLAGS =
    RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH |
    RAY_FLAG_SKIP_CLOSEST_HIT_SHADER |
    RAY_FLAG_FORCE_OPAQUE;

// RayQuery Visibility Helpers (DCE-eliminated when unreferenced by pure shading shaders)
public bool traceShadowRay(RaytracingAccelerationStructure tlas, inout RayQuery<RAY_FLAG_NONE> rq, float3 origin, float3 normal, float3 lightDir, float maxDist) {
    if (maxDist <= 0.05) return false;
    float3 rayOrig = origin + normal * 0.12;
    RayDesc ray = { rayOrig, 0.05, lightDir, maxDist };
    rq.TraceRayInline(tlas, SHADOW_RAY_FLAGS, 0xFFu, ray);
    while (rq.Proceed()) {}
    return (rq.CommittedStatus() != COMMITTED_NOTHING);
}

public float traceAreaShadow(RaytracingAccelerationStructure tlas, inout RayQuery<RAY_FLAG_NONE> rq, float3 origin, float3 normal, float3 lightDir, float maxDist, float lightRadius, inout uint rng, int numSamples) {
    float3 up = abs(lightDir.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    float3 tangent = normalize(cross(up, lightDir));
    float3 bitangent = cross(lightDir, tangent);

    float unoccluded = 0.0;
    float3 rayOrig = origin + normal * 0.12;

    for (int i = 0; i < numSamples; ++i) {
        float phi = float(i) * 2.3999632 + randFloat(rng) * 0.5;
        float r = sqrt((float(i) + 0.5) / float(numSamples)) * tan(lightRadius);
        float3 jitterDir = normalize(lightDir + tangent * (cos(phi) * r) + bitangent * (sin(phi) * r));

        if (dot(normal, jitterDir) <= 0.0) {
            continue;
        }

        RayDesc ray = { rayOrig, 0.05, jitterDir, maxDist };
        rq.TraceRayInline(tlas, SHADOW_RAY_FLAGS, 0xFFu, ray);
        while (rq.Proceed()) {}
        bool hit = (rq.CommittedStatus() != COMMITTED_NOTHING);
        if (!hit) {
            unoccluded += 1.0;
        }
    }

    return unoccluded / float(numSamples);
}

public float computeRTAO(RaytracingAccelerationStructure tlas, inout RayQuery<RAY_FLAG_NONE> rq, float3 origin, float3 normal, inout uint rng, int numSamples, float maxDist) {
    float occlusion = 0.0;
    float3 rayOrig = origin + normal * 0.08;

    float3 up = abs(normal.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    float3 tangent = normalize(cross(up, normal));
    float3 bitangent = cross(normal, tangent);

    for (int i = 0; i < numSamples; ++i) {
        float u1 = (float(i) + randFloat(rng)) / float(numSamples);
        float u2 = randFloat(rng);
        float r = sqrt(u1);
        float theta = 6.28318530718 * u2;
        float3 localDir = float3(r * cos(theta), r * sin(theta), sqrt(max(0.0, 1.0 - u1)));
        float3 aoDir = tangent * localDir.x + bitangent * localDir.y + normal * localDir.z;

        bool hit = false;
        float hitT = maxDist;
        RayDesc ray = { rayOrig, 0.02, aoDir, maxDist };
        rq.TraceRayInline(tlas, SHADOW_RAY_FLAGS, 0xFFu, ray);
        while (rq.Proceed()) {}
        if (rq.CommittedStatus() != COMMITTED_NOTHING) {
            hit = true;
            hitT = rq.CommittedRayT();
        }

        if (hit) {
            float atten = clamp(1.0 - hitT / maxDist, 0.0, 1.0);
            occlusion += atten;
        }
    }

    return clamp(1.0 - (occlusion / float(numSamples)), 0.0, 1.0);
}

public float3 computeGltfVisibility(
    RaytracingAccelerationStructure tlas,
    inout RayQuery<RAY_FLAG_NONE> rq,
    GltfMaterialGpu mat,
    uint arch,
    float3 hitPos,
    float3 geomNormal,
    float3 faceNormal,
    float3 inDir,
    uint sceneType,
    bool enableShadows,
    inout uint rng
) {
    float sunShadow = 1.0;
    float rtao = 1.0;
    float waterDepth = -1.0;

    if (!enableShadows) {
        return float3(sunShadow, rtao, waterDepth);
    }

    if (sceneType == SCENE_INDOOR_ATRIUM) {
        float3 sunDir = normalize(float3(0.35, 0.90, -0.25));
        if (dot(geomNormal, sunDir) > 0.0) {
            sunShadow = traceAreaShadow(tlas, rq, hitPos, geomNormal, sunDir, 10000.0, 0.035, rng, 4);
        }
        rtao = computeRTAO(tlas, rq, hitPos, geomNormal, rng, 4, 350.0);
    } else if (sceneType == SCENE_SHOWROOM_STUDIO) {
        if (arch != 2u) {
            float3 studioDir0 = normalize(float3(0.6, 0.8, 0.5));
            if (dot(geomNormal, studioDir0) > 0.0) {
                sunShadow = traceAreaShadow(tlas, rq, hitPos, geomNormal, studioDir0, 5000.0, 0.065, rng, 4);
            }
            rtao = computeRTAO(tlas, rq, hitPos, geomNormal, rng, 4, 30.0);
        }
    } else if (sceneType == SCENE_AAA_FOREST) {
        float3 sunDir = normalize(float3(0.42, 0.38, 0.78));
        if (dot(geomNormal, sunDir) > 0.0) {
            sunShadow = traceAreaShadow(tlas, rq, hitPos, geomNormal, sunDir, 3500.0, 0.045, rng, 4);
        }
        rtao = computeRTAO(tlas, rq, hitPos, geomNormal, rng, 4, 35.0);
        if (mat.transmissionFactor > 0.80) {
            float2 waveUV = hitPos.xy * 0.08;
            float w1 = sin(waveUV.x * 3.2 + waveUV.y * 2.1) * 0.025;
            float w2 = cos(waveUV.x * 6.5 - waveUV.y * 4.2) * 0.015;
            float3 waterNorm = normalize(faceNormal + float3(w1, w2, 0.0));
            float3 refrDir = refract(inDir, waterNorm, 1.0 / 1.333);
            if (length(refrDir) < 0.01) refrDir = reflect(inDir, waterNorm);
            RayDesc wRay = { hitPos + refrDir * 0.05, 0.001, refrDir, 25.0 };
            rq.TraceRayInline(tlas, RAY_FLAG_FORCE_OPAQUE, 0xFFu, wRay);
            while (rq.Proceed()) {}
            if (rq.CommittedStatus() != COMMITTED_NOTHING) {
                waterDepth = rq.CommittedRayT();
            }
        }
    }

    return float3(sunShadow, rtao, waterDepth);
}

public void computeProceduralVisibility(
    RaytracingAccelerationStructure tlas,
    inout RayQuery<RAY_FLAG_NONE> rq,
    uint sceneType,
    uint matId,
    float3 hitPos,
    float3 normal,
    float3 inDir,
    bool enableShadows,
    inout uint rng,
    out float shadowOut,
    out uint3 auxOut
) {
    shadowOut = 1.0;
    auxOut = uint3(0xFFFFFFFFu, 0u, 0u);
    if (!enableShadows) {
        return;
    }

    if (sceneType == SCENE_OUTDOOR_TERRAIN) {
        float3 sunDir = normalize(float3(0.45, 0.35, 0.82));
        float3 effNorm = normal;
        if (matId == 2u) {
            float2 waveUV = hitPos.xy * 0.10;
            float w1 = sin(waveUV.x * 2.8 + waveUV.y * 1.9) * 0.030;
            float w2 = cos(waveUV.x * 5.4 - waveUV.y * 3.8) * 0.018;
            float w3 = sin(waveUV.x * 10.5 + waveUV.y * 8.7) * 0.009;
            effNorm = normalize(normal + float3(w1 + w3, w2 + w3, 0.0));
        }
        if (dot(effNorm, sunDir) > 0.0 && traceShadowRay(tlas, rq, hitPos, effNorm, sunDir, 2500.0)) {
            shadowOut = 0.0;
        }
    } else {
        shadowOut = (matId != 2u) ? traceAreaShadow(tlas, rq, hitPos, normal, normalize(float3(0.28, -0.65, 0.70)), 25.0, 0.045, rng, 4) : 1.0;
        float rtao0 = 1.0, rtao1 = 1.0, rtao2 = 1.0, rtao3 = 1.0;
        if (matId == 5u) {
            rtao0 = computeRTAO(tlas, rq, hitPos, normal, rng, 1, 15.0);
            rtao1 = computeRTAO(tlas, rq, hitPos, normal, rng, 1, 15.0);
            rtao2 = computeRTAO(tlas, rq, hitPos, normal, rng, 1, 15.0);
            rtao3 = computeRTAO(tlas, rq, hitPos, normal, rng, 1, 15.0);
        }
        uint secPrim = 0xFFFFFFFFu;
        if (matId == 2u) {
            float3 refrDir = refract(inDir, normal, 1.0 / 1.52);
            if (length(refrDir) < 0.01) refrDir = reflect(inDir, normal);
            RayDesc sRay = { hitPos + refrDir * 0.02, 0.001, refrDir, 25.0 };
            rq.TraceRayInline(tlas, RAY_FLAG_FORCE_OPAQUE, 0xFFu, sRay);
            while (rq.Proceed()) {}
            if (rq.CommittedStatus() != COMMITTED_NOTHING) {
                secPrim = rq.CommittedPrimitiveIndex();
            }
        } else if (matId == 7u || (matId == 5u && hitPos.y <= 3.0)) {
            float3 reflDir = reflect(inDir, normal);
            RayDesc sRay = { hitPos + normal * 0.015, 0.001, reflDir, 25.0 };
            rq.TraceRayInline(tlas, RAY_FLAG_FORCE_OPAQUE, 0xFFu, sRay);
            while (rq.Proceed()) {}
            if (rq.CommittedStatus() != COMMITTED_NOTHING) {
                secPrim = rq.CommittedPrimitiveIndex();
            }
        }
        auxOut = uint3(secPrim, packHalf2x16_spv(float2(rtao0, rtao1)), packHalf2x16_spv(float2(rtao2, rtao3)));
    }
}

public float3 evalPathBounceDirectLight(
    RaytracingAccelerationStructure tlas,
    inout RayQuery<RAY_FLAG_NONE> rq,
    uint sceneType,
    uint isGltf,
    float3 hitPos,
    float3 normal,
    float3 albedo,
    bool multiplyAlbedo
) {
    float3 rad = float3(0.0);
    float3 albTerm = multiplyAlbedo ? albedo : float3(1.0);

    if (sceneType == SCENE_SHOWROOM_STUDIO) {
        for (int li = 0; li < 3; ++li) {
            float3 sDir, sCol;
            getStudioLight(li, sDir, sCol);
            float NdotL = max(0.0, dot(normal, sDir));
            if (NdotL > 0.0) {
                if (!traceShadowRay(tlas, rq, hitPos, normal, sDir, 5000.0)) {
                    rad += (albTerm / 3.14159) * sCol * NdotL;
                }
            }
        }
    } else {
        float3 sunDir, sunColor;
        getSceneSunParams(sceneType, sunDir, sunColor);
        float NdotL = max(0.0, dot(normal, sunDir));
        if (NdotL > 0.0) {
            if (!traceShadowRay(tlas, rq, hitPos, normal, sunDir, (isGltf != 0u) ? 5000.0 : 2000.0)) {
                rad += (albTerm / 3.14159) * sunColor * NdotL;
            }
        }
    }

    if (sceneType == SCENE_INDOOR_ATRIUM) {
        for (int li = 0; li < 3; ++li) {
            float3 lPos, lCol;
            getAtriumLantern(li, lPos, lCol);
            float3 toL = lPos - hitPos;
            float ndotToL = dot(normal, toL);
            if (ndotToL > 0.0) {
                float bd2 = dot(toL, toL);
                float bd = sqrt(bd2);
                float invBd = 1.0 / max(bd, 0.001);
                float3 L = toL * invBd;
                if (!traceShadowRay(tlas, rq, hitPos, normal, L, bd - 0.1)) {
                    float lNdotL = ndotToL * invBd;
                    float atten = 1.0 / (1.0 + 0.000004 * bd2);
                    rad += (albTerm / 3.14159) * lCol * atten * lNdotL;
                }
            }
        }
    }

    rad += albTerm * evalSky(sceneType, normal) * 0.25;
    return rad;
}

// Pure BRDF & BSDF Evaluators (ZERO RayQuery dependency)
public float3 evaluateCookTorranceGGX(
    float3 baseColor,
    float roughness,
    float metallic,
    float3 N,
    float3 V,
    float3 L,
    float3 lightRadiance
) {
    float NdotL = max(dot(N, L), 0.0);
    if (NdotL <= 0.0) return float3(0.0);
    float NdotV = max(dot(N, V), 1e-4);
    float3 H = normalize(V + L);
    float NdotH = max(dot(N, H), 0.0);
    float VdotH = max(dot(V, H), 0.0);

    float3 F0 = lerp(float3(0.04), baseColor, metallic);

    float a = roughness * roughness;
    float a2 = a * a;
    float denomD = (NdotH * NdotH * (a2 - 1.0) + 1.0);
    float D = a2 / (3.1415926535 * denomD * denomD + 1e-7);

    float k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    float G1_V = NdotV / (NdotV * (1.0 - k) + k);
    float G1_L = NdotL / (NdotL * (1.0 - k) + k);
    float G = G1_V * G1_L;

    float3 F = F0 + (1.0 - F0) * pow(clamp(1.0 - VdotH, 0.0, 1.0), 5.0);
    float3 specular = (D * G * F) / max(4.0 * NdotV * NdotL, 1e-4);
    float3 kD = (float3(1.0) - F) * (1.0 - metallic);
    float3 diffuse = kD * (baseColor / 3.1415926535);

    return (diffuse + specular) * lightRadiance * NdotL;
}

public float3 evalCookTorranceDirect(
    float3 N,
    float3 V,
    float3 L,
    float3 lightRadiance,
    float3 albedo,
    float roughness,
    float metallic
) {
    float NdotL = max(0.0, dot(N, L));
    if (NdotL <= 0.0) return float3(0.0);
    float NdotV = max(0.001, dot(N, V));
    float3 H = normalize(V + L);
    float NdotH = max(0.0, dot(N, H));
    float VdotH = max(0.0, dot(V, H));

    float alpha = max(0.04, roughness * roughness);
    float alpha2 = alpha * alpha;
    float denomD = (NdotH * NdotH * (alpha2 - 1.0) + 1.0);
    float D = alpha2 / (3.14159265 * denomD * denomD);

    float3 F0 = lerp(float3(0.04), albedo, metallic);
    float3 F = F0 + (1.0 - F0) * pow(clamp(1.0 - VdotH, 0.0, 1.0), 5.0);

    float k = (roughness + 1.0);
    k = (k * k) / 8.0;
    float G = (NdotV / (NdotV * (1.0 - k) + k)) * (NdotL / (NdotL * (1.0 - k) + k));

    float3 spec = (D * F * G) / max(0.001, 4.0 * NdotV * NdotL);
    float3 diff = (float3(1.0) - F) * (1.0 - metallic) * (albedo / 3.14159265);

    return (diff + spec) * lightRadiance * NdotL;
}

public float3 evaluateMaterialArchetypeDirect(
    uint archetype,
    uint specializedArch,
    float3 hitPos,
    float3 baseColor,
    float roughness,
    float metallic,
    float ior,
    float3 N,
    float3 V,
    float3 L,
    float3 lightRadiance,
    float ao
) {
    uint arch = (specializedArch != 0xFFFFFFFFu && specializedArch < 8u) ? specializedArch : archetype;

    switch (arch) {
        case 0u: {
            return evaluateCookTorranceGGX(baseColor, roughness, metallic, N, V, L, lightRadiance);
        }
        case 1u: {
            float NdotL = max(dot(N, L), 0.0);
            float wrappedNdotL = clamp((dot(N, L) + 0.40) / 1.40, 0.0, 1.0);
            float3 sssColor = exp(-float3(0.40, 0.08, 0.18) * (1.0 - wrappedNdotL) * 2.0) * baseColor;
            float3 H = normalize(V + L);
            float NdotH = max(dot(N, H), 0.0);
            float VdotH = max(dot(V, H), 0.0);
            float d_j = NdotH * NdotH * (0.008 - 1.0) + 1.0;
            float D_j = 0.008 / (3.14159265 * d_j * d_j + 0.0001);
            float F_j = 0.05 + 0.95 * pow(clamp(1.0 - VdotH, 0.0, 1.0), 5.0);
            float NdotV = max(dot(N, V), 1e-4);
            float3 specJade = float3(D_j * F_j * 0.65 / (4.0 * max(NdotL, 0.01) * NdotV + 0.001) * max(NdotL, 0.01));
            return (sssColor * wrappedNdotL * 0.95 + specJade) * lightRadiance;
        }
        case 2u: {
            float3 H = normalize(V + L);
            float NdotH = max(dot(N, H), 0.0);
            float VdotH = max(dot(V, H), 0.0);
            float NdotL = max(dot(N, L), 0.0);
            float NdotV = max(dot(N, V), 1e-4);
            if (NdotL <= 0.0) return float3(0.0);
            float d = NdotH * NdotH * (0.0004 - 1.0) + 1.0;
            float D = 0.0004 / (3.14159265 * d * d + 0.0001);
            float F_l = 0.04 + 0.96 * pow(clamp(1.0 - VdotH, 0.0, 1.0), 5.0);
            return lightRadiance * (D * F_l / (4.0 * NdotL * NdotV + 0.001) * NdotL);
        }
        case 3u: {
            float NdotL = max(dot(N, L), 0.0);
            float3 H = normalize(V + L);
            float NdotH = max(dot(N, H), 0.0);
            float VdotH = max(dot(V, H), 0.0);
            float invAlpha = 1.0 / max(roughness, 0.25);
            float sinThetaH = sqrt(max(0.0, 1.0 - NdotH * NdotH));
            float D_charlie = (2.0 + invAlpha) * pow(sinThetaH, invAlpha) / 6.283185;
            float F_velvet = pow(clamp(1.0 - VdotH, 0.0, 1.0), 4.0);
            float3 sheen = float3(0.95, 0.85, 0.95) * (D_charlie * F_velvet * 0.75 * NdotL);
            return (baseColor * (NdotL * 0.40 + 0.06) + sheen) * lightRadiance;
        }
        case 4u: {
            float rustNoise = rustFBM3D(hitPos * 0.05);
            float rustFine = noise3D(hitPos * 0.2) * 0.5 + 0.5;
            float rustVal = rustNoise * 0.80 + rustFine * 0.20;
            float rustMask = smoothstep(0.44, 0.54, rustVal);
            float3 rustAlbedo = lerp(float3(0.85, 0.38, 0.12), float3(0.35, 0.16, 0.08), rustFine);
            float3 finalAlbedo = lerp(baseColor, rustAlbedo, rustMask);
            float finalRough = lerp(roughness, 0.85, rustMask);
            float finalMetal = lerp(metallic, 0.0, rustMask);
            return evaluateCookTorranceGGX(finalAlbedo, finalRough, finalMetal, N, V, L, lightRadiance);
        }
        case 5u: {
            float NdotL = max(dot(N, L), 0.0);
            float3 H = normalize(V + L);
            float NdotH = max(dot(N, H), 0.0);
            float VdotH = max(dot(V, H), 0.0);
            float NdotV = max(dot(N, V), 1e-4);
            float3 diff = baseColor * (NdotL / 3.14159265);
            float alpha2 = 0.008;
            float d_f = NdotH * NdotH * (alpha2 - 1.0) + 1.0;
            float D_f = alpha2 / (3.14159265 * d_f * d_f + 0.0001);
            float F_f = 0.06 + 0.94 * pow(clamp(1.0 - VdotH, 0.0, 1.0), 5.0);
            float3 coatSpec = float3((D_f * F_f * 0.70) / (4.0 * max(NdotL, 0.01) * NdotV + 0.001) * NdotL);
            return (diff + coatSpec) * lightRadiance;
        }
        case 6u: {
            float NdotL = max(dot(N, L), 0.0);
            float3 H = normalize(V + L);
            float NdotH = max(dot(N, H), 0.0);
            float VdotH = max(dot(V, H), 0.0);
            float NdotV = max(dot(N, V), 1e-4);
            float D_c = NdotH * NdotH * (0.0003 - 1.0) + 1.0;
            float D_clear = 0.0003 / (3.14159265 * D_c * D_c + 0.0001);
            float F_clear = 0.04 + 0.96 * pow(clamp(1.0 - VdotH, 0.0, 1.0), 5.0);
            float3 specClear = float3((D_clear * F_clear * 0.90) / (4.0 * max(NdotL, 0.01) * NdotV + 0.001) * NdotL);
            float3 baseSpec = evaluateCookTorranceGGX(baseColor, 0.25, 0.85, N, V, L, lightRadiance);
            float flakeNoise = hash31(floor(hitPos * 12.0));
            float flakeGlint = pow(NdotH, 140.0) * step(0.65, flakeNoise) * 4.0;
            return baseSpec + (specClear + float3(flakeGlint)) * lightRadiance;
        }
        case 7u: {
            float NdotL = dot(N, L);
            float wrappedNdotL = clamp((NdotL + 0.35) / 1.35, 0.0, 1.0);
            float backTrans = clamp((-NdotL + 0.30) / 1.30, 0.0, 1.0);
            float3 transColor = baseColor * float3(1.1, 1.3, 0.4) * backTrans * 0.70;
            float3 H = normalize(V + L);
            float NdotH = max(dot(N, H), 0.0);
            float D_wax = pow(NdotH, 18.0) * 0.25;
            return (baseColor * (wrappedNdotL * 0.65 + 0.08) + transColor + float3(D_wax)) * lightRadiance;
        }
        default:
            return float3(0.0);
    }
}

public float3 evaluateGltfPbrShading(
    StructuredBuffer<TextureHeaderGpu> texHeaders,
    StructuredBuffer<uint> texPixels,
    uint specializedArch,
    GltfMaterialGpu mat,
    float3 hitPos,
    float3 geomNormal,
    float3 smoothNormal,
    float4 tangent,
    float2 uv,
    float3 inDir,
    uint sceneType,
    bool enableShadows,
    bool directOnly,
    float3 precomputedVis
) {
    float3 V = -inDir;

    float3 N = smoothNormal;
    if (mat.normalTexIdx >= 0 && length(tangent.xyz) > 0.01) {
        float3 mapNorm = sampleTexture(texHeaders, texPixels, uint(mat.normalTexIdx), uv).rgb * 2.0 - 1.0;
        mapNorm.xy *= mat.normalScale;
        float3 T = normalize(tangent.xyz - dot(tangent.xyz, N) * N);
        float3 B = cross(N, T) * tangent.w;
        N = normalize(T * mapNorm.x + B * mapNorm.y + N * mapNorm.z);
    }
    if (dot(N, inDir) > 0.0) N = -N;

    float4 baseSample = (mat.baseColorTexIdx >= 0) ? sampleTexture(texHeaders, texPixels, uint(mat.baseColorTexIdx), uv) : float4(1.0);
    float3 baseColor = pow(baseSample.rgb, float3(2.2)) * mat.baseColorFactor.rgb;

    float roughness = mat.roughnessFactor;
    float metallic = mat.metallicFactor;
    if (mat.metallicRoughnessTexIdx >= 0) {
        float4 mr = sampleTexture(texHeaders, texPixels, uint(mat.metallicRoughnessTexIdx), uv);
        roughness *= mr.g;
        metallic *= mr.b;
    }
    roughness = clamp(roughness, 0.04, 1.0);
    metallic = clamp(metallic, 0.0, 1.0);

    float ao = 1.0;
    if (mat.occlusionTexIdx >= 0) {
        ao = sampleTexture(texHeaders, texPixels, uint(mat.occlusionTexIdx), uv).r;
    }

    float3 emissive = mat.emissiveFactor.rgb;
    if (mat.emissiveTexIdx >= 0) {
        emissive *= pow(sampleTexture(texHeaders, texPixels, uint(mat.emissiveTexIdx), uv).rgb, float3(2.2));
    }

    uint arch = (specializedArch != 0xFFFFFFFFu && specializedArch < 8u) ? specializedArch : mat.archetype;
    float3 directColor = baseColor;
    float directRough = roughness;
    float directMetal = metallic;
    uint directArch = arch;
    if (arch == 4u) {
        float rustNoise = rustFBM3D(hitPos * 0.05);
        float rustFine = noise3D(hitPos * 0.2) * 0.5 + 0.5;
        float rustVal = rustNoise * 0.80 + rustFine * 0.20;
        float rustMask = smoothstep(0.44, 0.54, rustVal);
        float3 rustAlbedo = lerp(float3(0.85, 0.38, 0.12), float3(0.35, 0.16, 0.08), rustFine);
        directColor = lerp(baseColor, rustAlbedo, rustMask);
        directRough = lerp(roughness, 0.85, rustMask);
        directMetal = lerp(metallic, 0.0, rustMask);
        directArch = 0u;
    }
    uint directSpecArch = (specializedArch == 4u) ? 0u : specializedArch;
    float3 totalRadiance = float3(0.0);

    if (sceneType == SCENE_INDOOR_ATRIUM) {
        float3 sunDir = normalize(float3(0.35, 0.90, -0.25));
        float sunShadow = enableShadows ? precomputedVis.x : 1.0;
        float3 sunRadiance = float3(1.0, 0.95, 0.88) * 4.5 * sunShadow;

        for (int i = 0; i < 4; ++i) {
            float3 L;
            float3 lRad;
            if (i == 0) {
                L = sunDir;
                lRad = sunRadiance;
            } else {
                float3 lPos, lCol;
                getAtriumLantern(i - 1, lPos, lCol);
                float3 toL = lPos - hitPos;
                float dist2 = dot(toL, toL);
                L = normalize(toL);
                float atten = 1.0 / (1.0 + 0.000004 * dist2);
                lRad = lCol * atten;
            }
            totalRadiance += evaluateMaterialArchetypeDirect(directArch, directSpecArch, hitPos, directColor, directRough, directMetal, mat.ior, N, V, L, lRad, ao);
        }

        float rtao = enableShadows ? precomputedVis.y : 1.0;
        ao *= rtao;
        if (directOnly) {
            return totalRadiance * ao + emissive;
        }

        float3 skyAmb = lerp(float3(0.12, 0.15, 0.22), float3(0.70, 0.78, 0.95), clamp(N.y * 0.5 + 0.5, 0.0, 1.0)) * 0.9 * ao;
        totalRadiance += (1.0 - metallic) * baseColor * skyAmb;

        float3 R = reflect(inDir, N);
        float3 F0 = lerp(float3(0.04), baseColor, metallic);
        float3 F_env = F0 + (max(float3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - max(dot(N, V), 0.0), 0.0, 1.0), 5.0);
        float3 specEnv = lerp(float3(0.12, 0.15, 0.20), float3(0.80, 0.88, 1.0), clamp(R.y * 0.5 + 0.5, 0.0, 1.0)) * F_env * (1.0 - roughness * 0.7) * 0.6 * ao;
        totalRadiance += specEnv;
    } else if (sceneType == SCENE_SHOWROOM_STUDIO) {
        if (arch == 2u) {
            float3 cabinInterior = float3(0.02, 0.025, 0.03);
            float3 R = reflect(inDir, N);
            float3 refrDirR = refract(inDir, N, 1.0 / (mat.ior - 0.02));
            float3 refrDirG = refract(inDir, N, 1.0 / mat.ior);
            float3 refrDirB = refract(inDir, N, 1.0 / (mat.ior + 0.02));
            if (length(refrDirG) < 0.01) { refrDirR = R; refrDirG = R; refrDirB = R; }

            float3 envRefl = lerp(float3(0.12, 0.13, 0.16), float3(0.40, 0.44, 0.50), clamp(R.y * 1.5, 0.0, 1.0));
            float3 envRefr;
            envRefr.r = lerp(float3(0.06, 0.07, 0.09), float3(0.18, 0.20, 0.24), clamp(refrDirR.y * 1.5, 0.0, 1.0)).r;
            envRefr.g = lerp(float3(0.06, 0.07, 0.09), float3(0.18, 0.20, 0.24), clamp(refrDirG.y * 1.5, 0.0, 1.0)).g;
            envRefr.b = lerp(float3(0.06, 0.07, 0.09), float3(0.18, 0.20, 0.24), clamp(refrDirB.y * 1.5, 0.0, 1.0)).b;

            float3 glassTint = float3(0.88, 0.92, 0.94);
            float3 transmittedLight = lerp(cabinInterior, envRefr, 0.35) * glassTint;
            float F = 0.04 + 0.96 * pow(clamp(1.0 - max(dot(N, V), 0.0), 0.0, 1.0), 5.0);

            float3 specLights = float3(0.0);
            for (int i = 0; i < 3; ++i) {
                float3 sDir, sCol;
                getStudioLight(i, sDir, sCol);
                specLights += evaluateMaterialArchetypeDirect(2u, specializedArch, hitPos, baseColor, roughness, metallic, mat.ior, N, V, sDir, sCol, ao);
            }
            if (directOnly) {
                return specLights + cabinInterior + emissive;
            }
            return lerp(transmittedLight, envRefl, F) + specLights;
        }

        float keyShadow = enableShadows ? precomputedVis.x : 1.0;
        for (int i = 0; i < 3; ++i) {
            float3 sDir, sCol;
            getStudioLight(i, sDir, sCol);
            float3 lCol = (i == 0) ? (sCol * keyShadow) : sCol;
            totalRadiance += evaluateMaterialArchetypeDirect(directArch, directSpecArch, hitPos, directColor, directRough, directMetal, mat.ior, N, V, sDir, lCol, ao);
        }

        if (enableShadows && hitPos.y < 35.0 && hitPos.y > 0.0) {
            float contactOcc = lerp(0.35, 1.0, clamp(hitPos.y / 35.0, 0.0, 1.0));
            ao *= contactOcc;
        }

        float rtao = enableShadows ? precomputedVis.y : 1.0;
        ao *= rtao;
        if (directOnly) {
            return totalRadiance + emissive;
        }

        float3 studioAmb = lerp(float3(0.06, 0.06, 0.07), float3(0.25, 0.26, 0.28), clamp(N.y * 0.5 + 0.5, 0.0, 1.0)) * ao;
        totalRadiance += (1.0 - metallic) * baseColor * studioAmb;

        float3 R = reflect(inDir, N);
        float3 F0 = lerp(float3(0.04), baseColor, metallic);
        float3 F_env = F0 + (max(float3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - max(dot(N, V), 0.0), 0.0, 1.0), 5.0);
        float3 envRefl = lerp(float3(0.12, 0.13, 0.16), float3(0.24, 0.26, 0.30), clamp(R.y * 1.5, 0.0, 1.0));
        float3 specEnv = envRefl * F_env * (1.0 - roughness * 0.5) * ao;
        totalRadiance += specEnv;
    } else if (sceneType == SCENE_AAA_FOREST) {
        float3 sunDir = normalize(float3(0.42, 0.38, 0.78));
        float3 sunColor = float3(1.0, 0.94, 0.84) * 3.8;

        float sunShadow = enableShadows ? precomputedVis.x : 1.0;
        float rtao = enableShadows ? precomputedVis.y : 1.0;
        ao *= rtao;
        if (directOnly) {
            return evaluateCookTorranceGGX(baseColor, roughness, metallic, N, V, sunDir, sunColor * sunShadow) * ao + emissive;
        }

        float3 skyRadiance = evalOutdoorSky(N) * 0.45 * ao;

        if (!enableShadows) {
            float3 skyFillDir1 = normalize(float3(-0.6, 0.5, 0.6));
            float3 skyFillDir2 = normalize(float3(0.2, -0.7, 0.5));
            if (arch == 2u) {
                float3 reflSky = evalOutdoorSky(reflect(inDir, N));
                float F_w = 0.02 + 0.98 * pow(clamp(1.0 - max(dot(N, V), 0.0), 0.0, 1.0), 5.0);
                float3 specLights = evaluateMaterialArchetypeDirect(2u, specializedArch, hitPos, directColor, directRough, directMetal, mat.ior, N, V, sunDir, sunColor, ao)
                                  + evaluateMaterialArchetypeDirect(2u, specializedArch, hitPos, directColor, directRough, directMetal, mat.ior, N, V, skyFillDir1, skyRadiance, ao)
                                  + evaluateMaterialArchetypeDirect(2u, specializedArch, hitPos, directColor, directRough, directMetal, mat.ior, N, V, skyFillDir2, skyRadiance * 0.5, ao);
                return lerp(float3(0.02, 0.08, 0.12), reflSky, F_w) + specLights + emissive;
            }
            totalRadiance += evaluateMaterialArchetypeDirect(directArch, directSpecArch, hitPos, directColor, directRough, directMetal, mat.ior, N, V, sunDir, sunColor, ao);
            totalRadiance += evaluateMaterialArchetypeDirect(directArch, directSpecArch, hitPos, directColor, directRough, directMetal, mat.ior, N, V, skyFillDir1, skyRadiance, ao);
            totalRadiance += evaluateMaterialArchetypeDirect(directArch, directSpecArch, hitPos, directColor, directRough, directMetal, mat.ior, N, V, skyFillDir2, skyRadiance * 0.5, ao);
            return totalRadiance + baseColor * skyRadiance + emissive;
        }

        if (mat.transmissionFactor > 0.80) {
            float2 waveUV = hitPos.xy * 0.08;
            float w1 = sin(waveUV.x * 3.2 + waveUV.y * 2.1) * 0.025;
            float w2 = cos(waveUV.x * 6.5 - waveUV.y * 4.2) * 0.015;
            float3 waterNorm = normalize(N + float3(w1, w2, 0.0));

            float3 riverbedColor;
            if (precomputedVis.z >= 0.0) {
                float depth = precomputedVis.z;
                float3 bedBase = float3(0.18, 0.22, 0.20);
                float3 waterExtinction = exp(-float3(0.15, 0.04, 0.02) * depth * 1.5);
                riverbedColor = bedBase * waterExtinction * (sunColor * 0.5 * sunShadow + skyRadiance);
            } else {
                riverbedColor = float3(0.02, 0.08, 0.12);
            }

            float F_w = 0.02 + 0.98 * pow(clamp(1.0 - max(dot(waterNorm, V), 0.0), 0.0, 1.0), 5.0);
            float3 reflSky = evalOutdoorSky(reflect(inDir, waterNorm));
            float3 halfVec = normalize(V + sunDir);
            float sunGlint = pow(max(dot(waterNorm, halfVec), 0.0), 180.0) * 12.0 * sunShadow;
            totalRadiance = lerp(riverbedColor, reflSky, F_w) + sunColor * sunGlint;
        } else if (mat.transmissionFactor > 0.25) {
            float backNdotL = max(dot(-N, sunDir), 0.0);
            float3 transmission = baseColor * float3(1.1, 1.4, 0.6) * backNdotL * sunColor * sunShadow * 0.85;
            float F_sheen = pow(clamp(1.0 - max(dot(N, V), 0.0), 0.0, 1.0), 4.0) * 0.45;
            totalRadiance = evaluateCookTorranceGGX(baseColor, roughness, metallic, N, V, sunDir, sunColor * sunShadow)
                          + transmission + float3(F_sheen) * skyRadiance + baseColor * skyRadiance;
        } else if (roughness < 0.35 && baseColor.r > 0.85) {
            float3 halfVec = normalize(V + sunDir);
            float NdotH = max(dot(N, halfVec), 0.0);
            float glint = pow(NdotH, 120.0) * 8.0 * sunShadow;
            totalRadiance = evaluateCookTorranceGGX(baseColor, 0.35, 0.0, N, V, sunDir, sunColor * sunShadow)
                          + float3(glint) * sunColor + baseColor * skyRadiance;
        } else {
            float3 effColor = baseColor;
            float effRough = roughness;
            if (hitPos.z <= 2.2 && baseColor.g < 0.40) {
                float wetness = clamp(1.0 - hitPos.z / 2.2, 0.0, 1.0);
                effColor = lerp(effColor, effColor * 0.55, wetness);
                effRough = lerp(effRough, 0.15, wetness);
            }
            float2 fuv = hitPos.xy * 0.15;
            float texVar = sin(fuv.x * 3.7 + fuv.y * 2.3) * cos(fuv.x * 1.9 - fuv.y * 4.1) * 0.12;
            effColor = clamp(effColor * (1.0 + texVar), 0.0, 1.0);

            totalRadiance = evaluateCookTorranceGGX(effColor, effRough, metallic, N, V, sunDir, sunColor * sunShadow)
                          + effColor * skyRadiance;
        }

        float3 camPos = float3(-35.0, -50.0, 20.0);
        float camDist = length(hitPos - camPos);
        float3 rayDirToHit = (camDist > 0.001) ? ((hitPos - camPos) / camDist) : inDir;
        float3 skyHorizon = evalOutdoorSky(rayDirToHit);
        float hazeFactor = clamp(1.0 - exp(-camDist * 0.00065), 0.0, 1.0);
        if (camDist > 1400.0) {
            float fadeToEnd = smoothstep(1400.0, 2200.0, camDist);
            hazeFactor = lerp(hazeFactor, 1.0, fadeToEnd);
        }
        totalRadiance = lerp(totalRadiance, skyHorizon, hazeFactor);
    }

    totalRadiance += emissive;
    return totalRadiance;
}

public float3 evaluateGltfPbrWithRayQuery(
    RaytracingAccelerationStructure tlas,
    inout RayQuery<RAY_FLAG_NONE> rq,
    StructuredBuffer<TextureHeaderGpu> texHeaders,
    StructuredBuffer<uint> texPixels,
    uint specializedArch,
    GltfMaterialGpu mat,
    float3 hitPos,
    float3 geomNormal,
    float3 smoothNormal,
    float4 tangent,
    float2 uv,
    float3 inDir,
    uint sceneType,
    bool enableShadows,
    bool directOnly,
    inout uint rng
) {
    float3 N = smoothNormal;
    if (mat.normalTexIdx >= 0 && length(tangent.xyz) > 0.01) {
        float3 mapNorm = sampleTexture(texHeaders, texPixels, uint(mat.normalTexIdx), uv).rgb * 2.0 - 1.0;
        mapNorm.xy *= mat.normalScale;
        float3 T = normalize(tangent.xyz - dot(tangent.xyz, N) * N);
        float3 B = cross(N, T) * tangent.w;
        N = normalize(T * mapNorm.x + B * mapNorm.y + N * mapNorm.z);
    }
    if (dot(N, inDir) > 0.0) N = -N;

    uint arch = (specializedArch != 0xFFFFFFFFu && specializedArch < 8u) ? specializedArch : mat.archetype;
    float3 vis = computeGltfVisibility(tlas, rq, mat, arch, hitPos, geomNormal, N, inDir, sceneType, enableShadows, rng);
    return evaluateGltfPbrShading(texHeaders, texPixels, specializedArch, mat, hitPos, geomNormal, smoothNormal, tangent, uv, inDir, sceneType, enableShadows, directOnly, vis);
}

public float3 evalProceduralMaterialBSDF(
    uint sceneType,
    uint matId,
    uint specializedArch,
    float3 normal,
    float3 inDir,
    float3 hitPos,
    bool enableShadows,
    bool directOnly,
    float precomputedShadow,
    uint3 precomputedAux
) {
    float3 viewDir = -inDir;
    float NdotV = max(dot(normal, viewDir), 0.001);
    uint archetype = (specializedArch != 0xFFFFFFFFu && specializedArch < 8u) ? specializedArch : matId;

    if (sceneType == SCENE_OUTDOOR_TERRAIN) {
        float3 sunDir = normalize(float3(0.45, 0.35, 0.82));
        float3 sunColor = float3(1.0, 0.96, 0.88) * 2.2;
        float3 skyColor = float3(0.35, 0.55, 0.85) * 0.50;
        float3 groundColor = float3(0.20, 0.18, 0.12) * 0.25;

        if (archetype == 2u) {
            float2 waveUV = hitPos.xy * 0.10;
            float w1 = sin(waveUV.x * 2.8 + waveUV.y * 1.9) * 0.030;
            float w2 = cos(waveUV.x * 5.4 - waveUV.y * 3.8) * 0.018;
            float w3 = sin(waveUV.x * 10.5 + waveUV.y * 8.7) * 0.009;
            normal = normalize(normal + float3(w1 + w3, w2 + w3, 0.0));
            NdotV = max(dot(normal, viewDir), 0.0);
        }

        float NdotL = max(dot(normal, sunDir), 0.0);
        float3 halfVec = normalize(viewDir + sunDir);
        float NdotH = max(dot(normal, halfVec), 0.0);
        float VdotH = max(dot(viewDir, halfVec), 0.0);

        float3 color = float3(0.0);
        float3 albedo = float3(0.5);
        float3 fbmPos = hitPos * 0.32;
        float3 noisePos = hitPos * 1.40;
        switch (archetype) {
            case 0u: fbmPos = float3(hitPos.xy * 0.08, hitPos.z * 0.35); noisePos = hitPos * 0.65; break;
            case 1u: fbmPos = hitPos * 0.45; noisePos = hitPos * 1.80; break;
            case 2u: fbmPos = float3(hitPos.xy * 0.35, 0.0); break;
            case 3u: fbmPos = hitPos * 0.12; noisePos = hitPos * 2.40; break;
            case 4u: fbmPos = hitPos * 0.14; noisePos = hitPos * 0.95; break;
            case 5u: fbmPos = hitPos * 0.09; noisePos = hitPos * 0.75; break;
            case 6u: fbmPos = float3(hitPos.xy * 0.85, hitPos.z * 0.18); noisePos = hitPos * 1.35; break;
            default: break;
        }
        float fbmVal = rustFBM3D(fbmPos);
        float noiseVal = (archetype != 2u) ? noise3D(noisePos) : 0.0;

        switch (archetype) {
            case 0u: {
                albedo = lerp(float3(0.58, 0.45, 0.34), float3(0.72, 0.59, 0.46), clamp(fbmVal * 0.75 + noiseVal * 0.25, 0.0, 1.0));
                float D_r = pow(NdotH, 8.0) * (0.12 + 0.08 * noiseVal);
                color = albedo * (NdotL * 0.75 + 0.15) + float3(D_r);
                break;
            }
            case 1u: {
                albedo = lerp(float3(0.06, 0.26, 0.09), float3(0.11, 0.38, 0.15), clamp(fbmVal * 0.70 + noiseVal * 0.30, 0.0, 1.0));
                float wrappedNdotL = clamp((dot(normal, sunDir) + 0.45) / 1.45, 0.0, 1.0);
                float3 transmission = float3(0.18, 0.48, 0.10) * wrappedNdotL * (0.85 + 0.30 * noiseVal);
                float D_f = pow(NdotH, 30.0) * 0.4;
                color = albedo * (NdotL * 0.60 + 0.10) + transmission + float3(D_f);
                break;
            }
            case 2u: {
                albedo = lerp(float3(0.010, 0.038, 0.060), float3(0.015, 0.048, 0.072), fbmVal);
                break;
            }
            case 3u: {
                albedo = lerp(float3(0.88, 0.92, 0.97), float3(0.96, 0.98, 1.00), fbmVal);
                float snowWrap = clamp((dot(normal, sunDir) + 0.3) / 1.3, 0.0, 1.0);
                float snowGlint = pow(NdotH, 60.0) * (0.65 + 0.45 * noiseVal);
                color = albedo * (snowWrap * 0.85 + 0.20) + float3(snowGlint);
                break;
            }
            case 4u: {
                albedo = lerp(float3(0.24, 0.25, 0.28), float3(0.36, 0.37, 0.40), fbmVal);
                float cliffNdotL = NdotL * 0.80 + 0.12;
                float micaSpec = pow(NdotH, 45.0) * smoothstep(0.65, 0.95, noiseVal) * 0.35;
                color = albedo * cliffNdotL + float3(micaSpec);
                break;
            }
            case 5u: {
                albedo = lerp(float3(0.19, 0.38, 0.14), float3(0.29, 0.50, 0.22), clamp(fbmVal * 0.75 + noiseVal * 0.25, 0.0, 1.0));
                float sheen = pow(clamp(1.0 - VdotH, 0.0, 1.0), 3.0) * (0.24 + 0.12 * noiseVal);
                color = albedo * (NdotL * 0.70 + 0.18) + float3(sheen);
                break;
            }
            case 6u: {
                albedo = lerp(float3(0.18, 0.11, 0.07), float3(0.32, 0.22, 0.14), clamp(fbmVal * 0.70 + noiseVal * 0.30, 0.0, 1.0));
                color = albedo * (NdotL * 0.65 + 0.10);
                break;
            }
            case 7u: {
                albedo = lerp(float3(0.14, 0.16, 0.18), float3(0.23, 0.25, 0.27), clamp(fbmVal * 0.70 + noiseVal * 0.30, 0.0, 1.0));
                float F_wet = 0.06 + 0.94 * pow(clamp(1.0 - VdotH, 0.0, 1.0), 5.0);
                float D_wet = pow(NdotH, 90.0) * 2.5;
                color = albedo * (NdotL * 0.50 + 0.10) + float3(D_wet * F_wet);
                break;
            }
        }

        float sunShadow = enableShadows ? precomputedShadow : 1.0;
        float3 totalRadiance;
        if (archetype == 2u) {
            float F_w = 0.02 + 0.98 * pow(clamp(1.0 - NdotV, 0.0, 1.0), 5.0);
            float3 reflDir = reflect(inDir, normal);
            float3 reflSky = evalOutdoorSky(reflDir);
            float sunGlint = pow(NdotH, 180.0) * 10.0 * sunShadow;
            float3 deepWater = float3(0.012, 0.042, 0.065);
            totalRadiance = lerp(deepWater, reflSky, F_w) + sunColor * sunGlint;
        } else {
            totalRadiance = color * sunColor * sunShadow + albedo * skyColor * max(normal.z, 0.0) + albedo * groundColor * max(-normal.z, 0.0);
        }

        if (directOnly) {
            return totalRadiance;
        }

        Camera cam = getCamera(sceneType, 0u);
        float3 rayVec = hitPos - cam.pos;
        float camDist = length(rayVec);
        float3 rayDirToHit = rayVec / max(camDist, 0.001);

        float3 skyHorizon = evalOutdoorSky(rayDirToHit);
        float3 hazeColor = lerp(float3(0.65, 0.76, 0.90), skyHorizon, 0.75);

        float hazeFactor = clamp(1.0 - exp(-camDist * 0.00085), 0.0, 1.0);
        if (camDist > 1250.0) {
            float fadeToEnd = smoothstep(1250.0, 1750.0, camDist);
            hazeFactor = lerp(hazeFactor, 1.0, fadeToEnd);
            hazeColor = skyHorizon;
        }
        return lerp(totalRadiance, hazeColor, hazeFactor);
    } else {
        float3 totalRadiance = float3(0.0);
        float2 rtao01 = unpackHalf2x16_spv(precomputedAux.y);
        float2 rtao23 = unpackHalf2x16_spv(precomputedAux.z);

        for (int l = 0; l < 4; ++l) {
            float3 slDir, slCol;
            float slInt;
            getSceneLight(l, slDir, slCol, slInt);
            float3 lightDir = normalize(slDir);
            float3 lightCol = slCol * slInt;
            if (enableShadows && l == 0 && archetype != 2u) {
                lightCol *= precomputedShadow;
            }
            float3 halfVec = normalize(viewDir + lightDir);
            float NdotL = max(dot(normal, lightDir), 0.0);
            float NdotH = max(dot(normal, halfVec), 0.0);
            float VdotH = max(dot(viewDir, halfVec), 0.0);

            if (NdotL <= 0.0 && archetype != 1u && archetype != 2u && archetype != 5u) continue;

            float3 color = float3(0.0);
            switch (archetype) {
                case 0u: {
                    float3 carColor = float3(0.68, 0.01, 0.02);
                    float D_c = NdotH * NdotH * (0.0003 - 1.0) + 1.0;
                    float D_clear = 0.0003 / (3.14159 * D_c * D_c + 0.0001);
                    float F_clear = 0.04 + 0.96 * pow(clamp(1.0 - VdotH, 0.0, 1.0), 5.0);
                    float3 specClear = float3((D_clear * F_clear * 0.85) / (4.0 * NdotL * NdotV + 0.001) * NdotL);
                    float D_m = NdotH * NdotH * (0.035 - 1.0) + 1.0;
                    float D_base = 0.035 / (3.14159 * D_m * D_m + 0.0001);
                    float3 specBase = carColor * (D_base * 0.70 / (4.0 * NdotL * NdotV + 0.001) * NdotL);
                    color = (carColor * (NdotL * 0.65 + 0.08)) + specBase + specClear;
                    break;
                }
                case 1u: {
                    float3 jadeBase = float3(0.08, 0.88, 0.74);
                    float wrappedNdotL = clamp((dot(normal, lightDir) + 0.40) / 1.40, 0.0, 1.0);
                    float3 sssCol = exp(-float3(0.40, 0.08, 0.18) * (1.0 - wrappedNdotL) * 2.0) * jadeBase;
                    float d_j = NdotH * NdotH * (0.006 - 1.0) + 1.0;
                    float D_j = 0.006 / (3.14159 * d_j * d_j + 0.0001);
                    float F_j = 0.05 + 0.95 * pow(clamp(1.0 - VdotH, 0.0, 1.0), 5.0);
                    float3 specJade = float3(D_j * F_j * 0.65 / (4.0 * NdotL * NdotV + 0.001) * NdotL);
                    color = sssCol * wrappedNdotL * 0.95 + specJade;
                    break;
                }
                case 2u: {
                    float d_g = NdotH * NdotH * (0.0004 - 1.0) + 1.0;
                    float D_glass = 0.0004 / (3.14159 * d_g * d_g + 0.0001);
                    float F_g = 0.04 + 0.96 * pow(clamp(1.0 - VdotH, 0.0, 1.0), 5.0);
                    color = float3(D_glass * F_g * 1.0 / (4.0 * NdotL * NdotV + 0.001) * NdotL);
                    break;
                }
                case 3u: {
                    float3 velvetBase = float3(0.60, 0.02, 0.42);
                    float invAlpha = 1.0 / 0.35;
                    float sinThetaH = sqrt(max(0.0, 1.0 - NdotH * NdotH));
                    float D_charlie = (2.0 + invAlpha) * pow(sinThetaH, invAlpha) / 6.28318;
                    float F_velvet = pow(clamp(1.0 - VdotH, 0.0, 1.0), 4.0);
                    float3 sheen = float3(0.95, 0.35, 0.85) * (D_charlie * F_velvet * 0.75 * NdotL);
                    color = velvetBase * (NdotL * 0.40 + 0.06) + sheen;
                    break;
                }
                case 4u: {
                    float rustNoise = rustFBM3D(hitPos * 5.5);
                    float rustFine = noise3D(hitPos * 24.0) * 0.5 + 0.5;
                    float rustVal = rustNoise * 0.80 + rustFine * 0.20;
                    float rustMask = smoothstep(0.46, 0.52, rustVal);

                    float3 steelCol = float3(0.24, 0.26, 0.30);
                    float3 rustCol = float3(0.95, 0.42, 0.14);
                    float3 darkRust = float3(0.35, 0.16, 0.08);
                    float3 patinaCol = float3(0.20, 0.55, 0.48);

                    float patinaMask = smoothstep(0.52, 0.56, rustNoise) * (1.0 - smoothstep(0.56, 0.62, rustNoise));
                    float3 activeRust = lerp(rustCol, darkRust, rustFine * 0.5);
                    activeRust = lerp(activeRust, patinaCol, patinaMask * 0.50);

                    float d_s = NdotH * NdotH * (0.015 - 1.0) + 1.0;
                    float D_s = 0.015 / (3.14159 * d_s * d_s + 0.0001);
                    float3 steelSpec = steelCol * (D_s * 0.90 / (4.0 * NdotL * NdotV + 0.001) * NdotL);
                    float3 steelBase = steelCol * (NdotL * 0.55 + 0.08) + steelSpec;

                    float d_r = NdotH * NdotH * (0.50 - 1.0) + 1.0;
                    float D_r = 0.50 / (3.14159 * d_r * d_r + 0.0001);
                    float3 rustSpec = activeRust * (D_r * 0.10 / (4.0 * NdotL * NdotV + 0.001) * NdotL);
                    float3 rustBase = activeRust * (NdotL * 0.75 + 0.12) + rustSpec;

                    color = lerp(steelBase, rustBase, rustMask);
                    break;
                }
                case 5u: {
                    float3 floorBase = float3(0.24, 0.25, 0.28);
                    float floorOcc = computeFloorOcclusion(hitPos);
                    if (enableShadows) {
                        float rtao = (l == 0) ? rtao01.x : ((l == 1) ? rtao01.y : ((l == 2) ? rtao23.x : rtao23.y));
                        floorOcc *= rtao;
                    }
                    float wallFactor = smoothstep(2.0, 6.0, hitPos.y);
                    float wallAtten = lerp(1.0, 0.85, wallFactor);
                    float effectiveNdotL = NdotL * ((l == 0) ? floorOcc : clamp(floorOcc * 1.15, 0.0, 1.0)) * wallAtten;

                    float3 floorSpec = float3(0.0);
                    if (l == 0 && wallFactor < 0.5) {
                        float alpha2 = 0.015;
                        float d_f = NdotH * NdotH * (alpha2 - 1.0) + 1.0;
                        float D_f = alpha2 / (3.14159 * d_f * d_f + 0.0001);
                        float F_f = 0.08 + 0.92 * pow(clamp(1.0 - VdotH, 0.0, 1.0), 5.0);
                        floorSpec = float3((D_f * F_f * 0.45) / (4.0 * NdotL * NdotV + 0.001) * effectiveNdotL);
                    }
                    color = floorBase * (effectiveNdotL * 0.70 + 0.18) + floorSpec;
                    break;
                }
                case 6u: {
                    float3 goldBase = float3(1.00, 0.78, 0.32);
                    float d_g = NdotH * NdotH * (0.003 - 1.0) + 1.0;
                    float D_gold = 0.003 / (3.14159 * d_g * d_g + 0.0001);
                    float F_gold = 0.88 + 0.12 * pow(clamp(1.0 - VdotH, 0.0, 1.0), 5.0);
                    float3 specGold = goldBase * (D_gold * F_gold / (4.0 * NdotL * NdotV + 0.001) * NdotL);
                    color = goldBase * (NdotL * 0.40 + 0.05) + specGold * 1.2;
                    break;
                }
                case 7u: {
                    float3 chromeBase = float3(0.96, 0.96, 0.98);
                    float d_c = NdotH * NdotH * (0.0001 - 1.0) + 1.0;
                    float D_c = 0.0001 / (3.14159 * d_c * d_c + 0.0001);
                    color = chromeBase * (D_c * 0.98 / (4.0 * NdotL * NdotV + 0.001) * NdotL);
                    break;
                }
            }
            totalRadiance += color * lightCol;
        }

        if (directOnly) {
            return totalRadiance;
        }

        float3 reflDir = reflect(inDir, normal);
        float3 envRefl = evalStudioEnvironment(reflDir);
        uint rPrim = precomputedAux.x;
        if (archetype == 0u) {
            totalRadiance += envRefl * 0.50 * pow(clamp(1.0 - NdotV, 0.0, 1.0), 4.0);
        } else if (archetype == 1u) {
            totalRadiance += envRefl * 0.25 * pow(clamp(1.0 - NdotV, 0.0, 1.0), 5.0);
        } else if (archetype == 2u) {
            float3 refrDir = refract(inDir, normal, 1.0 / 1.52);
            if (length(refrDir) < 0.01) refrDir = reflDir;
            float3 transmittedCol = (enableShadows && rPrim != 0xFFFFFFFFu)
                ? getProceduralSecondaryColor(rPrim, true)
                : evalStudioEnvironment(refrDir);
            float F_glass = 0.04 + 0.96 * pow(clamp(1.0 - NdotV, 0.0, 1.0), 5.0);
            totalRadiance = lerp(transmittedCol * float3(0.90, 0.94, 0.96), envRefl, F_glass) + totalRadiance;
        } else if (archetype == 3u) {
            totalRadiance += envRefl * 0.15 * pow(clamp(1.0 - NdotV, 0.0, 1.0), 3.0);
        } else if (archetype == 6u) {
            float3 goldBase = float3(1.00, 0.78, 0.32);
            totalRadiance += envRefl * goldBase * (0.40 + 0.60 * pow(clamp(1.0 - NdotV, 0.0, 1.0), 5.0));
        } else if (archetype == 7u) {
            float3 reflColor = (enableShadows && rPrim != 0xFFFFFFFFu)
                ? getProceduralSecondaryColor(rPrim, false)
                : envRefl;
            totalRadiance = lerp(totalRadiance, reflColor, 0.96);
        } else if (archetype == 5u && hitPos.y <= 3.0) {
            float3 reflColor = (enableShadows && rPrim != 0xFFFFFFFFu)
                ? getProceduralSecondaryColor(rPrim, false)
                : (envRefl * 0.25 + float3(0.04, 0.04, 0.05));
            float fresnelFloor = pow(clamp(1.0 - NdotV, 0.0, 1.0), 5.0);
            totalRadiance += reflColor * (0.03 + 0.15 * fresnelFloor);
        }

        return totalRadiance;
    }
}

public float3 evalProceduralMaterialBSDFWithRayQuery(
    RaytracingAccelerationStructure tlas,
    inout RayQuery<RAY_FLAG_NONE> rq,
    uint sceneType,
    uint matId,
    uint specializedArch,
    float3 normal,
    float3 inDir,
    float3 hitPos,
    bool enableShadows,
    bool directOnly,
    inout uint rng
) {
    float shadowVis;
    uint3 aux;
    computeProceduralVisibility(tlas, rq, sceneType, matId, hitPos, normal, inDir, enableShadows, rng, shadowVis, aux);
    return evalProceduralMaterialBSDF(sceneType, matId, specializedArch, normal, inDir, hitPos, enableShadows, directOnly, shadowVis, aux);
}
