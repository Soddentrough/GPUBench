// GPUBench Industry-Standard PBR Shading Pipeline (8 Production AAA BSDF Archetypes)
// Shared identically across Traditional Megakernel and Work Lists / DGC pathways.

struct GltfVertexGpu {
    float px, py, pz;
    float nx, ny, nz;
    float tx, ty, tz, tw;
    float u, v;
};

struct GltfMaterialGpu {
    vec4 baseColorFactor;
    float metallicFactor;
    float roughnessFactor;
    float transmissionFactor;
    float ior;
    vec4 emissiveFactor;
    int baseColorTexIdx;
    int metallicRoughnessTexIdx;
    int normalTexIdx;
    int occlusionTexIdx;
    int emissiveTexIdx;
    float normalScale;
    float alphaCutoff;
    uint alphaMode;
    uint archetype;
    uint extraParam1;
    uint extraParam2;
    uint pad1;
};

struct TextureHeaderGpu {
    uint width;
    uint height;
    uint offset;
    uint pad;
};

#ifdef BINDING_MAT_BUF
layout(set = 0, binding = BINDING_MAT_BUF) readonly buffer MaterialBuffer {
    GltfMaterialGpu materials[];
} matBuf;
#endif

#ifdef BINDING_TRI_MAT_BUF
layout(set = 0, binding = BINDING_TRI_MAT_BUF) readonly buffer TriangleMaterialBuffer {
    uint triangleMats[];
} triMatBuf;
#endif

#ifdef BINDING_TEX_HDR_BUF
layout(set = 0, binding = BINDING_TEX_HDR_BUF) readonly buffer TexHeaderBuffer {
    TextureHeaderGpu headers[];
} texHeaders;
#endif

#ifdef BINDING_TEX_PIX_BUF
layout(set = 0, binding = BINDING_TEX_PIX_BUF) readonly buffer TexPixelBuffer {
    uint pixels[];
} texPixels;
#endif

layout(constant_id = 0) const uint SPECIALIZED_ARCHETYPE = 0xFFFFFFFFu;

// Procedural 3D noise for authentic micro-structure weathering without VRAM texture expansion
float hash31_common(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}

float noise3D_common(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float n000 = hash31_common(i + vec3(0, 0, 0));
    float n100 = hash31_common(i + vec3(1, 0, 0));
    float n010 = hash31_common(i + vec3(0, 1, 0));
    float n110 = hash31_common(i + vec3(1, 1, 0));
    float n011 = hash31_common(i + vec3(0, 0, 1));
    float n101 = hash31_common(i + vec3(1, 0, 1));
    float n011b = hash31_common(i + vec3(0, 1, 1));
    float n111 = hash31_common(i + vec3(1, 1, 1));
    return mix(
        mix(mix(n000, n100, f.x), mix(n010, n110, f.x), f.y),
        mix(mix(n011, n101, f.x), mix(n011b, n111, f.x), f.y),
        f.z
    );
}

float rustFBM3D_common(vec3 p) {
    float v = 0.0;
    float a = 0.5;
    mat3 rot = mat3(
         0.00,  0.80,  0.60,
        -0.80,  0.36, -0.48,
        -0.60, -0.48,  0.64
    );
    for (int i = 0; i < 4; ++i) {
        v += a * noise3D_common(p);
        p = rot * p * 2.02 + vec3(1.7, 9.2, 0.4);
        a *= 0.5;
    }
    return v;
}

// Industry-Standard Cook-Torrance GGX Microfacet BRDF
vec3 evaluateCookTorranceGGX(
    vec3 baseColor,
    float roughness,
    float metallic,
    vec3 N,
    vec3 V,
    vec3 L,
    vec3 lightRadiance
) {
    float NdotL = max(dot(N, L), 0.0);
    if (NdotL <= 0.0) return vec3(0.0);
    float NdotV = max(dot(N, V), 1e-4);
    vec3 H = normalize(V + L);
    float NdotH = max(dot(N, H), 0.0);
    float VdotH = max(dot(V, H), 0.0);

    // Dielectric F0 = 0.04, metal F0 = baseColor
    vec3 F0 = mix(vec3(0.04), baseColor, metallic);

    // 1. D: GGX Normal Distribution Function
    float a = roughness * roughness;
    float a2 = a * a;
    float denomD = (NdotH * NdotH * (a2 - 1.0) + 1.0);
    float D = a2 / (3.1415926535 * denomD * denomD + 1e-7);

    // 2. G: Smith Height-Correlated Masking-Shadowing Function (Schlick-GGX)
    float k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    float G1_V = NdotV / (NdotV * (1.0 - k) + k);
    float G1_L = NdotL / (NdotL * (1.0 - k) + k);
    float G = G1_V * G1_L;

    // 3. F: Fresnel-Schlick
    vec3 F = F0 + (1.0 - F0) * pow(clamp(1.0 - VdotH, 0.0, 1.0), 5.0);

    // Specular BRDF
    vec3 specular = (D * G * F) / max(4.0 * NdotV * NdotL, 1e-4);

    // Diffuse component (Lambertian with energy conservation)
    vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);
    vec3 diffuse = kD * (baseColor / 3.1415926535);

    return (diffuse + specular) * lightRadiance * NdotL;
}

// 8 Production AAA BSDF Archetype Evaluator
vec3 evaluateMaterialArchetypeDirect(
    uint archetype,
    vec3 hitPos,
    vec3 baseColor,
    float roughness,
    float metallic,
    float ior,
    vec3 N,
    vec3 V,
    vec3 L,
    vec3 lightRadiance,
    float ao
) {
    uint arch = (SPECIALIZED_ARCHETYPE != 0xFFFFFFFFu) ? SPECIALIZED_ARCHETYPE : archetype;

    switch (arch) {
        case 0: { // Archetype 0: Standard Dielectric/Conductor PBR (Cook-Torrance GGX)
            return evaluateCookTorranceGGX(baseColor, roughness, metallic, N, V, L, lightRadiance);
        }
        case 1: { // Archetype 1: Subsurface Scattering (Diffusion Profile)
            float NdotL = max(dot(N, L), 0.0);
            float wrappedNdotL = clamp((dot(N, L) + 0.40) / 1.40, 0.0, 1.0);
            vec3 sssColor = exp(-vec3(0.40, 0.08, 0.18) * (1.0 - wrappedNdotL) * 2.0) * baseColor;
            vec3 H = normalize(V + L);
            float NdotH = max(dot(N, H), 0.0);
            float VdotH = max(dot(V, H), 0.0);
            float d_j = NdotH * NdotH * (0.008 - 1.0) + 1.0;
            float D_j = 0.008 / (3.14159265 * d_j * d_j + 0.0001);
            float F_j = 0.05 + 0.95 * pow(clamp(1.0 - VdotH, 0.0, 1.0), 5.0);
            float NdotV = max(dot(N, V), 1e-4);
            vec3 specJade = vec3(D_j * F_j * 0.65 / (4.0 * max(NdotL, 0.01) * NdotV + 0.001) * max(NdotL, 0.01));
            return (sssColor * wrappedNdotL * 0.95 + specJade) * lightRadiance;
        }
        case 2: { // Archetype 2: Dielectric Transmission / Glass
            vec3 H = normalize(V + L);
            float NdotH = max(dot(N, H), 0.0);
            float VdotH = max(dot(V, H), 0.0);
            float NdotL = max(dot(N, L), 0.0);
            float NdotV = max(dot(N, V), 1e-4);
            if (NdotL <= 0.0) return vec3(0.0);
            float d = NdotH * NdotH * (0.0004 - 1.0) + 1.0;
            float D = 0.0004 / (3.14159265 * d * d + 0.0001);
            float F_l = 0.04 + 0.96 * pow(clamp(1.0 - VdotH, 0.0, 1.0), 5.0);
            return lightRadiance * (D * F_l / (4.0 * NdotL * NdotV + 0.001) * NdotL);
        }
        case 3: { // Archetype 3: Anisotropic Velvet / Fabric Sheen (Charlie Micro-Fiber)
            float NdotL = max(dot(N, L), 0.0);
            vec3 H = normalize(V + L);
            float NdotH = max(dot(N, H), 0.0);
            float VdotH = max(dot(V, H), 0.0);
            float invAlpha = 1.0 / max(roughness, 0.25);
            float sinThetaH = sqrt(max(0.0, 1.0 - NdotH * NdotH));
            float D_charlie = (2.0 + invAlpha) * pow(sinThetaH, invAlpha) / 6.283185;
            float F_velvet = pow(clamp(1.0 - VdotH, 0.0, 1.0), 4.0);
            vec3 sheen = vec3(0.95, 0.85, 0.95) * (D_charlie * F_velvet * 0.75 * NdotL);
            return (baseColor * (NdotL * 0.40 + 0.06) + sheen) * lightRadiance;
        }
        case 4: { // Archetype 4: Weathered Multi-Layered Conductor/Dielectric (Dynamic Rust & Patina)
            float rustNoise = rustFBM3D_common(hitPos * 0.05);
            float rustFine = noise3D_common(hitPos * 0.2) * 0.5 + 0.5;
            float rustVal = rustNoise * 0.80 + rustFine * 0.20;
            float rustMask = smoothstep(0.44, 0.54, rustVal);
            vec3 rustAlbedo = mix(vec3(0.85, 0.38, 0.12), vec3(0.35, 0.16, 0.08), rustFine);
            vec3 finalAlbedo = mix(baseColor, rustAlbedo, rustMask);
            float finalRough = mix(roughness, 0.85, rustMask);
            float finalMetal = mix(metallic, 0.0, rustMask);
            return evaluateCookTorranceGGX(finalAlbedo, finalRough, finalMetal, N, V, L, lightRadiance);
        }
        case 5: { // Archetype 5: Polished Architectural Stone / Terrazzo
            float NdotL = max(dot(N, L), 0.0);
            vec3 H = normalize(V + L);
            float NdotH = max(dot(N, H), 0.0);
            float VdotH = max(dot(V, H), 0.0);
            float NdotV = max(dot(N, V), 1e-4);
            vec3 diff = baseColor * (NdotL / 3.14159265);
            float alpha2 = 0.008;
            float d_f = NdotH * NdotH * (alpha2 - 1.0) + 1.0;
            float D_f = alpha2 / (3.14159265 * d_f * d_f + 0.0001);
            float F_f = 0.06 + 0.94 * pow(clamp(1.0 - VdotH, 0.0, 1.0), 5.0);
            vec3 coatSpec = vec3((D_f * F_f * 0.70) / (4.0 * max(NdotL, 0.01) * NdotV + 0.001) * NdotL);
            return (diff + coatSpec) * lightRadiance;
        }
        case 6: { // Archetype 6: Clearcoat Automotive Paint with Voronoi Micro-Flakes
            float NdotL = max(dot(N, L), 0.0);
            vec3 H = normalize(V + L);
            float NdotH = max(dot(N, H), 0.0);
            float VdotH = max(dot(V, H), 0.0);
            float NdotV = max(dot(N, V), 1e-4);
            float D_c = NdotH * NdotH * (0.0003 - 1.0) + 1.0;
            float D_clear = 0.0003 / (3.14159265 * D_c * D_c + 0.0001);
            float F_clear = 0.04 + 0.96 * pow(clamp(1.0 - VdotH, 0.0, 1.0), 5.0);
            vec3 specClear = vec3((D_clear * F_clear * 0.90) / (4.0 * max(NdotL, 0.01) * NdotV + 0.001) * NdotL);
            vec3 baseSpec = evaluateCookTorranceGGX(baseColor, 0.25, 0.85, N, V, L, lightRadiance);
            float flakeNoise = hash31_common(floor(hitPos * 12.0));
            float flakeGlint = pow(NdotH, 140.0) * step(0.65, flakeNoise) * 4.0;
            return baseSpec + (specClear + vec3(flakeGlint)) * lightRadiance;
        }
        case 7: { // Archetype 7: Alpha-Tested Foliage & Thin-Sheet Transmission
            float NdotL = dot(N, L);
            float wrappedNdotL = clamp((NdotL + 0.35) / 1.35, 0.0, 1.0);
            float backTrans = clamp((-NdotL + 0.30) / 1.30, 0.0, 1.0);
            vec3 transColor = baseColor * vec3(1.1, 1.3, 0.4) * backTrans * 0.70;
            vec3 H = normalize(V + L);
            float NdotH = max(dot(N, H), 0.0);
            float D_wax = pow(NdotH, 18.0) * 0.25;
            return (baseColor * (wrappedNdotL * 0.65 + 0.08) + transColor + vec3(D_wax)) * lightRadiance;
        }
    }
    return vec3(0.0);
}

#if defined(BINDING_TEX_HDR_BUF) && defined(BINDING_TEX_PIX_BUF)
vec4 sampleTexture(uint texIdx, vec2 uv) {
    TextureHeaderGpu h = texHeaders.headers[texIdx];
    if (h.width == 0u || h.height == 0u) return vec4(1.0);
    float fx = fract(uv.x) * float(h.width) - 0.5;
    float fy = fract(uv.y) * float(h.height) - 0.5;
    int x0 = int(floor(fx));
    int y0 = int(floor(fy));
    float wx = fract(fx);
    float wy = fract(fy);
    
    int x1 = (x0 + 1) % int(h.width);
    int y1 = (y0 + 1) % int(h.height);
    x0 = (x0 < 0) ? (x0 + int(h.width)) : x0;
    y0 = (y0 < 0) ? (y0 + int(h.height)) : y0;
    
    vec4 c00 = unpackUnorm4x8(texPixels.pixels[h.offset + uint(y0) * h.width + uint(x0)]);
    vec4 c10 = unpackUnorm4x8(texPixels.pixels[h.offset + uint(y0) * h.width + uint(x1)]);
    vec4 c01 = unpackUnorm4x8(texPixels.pixels[h.offset + uint(y1) * h.width + uint(x0)]);
    vec4 c11 = unpackUnorm4x8(texPixels.pixels[h.offset + uint(y1) * h.width + uint(x1)]);
    
    return mix(mix(c00, c10, wx), mix(c01, c11, wx), wy);
}

bool traceShadowRay(vec3 origin, vec3 normal, vec3 lightDir, float maxDist) {
    rayQueryEXT sQuery;
    vec3 rayOrig = origin + normal * 0.12;
    rayQueryInitializeEXT(sQuery, topLevelAS,
        gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT | gl_RayFlagsOpaqueEXT,
        0xFF, rayOrig, 0.05, lightDir, maxDist);
    while (rayQueryProceedEXT(sQuery)) {}
    return (rayQueryGetIntersectionTypeEXT(sQuery, true) != gl_RayQueryCommittedIntersectionNoneEXT);
}

vec3 evaluateGltfPbr(
    GltfMaterialGpu mat,
    vec3 hitPos,
    vec3 geomNormal,
    vec3 smoothNormal,
    vec4 tangent,
    vec2 uv,
    vec3 inDir,
    uint sceneType,
    bool enableShadows
) {
    vec3 V = -inDir;

    // Normal Perturbation from normal map
    vec3 N = smoothNormal;
    if (mat.normalTexIdx >= 0 && length(tangent.xyz) > 0.01) {
        vec3 mapNorm = sampleTexture(uint(mat.normalTexIdx), uv).rgb * 2.0 - 1.0;
        mapNorm.xy *= mat.normalScale;
        vec3 T = normalize(tangent.xyz - dot(tangent.xyz, N) * N);
        vec3 B = cross(N, T) * tangent.w;
        mat3 TBN = mat3(T, B, N);
        N = normalize(TBN * mapNorm);
    }
    if (dot(N, inDir) > 0.0) N = -N;

    // Base Color & Alpha
    vec4 baseSample = (mat.baseColorTexIdx >= 0) ? sampleTexture(uint(mat.baseColorTexIdx), uv) : vec4(1.0);
    vec3 baseColor = pow(baseSample.rgb, vec3(2.2)) * mat.baseColorFactor.rgb;
    float alpha = baseSample.a * mat.baseColorFactor.a;
    if (mat.alphaMode == 1u && alpha < mat.alphaCutoff) return vec3(0.0);

    // Metallic & Roughness
    float roughness = mat.roughnessFactor;
    float metallic = mat.metallicFactor;
    if (mat.metallicRoughnessTexIdx >= 0) {
        vec4 mr = sampleTexture(uint(mat.metallicRoughnessTexIdx), uv);
        roughness *= mr.g;
        metallic *= mr.b;
    }
    roughness = clamp(roughness, 0.04, 1.0);
    metallic = clamp(metallic, 0.0, 1.0);

    // Ambient Occlusion
    float ao = 1.0;
    if (mat.occlusionTexIdx >= 0) {
        ao = sampleTexture(uint(mat.occlusionTexIdx), uv).r;
    }

    // Emissive
    vec3 emissive = mat.emissiveFactor.rgb;
    if (mat.emissiveTexIdx >= 0) {
        emissive *= pow(sampleTexture(uint(mat.emissiveTexIdx), uv).rgb, vec3(2.2));
    }

    uint arch = (SPECIALIZED_ARCHETYPE != 0xFFFFFFFFu) ? SPECIALIZED_ARCHETYPE : mat.archetype;
    vec3 totalRadiance = vec3(0.0);

    if (sceneType == 2u) {
        // Sponza Indoor Atrium: Direct Sun Streaming through vaulted ceiling with Ray-Traced Shadow
        vec3 sunDir = normalize(vec3(0.35, 0.90, -0.25));
        float sunShadow = 1.0;
        if (enableShadows && dot(geomNormal, sunDir) > 0.0) {
            if (traceShadowRay(hitPos, geomNormal, sunDir, 10000.0)) {
                sunShadow = 0.0;
            }
        }
        vec3 sunRadiance = vec3(1.0, 0.95, 0.88) * 4.5 * sunShadow;
        totalRadiance += evaluateMaterialArchetypeDirect(arch, hitPos, baseColor, roughness, metallic, mat.ior, N, V, sunDir, sunRadiance, ao);

        // Warm Colonnade Lanterns
        const vec3 lightPos[3] = vec3[3](
            vec3(0.0, 450.0, 0.0),
            vec3(-650.0, 250.0, 0.0),
            vec3(650.0, 250.0, 0.0)
        );
        const vec3 lightColor[3] = vec3[3](
            vec3(1.0, 0.85, 0.60) * 2.2,
            vec3(1.0, 0.80, 0.50) * 1.8,
            vec3(1.0, 0.80, 0.50) * 1.8
        );
        for (int i = 0; i < 3; ++i) {
            vec3 toL = lightPos[i] - hitPos;
            float dist2 = dot(toL, toL);
            vec3 L = normalize(toL);
            float atten = 1.0 / (1.0 + 0.000004 * dist2);
            totalRadiance += evaluateMaterialArchetypeDirect(arch, hitPos, baseColor, roughness, metallic, mat.ior, N, V, L, lightColor[i] * atten, ao);
        }

        // Vaulted Atrium Sky Ambient & Floor Bounce
        vec3 skyAmb = mix(vec3(0.12, 0.15, 0.22), vec3(0.70, 0.78, 0.95), clamp(N.y * 0.5 + 0.5, 0.0, 1.0)) * 0.9 * ao;
        vec3 diffAmb = (1.0 - metallic) * baseColor * skyAmb;
        totalRadiance += diffAmb;

        // Specular IBL Approximation
        vec3 R = reflect(inDir, N);
        vec3 F0 = mix(vec3(0.04), baseColor, metallic);
        vec3 F_env = F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - max(dot(N, V), 0.0), 0.0, 1.0), 5.0);
        vec3 specEnv = mix(vec3(0.12, 0.15, 0.20), vec3(0.80, 0.88, 1.0), clamp(R.y * 0.5 + 0.5, 0.0, 1.0)) * F_env * (1.0 - roughness * 0.7) * 0.6 * ao;
        totalRadiance += specEnv;
    } else {
        // ToyCar Automotive Showroom: 3-Point Studio Softbox Rig with Contact Grounding
        const vec3 studioDir[3] = vec3[3](
            normalize(vec3(0.6, 0.8, 0.5)),
            normalize(vec3(-0.7, 0.5, 0.4)),
            normalize(vec3(-0.2, 0.4, -0.9))
        );
        const vec3 studioColor[3] = vec3[3](
            vec3(1.0, 0.97, 0.92) * 3.8,
            vec3(0.75, 0.88, 1.0) * 1.8,
            vec3(1.0, 1.0, 1.0) * 2.5
        );

        if (arch == 2u) {
            // Automotive Dielectric Safety Glass (Transmission + Cauchy Chromatic Dispersion)
            vec3 cabinInterior = vec3(0.02, 0.025, 0.03);
            vec3 R = reflect(inDir, N);
            vec3 refrDirR = refract(inDir, N, 1.0 / (mat.ior - 0.02));
            vec3 refrDirG = refract(inDir, N, 1.0 / mat.ior);
            vec3 refrDirB = refract(inDir, N, 1.0 / (mat.ior + 0.02));
            if (length(refrDirG) < 0.01) { refrDirR = R; refrDirG = R; refrDirB = R; }

            vec3 envRefl = mix(vec3(0.12, 0.13, 0.16), vec3(0.40, 0.44, 0.50), clamp(R.y * 1.5, 0.0, 1.0));
            vec3 envRefr;
            envRefr.r = mix(vec3(0.06, 0.07, 0.09), vec3(0.18, 0.20, 0.24), clamp(refrDirR.y * 1.5, 0.0, 1.0)).r;
            envRefr.g = mix(vec3(0.06, 0.07, 0.09), vec3(0.18, 0.20, 0.24), clamp(refrDirG.y * 1.5, 0.0, 1.0)).g;
            envRefr.b = mix(vec3(0.06, 0.07, 0.09), vec3(0.18, 0.20, 0.24), clamp(refrDirB.y * 1.5, 0.0, 1.0)).b;

            vec3 glassTint = vec3(0.88, 0.92, 0.94);
            vec3 transmittedLight = mix(cabinInterior, envRefr, 0.35) * glassTint;
            float F = 0.04 + 0.96 * pow(clamp(1.0 - max(dot(N, V), 0.0), 0.0, 1.0), 5.0);

            vec3 specLights = vec3(0.0);
            for (int i = 0; i < 3; ++i) {
                specLights += evaluateMaterialArchetypeDirect(2u, hitPos, baseColor, roughness, metallic, mat.ior, N, V, studioDir[i], studioColor[i], ao);
            }
            return mix(transmittedLight, envRefl, F) + specLights;
        }

        float keyShadow = 1.0;
        if (enableShadows && dot(geomNormal, studioDir[0]) > 0.0) {
            if (traceShadowRay(hitPos, geomNormal, studioDir[0], 5000.0)) {
                keyShadow = 0.0;
            }
        }
        totalRadiance += evaluateMaterialArchetypeDirect(arch, hitPos, baseColor, roughness, metallic, mat.ior, N, V, studioDir[0], studioColor[0] * keyShadow, ao);
        totalRadiance += evaluateMaterialArchetypeDirect(arch, hitPos, baseColor, roughness, metallic, mat.ior, N, V, studioDir[1], studioColor[1], ao);
        totalRadiance += evaluateMaterialArchetypeDirect(arch, hitPos, baseColor, roughness, metallic, mat.ior, N, V, studioDir[2], studioColor[2], ao);

        // Contact proximity shadow under vehicle chassis onto cloth pedestal
        if (enableShadows && hitPos.y < 35.0 && hitPos.y > 0.0) {
            float contactOcc = mix(0.35, 1.0, clamp(hitPos.y / 35.0, 0.0, 1.0));
            ao *= contactOcc;
        }

        // Studio Floor & Ceiling Ambient
        vec3 studioAmb = mix(vec3(0.06, 0.06, 0.07), vec3(0.25, 0.26, 0.28), clamp(N.y * 0.5 + 0.5, 0.0, 1.0)) * ao;
        totalRadiance += (1.0 - metallic) * baseColor * studioAmb;

        // Automotive Clearcoat & Metallic Reflection
        vec3 R = reflect(inDir, N);
        vec3 F0 = mix(vec3(0.04), baseColor, metallic);
        vec3 F_env = F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - max(dot(N, V), 0.0), 0.0, 1.0), 5.0);
        vec3 envRefl = mix(vec3(0.12, 0.13, 0.16), vec3(0.24, 0.26, 0.30), clamp(R.y * 1.5, 0.0, 1.0));
        vec3 specEnv = envRefl * F_env * (1.0 - roughness * 0.5) * ao;
        totalRadiance += specEnv;
    }

    totalRadiance += emissive;
    return totalRadiance;
}

vec3 evaluateGltfPbr(
    GltfMaterialGpu mat,
    vec3 hitPos,
    vec3 geomNormal,
    vec3 smoothNormal,
    vec4 tangent,
    vec2 uv,
    vec3 inDir,
    uint sceneType
) {
    return evaluateGltfPbr(mat, hitPos, geomNormal, smoothNormal, tangent, uv, inDir, sceneType, true);
}
#endif
