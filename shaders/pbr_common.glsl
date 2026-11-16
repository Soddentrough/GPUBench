// GPUBench Industry-Standard PBR Shading Pipeline (8 Production AAA BSDF Archetypes)
// Shared across Traditional Megakernel, Persistent Megakernel, DGC, and Hardware RT / SER pathways.

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

#ifndef EVAL_OUTDOOR_SKY_DEFINED
#define EVAL_OUTDOOR_SKY_DEFINED
vec3 evalOutdoorSky(vec3 dir) {
    vec3 sunDir = normalize(vec3(0.45, 0.35, 0.82));
    float sunDot = max(dot(dir, sunDir), 0.0);
    vec3 sky = mix(vec3(0.70, 0.82, 0.95), vec3(0.18, 0.42, 0.82), clamp(dir.z, 0.0, 1.0));
    if (dir.z < 0.0) {
        sky = mix(vec3(0.70, 0.82, 0.95), vec3(0.35, 0.30, 0.25), clamp(-dir.z * 2.0, 0.0, 1.0));
    }
    // Sun disc and golden glow
    sky += vec3(1.0, 0.95, 0.85) * pow(sunDot, 120.0) * 8.0;
    sky += vec3(1.0, 0.85, 0.60) * pow(sunDot, 12.0) * 0.8;
    return sky;
}
#endif

#ifndef EVAL_REACTOR_ENV_DEFINED
#define EVAL_REACTOR_ENV_DEFINED
vec3 evalReactorEnvironment(vec3 dir) {
    vec3 sunDir = normalize(vec3(0.24, -0.22, 0.945));
    float sunDot = max(dot(dir, sunDir), 0.0);
    vec3 lowerBay = mix(vec3(0.05, 0.07, 0.10), vec3(0.12, 0.16, 0.22), smoothstep(-0.35, 0.15, dir.z));
    vec3 skyDome = mix(vec3(0.20, 0.30, 0.44), vec3(0.46, 0.68, 0.94), clamp(dir.z, 0.0, 1.0));
    vec3 env = mix(lowerBay, skyDome, smoothstep(0.08, 0.26, dir.z));
    env += vec3(1.0, 0.95, 0.88) * pow(sunDot, 96.0) * 6.0;
    env += vec3(0.22, 0.78, 1.00) * (0.14 * max(0.0, 1.0 - abs(dir.z)));
    return env;
}
#endif

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

#if defined(ENABLE_SPECIALIZED_ARCHETYPE)
layout(constant_id = 0) const uint SPECIALIZED_ARCHETYPE = 0xFFFFFFFFu;
#else
const uint SPECIALIZED_ARCHETYPE = 0xFFFFFFFFu;
#endif

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

    // 2. G: Separable Smith Masking-Shadowing Function (Schlick-GGX)
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

// 8 Production AAA BSDF Archetype Evaluator (NaN-safe, rich multi-lobe shader diversity)
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
    uint arch = (SPECIALIZED_ARCHETYPE != 0xFFFFFFFFu && SPECIALIZED_ARCHETYPE < 8u) ? SPECIALIZED_ARCHETYPE : archetype;

    vec3 vpl = V + L;
    vec3 H = vpl * inversesqrt(max(dot(vpl, vpl), 1e-12));
    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 1e-4);
    float NdotH = max(dot(N, H), 0.0);
    float VdotH = max(dot(V, H), 0.0);
    float fresnel5 = pow(clamp(1.0 - VdotH, 0.0, 1.0), 5.0);

    vec3 upRef = (abs(N.z) < 0.99) ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 T_raw = cross(upRef, N);
    vec3 T = T_raw * inversesqrt(max(dot(T_raw, T_raw), 1e-12));
    vec3 B = cross(N, T);
    float TdotH = dot(T, H);
    float BdotH = dot(B, H);
    float TdotV = dot(T, V);
    float BdotV = dot(B, V);
    float TdotL = dot(T, L);
    float BdotL = dot(B, L);

    switch (arch) {
        case 0: { // Archetype 0: Brushed Aerospace Titanium (Anisotropic GGX + Multi-Scattering Energy Comp)
            if (NdotL <= 0.0) return vec3(0.0);
            float microGrain = sin(hitPos.x * 38.0 + hitPos.z * 26.0) * cos(hitPos.y * 22.0) * 0.035;
            float rEff = clamp(roughness + microGrain, 0.06, 0.95);
            float a = rEff * rEff;
            float ax = max(a * 1.45, 0.012);
            float ay = max(a * 0.45, 0.008);
            float dAniso = (TdotH * TdotH) / (ax * ax) + (BdotH * BdotH) / (ay * ay) + NdotH * NdotH;
            float D = 1.0 / (3.14159265 * ax * ay * max(dAniso * dAniso, 1e-7));
            float lambdaV = NdotL * sqrt(ax * ax * TdotV * TdotV + ay * ay * BdotV * BdotV + NdotV * NdotV);
            float lambdaL = NdotV * sqrt(ax * ax * TdotL * TdotL + ay * ay * BdotL * BdotL + NdotL * NdotL);
            float Vis = 0.5 / max(lambdaV + lambdaL, 1e-4);
            vec3 F0 = mix(vec3(0.04), baseColor, metallic);
            vec3 F = F0 + (1.0 - F0) * fresnel5;
            vec3 multiComp = 1.0 + F0 * (0.38 * rEff);
            vec3 spec = (D * Vis * F) * multiComp;
            vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);
            vec3 diff = kD * (baseColor / 3.1415926535);
            return (diff + spec) * lightRadiance * NdotL;
        }
        case 1: { // Archetype 1: Cryo Ceramic Dielectric (Subsurface Diffusion Profile + SiO2 Glaze Topcoat)
            float rawNdotL = dot(N, L);
            float wrappedNdotL = clamp((rawNdotL + 0.42) / 1.42, 0.0, 1.0);
            vec3 sssColor = exp(-vec3(0.32, 0.14, 0.06) * (1.0 - wrappedNdotL) * 2.1) * baseColor;
            vec3 baseLobe = evaluateCookTorranceGGX(sssColor, max(roughness, 0.25), 0.0, N, V, L, vec3(1.0));
            float f0_coat = pow((max(ior, 1.2) - 1.0) / (max(ior, 1.2) + 1.0), 2.0);
            float a2c = 0.012;
            float dc = NdotH * NdotH * (a2c - 1.0) + 1.0;
            float Dc = a2c / (3.14159265 * dc * dc + 1e-5);
            float kc = 0.125 * 1.12 * 1.12;
            float Gc = (NdotV / (NdotV * (1.0 - kc) + kc)) * (max(NdotL, 0.01) / (max(NdotL, 0.01) * (1.0 - kc) + kc));
            float Fc = f0_coat + (1.0 - f0_coat) * fresnel5;
            vec3 glazeSpec = vec3((Dc * Gc * Fc * 0.72) / max(4.0 * NdotV * max(NdotL, 0.01), 1e-4) * NdotL);
            return (sssColor * wrappedNdotL * 0.62 + baseLobe * 0.45 + glazeSpec) * lightRadiance;
        }
        case 2: { // Archetype 2: Armored Borosilicate Glass (Exact IOR Fresnel + Dual-Lobe Dispersion Specular)
            if (NdotL <= 0.0) return vec3(0.0);
            float iorSafe = max(ior, 1.05);
            float f0_g = pow((iorSafe - 1.0) / (iorSafe + 1.0), 2.0);
            float F_g = f0_g + (1.0 - f0_g) * fresnel5;
            float a2p = 0.00045;
            float d1 = NdotH * NdotH * (a2p - 1.0) + 1.0;
            float D1 = a2p / (3.14159265 * d1 * d1 + 1e-5);
            float a2b = 0.0144;
            float d2 = NdotH * NdotH * (a2b - 1.0) + 1.0;
            float D2 = a2b / (3.14159265 * d2 * d2 + 1e-5);
            float Vis = 1.0 / max(4.0 * NdotL * NdotV, 1e-3);
            vec3 arTint = vec3(0.82, 0.96, 1.00) + 0.18 * sin(vec3(0.0, 2.094, 4.188) + VdotH * 6.28318);
            vec3 spec = arTint * ((D1 * 0.85 + D2 * 0.15) * F_g * Vis * NdotL);
            return lightRadiance * spec;
        }
        case 3: { // Archetype 3: Kapton MLI Foil (3D Micro-Crinkle Normal Perturb + Metallic GGX + Grazing Sheen)
            float c1 = noise3D_common(hitPos * 3.8) * 2.0 - 1.0;
            float c2 = noise3D_common(hitPos * 9.5 + vec3(4.1, 7.3, 1.9)) * 2.0 - 1.0;
            vec3 nc_raw = N + T * (c1 * 0.22) + B * (c2 * 0.22);
            vec3 Nc = nc_raw * inversesqrt(max(dot(nc_raw, nc_raw), 1e-12));
            vec3 baseSpec = evaluateCookTorranceGGX(baseColor, clamp(roughness * 0.85, 0.10, 0.70), metallic, Nc, V, L, lightRadiance);
            float invAlpha = 1.0 / max(roughness, 0.25);
            float sinThetaH = sqrt(max(0.0, 1.0 - NdotH * NdotH));
            float D_charlie = (2.0 + invAlpha) * pow(max(sinThetaH, 0.0), invAlpha) / 6.283185;
            float F_velvet = pow(clamp(1.0 - VdotH, 0.0, 1.0), 4.0);
            vec3 crinkleSheen = mix(baseColor, vec3(1.0, 0.86, 0.54), 0.5) * (D_charlie * F_velvet * 0.52 * NdotL);
            return baseSpec + crinkleSheen * lightRadiance;
        }
        case 4: { // Archetype 4: Weathered Industrial Ferrocrete (FBM Oxide/Spall Layering + Oren-Nayar Rough Diffuse + GGX)
            float rustNoise = rustFBM3D_common(hitPos * 0.10);
            float rustFine = noise3D_common(hitPos * 1.4) * 0.5 + 0.5;
            float spallGrain = hash31_common(floor(hitPos * 14.0));
            float rustVal = rustNoise * 0.72 + rustFine * 0.20 + spallGrain * 0.08;
            float rustMask = smoothstep(0.44, 0.56, rustVal);
            vec3 rustAlbedo = mix(vec3(0.58, 0.28, 0.14), vec3(0.24, 0.15, 0.10), rustFine);
            vec3 finalAlbedo = mix(baseColor, rustAlbedo, rustMask * 0.85);
            float finalRough = mix(roughness, 0.88, rustMask);
            float finalMetal = mix(metallic, 0.0, rustMask);
            float sigma2 = finalRough * finalRough;
            float onA = 1.0 - 0.5 * (sigma2 / (sigma2 + 0.33));
            float onB = 0.45 * (sigma2 / (sigma2 + 0.09));
            float VdotL_proj = dot(V - N * NdotV, L - N * NdotL);
            float s = max(NdotL, NdotV);
            float t = mix(1.0, max(s, 1e-4), step(0.0, VdotL_proj));
            vec3 orenDiff = finalAlbedo * (1.0 - finalMetal) / 3.14159265 * (onA + onB * max(0.0, VdotL_proj) / max(t, 1e-4)) * NdotL;
            vec3 dampSpec = evaluateCookTorranceGGX(finalAlbedo, clamp(finalRough * 0.82, 0.18, 0.95), finalMetal, N, V, L, vec3(1.0));
            return (orenDiff * 0.55 + dampSpec * 0.65) * lightRadiance;
        }
        case 5: { // Archetype 5: Cryo-Plasma Conduit Jacket (Thin-Film Interference Specular + Forward Translucent Scatter)
            float opd = 2.0 * 1.38 * 0.42 * max(VdotH, 0.05);
            vec3 filmTint = 0.55 + 0.45 * cos(6.283185 * (opd + vec3(0.0, 0.333, 0.667)));
            vec3 coatSpec = evaluateCookTorranceGGX(mix(baseColor, filmTint, 0.42), clamp(roughness, 0.05, 0.55), metallic, N, V, L, vec3(1.0));
            float fwdScatter = pow(clamp(dot(-V, L), 0.0, 1.0), 4.0) * 0.35;
            float wrappedNdotL = clamp((dot(N, L) + 0.25) / 1.25, 0.0, 1.0);
            vec3 transDiff = baseColor * (wrappedNdotL * 0.45 + fwdScatter) * filmTint;
            return (coatSpec + transDiff * 0.35) * lightRadiance;
        }
        case 6: { // Archetype 6: Industrial Hazard Clearcoat Enamel (Metallic Primer GGX + Smith Clearcoat + Grid Micro-Flakes)
            vec3 baseSpec = evaluateCookTorranceGGX(baseColor, clamp(roughness + 0.14, 0.18, 0.65), 0.80, N, V, L, vec3(1.0));
            float a2c = 0.00035;
            float D_c = NdotH * NdotH * (a2c - 1.0) + 1.0;
            float D_clear = a2c / (3.14159265 * D_c * D_c + 1e-5);
            float kc = 0.025;
            float G_clear = (NdotV / (NdotV * (1.0 - kc) + kc)) * (max(NdotL, 0.01) / (max(NdotL, 0.01) * (1.0 - kc) + kc));
            float F_clear = 0.04 + 0.96 * fresnel5;
            vec3 specClear = vec3((D_clear * G_clear * F_clear * 0.90) / max(4.0 * max(NdotL, 0.01) * NdotV, 1e-4) * NdotL);
            float flakeNoise = hash31_common(floor(hitPos * 16.0));
            float flakeGlint = pow(max(NdotH, 0.0), 96.0) * step(0.62, flakeNoise) * 3.2 * NdotL;
            return (baseSpec * (1.0 - F_clear * 0.85) + specClear + vec3(flakeGlint)) * lightRadiance;
        }
        case 7: { // Archetype 7: Perforated Anisotropic Hex-Grate (Cross-Wire Aniso GGX + Diffraction Rim + Cavity Atten)
            float rawNdotL = dot(N, L);
            float wrappedNdotL = clamp((rawNdotL + 0.35) / 1.35, 0.0, 1.0);
            float backTrans = clamp((-rawNdotL + 0.25) / 1.25, 0.0, 1.0);
            float cav = mix(0.48, 1.0, ao);
            vec3 meshRim = baseColor * vec3(0.45, 0.72, 0.96) * (backTrans * 0.38 * cav);
            float ax = max(roughness * roughness * 1.6, 0.02);
            float ay = max(roughness * roughness * 0.35, 0.008);
            float dAniso = (TdotH * TdotH) / (ax * ax) + (BdotH * BdotH) / (ay * ay) + NdotH * NdotH;
            float D_wire = 1.0 / (3.14159265 * ax * ay * max(dAniso * dAniso, 1e-6));
            float diffGlint = (0.35 + 0.65 * pow(max(sin(TdotH * 14.0), 0.0), 2.0)) * D_wire * (0.04 + 0.96 * fresnel5) * 0.25;
            vec3 wireSpec = baseColor * (diffGlint / max(4.0 * NdotV * max(NdotL, 0.05), 1e-3)) * NdotL;
            return (baseColor * (wrappedNdotL * 0.58 + 0.08) * cav + meshRim + wireSpec) * lightRadiance;
        }
    }
    return vec3(0.0);
}

// Deterministic AAA Industrial Trim-Sheet & Sub-Panel Hardware Material Resolver
// Preserves ~64% primary material identity (and 100% pure borosilicate glass on baseMatId == 2u)
// while adding physical 12.5 cm bolted titanium seam frames, ceramic bushings, and 4.17 cm micro-ribs/louvers.
uint resolveReactorBayMaterialId(uint baseMatId, vec3 hitPos, vec3 geomNormal) {
    if (baseMatId == 2u) return 2u;
    vec3 an = abs(geomNormal);
    vec2 suv = (an.z >= an.x && an.z >= an.y) ? hitPos.xy
             : ((an.x >= an.y) ? hitPos.yz : hitPos.xz);
    vec2 panelCell = fract(suv * 8.0);
    vec2 microCell = fract(suv * 24.0);
    float diagStripe = fract((suv.x + suv.y) * 10.0);
    uint cellHash = (uint(floor(suv.x * 8.0) + 1024.0) * 1664525u) ^
                    (uint(floor(suv.y * 8.0) + 1024.0) * 1013904223u) ^
                    (baseMatId * 374761393u);

    if (panelCell.x < 0.09 || panelCell.y < 0.09) {
        if (panelCell.x < 0.09 && panelCell.y < 0.09) {
            return 1u;
        }
        return (baseMatId == 0u) ? 4u : 0u;
    }
    if (microCell.x < 0.16 && microCell.y > 0.22 && microCell.y < 0.78) {
        uint sel = (cellHash >> 28u) & 3u;
        if (sel == 0u) return 7u;
        if (sel == 1u) return 6u;
        if (sel == 2u) return 3u;
        return (baseMatId == 5u) ? 0u : 5u;
    }
    if (microCell.y < 0.13 && ((cellHash & 1u) != 0u)) {
        return (diagStripe < 0.35) ? 6u : ((baseMatId == 4u) ? 0u : 4u);
    }
    return baseMatId;
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

uint pcg_hash_pbr(inout uint state) {
    uint oldstate = state;
    state = oldstate * 747796405u + 2891336453u;
    uint word = ((oldstate >> ((oldstate >> 28u) + 4u)) ^ oldstate) * 277803737u;
    return (word >> 22u) ^ word;
}

float rand_float_pbr(inout uint state) {
    return float(pcg_hash_pbr(state) & 0x00FFFFFFu) / 16777216.0;
}

#ifndef PBR_NO_RAY_QUERY
bool traceShadowRay(vec3 origin, vec3 normal, vec3 lightDir, float maxDist) {
    if (maxDist <= 0.05) return false;
    rayQueryEXT sQuery;
    vec3 rayOrig = origin + normal * 0.04;
    rayQueryInitializeEXT(sQuery, topLevelAS,
        gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT | gl_RayFlagsOpaqueEXT,
        0xFF, rayOrig, 0.02, lightDir, maxDist);
    while (rayQueryProceedEXT(sQuery)) {}
    return (rayQueryGetIntersectionTypeEXT(sQuery, true) != gl_RayQueryCommittedIntersectionNoneEXT);
}
#endif

void getSceneSunParams(uint sceneType, out vec3 sunDir, out vec3 sunColor) {
    if (sceneType == 2u) { // Indoor Atrium
        sunDir = normalize(vec3(0.35, 0.90, -0.25));
        sunColor = vec3(1.0, 0.95, 0.88) * 4.5;
    } else if (sceneType == 0u) { // Showroom Studio
        sunDir = normalize(vec3(0.6, 0.8, 0.5));
        sunColor = vec3(1.0, 0.97, 0.92) * 3.5;
    } else if (sceneType == 3u) { // Zero-Day Reactor Bay (Skylight Sun)
        sunDir = normalize(vec3(0.24, -0.22, 0.945));
        sunColor = vec3(1.0, 0.95, 0.88) * 4.2;
    } else {
        sunDir = normalize(vec3(0.45, 0.35, 0.82));
        sunColor = vec3(1.0, 0.95, 0.85) * 3.5;
    }
}

vec3 getSceneArchetypeAlbedo(uint sceneType, uint archetype) {
    if (sceneType == 1u) { // SCENE_OUTDOOR_TERRAIN
        if (archetype == 0u) return vec3(0.65, 0.52, 0.40); // Scree slope / weathered sandstone
        if (archetype == 1u) return vec3(0.08, 0.32, 0.12); // Translucent conifer foliage / needles
        if (archetype == 2u) return vec3(0.012, 0.042, 0.065); // Alpine lake / river water
        if (archetype == 3u) return vec3(0.92, 0.95, 0.98); // High mountain snow & glacial ice (Crisp white snow)
        if (archetype == 4u) return vec3(0.30, 0.31, 0.34); // Exposed granite cliff face
        if (archetype == 5u) return vec3(0.24, 0.44, 0.18); // Valley grassland / alpine meadow (Lush green)
        if (archetype == 6u) return vec3(0.25, 0.17, 0.11); // Pine bark / timber trunk
        return vec3(0.18, 0.20, 0.22);                      // Wet river stones / shoreline gravel
    } else {
        // Showroom Knot / Atrium
        if (archetype == 0u) return vec3(0.65, 0.52, 0.40); // Ruby paint base
        if (archetype == 1u) return vec3(0.18, 0.55, 0.34); // Jade subsurface
        if (archetype == 2u) return vec3(0.95, 0.95, 0.98); // Glass
        if (archetype == 3u) return vec3(0.85, 0.12, 0.22); // Magenta velvet
        if (archetype == 4u) return vec3(0.55, 0.45, 0.35); // Weathered rust
        if (archetype == 5u) return vec3(0.80, 0.80, 0.82); // Cyclorama floor
        if (archetype == 6u) return vec3(0.88, 0.20, 0.10); // Pedestal / Gold
        return vec3(0.12, 0.45, 0.15);                      // Chrome / accent
    }
}

#ifndef PBR_NO_RAY_QUERY
float traceAreaShadow(vec3 origin, vec3 normal, vec3 lightDir, float maxDist, float lightRadius, inout uint rng, int numSamples) {
    vec3 up = abs(lightDir.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(up, lightDir));
    vec3 bitangent = cross(lightDir, tangent);

    float unoccluded = 0.0;
    vec3 rayOrig = origin + normal * 0.04;

    for (int i = 0; i < numSamples; ++i) {
        float phi = float(i) * 2.3999632 + rand_float_pbr(rng) * 0.5;
        float r = sqrt((float(i) + 0.5) / float(numSamples)) * tan(lightRadius);
        vec3 jitterDir = normalize(lightDir + tangent * (cos(phi) * r) + bitangent * (sin(phi) * r));

        if (dot(normal, jitterDir) <= 0.0) {
            continue;
        }

        rayQueryEXT sQuery;
        rayQueryInitializeEXT(sQuery, topLevelAS,
            gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT | gl_RayFlagsOpaqueEXT,
            0xFF, rayOrig, 0.02, jitterDir, maxDist);
        while (rayQueryProceedEXT(sQuery)) {}

        if (rayQueryGetIntersectionTypeEXT(sQuery, true) == gl_RayQueryCommittedIntersectionNoneEXT) {
            unoccluded += 1.0;
        }
    }

    return unoccluded / float(numSamples);
}

float computeRTAO(vec3 origin, vec3 normal, inout uint rng, int numSamples, float maxDist) {
    float occlusion = 0.0;
    vec3 rayOrig = origin + normal * 0.04;

    vec3 up = abs(normal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(up, normal));
    vec3 bitangent = cross(normal, tangent);

    for (int i = 0; i < numSamples; ++i) {
        float u1 = (float(i) + rand_float_pbr(rng)) / float(numSamples);
        float u2 = rand_float_pbr(rng);
        float r = sqrt(u1);
        float theta = 6.28318530718 * u2;
        vec3 localDir = vec3(r * cos(theta), r * sin(theta), sqrt(max(0.0, 1.0 - u1)));
        vec3 aoDir = tangent * localDir.x + bitangent * localDir.y + normal * localDir.z;

        rayQueryEXT aoQuery;
        rayQueryInitializeEXT(aoQuery, topLevelAS,
            gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT | gl_RayFlagsOpaqueEXT,
            0xFF, rayOrig, 0.015, aoDir, maxDist);
        while (rayQueryProceedEXT(aoQuery)) {}

        if (rayQueryGetIntersectionTypeEXT(aoQuery, true) != gl_RayQueryCommittedIntersectionNoneEXT) {
            float t = rayQueryGetIntersectionTEXT(aoQuery, true);
            float atten = clamp(1.0 - t / maxDist, 0.0, 1.0);
            occlusion += atten;
        }
    }

    return clamp(1.0 - (occlusion / float(numSamples)), 0.0, 1.0);
}
#endif

vec3 evaluateGltfPbrInternal(
    GltfMaterialGpu mat,
    vec3 hitPos,
    vec3 geomNormal,
    vec3 smoothNormal,
    vec4 tangent,
    vec2 uv,
    vec3 inDir,
    uint sceneType,
    bool enableShadows,
    bool directOnly,
    bool usePrecomputedVis,
    vec3 precomputedVis,
    inout uint rng
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

    // Base Color
    vec4 baseSample = (mat.baseColorTexIdx >= 0) ? sampleTexture(uint(mat.baseColorTexIdx), uv) : vec4(1.0);
    vec3 baseColor = pow(baseSample.rgb, vec3(2.2)) * mat.baseColorFactor.rgb;

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

    uint arch = (SPECIALIZED_ARCHETYPE != 0xFFFFFFFFu && SPECIALIZED_ARCHETYPE < 8u) ? SPECIALIZED_ARCHETYPE : mat.archetype;
    vec3 directColor = baseColor;
    float directRough = roughness;
    float directMetal = metallic;
    uint directArch = arch;
    vec3 totalRadiance = vec3(0.0);

    if (sceneType == 3u) {
        // Zero-Day Reactor Bay: Overhead Skylight Sun + 3 Hangar Point Lights + Refractive Armored Glass
        vec3 sunDir = normalize(vec3(0.24, -0.22, 0.945));
        vec3 sunColor = vec3(1.0, 0.95, 0.88) * 4.2;

        float sunShadow = 1.0;
        if (enableShadows) {
            if (usePrecomputedVis) {
                sunShadow = precomputedVis.x;
            }
#ifndef PBR_NO_RAY_QUERY
            else if (dot(geomNormal, sunDir) > 0.0) {
                sunShadow = traceAreaShadow(hitPos, geomNormal, sunDir, 2500.0, 0.040, rng, 4);
            }
#endif
        }

        // RTAO (Ray-Traced Ambient Occlusion) inside industrial bay corners & catwalk structures
        float rtao = 1.0;
        if (enableShadows) {
            if (usePrecomputedVis) {
                rtao = precomputedVis.y;
            }
#ifndef PBR_NO_RAY_QUERY
            else {
                rtao = computeRTAO(hitPos, geomNormal, rng, 4, 6.5);
            }
#endif
        }
        ao *= rtao;

        const vec3 rbLightPos[15] = vec3[15](
            vec3( 5.8,   5.2,  7.6),
            vec3(-5.8,   5.2,  7.6),
            vec3( 0.0,  -0.6,  9.2),
            vec3( 0.0,  11.0,  6.0),
            vec3(-11.5, -5.5,  7.2),
            vec3( 11.5, -5.5,  7.2),
            vec3(-11.5, 12.5,  8.5),
            vec3( 11.5, 12.5,  8.5),
            vec3( 12.5,  2.5,  8.0),
            vec3(-14.0,  3.0,  7.8),
            vec3(  0.0, 24.0,  9.5),
            vec3( -4.3, -12.0, 1.4),
            vec3(  4.3, -12.0, 1.4),
            vec3( -8.5, -16.5, 5.0),
            vec3(  8.5, -16.5, 5.0)
        );
        const vec3 rbLightCol[15] = vec3[15](
            vec3(0.18, 0.82, 1.00) * 18.0,
            vec3(0.18, 0.82, 1.00) * 18.0,
            vec3(0.22, 0.86, 1.00) * 16.0,
            vec3(0.22, 0.86, 1.00) * 16.0,
            vec3(1.00, 0.56, 0.16) * 16.0,
            vec3(1.00, 0.56, 0.16) * 16.0,
            vec3(1.00, 0.58, 0.20) * 15.0,
            vec3(1.00, 0.58, 0.20) * 15.0,
            vec3(0.42, 0.92, 0.72) * 15.0,
            vec3(0.42, 0.92, 0.72) * 15.0,
            vec3(0.45, 0.90, 0.78) * 14.0,
            vec3(0.26, 0.78, 0.98) * 12.0,
            vec3(0.26, 0.78, 0.98) * 12.0,
            vec3(1.00, 0.72, 0.38) * 13.0,
            vec3(1.00, 0.72, 0.38) * 13.0
        );

        vec3 envAmb = evalReactorEnvironment(N) * ao;

        if (!enableShadows) {
            // Pure material-shading microbenchmark (mode 4 MK / mode 1 DGC):
            // exercise all 8 divergent BSDF archetypes across 16 Reactor Bay lights
            vec3 reflEnv = evalReactorEnvironment(reflect(inDir, N));
            vec3 shadeLights = vec3(0.0);
#ifdef GL_EXT_control_flow_attributes
            [[dont_unroll]]
#endif
            for (uint li = 0u; li < 16u; ++li) {
                vec3 lDir;
                vec3 lRad;
                if (li == 0u) {
                    lDir = sunDir;
                    lRad = sunColor;
                } else {
                    vec3 toL = rbLightPos[li - 1u] - hitPos;
                    float dist2 = max(dot(toL, toL), 0.25);
                    lDir = toL * inversesqrt(dist2);
                    lRad = rbLightCol[li - 1u] / (1.0 + 0.022 * dist2);
                }
                shadeLights += evaluateMaterialArchetypeDirect(directArch, hitPos, directColor, directRough, directMetal, mat.ior, N, V, lDir, lRad, ao);
            }
            if (arch == 2u) {
                float F_g = 0.04 + 0.96 * pow(clamp(1.0 - max(dot(N, V), 0.0), 0.0, 1.0), 5.0);
                return mix(vec3(0.03, 0.11, 0.16), reflEnv, F_g) + shadeLights + emissive;
            }
            return shadeLights + (1.0 - directMetal) * directColor * envAmb * 0.55 + emissive;
        }

        if (mat.transmissionFactor > 0.80) {
            // Refractive Armored Control-Booth Glass & Coolant Covers (Snell + Beer-Lambert)
            vec3 refrDir = refract(inDir, N, 1.0 / max(mat.ior, 1.01));
            if (length(refrDir) < 0.01) refrDir = reflect(inDir, N);

            vec3 transmittedCol = vec3(0.03, 0.10, 0.14);
            float refrDepth = -1.0;
            if (usePrecomputedVis) {
                refrDepth = precomputedVis.z;
            }
#ifndef PBR_NO_RAY_QUERY
            else {
                rayQueryEXT glassQuery;
                rayQueryInitializeEXT(glassQuery, topLevelAS, gl_RayFlagsOpaqueEXT, 0xFF, hitPos + refrDir * 0.05, 0.001, refrDir, 25.0);
                while (rayQueryProceedEXT(glassQuery)) {}
                if (rayQueryGetIntersectionTypeEXT(glassQuery, true) != 0) {
                    refrDepth = rayQueryGetIntersectionTEXT(glassQuery, true);
                }
            }
#endif
            if (refrDepth >= 0.0) {
                vec3 glassExtinction = exp(-vec3(0.18, 0.05, 0.03) * clamp(refrDepth, 0.0, 20.0) * 0.85);
                vec3 interiorBase = mix(vec3(0.10, 0.18, 0.24), vec3(0.22, 0.42, 0.54), clamp(1.0 - refrDepth * 0.08, 0.0, 1.0));
                transmittedCol = interiorBase * glassExtinction * (sunColor * 0.35 * sunShadow + vec3(0.35, 0.55, 0.72));
            }

            vec3 specLights = vec3(0.0);
#ifdef GL_EXT_control_flow_attributes
            [[dont_unroll]]
#endif
            for (uint li = 0u; li < 16u; ++li) {
                vec3 lDir;
                vec3 lRad;
                if (li == 0u) {
                    lDir = sunDir;
                    lRad = sunColor * sunShadow;
                } else {
                    vec3 toL = rbLightPos[li - 1u] - hitPos;
                    float dist2 = max(dot(toL, toL), 0.25);
                    lDir = toL * inversesqrt(dist2);
                    lRad = rbLightCol[li - 1u] / (1.0 + 0.022 * dist2);
                }
                specLights += evaluateMaterialArchetypeDirect(2u, hitPos, directColor, directRough, directMetal, mat.ior, N, V, lDir, lRad, ao);
            }
            if (directOnly) {
                return specLights + emissive;
            }
            float F_g = 0.04 + 0.96 * pow(clamp(1.0 - max(dot(N, V), 0.0), 0.0, 1.0), 5.0);
            vec3 reflEnv = evalReactorEnvironment(reflect(inDir, N));
            totalRadiance = mix(transmittedCol, reflEnv, F_g) + specLights;
        } else {
            // Opaque Reactor Bay PBR (Titanium, Ceramic/Cryo, MLI Foil, Ferrocrete, Coolant Coils, Clearcoat, Perf Grate)
            vec3 effColor = baseColor;
            float effRough = roughness;
            vec2 fuv = hitPos.xy * 0.20;
            float panelVar = sin(fuv.x * 2.7 + fuv.y * 1.9) * cos(fuv.x * 1.5 - fuv.y * 3.1) * 0.08;
            effColor = clamp(effColor * (1.0 + panelVar), 0.0, 1.0);

#ifdef GL_EXT_control_flow_attributes
            [[dont_unroll]]
#endif
            for (uint li = 0u; li < 16u; ++li) {
                vec3 lDir;
                vec3 lRad;
                if (li == 0u) {
                    lDir = sunDir;
                    lRad = sunColor * sunShadow;
                } else {
                    vec3 toL = rbLightPos[li - 1u] - hitPos;
                    float dist2 = max(dot(toL, toL), 0.25);
                    lDir = toL * inversesqrt(dist2);
                    lRad = rbLightCol[li - 1u] / (1.0 + 0.022 * dist2);
                }
                totalRadiance += evaluateMaterialArchetypeDirect(directArch, hitPos, effColor, effRough, directMetal, mat.ior, N, V, lDir, lRad, ao);
            }

            if (directOnly) {
                return totalRadiance * ao + emissive;
            }

            totalRadiance += (1.0 - directMetal) * effColor * envAmb * 0.55;
            vec3 R = reflect(inDir, N);
            vec3 F0 = mix(vec3(0.04), effColor, directMetal);
            vec3 F_env = F0 + (max(vec3(1.0 - effRough), F0) - F0) * pow(clamp(1.0 - max(dot(N, V), 0.0), 0.0, 1.0), 5.0);
            vec3 specEnv = evalReactorEnvironment(R) * F_env * (1.0 - effRough * 0.55) * 0.65 * ao;
            totalRadiance += specEnv;
        }
    }

    totalRadiance += emissive;
    return totalRadiance;
}

vec3 evaluateGltfPbrPrecomputedVis(
    GltfMaterialGpu mat,
    vec3 hitPos,
    vec3 geomNormal,
    vec3 smoothNormal,
    vec4 tangent,
    vec2 uv,
    vec3 inDir,
    uint sceneType,
    bool enableShadows,
    vec3 precomputedVis,
    inout uint rng
) {
    return evaluateGltfPbrInternal(mat, hitPos, geomNormal, smoothNormal, tangent, uv, inDir, sceneType, enableShadows, false, true, precomputedVis, rng);
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
    bool enableShadows,
    inout uint rng
) {
    return evaluateGltfPbrInternal(mat, hitPos, geomNormal, smoothNormal, tangent, uv, inDir, sceneType, enableShadows, false, false, vec3(1.0, 1.0, -1.0), rng);
}

vec3 evaluateGltfPbrDirect(
    GltfMaterialGpu mat,
    vec3 hitPos,
    vec3 geomNormal,
    vec3 smoothNormal,
    vec4 tangent,
    vec2 uv,
    vec3 inDir,
    uint sceneType,
    inout uint rng
) {
    return evaluateGltfPbrInternal(mat, hitPos, geomNormal, smoothNormal, tangent, uv, inDir, sceneType, true, true, false, vec3(1.0, 1.0, -1.0), rng);
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
    inout uint rng
) {
    return evaluateGltfPbrInternal(mat, hitPos, geomNormal, smoothNormal, tangent, uv, inDir, sceneType, true, false, false, vec3(1.0, 1.0, -1.0), rng);
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
    uint dummyRng = 123456789u;
    return evaluateGltfPbr(mat, hitPos, geomNormal, smoothNormal, tangent, uv, inDir, sceneType, enableShadows, dummyRng);
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
    uint dummyRng = 123456789u;
    return evaluateGltfPbr(mat, hitPos, geomNormal, smoothNormal, tangent, uv, inDir, sceneType, true, dummyRng);
}

void getMultiLight(uint idx, uint totalLights, uint sceneType, out vec3 lightDir, out vec3 lightColor) {
    if (totalLights <= 1u) {
        if (sceneType == 2u) {
            lightDir = normalize(vec3(0.35, 0.90, -0.25));
        } else if (sceneType == 0u) {
            lightDir = normalize(vec3(0.6, 0.8, 0.5));
        } else if (sceneType == 3u) {
            lightDir = normalize(vec3(0.24, -0.22, 0.945));
        } else {
            lightDir = normalize(vec3(0.45, 0.35, 0.82));
        }
        lightColor = vec3(1.0, 0.95, 0.9) * 3.5;
    } else {
        // 8-Octant Clustered Light Distribution (16 lights per octant bin b = 0..7)
        uint b = (idx >> 4u) & 7u;
        uint li = idx & 15u;
        float u = (float(li) + 0.5) / 16.0;
        float v = fract(float(li) * 0.618033988749895 + float(b) * 0.125);
        float phi = 0.15 + 1.27079632679 * v; // [0.15, pi/2 - 0.15] inside positive octant
        float cosTheta = 0.20 + 0.70 * u;     // [0.20, 0.90]
        float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));

        vec3 posOct = vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
        vec3 octSign = vec3(
            (b & 1u) != 0u ? 1.0 : -1.0,
            (b & 2u) != 0u ? 1.0 : -1.0,
            (b & 4u) != 0u ? 1.0 : -1.0
        );
        lightDir = normalize(posOct * octSign);

        float t = float(idx) / float(totalLights);
        vec3 col = mix(vec3(1.0, 0.74, 0.42), vec3(0.32, 0.82, 1.0), fract(t * 5.0));
        lightColor = col * (18.0 / float(totalLights));
    }
}

vec3 evalCookTorranceDirect(vec3 N, vec3 V, vec3 L, vec3 lightRadiance, vec3 albedo, float roughness, float metallic) {
    float NdotL = max(0.0, dot(N, L));
    if (NdotL <= 0.0) return vec3(0.0);
    float NdotV = max(0.001, dot(N, V));
    vec3 H = normalize(V + L);
    float NdotH = max(0.0, dot(N, H));
    float VdotH = max(0.0, dot(V, H));

    // D: GGX Trowbridge-Reitz
    float alpha = max(0.04, roughness * roughness);
    float alpha2 = alpha * alpha;
    float denomD = (NdotH * NdotH * (alpha2 - 1.0) + 1.0);
    float D = alpha2 / (3.14159265 * denomD * denomD);

    // F: Fresnel-Schlick
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    vec3 F = F0 + (1.0 - F0) * pow(clamp(1.0 - VdotH, 0.0, 1.0), 5.0);

    // G: Smith GGX
    float k = (roughness + 1.0);
    k = (k * k) / 8.0;
    float G = (NdotV / (NdotV * (1.0 - k) + k)) * (NdotL / (NdotL * (1.0 - k) + k));

    vec3 spec = (D * F * G) / max(0.001, 4.0 * NdotV * NdotL);
    vec3 diff = (vec3(1.0) - F) * (1.0 - metallic) * (albedo / 3.14159265);

    return (diff + spec) * lightRadiance * NdotL;
}
#endif

