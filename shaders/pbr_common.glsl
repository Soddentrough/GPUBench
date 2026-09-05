// GPUBench Industry-Standard PBR Shading Pipeline (Cook-Torrance GGX)
// Shared across Traditional Megakernel and Work Lists / DGC pathways.

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

bool traceShadowRay(vec3 origin, vec3 normal, vec3 lightDir, float maxDist) {
    rayQueryEXT sQuery;
    vec3 rayOrig = origin + normal * 0.12;
    rayQueryInitializeEXT(sQuery, topLevelAS,
        gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT | gl_RayFlagsOpaqueEXT,
        0xFF, rayOrig, 0.05, lightDir, maxDist);
    while (rayQueryProceedEXT(sQuery)) {}
    return (rayQueryGetIntersectionTypeEXT(sQuery, true) != gl_RayQueryCommittedIntersectionNoneEXT);
}

float traceAreaShadow(vec3 origin, vec3 normal, vec3 lightDir, float maxDist, float lightRadius, inout uint rng, int numSamples) {
    vec3 up = abs(lightDir.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(up, lightDir));
    vec3 bitangent = cross(lightDir, tangent);

    float unoccluded = 0.0;
    vec3 rayOrig = origin + normal * 0.12;

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
            0xFF, rayOrig, 0.05, jitterDir, maxDist);
        while (rayQueryProceedEXT(sQuery)) {}

        if (rayQueryGetIntersectionTypeEXT(sQuery, true) == gl_RayQueryCommittedIntersectionNoneEXT) {
            unoccluded += 1.0;
        }
    }

    return unoccluded / float(numSamples);
}

float computeRTAO(vec3 origin, vec3 normal, inout uint rng, int numSamples, float maxDist) {
    float occlusion = 0.0;
    vec3 rayOrig = origin + normal * 0.08;

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
            0xFF, rayOrig, 0.02, aoDir, maxDist);
        while (rayQueryProceedEXT(aoQuery)) {}

        if (rayQueryGetIntersectionTypeEXT(aoQuery, true) != gl_RayQueryCommittedIntersectionNoneEXT) {
            float t = rayQueryGetIntersectionTEXT(aoQuery, true);
            float atten = clamp(1.0 - t / maxDist, 0.0, 1.0);
            occlusion += atten;
        }
    }

    return clamp(1.0 - (occlusion / float(numSamples)), 0.0, 1.0);
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

    vec3 totalRadiance = vec3(0.0);

    if (sceneType == 2u) {
        // Sponza Indoor Atrium: Direct Sun Streaming through vaulted ceiling with Area Light Soft Shadow
        vec3 sunDir = normalize(vec3(0.35, 0.90, -0.25));
        float sunShadow = 1.0;
        if (enableShadows && dot(geomNormal, sunDir) > 0.0) {
            sunShadow = traceAreaShadow(hitPos, geomNormal, sunDir, 10000.0, 0.035, rng, 4);
        }
        vec3 sunRadiance = vec3(1.0, 0.95, 0.88) * 4.5 * sunShadow;
        totalRadiance += evaluateCookTorranceGGX(baseColor, roughness, metallic, N, V, sunDir, sunRadiance);

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
            totalRadiance += evaluateCookTorranceGGX(baseColor, roughness, metallic, N, V, L, lightColor[i] * atten);
        }

        // Ray-Traced Ambient Occlusion (RTAO) with 4 stratified rays
        float rtao = computeRTAO(hitPos, geomNormal, rng, 4, 350.0);
        ao *= rtao;

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
    } else if (sceneType == 0u) {
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

        if (mat.transmissionFactor > 0.05) {
            // Automotive Dielectric Safety Glass (Physical Snell's Law Refraction Bounce)
            vec3 R = reflect(inDir, N);
            float ior = (mat.ior > 1.0) ? mat.ior : 1.52;
            bool entering = dot(inDir, N) < 0.0;
            vec3 refrNormal = entering ? N : -N;
            float eta = entering ? (1.0 / ior) : ior;
            vec3 refrDir = refract(inDir, refrNormal, eta);

            // Total Internal Reflection (TIR)
            bool tir = (dot(refrDir, refrDir) < 0.01);
            if (tir) refrDir = R;

            vec3 envRefl = mix(vec3(0.12, 0.13, 0.16), vec3(0.40, 0.44, 0.50), clamp(R.y * 1.5, 0.0, 1.0));
            vec3 envRefr = mix(vec3(0.06, 0.07, 0.09), vec3(0.18, 0.20, 0.24), clamp(refrDir.y * 1.5, 0.0, 1.0));

            // Trace secondary refraction ray through the glass into interior/backdrop
            vec3 transmittedRadiance = envRefr;
            if (!tir) {
                rayQueryEXT refrQuery;
                vec3 refrOrig = hitPos + refrDir * 0.05;
                rayQueryInitializeEXT(refrQuery, topLevelAS, gl_RayFlagsOpaqueEXT, 0xFF, refrOrig, 0.02, refrDir, 2500.0);
                while (rayQueryProceedEXT(refrQuery)) {}

                if (rayQueryGetIntersectionTypeEXT(refrQuery, true) == gl_RayQueryCommittedIntersectionTriangleEXT) {
                    float tRefr = rayQueryGetIntersectionTEXT(refrQuery, true);
                    uint primRefr = rayQueryGetIntersectionPrimitiveIndexEXT(refrQuery, true);
                    vec2 baryRefr = rayQueryGetIntersectionBarycentricsEXT(refrQuery, true);

                    // Interpolate interior hit attributes
                    uint baseR = primRefr * 36u;
                    vec3 rp0 = vec3(vbuf.vertices[baseR + 0], vbuf.vertices[baseR + 1], vbuf.vertices[baseR + 2]);
                    vec3 rn0 = vec3(vbuf.vertices[baseR + 3], vbuf.vertices[baseR + 4], vbuf.vertices[baseR + 5]);
                    vec2 ruv0 = vec2(vbuf.vertices[baseR + 10], vbuf.vertices[baseR + 11]);

                    baseR += 12u;
                    vec3 rp1 = vec3(vbuf.vertices[baseR + 0], vbuf.vertices[baseR + 1], vbuf.vertices[baseR + 2]);
                    vec3 rn1 = vec3(vbuf.vertices[baseR + 3], vbuf.vertices[baseR + 4], vbuf.vertices[baseR + 5]);
                    vec2 ruv1 = vec2(vbuf.vertices[baseR + 10], vbuf.vertices[baseR + 11]);

                    baseR += 12u;
                    vec3 rp2 = vec3(vbuf.vertices[baseR + 0], vbuf.vertices[baseR + 1], vbuf.vertices[baseR + 2]);
                    vec3 rn2 = vec3(vbuf.vertices[baseR + 3], vbuf.vertices[baseR + 4], vbuf.vertices[baseR + 5]);
                    vec2 ruv2 = vec2(vbuf.vertices[baseR + 10], vbuf.vertices[baseR + 11]);

                    float rw = 1.0 - baryRefr.x - baryRefr.y;
                    vec3 rHitPos = rw * rp0 + baryRefr.x * rp1 + baryRefr.y * rp2;
                    vec3 rNorm = normalize(rw * rn0 + baryRefr.x * rn1 + baryRefr.y * rn2);
                    vec2 rUv = rw * ruv0 + baryRefr.x * ruv1 + baryRefr.y * ruv2;

                    uint rMatId = triMatBuf.triangleMats[primRefr];
                    GltfMaterialGpu rMat = matBuf.materials[rMatId];
                    vec4 rBaseCol = (rMat.baseColorTexIdx >= 0) ? sampleTexture(uint(rMat.baseColorTexIdx), rUv) : vec4(1.0);
                    vec3 interiorAlbedo = pow(rBaseCol.rgb, vec3(2.2)) * rMat.baseColorFactor.rgb;

                    // Light the interior surface with studio key light & softbox ambient
                    float rNdotL = max(dot(rNorm, studioDir[0]), 0.0);
                    float rShadow = (rNdotL > 0.0) ? (traceShadowRay(rHitPos, rNorm, studioDir[0], 5000.0) ? 0.2 : 1.0) : 0.2;
                    vec3 interiorLight = interiorAlbedo * (studioColor[0] * rNdotL * rShadow + vec3(0.35, 0.38, 0.42));

                    // Beer-Lambert physical volumetric absorption through glass thickness
                    vec3 absorption = vec3(0.08, 0.04, 0.03); // Slight dark smoke tint
                    vec3 transmissionColor = exp(-absorption * min(tRefr, 50.0));
                    transmittedRadiance = interiorLight * transmissionColor;
                }
            }

            // Fresnel reflection
            float F0_g = ((1.0 - ior) / (1.0 + ior)) * ((1.0 - ior) / (1.0 + ior));
            float F = F0_g + (1.0 - F0_g) * pow(clamp(1.0 - max(dot(N, V), 0.0), 0.0, 1.0), 5.0);

            // Studio softbox specular highlights on the glass surface
            vec3 specLights = vec3(0.0);
            for (int i = 0; i < 3; ++i) {
                vec3 H = normalize(V + studioDir[i]);
                float NdotH = max(dot(N, H), 0.0);
                float VdotH = max(dot(V, H), 0.0);
                float NdotL = max(dot(N, studioDir[i]), 0.0);
                float NdotV = max(dot(N, V), 0.0);
                if (NdotL > 0.0) {
                    float d = NdotH * NdotH * (0.0004 - 1.0) + 1.0;
                    float D = 0.0004 / (3.14159 * d * d + 0.0001);
                    float F_l = F0_g + (1.0 - F0_g) * pow(clamp(1.0 - VdotH, 0.0, 1.0), 5.0);
                    specLights += studioColor[i] * (D * F_l / (4.0 * NdotL * NdotV + 0.001) * NdotL);
                }
            }

            return mix(transmittedRadiance, envRefl, F) + specLights;
        }

        // Studio key light with Area Light Soft Shadow
        float keyShadow = 1.0;
        if (enableShadows && dot(geomNormal, studioDir[0]) > 0.0) {
            keyShadow = traceAreaShadow(hitPos, geomNormal, studioDir[0], 5000.0, 0.065, rng, 4);
        }
        totalRadiance += evaluateCookTorranceGGX(baseColor, roughness, metallic, N, V, studioDir[0], studioColor[0] * keyShadow);
        totalRadiance += evaluateCookTorranceGGX(baseColor, roughness, metallic, N, V, studioDir[1], studioColor[1]);
        totalRadiance += evaluateCookTorranceGGX(baseColor, roughness, metallic, N, V, studioDir[2], studioColor[2]);

        // Contact proximity shadow under vehicle chassis onto cloth pedestal
        if (enableShadows && hitPos.y < 35.0 && hitPos.y > 0.0) {
            float contactOcc = mix(0.35, 1.0, clamp(hitPos.y / 35.0, 0.0, 1.0));
            ao *= contactOcc;
        }

        // Studio RTAO (Ray-Traced Ambient Occlusion)
        float rtao = computeRTAO(hitPos, geomNormal, rng, 4, 30.0);
        ao *= rtao;

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
    } else if (sceneType == 3u) {
        // AAA Open-World Alpine Forest: Physical Outdoor Sun, Sky, Nature PBR & Water Refraction
        vec3 sunDir = normalize(vec3(0.42, 0.38, 0.78));
        vec3 sunColor = vec3(1.0, 0.94, 0.84) * 3.8;

        float sunShadow = 1.0;
        if (enableShadows && dot(geomNormal, sunDir) > 0.0) {
            sunShadow = traceAreaShadow(hitPos, geomNormal, sunDir, 3500.0, 0.045, rng, 4);
        }

        // RTAO (Ray-Traced Ambient Occlusion) under canopy and crevices
        float rtao = computeRTAO(hitPos, geomNormal, rng, 4, 35.0);
        ao *= rtao;

        vec3 skyRadiance = evalOutdoorSky(N) * 0.45 * ao;

        if (mat.transmissionFactor > 0.80) {
            // Material 5: River Water Surface with Snell's Law Refraction & Riverbed Bathymetry
            vec2 waveUV = hitPos.xy * 0.08;
            float w1 = sin(waveUV.x * 3.2 + waveUV.y * 2.1) * 0.025;
            float w2 = cos(waveUV.x * 6.5 - waveUV.y * 4.2) * 0.015;
            vec3 waterNorm = normalize(N + vec3(w1, w2, 0.0));

            vec3 refrDir = refract(inDir, waterNorm, 1.0 / 1.333);
            if (length(refrDir) < 0.01) refrDir = reflect(inDir, waterNorm);

            rayQueryEXT waterQuery;
            rayQueryInitializeEXT(waterQuery, topLevelAS, gl_RayFlagsOpaqueEXT, 0xFF, hitPos + refrDir * 0.05, 0.001, refrDir, 25.0);
            while (rayQueryProceedEXT(waterQuery)) {}

            vec3 riverbedColor;
            if (rayQueryGetIntersectionTypeEXT(waterQuery, true) != 0) {
                float depth = rayQueryGetIntersectionTEXT(waterQuery, true);
                vec3 bedBase = vec3(0.18, 0.22, 0.20); // Submerged river stones
                vec3 waterExtinction = exp(-vec3(0.15, 0.04, 0.02) * depth * 1.5);
                riverbedColor = bedBase * waterExtinction * (sunColor * 0.5 * sunShadow + skyRadiance);
            } else {
                riverbedColor = vec3(0.02, 0.08, 0.12);
            }

            float F_w = 0.02 + 0.98 * pow(clamp(1.0 - max(dot(waterNorm, V), 0.0), 0.0, 1.0), 5.0);
            vec3 reflSky = evalOutdoorSky(reflect(inDir, waterNorm));
            vec3 halfVec = normalize(V + sunDir);
            float sunGlint = pow(max(dot(waterNorm, halfVec), 0.0), 180.0) * 12.0 * sunShadow;
            totalRadiance = mix(riverbedColor, reflSky, F_w) + sunColor * sunGlint;
        } else if (mat.transmissionFactor > 0.25) {
            // Material 0: Canopy Leaves & Needles (Two-sided thin-surface transmission)
            float backNdotL = max(dot(-N, sunDir), 0.0);
            vec3 transmission = baseColor * vec3(1.1, 1.4, 0.6) * backNdotL * sunColor * sunShadow * 0.85;
            float F_sheen = pow(clamp(1.0 - max(dot(N, V), 0.0), 0.0, 1.0), 4.0) * 0.45;
            totalRadiance = evaluateCookTorranceGGX(baseColor, roughness, metallic, N, V, sunDir, sunColor * sunShadow)
                          + transmission + vec3(F_sheen) * skyRadiance + baseColor * skyRadiance;
        } else if (roughness < 0.35 && baseColor.r > 0.85) {
            // Material 6: Alpine Snow & Frost with Crystalline Micro-Glints
            vec3 halfVec = normalize(V + sunDir);
            float NdotH = max(dot(N, halfVec), 0.0);
            float glint = pow(NdotH, 120.0) * 8.0 * sunShadow;
            totalRadiance = evaluateCookTorranceGGX(baseColor, 0.35, 0.0, N, V, sunDir, sunColor * sunShadow)
                          + vec3(glint) * sunColor + baseColor * skyRadiance;
        } else {
            // Materials 1, 2, 3, 4, 7: Bark, Granite Rock, Dirt, Grass, Timber
            vec3 effColor = baseColor;
            float effRough = roughness;
            // Moisture darkening on dirt/rocks near waterline
            if (hitPos.z <= 2.2 && baseColor.g < 0.40) {
                float wetness = clamp(1.0 - hitPos.z / 2.2, 0.0, 1.0);
                effColor = mix(effColor, effColor * 0.55, wetness);
                effRough = mix(effRough, 0.15, wetness);
            }
            // Multi-frequency natural procedural surface variation
            vec2 fuv = hitPos.xy * 0.15;
            float texVar = sin(fuv.x * 3.7 + fuv.y * 2.3) * cos(fuv.x * 1.9 - fuv.y * 4.1) * 0.12;
            effColor = clamp(effColor * (1.0 + texVar), 0.0, 1.0);

            totalRadiance = evaluateCookTorranceGGX(effColor, effRough, metallic, N, V, sunDir, sunColor * sunShadow)
                          + effColor * skyRadiance;
        }

        // Atmospheric Aerial Perspective Haze
        vec3 camPos = vec3(-35.0, -50.0, 20.0);
        float camDist = length(hitPos - camPos);
        vec3 rayDirToHit = (camDist > 0.001) ? ((hitPos - camPos) / camDist) : inDir;
        vec3 skyHorizon = evalOutdoorSky(rayDirToHit);
        float hazeFactor = clamp(1.0 - exp(-camDist * 0.00065), 0.0, 1.0);
        if (camDist > 1400.0) {
            float fadeToEnd = smoothstep(1400.0, 2200.0, camDist);
            hazeFactor = mix(hazeFactor, 1.0, fadeToEnd);
        }
        totalRadiance = mix(totalRadiance, skyHorizon, hazeFactor);
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
    uint sceneType,
    inout uint rng
) {
    return evaluateGltfPbr(mat, hitPos, geomNormal, smoothNormal, tangent, uv, inDir, sceneType, true, rng);
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
#endif
