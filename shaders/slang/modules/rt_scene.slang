module rt_scene;

import rt_types;
import rt_math;

public struct Camera {
    public float3 pos;
    public float3 forward;
    public float3 right;
    public float3 up;
    public float fov;
};

public Camera getCamera(uint sceneType, uint isGltf) {
    Camera cam;
    if (isGltf != 0u) {
        if (sceneType == SCENE_AAA_FOREST) {
            // AAA Open-World Woodland Forest: Eye-level trail camera gazing through dense canopy
            cam.pos = float3(0.0, -18.0, 0.55);
            cam.forward = normalize(float3(0.0, 33.0, 1.2));
            cam.right = normalize(cross(cam.forward, float3(0.0, 0.0, 1.0)));
            cam.up = cross(cam.right, cam.forward);
            cam.fov = 0.65;
            return cam;
        } else if (sceneType == SCENE_INDOOR_ATRIUM) {
            // Sponza Colonnade Nave Grand View
            cam.pos = float3(-1150.0, 280.0, -40.0);
            cam.forward = normalize(float3(1.0, 0.02, 0.0));
            cam.right = float3(0.0, 0.0, -1.0);
            cam.up = float3(0.0, 1.0, 0.0);
            cam.fov = 0.55;
            return cam;
        } else if (sceneType == SCENE_SHOWROOM_STUDIO) {
            // ToyCar Automotive 3/4 Turntable View (+Y Up)
            cam.pos = float3(360.0, 160.0, 420.0);
            float3 target = float3(0.0, 30.0, 0.0);
            cam.forward = normalize(target - cam.pos);
            cam.right = normalize(cross(cam.forward, float3(0.0, 1.0, 0.0)));
            cam.up = cross(cam.right, cam.forward);
            cam.fov = 0.45;
            return cam;
        }
    }
    if (sceneType == SCENE_OUTDOOR_TERRAIN) {
        // Outdoor Landscape: Mountain Valley Vista
        cam.pos = float3(0.0, -320.0, 75.0);
        cam.forward = float3(0.0, 0.997375, -0.072412);
        cam.right = float3(1.0, 0.0, 0.0);
        cam.up = float3(0.0, 0.072412, 0.997375);
        cam.fov = 0.45;
    } else if (sceneType == SCENE_INDOOR_ATRIUM) {
        // Indoor Atrium: Colonnade & Vaulted Ceiling Grand View
        cam.pos = float3(0.0, -9.6, 1.2);
        cam.forward = normalize(float3(0.0, 9.6, 0.6));
        cam.right = float3(1.0, 0.0, 0.0);
        cam.up = normalize(cross(cam.right, cam.forward));
        cam.fov = 0.65;
    } else {
        // Showroom Studio: Central Sculpture Focus
        cam.pos = float3(-0.5081, -8.1276, 1.5591);
        cam.forward = float3(0.0562, 0.9852, -0.1622);
        cam.right = float3(0.9984, -0.0570, 0.0000);
        cam.up = float3(0.0092, 0.1619, 0.9868);
        cam.fov = 0.2155;
    }
    return cam;
}

public float3 makePrimaryRayDir(Camera cam, uint px_u, uint py_u, uint width, uint height, float2 jitter = float2(0.0, 0.0)) {
    float halfH = float(height) * 0.5;
    float px = float(px_u) - float(width) * 0.5 + 0.5 + jitter.x;
    float py = float(py_u) - halfH + 0.5 + jitter.y;
    return normalize(cam.forward + (px / halfH * cam.fov) * cam.right - (py / halfH * cam.fov) * cam.up);
}

public void getPrimaryRayRange(uint sceneType, uint isGltf, out float tMin, out float tMax) {
    if (isGltf != 0u) {
        tMin = 0.1;
        tMax = 5000.0;
    } else if (sceneType == SCENE_OUTDOOR_TERRAIN) {
        tMin = 0.05;
        tMax = 2500.0;
    } else {
        tMin = 0.01;
        tMax = 50.0;
    }
}

public void getSceneSunParams(uint sceneType, out float3 sunDir, out float3 sunColor) {
    if (sceneType == SCENE_INDOOR_ATRIUM) {
        sunDir = normalize(float3(0.35, 0.90, -0.25));
        sunColor = float3(1.0, 0.95, 0.88) * 4.5;
    } else if (sceneType == SCENE_SHOWROOM_STUDIO) {
        sunDir = normalize(float3(0.6, 0.8, 0.5));
        sunColor = float3(1.0, 0.97, 0.92) * 3.5;
    } else if (sceneType == SCENE_AAA_FOREST) {
        sunDir = normalize(float3(0.42, 0.38, 0.78));
        sunColor = float3(1.0, 0.94, 0.84) * 3.8;
    } else {
        sunDir = normalize(float3(0.45, 0.35, 0.82));
        sunColor = float3(1.0, 0.95, 0.85) * 3.5;
    }
}

public void getShadowLightParams(uint sceneType, uint lightIndex, out float3 lightDir, out float maxDist) {
    if (sceneType == SCENE_INDOOR_ATRIUM) {
        if (lightIndex == 1u) {
            lightDir = normalize(float3(0.32, 0.91, -0.22));
        } else if (lightIndex == 2u) {
            lightDir = normalize(float3(0.38, 0.89, -0.28));
        } else {
            lightDir = normalize(float3(0.35, 0.90, -0.25));
        }
        maxDist = 10000.0;
    } else if (sceneType == SCENE_SHOWROOM_STUDIO) {
        if (lightIndex == 1u) {
            lightDir = normalize(float3(0.57, 0.82, 0.47));
        } else if (lightIndex == 2u) {
            lightDir = normalize(float3(0.63, 0.78, 0.53));
        } else {
            lightDir = normalize(float3(0.6, 0.8, 0.5));
        }
        maxDist = 5000.0;
    } else {
        if (lightIndex == 1u) {
            lightDir = normalize(float3(0.42, 0.38, 0.82));
        } else if (lightIndex == 2u) {
            lightDir = normalize(float3(0.48, 0.32, 0.82));
        } else {
            lightDir = normalize(float3(0.45, 0.35, 0.82));
        }
        maxDist = 2500.0;
    }
}

public void getStudioLight(int idx, out float3 dir, out float3 color) {
    if (idx == 0) {
        dir = normalize(float3(0.6, 0.8, 0.5));
        color = float3(1.0, 0.97, 0.92) * 3.8;
    } else if (idx == 1) {
        dir = normalize(float3(-0.7, 0.5, 0.4));
        color = float3(0.75, 0.88, 1.0) * 1.8;
    } else {
        dir = normalize(float3(-0.2, 0.4, -0.9));
        color = float3(1.0, 1.0, 1.0) * 2.5;
    }
}

public void getAtriumLantern(int idx, out float3 pos, out float3 color) {
    if (idx == 0) {
        pos = float3(0.0, 450.0, 0.0);
        color = float3(1.0, 0.85, 0.60) * 2.2;
    } else if (idx == 1) {
        pos = float3(-650.0, 250.0, 0.0);
        color = float3(1.0, 0.80, 0.50) * 1.8;
    } else {
        pos = float3(650.0, 250.0, 0.0);
        color = float3(1.0, 0.80, 0.50) * 1.8;
    }
}

public void getSceneLight(int idx, out float3 dir, out float3 color, out float intensity) {
    if (idx == 0) {
        dir = float3(0.28, -0.65, 0.70);
        color = float3(1.0, 0.98, 0.94);
        intensity = 1.00;
    } else if (idx == 1) {
        dir = float3(-0.65, 0.55, 0.45);
        color = float3(0.85, 0.90, 1.0);
        intensity = 0.35;
    } else if (idx == 2) {
        dir = float3(-0.50, -0.80, 0.35);
        color = float3(1.0, 0.95, 0.85);
        intensity = 0.45;
    } else {
        dir = float3(0.0, 0.15, 0.98);
        color = float3(0.60, 0.65, 0.70);
        intensity = 0.20;
    }
}

public float3 evalStudioEnvironment(float3 dir) {
    float z = dir.z;
    float3 baseWall = lerp(float3(0.12, 0.13, 0.16), float3(0.24, 0.26, 0.30), clamp(z * 1.5, 0.0, 1.0));
    if (z < 0.0) {
        baseWall = lerp(float3(0.12, 0.13, 0.16), float3(0.06, 0.07, 0.08), clamp(-z * 2.0, 0.0, 1.0));
    }
    float3 softboxDir1 = normalize(float3(0.05, 0.30, 0.95));
    float sb1 = pow(clamp(dot(dir, softboxDir1), 0.0, 1.0), 80.0) * 6.0;
    float3 softboxDir2 = normalize(float3(-0.75, -0.20, 0.40));
    float sb2 = pow(clamp(dot(dir, softboxDir2), 0.0, 1.0), 45.0) * 3.5;
    float3 softboxDir3 = normalize(float3(0.65, 0.70, 0.30));
    float sb3 = pow(clamp(dot(dir, softboxDir3), 0.0, 1.0), 40.0) * 2.5;
    return baseWall + float3(1.0, 0.98, 0.95) * sb1 + float3(0.85, 0.92, 1.0) * sb2 + float3(1.0, 0.90, 0.80) * sb3;
}

public float3 evalAtriumEnvironment(float3 dir) {
    float z = dir.z;
    float3 baseWall = lerp(float3(0.16, 0.15, 0.15), float3(0.26, 0.24, 0.22), clamp(z * 1.5, 0.0, 1.0));
    if (z < 0.0) {
        baseWall = lerp(float3(0.16, 0.15, 0.15), float3(0.08, 0.07, 0.06), clamp(-z * 2.0, 0.0, 1.0));
    }
    float3 clerestoryDir = normalize(float3(0.10, 0.30, 0.95));
    float cs = pow(clamp(dot(dir, clerestoryDir), 0.0, 1.0), 30.0) * 3.5;
    return baseWall + float3(1.0, 0.95, 0.85) * cs;
}

public float3 evalOutdoorSky(float3 dir) {
    float3 sunDir = normalize(float3(0.45, 0.35, 0.82));
    float sunDot = max(dot(dir, sunDir), 0.0);
    float3 sky = lerp(float3(0.70, 0.82, 0.95), float3(0.18, 0.42, 0.82), clamp(dir.z, 0.0, 1.0));
    if (dir.z < 0.0) {
        sky = lerp(float3(0.70, 0.82, 0.95), float3(0.35, 0.30, 0.25), clamp(-dir.z * 2.0, 0.0, 1.0));
    }
    sky += float3(1.0, 0.95, 0.85) * pow(sunDot, 120.0) * 8.0;
    sky += float3(1.0, 0.85, 0.60) * pow(sunDot, 12.0) * 0.8;
    return sky;
}

public float3 evalSky(uint sceneType, float3 dir) {
    if (sceneType == SCENE_OUTDOOR_TERRAIN || sceneType == SCENE_AAA_FOREST) {
        return evalOutdoorSky(dir);
    } else if (sceneType == SCENE_INDOOR_ATRIUM) {
        return evalAtriumEnvironment(dir);
    }
    return evalStudioEnvironment(dir);
}

public float computeFloorOcclusion(float3 hitPos) {
    float occ = 1.0;
    float dCenter = length(hitPos.xy - float2(0.0, 0.2));
    occ *= clamp(0.20 + 0.80 * smoothstep(0.4, 2.5, dCenter), 0.0, 1.0);
    float dLeft = length(hitPos.xy - float2(-1.8, -1.2));
    occ *= clamp(0.15 + 0.85 * smoothstep(0.3, 2.0, dLeft), 0.0, 1.0);
    float dFront = length(float2(hitPos.x, (hitPos.y - (-1.5)) * 2.0));
    occ *= clamp(0.18 + 0.82 * smoothstep(0.4, 2.2, dFront), 0.0, 1.0);
    float dSphere = length(hitPos.xy - float2(-3.24, 0.39));
    occ *= clamp(0.10 + 0.90 * smoothstep(0.3, 1.4, dSphere), 0.0, 1.0);
    float dPed = length(hitPos.xy - float2(-0.46, 0.06));
    occ *= clamp(0.08 + 0.92 * smoothstep(0.3, 1.2, dPed), 0.0, 1.0);
    float dSuz = length(hitPos.xy - float2(2.71, -0.57));
    occ *= clamp(0.12 + 0.88 * smoothstep(0.3, 1.4, dSuz), 0.0, 1.0);
    return occ;
}

public uint computeMaterialId(uint sceneType, uint primId, float3 hitPos, float3 normal) {
    if (sceneType == SCENE_OUTDOOR_TERRAIN) {
        if (primId < PRIM_TERRAIN_END) {
            if (hitPos.z >= -0.2 && hitPos.z <= 2.2 && abs(hitPos.x) < 185.0 && hitPos.y >= -145.0 && hitPos.y <= 930.0) {
                return 7u; // Wet River Stones / Shoreline Boulders
            }
            if (normal.z >= 0.50) {
                float baseSnowAlt = 60.0 + (1.0 - normal.z) * 28.0;
                if (hitPos.z > baseSnowAlt - 16.0) {
                    if (hitPos.z > baseSnowAlt + 16.0) {
                        return 3u;
                    }
                    float snowHarmonics = sin(hitPos.x * 0.032 + hitPos.y * 0.018) * 9.5
                                        + cos(hitPos.x * 0.068 - hitPos.y * 0.045) * 4.5
                                        + sin(hitPos.x * 0.130 + hitPos.y * 0.110) * 2.0;
                    if (hitPos.z > baseSnowAlt + snowHarmonics) {
                        return 3u; // High Mountain Snow & Glacial Ice
                    }
                }
            }
            if (normal.z < 0.65) return 4u; // Exposed Granite Cliff Face
            if (normal.z < 0.82) return 0u; // Weathered Sandstone & Scree Slope
            return 5u;                      // Valley Grassland / Alpine Meadow
        }
        if (primId < PRIM_WATER_END) return 2u;
        uint treeLocalTri = (primId - PRIM_WATER_END) % 112u;
        if (treeLocalTri < 16u) return 6u;
        return 1u;
    } else {
        if (primId < PRIM_KNOT_PAINT_END)  return 0u;
        if (primId < PRIM_KNOT_JADE_END)   return 1u;
        if (primId < PRIM_KNOT_CHROME_END) return 7u;
        if (primId < PRIM_KNOT_VELVET_END) return 3u;
        if (primId < PRIM_KNOT_RUST_END)   return 4u;
        if (primId < PRIM_SPHERE_END)      return 1u;
        if (primId < PRIM_PEDESTAL_END)    return 6u;
        if (primId < PRIM_SUZANNE_END)     return 2u;
        if (primId < PRIM_FLOOR_END)       return 5u;
        if (primId < PRIM_WALLS_END)       return 4u;
        if (primId < PRIM_CEILING_END)     return 6u;
        return 1u;
    }
}

public float3 getSceneArchetypeAlbedo(uint sceneType, uint archetype) {
    if (sceneType == SCENE_OUTDOOR_TERRAIN) {
        if (archetype == 0u) return float3(0.65, 0.52, 0.40);
        if (archetype == 1u) return float3(0.08, 0.32, 0.12);
        if (archetype == 2u) return float3(0.012, 0.042, 0.065);
        if (archetype == 3u) return float3(0.92, 0.95, 0.98);
        if (archetype == 4u) return float3(0.30, 0.31, 0.34);
        if (archetype == 5u) return float3(0.24, 0.44, 0.18);
        if (archetype == 6u) return float3(0.25, 0.17, 0.11);
        return float3(0.18, 0.20, 0.22);
    } else {
        if (archetype == 0u) return float3(0.65, 0.52, 0.40);
        if (archetype == 1u) return float3(0.18, 0.55, 0.34);
        if (archetype == 2u) return float3(0.95, 0.95, 0.98);
        if (archetype == 3u) return float3(0.85, 0.12, 0.22);
        if (archetype == 4u) return float3(0.55, 0.45, 0.35);
        if (archetype == 5u) return float3(0.80, 0.80, 0.82);
        if (archetype == 6u) return float3(0.88, 0.20, 0.10);
        return float3(0.12, 0.45, 0.15);
    }
}

public void getProceduralTerrainMaterial(uint primId, float3 hitPos, float3 smoothNormal, out float3 albedo, out float roughness, out float metallic) {
    roughness = 0.5;
    metallic = 0.0;
    if (primId < PRIM_TERRAIN_END) {
        if (hitPos.z >= -0.2 && hitPos.z <= 2.2 && abs(hitPos.x) < 185.0 && hitPos.y >= -145.0 && hitPos.y <= 930.0) {
            albedo = float3(0.18, 0.20, 0.22);
            roughness = 0.20;
        } else {
            bool isSnow = false;
            if (smoothNormal.z >= 0.50) {
                float baseSnowAlt = 60.0 + (1.0 - smoothNormal.z) * 28.0;
                if (hitPos.z > baseSnowAlt + 16.0) {
                    isSnow = true;
                } else if (hitPos.z > baseSnowAlt - 16.0) {
                    float snowHarmonics = sin(hitPos.x * 0.032 + hitPos.y * 0.018) * 9.5
                                        + cos(hitPos.x * 0.068 - hitPos.y * 0.045) * 4.5
                                        + sin(hitPos.x * 0.130 + hitPos.y * 0.110) * 2.0;
                    isSnow = (hitPos.z > baseSnowAlt + snowHarmonics);
                }
            }
            if (isSnow) {
                albedo = float3(0.85, 0.88, 0.92);
                roughness = 0.35;
            } else if (smoothNormal.z < 0.65) {
                albedo = float3(0.30, 0.31, 0.34);
                roughness = 0.85;
            } else if (smoothNormal.z < 0.82) {
                albedo = float3(0.60, 0.48, 0.38);
                roughness = 0.75;
            } else {
                albedo = float3(0.24, 0.44, 0.18);
                roughness = 0.65;
            }
        }
    } else if (primId < PRIM_WATER_END) {
        albedo = float3(0.06, 0.22, 0.32);
        roughness = 0.08;
        metallic = 0.25;
    } else {
        uint treeLocalTri = (primId - PRIM_WATER_END) % 112u;
        if (treeLocalTri < 16u) {
            albedo = float3(0.25, 0.17, 0.11);
            roughness = 0.85;
        } else {
            albedo = float3(0.08, 0.32, 0.12);
            roughness = 0.50;
        }
    }
}

public float3 getProceduralSecondaryColor(uint rPrim, bool isGlass) {
    if (rPrim < PRIM_KNOT_PAINT_END)  return float3(0.68, 0.01, 0.02);
    if (rPrim < PRIM_KNOT_JADE_END)   return float3(0.06, 0.75, 0.65);
    if (rPrim < PRIM_KNOT_CHROME_END) return float3(0.85, 0.85, 0.88);
    if (rPrim < PRIM_KNOT_VELVET_END) return float3(0.60, 0.02, 0.42);
    if (rPrim < PRIM_KNOT_RUST_END)   return float3(0.80, 0.35, 0.12);
    if (rPrim < PRIM_SPHERE_END)      return float3(0.06, 0.75, 0.65);
    if (rPrim < PRIM_PEDESTAL_END)    return float3(0.90, 0.75, 0.30);
    if (rPrim < PRIM_SUZANNE_END)     return float3(0.75, 0.80, 0.85);
    return isGlass ? float3(0.24, 0.25, 0.28) : float3(0.20, 0.21, 0.23);
}

public void getMultiLight(uint idx, uint totalLights, uint sceneType, out float3 lightDir, out float3 lightColor) {
    if (totalLights <= 1u) {
        if (sceneType == SCENE_INDOOR_ATRIUM) {
            lightDir = normalize(float3(0.35, 0.90, -0.25));
        } else if (sceneType == SCENE_SHOWROOM_STUDIO) {
            lightDir = normalize(float3(0.6, 0.8, 0.5));
        } else {
            lightDir = normalize(float3(0.45, 0.35, 0.82));
        }
        lightColor = float3(1.0, 0.95, 0.9) * 3.5;
    } else {
        uint b = (idx >> 4u) & 7u;
        uint li = idx & 15u;
        float u = (float(li) + 0.5) / 16.0;
        float v = frac(float(li) * 0.618033988749895 + float(b) * 0.125);
        float phi = 0.15 + 1.27079632679 * v;
        float cosTheta = 0.20 + 0.70 * u;
        float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));

        float3 posOct = float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
        float3 octSign = float3(
            (b & 1u) != 0u ? 1.0 : -1.0,
            (b & 2u) != 0u ? 1.0 : -1.0,
            (b & 4u) != 0u ? 1.0 : -1.0
        );
        lightDir = normalize(posOct * octSign);

        float t = float(idx) / float(totalLights);
        float3 col = lerp(float3(1.0, 0.88, 0.70), float3(0.75, 0.88, 1.0), frac(t * 5.0));
        lightColor = col * (12.0 / float(totalLights));
    }
}

public float4 sampleTexture(StructuredBuffer<TextureHeaderGpu> texHeaders, StructuredBuffer<uint> texPixels, uint texIdx, float2 uv) {
    TextureHeaderGpu h = texHeaders[texIdx];
    if (h.width == 0u || h.height == 0u) return float4(1.0);
    float fx = frac(uv.x) * float(h.width) - 0.5;
    float fy = frac(uv.y) * float(h.height) - 0.5;
    int x0 = int(floor(fx));
    int y0 = int(floor(fy));
    float wx = frac(fx);
    float wy = frac(fy);

    int x1 = (x0 + 1) % int(h.width);
    int y1 = (y0 + 1) % int(h.height);
    x0 = (x0 < 0) ? (x0 + int(h.width)) : x0;
    y0 = (y0 < 0) ? (y0 + int(h.height)) : y0;

    float4 c00 = unpackUnorm4x8_spv(texPixels[h.offset + uint(y0) * h.width + uint(x0)]);
    float4 c10 = unpackUnorm4x8_spv(texPixels[h.offset + uint(y0) * h.width + uint(x1)]);
    float4 c01 = unpackUnorm4x8_spv(texPixels[h.offset + uint(y1) * h.width + uint(x0)]);
    float4 c11 = unpackUnorm4x8_spv(texPixels[h.offset + uint(y1) * h.width + uint(x1)]);

    return lerp(lerp(c00, c10, wx), lerp(c01, c11, wx), wy);
}

float3 rotKnot(float3 v) {
    return float3(0.929662, 0.326751, 0.170182) * v.x
         + float3(-0.239757, 0.887321, -0.393927) * v.y
         + float3(-0.279722, 0.325417, 0.903249) * v.z;
}

public float3 computeSmoothNormal(StructuredBuffer<float> vbuf, uint sceneType, uint isGltf, uint primId, float3 hitPos, float3 rayDir, float2 bary) {
    if (isGltf != 0u) {
        uint base = primId * 36u;
        float3 n0 = float3(vbuf[base + 3u], vbuf[base + 4u], vbuf[base + 5u]);
        float3 n1 = float3(vbuf[base + 15u], vbuf[base + 16u], vbuf[base + 17u]);
        float3 n2 = float3(vbuf[base + 27u], vbuf[base + 28u], vbuf[base + 29u]);

        float w = 1.0 - bary.x - bary.y;
        float3 smoothNorm = w * n0 + bary.x * n1 + bary.y * n2;
        float snLen2 = dot(smoothNorm, smoothNorm);
        float3 n;
        if (snLen2 > 1e-8) {
            n = smoothNorm * rsqrt(snLen2);
        } else {
            float3 p0 = float3(vbuf[base + 0u], vbuf[base + 1u], vbuf[base + 2u]);
            float3 p1 = float3(vbuf[base + 12u], vbuf[base + 13u], vbuf[base + 14u]);
            float3 p2 = float3(vbuf[base + 24u], vbuf[base + 25u], vbuf[base + 26u]);
            n = normalize(cross(p1 - p0, p2 - p0));
        }
        if (dot(n, rayDir) > 0.0) n = -n;
        return n;
    }
    float3 normal;
    if (sceneType == SCENE_OUTDOOR_TERRAIN) {
        if (primId >= PRIM_TERRAIN_END && primId < PRIM_WATER_END) {
            normal = float3(0.0, 0.0, 1.0);
        } else {
            uint base = primId * 9u;
            float3 p0 = float3(vbuf[base + 0u], vbuf[base + 1u], vbuf[base + 2u]);
            float3 p1 = float3(vbuf[base + 3u], vbuf[base + 4u], vbuf[base + 5u]);
            float3 p2 = float3(vbuf[base + 6u], vbuf[base + 7u], vbuf[base + 8u]);
            normal = normalize(cross(p1 - p0, p2 - p0));
        }
    } else {
        if (primId < PRIM_KNOT_RUST_END) {
            uint segI = primId / 48u;
            float u0 = (float(segI) / 160.0) * 6.283185307;
            float u1 = (float((segI + 1u) % 160u) / 160.0) * 6.283185307;

            float r0 = (2.0 + 0.8 * cos(3.0 * u0)) * 0.835;
            float3 unrot0 = float3(r0 * cos(2.0 * u0), r0 * sin(2.0 * u0), -0.8 * 0.835 * sin(3.0 * u0));
            float r1 = (2.0 + 0.8 * cos(3.0 * u1)) * 0.835;
            float3 unrot1 = float3(r1 * cos(2.0 * u1), r1 * sin(2.0 * u1), -0.8 * 0.835 * sin(3.0 * u1));

            float3 c0 = rotKnot(unrot0) + float3(0.0, 0.0, 0.385);
            float3 c1 = rotKnot(unrot1) + float3(0.0, 0.0, 0.385);

            float3 seg = c1 - c0;
            float segLen2 = dot(seg, seg);
            float tSeg = clamp(dot(hitPos - c0, seg) / max(segLen2, 0.00001), 0.0, 1.0);
            float3 centerPt = c0 + tSeg * seg;

            normal = normalize(hitPos - centerPt);
        } else if (primId < PRIM_SPHERE_END) {
            normal = normalize(hitPos - float3(-3.242, 0.391, -0.509));
        } else if (primId < PRIM_PEDESTAL_END) {
            if (hitPos.z >= -0.658) {
                normal = float3(0.0, 0.0, 1.0);
            } else if (hitPos.z <= -1.050) {
                normal = float3(0.0, 0.0, -1.0);
            } else {
                normal = normalize(float3(hitPos.x - (-0.460), hitPos.y - 0.060, 0.0));
            }
        } else if (primId < PRIM_SUZANNE_END) {
            uint base = primId * 9u;
            float3 p0 = float3(vbuf[base + 0u], vbuf[base + 1u], vbuf[base + 2u]);
            float3 p1 = float3(vbuf[base + 3u], vbuf[base + 4u], vbuf[base + 5u]);
            float3 p2 = float3(vbuf[base + 6u], vbuf[base + 7u], vbuf[base + 8u]);
            normal = normalize(cross(p1 - p0, p2 - p0));
        } else if (primId < PRIM_FLOOR_END) {
            float backCurve = (hitPos.y > 1.0) ? 0.044 * (hitPos.y - 1.0) : 0.0;
            normal = normalize(float3(0.0, -backCurve, 1.0));
        } else {
            uint base = primId * 9u;
            float3 p0 = float3(vbuf[base + 0u], vbuf[base + 1u], vbuf[base + 2u]);
            float3 p1 = float3(vbuf[base + 3u], vbuf[base + 4u], vbuf[base + 5u]);
            float3 p2 = float3(vbuf[base + 6u], vbuf[base + 7u], vbuf[base + 8u]);
            normal = normalize(cross(p1 - p0, p2 - p0));
        }
    }
    if (dot(normal, rayDir) > 0.0) normal = -normal;
    return normal;
}

public float3 computeGltfGeomNormal(StructuredBuffer<float> vbuf, uint primId, float3 rayDir) {
    uint base = primId * 36u;
    float3 p0 = float3(vbuf[base + 0u], vbuf[base + 1u], vbuf[base + 2u]);
    float3 p1 = float3(vbuf[base + 12u], vbuf[base + 13u], vbuf[base + 14u]);
    float3 p2 = float3(vbuf[base + 24u], vbuf[base + 25u], vbuf[base + 26u]);
    float3 geomNormal = normalize(cross(p1 - p0, p2 - p0));
    if (dot(geomNormal, rayDir) > 0.0) geomNormal = -geomNormal;
    return geomNormal;
}

public float2 unpackGltfUV(StructuredBuffer<float> vbuf, uint primId, float2 bary) {
    uint base = primId * 36u;
    float2 uv0 = float2(vbuf[base + 10u], vbuf[base + 11u]);
    float2 uv1 = float2(vbuf[base + 22u], vbuf[base + 23u]);
    float2 uv2 = float2(vbuf[base + 34u], vbuf[base + 35u]);
    float w = 1.0 - bary.x - bary.y;
    return w * uv0 + bary.x * uv1 + bary.y * uv2;
}

public void unpackGltfHit(
    StructuredBuffer<float> vbuf,
    uint primId,
    float2 bary,
    bool needTangent,
    out float3 interpHitPos,
    out float3 geomNormal,
    out float3 smoothNormal,
    out float4 tangent,
    out float2 uv
) {
    uint base = primId * 36u;
    float3 p0 = float3(vbuf[base + 0u], vbuf[base + 1u], vbuf[base + 2u]);
    float3 n0 = float3(vbuf[base + 3u], vbuf[base + 4u], vbuf[base + 5u]);
    float2 uv0 = float2(vbuf[base + 10u], vbuf[base + 11u]);

    uint base1 = base + 12u;
    float3 p1 = float3(vbuf[base1 + 0u], vbuf[base1 + 1u], vbuf[base1 + 2u]);
    float3 n1 = float3(vbuf[base1 + 3u], vbuf[base1 + 4u], vbuf[base1 + 5u]);
    float2 uv1 = float2(vbuf[base1 + 10u], vbuf[base1 + 11u]);

    uint base2 = base + 24u;
    float3 p2 = float3(vbuf[base2 + 0u], vbuf[base2 + 1u], vbuf[base2 + 2u]);
    float3 n2 = float3(vbuf[base2 + 3u], vbuf[base2 + 4u], vbuf[base2 + 5u]);
    float2 uv2 = float2(vbuf[base2 + 10u], vbuf[base2 + 11u]);

    float w = 1.0 - bary.x - bary.y;
    interpHitPos = w * p0 + bary.x * p1 + bary.y * p2;
    geomNormal = normalize(cross(p1 - p0, p2 - p0));
    float3 sn = w * n0 + bary.x * n1 + bary.y * n2;
    smoothNormal = (length(sn) > 1e-4) ? normalize(sn) : geomNormal;

    tangent = float4(0.0);
    if (needTangent) {
        float4 t0 = float4(vbuf[base + 6u], vbuf[base + 7u], vbuf[base + 8u], vbuf[base + 9u]);
        float4 t1 = float4(vbuf[base1 + 6u], vbuf[base1 + 7u], vbuf[base1 + 8u], vbuf[base1 + 9u]);
        float4 t2 = float4(vbuf[base2 + 6u], vbuf[base2 + 7u], vbuf[base2 + 8u], vbuf[base2 + 9u]);
        tangent = w * t0 + bary.x * t1 + bary.y * t2;
    }
    uv = w * uv0 + bary.x * uv1 + bary.y * uv2;
}

public uint computeBounceOctant(uint sceneType, float3 dir) {
    bool zUp = (sceneType == SCENE_OUTDOOR_TERRAIN || sceneType == SCENE_AAA_FOREST);
    float2 hDir = zUp ? dir.xy : dir.xz;
    float upCoord = zUp ? dir.z : dir.y;
    uint thirdBit = (sceneType == SCENE_INDOOR_ATRIUM) ? (upCoord >= 0.0 ? 4u : 0u) : (upCoord >= 0.45 ? 4u : 0u);
    return (hDir.x >= 0.0 ? 1u : 0u)
         | (hDir.y >= 0.0 ? 2u : 0u)
         | thirdBit;
}
