#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_ray_query : enable

layout(location = 0) rayPayloadInEXT vec3 payload;
hitAttributeEXT vec2 hitBarycentrics;

layout(set = 0, binding = 0) uniform accelerationStructureEXT topLevelAS;
layout(set = 0, binding = 1) buffer Results {
    uint totalRays;
    uint totalHits;
    float accumulatedLuminance;
} results;

layout(set = 0, binding = 2) buffer Framebuffer {
    vec4 pixels[];
} fb;

layout(set = 0, binding = 3) readonly buffer VertexBuffer {
    float vertices[];
} vbuf;

#define BINDING_MAT_BUF 4
#define BINDING_TRI_MAT_BUF 5
#define BINDING_TEX_HDR_BUF 6
#define BINDING_TEX_PIX_BUF 7
#include "pbr_common.glsl"

layout(push_constant) uniform PushConstants {
    uint rayCount;
    uint mode;
    uint bounces;
    uint seed;
    uint dumpRenders;
    uint width;
    uint height;
    uint spatialPattern;
    uint sceneType;
    uint isGltf;
    uint spp;
} pc;

const uint SCENE_SHOWROOM_STUDIO = 0u;
const uint SCENE_OUTDOOR_TERRAIN = 1u;
const uint SCENE_INDOOR_ATRIUM   = 2u;
const uint SCENE_AAA_FOREST       = 3u;

// Indoor Atrium Primitives
const uint PRIM_KNOT_PAINT_END  = 1536u;
const uint PRIM_KNOT_JADE_END   = 3072u;
const uint PRIM_KNOT_CHROME_END = 4608u;
const uint PRIM_KNOT_VELVET_END = 6144u;
const uint PRIM_KNOT_RUST_END   = 7680u;
const uint PRIM_SPHERE_END      = 9600u;
const uint PRIM_PEDESTAL_END    = 9728u;
const uint PRIM_SUZANNE_END     = 10696u;
const uint PRIM_FLOOR_END       = 27080u;
const uint PRIM_WALLS_END       = 28104u;
const uint PRIM_CEILING_END     = 29128u;
const uint PRIM_COLUMNS_END     = 35272u;

// Outdoor Landscape Primitives
const uint PRIM_TERRAIN_END     = 32768u;
const uint PRIM_WATER_END       = 34816u;
const uint PRIM_TREES_END       = 57216u;

uint pcg_hash(inout uint state) {
    uint oldstate = state;
    state = oldstate * 747796405u + 2891336453u;
    uint word = ((oldstate >> ((oldstate >> 28u) + 4u)) ^ oldstate) * 277803737u;
    return (word >> 22u) ^ word;
}

float rand_float(inout uint state) {
    return float(pcg_hash(state) & 0x00FFFFFFu) / 16777216.0;
}

vec3 cosine_sample_hemisphere(float u1, float u2) {
    float r = sqrt(u1);
    float theta = 6.28318530718 * u2;
    float x = r * cos(theta);
    float y = r * sin(theta);
    float z = sqrt(max(0.0, 1.0 - u1));
    return vec3(x, y, z);
}

vec3 align_to_normal(vec3 sample_dir, vec3 normal) {
    vec3 up = abs(normal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(up, normal));
    vec3 bitangent = cross(normal, tangent);
    return tangent * sample_dir.x + bitangent * sample_dir.y + normal * sample_dir.z;
}

uint computeMaterialId(uint sceneType, uint primId, vec3 hitPos, vec3 normal) {
    if (sceneType == SCENE_OUTDOOR_TERRAIN) {
        if (primId < PRIM_TERRAIN_END) {
            if (hitPos.z >= -0.2 && hitPos.z <= 2.2 && abs(hitPos.x) < 185.0 && hitPos.y >= -145.0 && hitPos.y <= 930.0) {
                return 7u;
            }
            float snowHarmonics = sin(hitPos.x * 0.032 + hitPos.y * 0.018) * 9.5
                                + cos(hitPos.x * 0.068 - hitPos.y * 0.045) * 4.5
                                + sin(hitPos.x * 0.130 + hitPos.y * 0.110) * 2.0;
            float snowAltitude = 60.0 + (1.0 - normal.z) * 28.0 + snowHarmonics;
            if (hitPos.z > snowAltitude && normal.z >= 0.50) return 3u;
            if (normal.z < 0.65) return 4u;
            if (normal.z < 0.82) return 0u;
            return 5u;
        }
        if (primId < PRIM_WATER_END) return 2u;
        uint treeLocalTri = (primId - PRIM_WATER_END) % 224u;
        if (treeLocalTri < 32u) return 6u;
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

vec3 computeSmoothNormal(uint sceneType, uint primId, vec3 hitPos, vec3 rayDir, vec2 bary) {
    if (pc.isGltf != 0u) {
        uint base = primId * 36u;
        vec3 p0 = vec3(vbuf.vertices[base + 0], vbuf.vertices[base + 1], vbuf.vertices[base + 2]);
        vec3 n0 = vec3(vbuf.vertices[base + 3], vbuf.vertices[base + 4], vbuf.vertices[base + 5]);
        base += 12u;
        vec3 p1 = vec3(vbuf.vertices[base + 0], vbuf.vertices[base + 1], vbuf.vertices[base + 2]);
        vec3 n1 = vec3(vbuf.vertices[base + 3], vbuf.vertices[base + 4], vbuf.vertices[base + 5]);
        base += 12u;
        vec3 p2 = vec3(vbuf.vertices[base + 0], vbuf.vertices[base + 1], vbuf.vertices[base + 2]);
        vec3 n2 = vec3(vbuf.vertices[base + 3], vbuf.vertices[base + 4], vbuf.vertices[base + 5]);

        float w = 1.0 - bary.x - bary.y;
        vec3 geomNorm = normalize(cross(p1 - p0, p2 - p0));
        vec3 smoothNorm = w * n0 + bary.x * n1 + bary.y * n2;
        vec3 n = (length(smoothNorm) > 1e-4) ? normalize(smoothNorm) : geomNorm;
        if (dot(n, rayDir) > 0.0) n = -n;
        return n;
    }
    uint base = primId * 9u;
    vec3 p0 = vec3(vbuf.vertices[base + 0], vbuf.vertices[base + 1], vbuf.vertices[base + 2]);
    vec3 p1 = vec3(vbuf.vertices[base + 3], vbuf.vertices[base + 4], vbuf.vertices[base + 5]);
    vec3 p2 = vec3(vbuf.vertices[base + 6], vbuf.vertices[base + 7], vbuf.vertices[base + 8]);
    vec3 n = normalize(cross(p1 - p0, p2 - p0));
    if (dot(n, rayDir) > 0.0) n = -n;
    return n;
}

void main() {
    uint primId = gl_PrimitiveID;
    float t = gl_HitTEXT;
    vec3 rayOrigin = gl_WorldRayOriginEXT;
    vec3 rayDir = gl_WorldRayDirectionEXT;
    vec3 hitPos = rayOrigin + rayDir * t;
    vec2 rawBary = hitBarycentrics;
    uint rngState = primId + pc.seed;

    vec3 pixelColor = vec3(0.0);

    if (pc.isGltf != 0u) {
        vec2 bary = unpackUnorm2x16(packUnorm2x16(rawBary));
        uint base = primId * 36u;
        vec3 p0 = vec3(vbuf.vertices[base + 0], vbuf.vertices[base + 1], vbuf.vertices[base + 2]);
        vec3 n0 = vec3(vbuf.vertices[base + 3], vbuf.vertices[base + 4], vbuf.vertices[base + 5]);
        vec4 t0 = vec4(vbuf.vertices[base + 6], vbuf.vertices[base + 7], vbuf.vertices[base + 8], vbuf.vertices[base + 9]);
        vec2 uv0 = vec2(vbuf.vertices[base + 10], vbuf.vertices[base + 11]);

        base += 12u;
        vec3 p1 = vec3(vbuf.vertices[base + 0], vbuf.vertices[base + 1], vbuf.vertices[base + 2]);
        vec3 n1 = vec3(vbuf.vertices[base + 3], vbuf.vertices[base + 4], vbuf.vertices[base + 5]);
        vec4 t1 = vec4(vbuf.vertices[base + 6], vbuf.vertices[base + 7], vbuf.vertices[base + 8], vbuf.vertices[base + 9]);
        vec2 uv1 = vec2(vbuf.vertices[base + 10], vbuf.vertices[base + 11]);

        base += 12u;
        vec3 p2 = vec3(vbuf.vertices[base + 0], vbuf.vertices[base + 1], vbuf.vertices[base + 2]);
        vec3 n2 = vec3(vbuf.vertices[base + 3], vbuf.vertices[base + 4], vbuf.vertices[base + 5]);
        vec4 t2 = vec4(vbuf.vertices[base + 6], vbuf.vertices[base + 7], vbuf.vertices[base + 8], vbuf.vertices[base + 9]);
        vec2 uv2 = vec2(vbuf.vertices[base + 10], vbuf.vertices[base + 11]);

        float w = 1.0 - bary.x - bary.y;
        vec3 interpHitPos = w * p0 + bary.x * p1 + bary.y * p2;
        vec3 geomNormal = normalize(cross(p1 - p0, p2 - p0));
        vec3 sn = w * n0 + bary.x * n1 + bary.y * n2;
        vec3 smoothNormal = (length(sn) > 1e-4) ? normalize(sn) : geomNormal;
        vec4 tangent = w * t0 + bary.x * t1 + bary.y * t2;
        vec2 uv = w * uv0 + bary.x * uv1 + bary.y * uv2;

        uint matId = triMatBuf.triangleMats[primId];
        GltfMaterialGpu mat = matBuf.materials[matId];
        pixelColor = evaluateGltfPbr(mat, interpHitPos, geomNormal, smoothNormal, tangent, uv, rayDir, pc.sceneType, rngState);
    } else {
        vec3 normal = computeSmoothNormal(pc.sceneType, primId, hitPos, rayDir, rawBary);
        uint matId = computeMaterialId(pc.sceneType, primId, hitPos, normal);
        vec3 sunDir, sunColor;
        getSceneSunParams(pc.sceneType, sunDir, sunColor);
        vec3 baseColor = vec3(0.5);
        if (matId == 0u) baseColor = vec3(0.65, 0.52, 0.40);
        else if (matId == 1u) baseColor = vec3(0.18, 0.55, 0.34);
        else if (matId == 2u) baseColor = vec3(0.95, 0.95, 0.98);
        else if (matId == 3u) baseColor = vec3(0.85, 0.12, 0.22);
        else if (matId == 4u) baseColor = vec3(0.55, 0.45, 0.35);
        else if (matId == 5u) baseColor = vec3(0.80, 0.80, 0.82);
        else if (matId == 6u) baseColor = vec3(0.88, 0.20, 0.10);
        else baseColor = vec3(0.12, 0.45, 0.15);
        pixelColor = evaluateMaterialArchetypeDirect(matId, hitPos, baseColor, 0.5, 0.0, 1.5, normal, -rayDir, sunDir, sunColor, 1.0);
    }

    payload = pixelColor;
}
