#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT vec3 payload;

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

vec3 evalOutdoorSky(vec3 dir) {
    vec3 sunDir = normalize(vec3(0.45, 0.35, 0.82));
    float sunDot = max(dot(dir, sunDir), 0.0);
    vec3 sky = mix(vec3(0.70, 0.82, 0.95), vec3(0.18, 0.42, 0.82), clamp(dir.z, 0.0, 1.0));
    if (dir.z < 0.0) {
        sky = mix(vec3(0.70, 0.82, 0.95), vec3(0.35, 0.30, 0.25), clamp(-dir.z * 2.0, 0.0, 1.0));
    }
    sky += vec3(1.0, 0.95, 0.85) * pow(sunDot, 120.0) * 8.0;
    sky += vec3(1.0, 0.85, 0.60) * pow(sunDot, 12.0) * 0.8;
    return sky;
}

vec3 evalStudioEnvironment(vec3 dir) {
    float z = dir.z;
    vec3 baseWall = mix(vec3(0.12, 0.13, 0.16), vec3(0.24, 0.26, 0.30), clamp(z * 1.5, 0.0, 1.0));
    if (z < 0.0) {
        baseWall = mix(vec3(0.12, 0.13, 0.16), vec3(0.06, 0.07, 0.08), clamp(-z * 2.0, 0.0, 1.0));
    }
    vec3 softboxDir1 = normalize(vec3(0.05, 0.30, 0.95));
    float sb1 = pow(clamp(dot(dir, softboxDir1), 0.0, 1.0), 80.0) * 6.0;
    vec3 softboxDir2 = normalize(vec3(-0.75, -0.20, 0.40));
    float sb2 = pow(clamp(dot(dir, softboxDir2), 0.0, 1.0), 45.0) * 3.5;
    vec3 softboxDir3 = normalize(vec3(0.65, 0.70, 0.30));
    float sb3 = pow(clamp(dot(dir, softboxDir3), 0.0, 1.0), 40.0) * 2.5;
    return baseWall + vec3(1.0, 0.98, 0.95) * sb1 + vec3(0.85, 0.92, 1.0) * sb2 + vec3(1.0, 0.90, 0.80) * sb3;
}

vec3 evalAtriumEnvironment(vec3 dir) {
    float z = dir.z;
    vec3 baseWall = mix(vec3(0.16, 0.15, 0.15), vec3(0.26, 0.24, 0.22), clamp(z * 1.5, 0.0, 1.0));
    if (z < 0.0) {
        baseWall = mix(vec3(0.16, 0.15, 0.15), vec3(0.08, 0.07, 0.06), clamp(-z * 2.0, 0.0, 1.0));
    }
    vec3 clerestoryDir = normalize(vec3(0.10, 0.30, 0.95));
    float cs = pow(clamp(dot(dir, clerestoryDir), 0.0, 1.0), 30.0) * 3.5;
    return baseWall + vec3(1.0, 0.95, 0.85) * cs;
}

void main() {
    if (pc.sceneType == SCENE_OUTDOOR_TERRAIN || pc.sceneType == SCENE_AAA_FOREST) {
        payload = evalOutdoorSky(gl_WorldRayDirectionEXT);
    } else if (pc.sceneType == SCENE_INDOOR_ATRIUM) {
        payload = evalAtriumEnvironment(gl_WorldRayDirectionEXT);
    } else {
        payload = evalStudioEnvironment(gl_WorldRayDirectionEXT);
    }
}
