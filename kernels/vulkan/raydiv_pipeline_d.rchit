#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT vec3 payload;

void main() {
    vec3 p = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    float val = 0.0;
    for(int i = 0; i < 150; i++) {
        val += fract(p.x * i) * fract(p.y * i);
    }
    payload = vec3(val, val * 0.4, val * 0.6);
}
