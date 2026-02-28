#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT vec3 payload;

void main() {
    vec3 p = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    
    // Shader E: Intermediate complexity
    float val = 0.5;
    for(int i = 0; i < 50; i++) {
        val = abs(sin(val + p.z * 0.1) * cos(p.x * i * 0.1));
    }
    
    payload = vec3(val * 0.5, 1.0 - val, val);
}
