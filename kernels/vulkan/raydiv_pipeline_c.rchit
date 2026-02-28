#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT vec3 payload;

void main() {
    vec3 p = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    
    float val = length(p);
    for(int i = 0; i < 75; i++) {
        val = pow(val, 1.01) * 0.99;
    }
    
    payload = vec3(val * 0.1, 0.0, val * 0.3);
}
