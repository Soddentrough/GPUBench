#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT vec3 payload;

void main() {
    vec3 p = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    
    float val = 1.0;
    for(int i = 1; i < 30; i++) {
        if (i % 2 == 0) {
            val = sqrt(val + abs(p.x));
        } else {
            val = exp(-abs(p.y)) * val;
        }
    }
    
    payload = vec3(0.0, val, 1.0 - val);
}
