#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT vec3 payload;

void main() {
    vec3 p = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    
    // Shader A: Short & Simple
    float val = 0.0;
    for(int i = 0; i < 10; i++) {
        val += sin(p.x * i) * cos(p.y * i);
    }
    float absVal = abs(val);
    payload = vec3(absVal + 1.0, absVal * 0.5, absVal * 0.25);
}
