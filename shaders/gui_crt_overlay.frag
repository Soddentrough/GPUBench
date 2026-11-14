#version 450

layout(location = 0) in vec2 vUv;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform CrtPush {
    float width;
    float height;
    float intensity;
    int paletteMode;
} pc;

void main() {
    vec2 res = max(vec2(pc.width, pc.height), vec2(1.0));
    vec2 uv = clamp(gl_FragCoord.xy / res, vec2(0.0), vec2(1.0));
    float strength = clamp(pc.intensity, 0.0, 1.5);

    if (strength <= 0.001) {
        outColor = vec4(0.0);
        return;
    }

    // Smooth corner bezel vignette
    vec2 edge = uv * (1.0 - uv);
    float vigFactor = clamp(16.0 * edge.x * edge.y, 0.0, 1.0);
    float vignette = 1.0 - pow(vigFactor, 0.30);

    if (pc.paletteMode == 2) {
        // Datasheet Paper mode: faint drafting micro-grid + warm edge patina
        float gx = step(0.96, fract(gl_FragCoord.x / 24.0));
        float gy = step(0.96, fract(gl_FragCoord.y / 24.0));
        float grid = max(gx, gy) * 0.035;
        vec3 col = mix(vec3(0.45, 0.42, 0.36), vec3(0.18, 0.16, 0.12), vignette);
        float alpha = clamp((grid + vignette * 0.10) * strength, 0.0, 0.16);
        outColor = vec4(col, alpha);
        return;
    }

    // Phosphor CRT modes (0: P3 Amber, 1: P1 Green)
    // Crisp 2-pixel horizontal scanline trough + subtle 3-pixel vertical aperture grille
    float scanTrough = 0.5 - 0.5 * cos(gl_FragCoord.y * 3.14159265);
    float grilleCol = mod(floor(gl_FragCoord.x), 3.0);
    float grille = (grilleCol == 0.0) ? 0.18 : 0.05;

    float darkMask = (scanTrough * 0.115 + grille * 0.055 + vignette * 0.17) * strength;
    float alpha = clamp(darkMask, 0.0, 0.28);

    // Subtle phosphor edge tint mixed into dark shadow veil so scanlines stay clean
    vec3 phosphorTint = (pc.paletteMode == 1)
        ? vec3(0.02, 0.08, 0.03)
        : vec3(0.08, 0.045, 0.005);
    vec3 finalRgb = mix(vec3(0.01, 0.012, 0.01), phosphorTint, (1.0 - scanTrough) * 0.35);

    outColor = vec4(finalRgb, alpha);
}
