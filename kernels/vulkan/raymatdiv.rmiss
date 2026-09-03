#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT vec3 payload;

// Analytic Atmospheric Rayleigh-Mie Sky Model for Ray Miss Evaluation
vec3 evalAnalyticSky(vec3 dir) {
    vec3 sunDir = normalize(vec3(0.3, 0.6, 0.7));
    float cosTheta = clamp(dir.y, 0.0, 1.0);
    float cosGamma = dot(dir, sunDir);

    // Rayleigh scattering gradient (zenith deep blue -> horizon haze)
    vec3 zenithCol = vec3(0.08, 0.25, 0.65);
    vec3 horizonCol = vec3(0.65, 0.75, 0.88);
    float rayleighPhase = 0.75 * (1.0 + cosGamma * cosGamma);
    vec3 skyGrad = mix(horizonCol, zenithCol, pow(cosTheta, 0.45));

    // Mie scattering (forward solar aureole via Henyey-Greenstein)
    const float g = 0.78;
    const float g2 = g * g;
    float miePhase = (1.5 * (1.0 - g2) / (2.0 + g2)) * (1.0 + cosGamma * cosGamma) / pow(max(1.0 + g2 - 2.0 * g * cosGamma, 0.001), 1.5);
    vec3 sunColor = vec3(1.0, 0.9, 0.7) * 3.0;

    // Ground bounce reflection below horizon
    if (dir.y < 0.0) {
        vec3 groundCol = vec3(0.12, 0.10, 0.09);
        return mix(horizonCol * 0.5, groundCol, clamp(-dir.y * 2.0, 0.0, 1.0));
    }

    return skyGrad * rayleighPhase + sunColor * (miePhase * 0.05);
}

void main() {
    payload = evalAnalyticSky(gl_WorldRayDirectionEXT);
}
