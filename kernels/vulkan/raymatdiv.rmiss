#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT vec3 payload;

// Procedural hash & value noise for atmospheric cloud coverage
float skyHash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

float skyNoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = skyHash21(i);
    float b = skyHash21(i + vec2(1.0, 0.0));
    float c = skyHash21(i + vec2(0.0, 1.0));
    float d = skyHash21(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float skyCloudFBM(vec2 p) {
    float v = 0.0;
    float a = 0.5;
    mat2 rot = mat2(0.8, -0.6, 0.6, 0.8);
    for (int i = 0; i < 3; ++i) {
        v += a * skyNoise(p);
        p = rot * p * 2.02 + vec2(1.7, 9.2);
        a *= 0.5;
    }
    return v;
}

// Analytic Atmospheric Rayleigh-Mie Sky Model with Procedural Clouds for Ray Miss Evaluation
vec3 evalAnalyticSky(vec3 dir) {
    vec3 sunDir = normalize(vec3(0.4, 0.55, 0.72));
    float cosTheta = clamp(dir.y, 0.0, 1.0);
    float cosGamma = dot(dir, sunDir);

    // Deep zenith gradient -> atmospheric haze
    vec3 zenithCol = vec3(0.06, 0.22, 0.62);
    vec3 horizonCol = vec3(0.58, 0.70, 0.84);
    float rayleighPhase = 0.75 * (1.0 + cosGamma * cosGamma);
    vec3 skyGrad = mix(horizonCol, zenithCol, pow(cosTheta, 0.55));

    // Sharp solar disc + Henyey-Greenstein corona
    float sunDisc = smoothstep(0.9985, 0.9995, cosGamma);
    const float g = 0.82;
    const float g2 = g * g;
    float miePhase = (1.5 * (1.0 - g2) / (2.0 + g2)) * (1.0 + cosGamma * cosGamma) 
                   / pow(max(1.0 + g2 - 2.0 * g * cosGamma, 0.001), 1.5);
    vec3 sunColor = vec3(1.0, 0.92, 0.75);

    vec3 sky = skyGrad * rayleighPhase + sunColor * (miePhase * 0.04) + sunColor * (sunDisc * 8.0);

    // Procedural planar cloud deck (projected to altitude plane)
    if (dir.y > 0.02) {
        vec2 cloudUV = (dir.xz / (dir.y + 0.05)) * 1.8;
        float cloudDensity = smoothstep(0.42, 0.75, skyCloudFBM(cloudUV));
        if (cloudDensity > 0.0) {
            float sunScatter = clamp(cosGamma, 0.0, 1.0);
            vec3 cloudLit = vec3(0.72, 0.76, 0.82) * 0.4 + vec3(1.0, 0.96, 0.90) * 1.4 * pow(sunScatter, 3.0);
            vec3 cloudShadow = vec3(0.35, 0.38, 0.45);
            vec3 cloudColor = mix(cloudShadow, cloudLit, 0.6 + 0.4 * cosTheta);
            sky = mix(sky, cloudColor, cloudDensity * 0.85);
        }
    } else {
        vec3 groundCol = vec3(0.10, 0.09, 0.08);
        sky = mix(horizonCol * 0.4, groundCol, clamp(-dir.y * 3.0, 0.0, 1.0));
    }

    return sky;
}

void main() {
    payload = evalAnalyticSky(gl_WorldRayDirectionEXT);
}
