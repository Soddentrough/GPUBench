module rt_math;

// Hardware-exact SPIR-V GLSL.std.450 packing/unpacking intrinsics
public uint packSnorm2x16_spv(float2 v) {
    return spirv_asm { result:$$uint = OpExtInst glsl450 PackSnorm2x16 $v; };
}

public float2 unpackSnorm2x16_spv(uint u) {
    return spirv_asm { result:$$float2 = OpExtInst glsl450 UnpackSnorm2x16 $u; };
}

public uint packUnorm2x16_spv(float2 v) {
    return spirv_asm { result:$$uint = OpExtInst glsl450 PackUnorm2x16 $v; };
}

public float2 unpackUnorm2x16_spv(uint u) {
    return spirv_asm { result:$$float2 = OpExtInst glsl450 UnpackUnorm2x16 $u; };
}

public uint packHalf2x16_spv(float2 v) {
    return spirv_asm { result:$$uint = OpExtInst glsl450 PackHalf2x16 $v; };
}

public float2 unpackHalf2x16_spv(uint u) {
    return spirv_asm { result:$$float2 = OpExtInst glsl450 UnpackHalf2x16 $u; };
}

public float4 unpackUnorm4x8_spv(uint u) {
    return spirv_asm { result:$$float4 = OpExtInst glsl450 UnpackUnorm4x8 $u; };
}

// PCG PRNG
public uint pcgHash(inout uint state) {
    uint oldstate = state;
    state = oldstate * 747796405u + 2891336453u;
    uint word = ((oldstate >> ((oldstate >> 28u) + 4u)) ^ oldstate) * 277803737u;
    return (word >> 22u) ^ word;
}

public float randFloat(inout uint state) {
    return float(pcgHash(state) & 0x00FFFFFFu) / 16777216.0;
}

public uint serHash(uint seed) {
    seed = (seed ^ 61u) ^ (seed >> 16u);
    seed *= 9u;
    seed = seed ^ (seed >> 4u);
    seed *= 0x27d4eb2du;
    seed = seed ^ (seed >> 15u);
    return seed;
}

// Octahedral Direction Codec
public float2 signNotZero(float2 v) {
    return float2(v.x >= 0.0 ? 1.0 : -1.0, v.y >= 0.0 ? 1.0 : -1.0);
}

public float2 octEncode(float3 n) {
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    return (n.z >= 0.0) ? n.xy : ((1.0 - abs(n.yx)) * signNotZero(n.xy));
}

public float3 octDecode(float2 f) {
    float3 n = float3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    float t = clamp(-n.z, 0.0, 1.0);
    n.x += (n.x >= 0.0) ? -t : t;
    n.y += (n.y >= 0.0) ? -t : t;
    return normalize(n);
}

public float3 quantizeOctDir(float3 dir) {
    return octDecode(unpackSnorm2x16_spv(packSnorm2x16_spv(octEncode(dir))));
}

// Hemisphere Sampling & Tangent Frame Alignment
public float3 cosineSampleHemisphere(float u1, float u2) {
    float r = sqrt(u1);
    float theta = 6.28318530718 * u2;
    float x = r * cos(theta);
    float y = r * sin(theta);
    float z = sqrt(max(0.0, 1.0 - u1));
    return float3(x, y, z);
}

public float3 alignToNormal(float3 sampleDir, float3 normal) {
    float3 up = abs(normal.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    float3 tangent = normalize(cross(up, normal));
    float3 bitangent = cross(normal, tangent);
    return tangent * sampleDir.x + bitangent * sampleDir.y + normal * sampleDir.z;
}

// Procedural 3D Noise & FBM (explicit column-vector basis to match GLSL mat3 identically)
public float hash31(float3 p) {
    p = frac(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return frac((p.x + p.y) * p.z);
}

public float3 hash33(float3 p) {
    float3 p3 = frac(p * float3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yxz + 33.33);
    return frac((p3.xxy + p3.yxx) * p3.zyx);
}

public float noise3D(float3 p) {
    float3 i = floor(p);
    float3 f = frac(p);
    f = f * f * (3.0 - 2.0 * f);
    float n000 = hash31(i + float3(0, 0, 0));
    float n100 = hash31(i + float3(1, 0, 0));
    float n010 = hash31(i + float3(0, 1, 0));
    float n110 = hash31(i + float3(1, 1, 0));
    float n011 = hash31(i + float3(0, 0, 1));
    float n101 = hash31(i + float3(1, 0, 1));
    float n011b = hash31(i + float3(0, 1, 1));
    float n111 = hash31(i + float3(1, 1, 1));
    return lerp(
        lerp(lerp(n000, n100, f.x), lerp(n010, n110, f.x), f.y),
        lerp(lerp(n011, n101, f.x), lerp(n011b, n111, f.x), f.y),
        f.z
    );
}

public float rustFBM3D(float3 p) {
    float v = 0.0;
    float a = 0.5;
    for (int i = 0; i < 4; ++i) {
        v += a * noise3D(p);
        float3 rotP = float3( 0.00,  0.80,  0.60) * p.x
                    + float3(-0.80,  0.36, -0.48) * p.y
                    + float3(-0.60, -0.48,  0.64) * p.z;
        p = rotP * 2.02 + float3(1.7, 9.2, 0.4);
        a *= 0.5;
    }
    return v;
}

// Color & Tonemapping
public float3 tonemapACES(float3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

public float3 turboColormap(float x) {
    x = clamp(x, 0.0, 1.0);
    const float4 kRedVec4   = float4(0.13572138,  4.61539260, -42.66032258, 132.13108234);
    const float4 kGreenVec4 = float4(0.09140261,  2.19418839,   4.84296658, -14.18503333);
    const float4 kBlueVec4  = float4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
    const float2 kRedVec2   = float2(-152.94239396, 59.28637943);
    const float2 kGreenVec2 = float2(    4.27729857,  2.82956604);
    const float2 kBlueVec2  = float2( -89.90310912, 27.34824973);

    float4 v4 = float4(1.0, x, x * x, x * x * x);
    float2 v2 = v4.zw * v4.z;

    return clamp(float3(
        dot(v4, kRedVec4)   + dot(v2, kRedVec2),
        dot(v4, kGreenVec4) + dot(v2, kGreenVec2),
        dot(v4, kBlueVec4)  + dot(v2, kBlueVec2)
    ), 0.0, 1.0);
}

public uint luminanceToUint(float3 color) {
    return uint(clamp(dot(color, float3(0.2126, 0.7152, 0.0722)), 0.0, 100.0) * 256.0);
}

// Spatial Ray Coordinate Layout (Linear 32x1, Tiled 8x4, Morton 8x4, Morton 4x8)
public void computePixelCoords(uint idx, uint tileIdx, uint inTile, uint pattern, uint width, uint height, uint isGltf, out uint px_u, out uint py_u) {
    if (pattern == 1u) {
        uint tilesX = (width + 7u) / 8u;
        uint tileX = tileIdx % tilesX;
        uint tileY = tileIdx / tilesX;
        px_u = tileX * 8u + (inTile & 7u);
        py_u = tileY * 4u + (inTile >> 3u);
    } else if (pattern == 2u) {
        uint tilesX = (width + 7u) >> 3u;
        uint tilesY = (height + 3u) >> 2u;
        uint tileX, tileY;
        if (isGltf != 0u && (tilesX & 7u) == 0u && (tilesY & 3u) == 0u) {
            uint macroX = tilesX >> 3u;
            uint macroIdx = tileIdx >> 5u;
            uint inMacro = tileIdx & 31u;
            uint subX = ((inMacro >> 0u) & 1u) | (((inMacro >> 2u) & 1u) << 1u) | (((inMacro >> 4u) & 1u) << 2u);
            uint subY = ((inMacro >> 1u) & 1u) | (((inMacro >> 3u) & 1u) << 1u);
            tileX = (macroIdx % macroX) * 8u + subX;
            tileY = (macroIdx / macroX) * 4u + subY;
        } else if ((tilesX & 3u) == 0u && (tilesY & 1u) == 0u) {
            uint macroX = tilesX >> 2u;
            uint macroIdx = tileIdx >> 3u;
            uint inMacro = tileIdx & 7u;
            tileX = (macroIdx % macroX) * 4u + (inMacro & 3u);
            tileY = (macroIdx / macroX) * 2u + (inMacro >> 2u);
        } else {
            tileX = tileIdx % tilesX;
            tileY = tileIdx / tilesX;
        }
        uint mx = ((inTile >> 0u) & 1u) | (((inTile >> 2u) & 1u) << 1u) | (((inTile >> 4u) & 1u) << 2u);
        uint my = ((inTile >> 1u) & 1u) | (((inTile >> 3u) & 1u) << 1u);
        px_u = tileX * 8u + mx;
        py_u = tileY * 4u + my;
    } else if (pattern == 3u) {
        uint tilesX = (width + 3u) / 4u;
        uint tileX = tileIdx % tilesX;
        uint tileY = tileIdx / tilesX;
        uint mx = ((inTile >> 0u) & 1u) | (((inTile >> 2u) & 1u) << 1u);
        uint my = ((inTile >> 1u) & 1u) | (((inTile >> 3u) & 1u) << 1u) | (((inTile >> 4u) & 1u) << 2u);
        px_u = tileX * 4u + mx;
        py_u = tileY * 8u + my;
    } else {
        px_u = idx % width;
        py_u = idx / width;
    }
}

public void computePixelCoords(uint idx, uint pattern, uint width, uint height, uint isGltf, out uint px_u, out uint py_u) {
    computePixelCoords(idx, idx >> 5u, idx & 31u, pattern, width, height, isGltf, px_u, py_u);
}

public uint ballotFindLSB(uint4 b) {
    return (b.x != 0u) ? firstbitlow(b.x) : (32u + firstbitlow(b.y));
}
