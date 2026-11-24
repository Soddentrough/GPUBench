#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace gpubench::showcase {

constexpr float kPi = 3.14159265358979323846f;

struct Vec2 {
  float x{0.0f}, y{0.0f};
  Vec2() = default;
  Vec2(float x_, float y_) : x(x_), y(y_) {}
  Vec2 operator+(const Vec2 &o) const { return {x + o.x, y + o.y}; }
  Vec2 operator-(const Vec2 &o) const { return {x - o.x, y - o.y}; }
  Vec2 operator*(float s) const { return {x * s, y * s}; }
};

struct Vec3 {
  float x{0.0f}, y{0.0f}, z{0.0f};
  Vec3() = default;
  Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
  Vec3 operator+(const Vec3 &o) const { return {x + o.x, y + o.y, z + o.z}; }
  Vec3 operator-(const Vec3 &o) const { return {x - o.x, y - o.y, z - o.z}; }
  Vec3 operator-() const { return {-x, -y, -z}; }
  Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
  Vec3 operator/(float s) const { return {x / s, y / s, z / s}; }
  Vec3 &operator+=(const Vec3 &o) {
    x += o.x;
    y += o.y;
    z += o.z;
    return *this;
  }
  Vec3 &operator-=(const Vec3 &o) {
    x -= o.x;
    y -= o.y;
    z -= o.z;
    return *this;
  }
};

struct Vec4 {
  float x{0.0f}, y{0.0f}, z{0.0f}, w{0.0f};
  Vec4() = default;
  Vec4(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) {}
  Vec4(const Vec3 &v, float w_) : x(v.x), y(v.y), z(v.z), w(w_) {}
};

inline float dot(const Vec3 &a, const Vec3 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3 cross(const Vec3 &a, const Vec3 &b) {
  return {a.y * b.z - a.z * b.y,
          a.z * b.x - a.x * b.z,
          a.x * b.y - a.y * b.x};
}

inline float length(const Vec3 &v) {
  return std::sqrt(dot(v, v));
}

inline Vec3 normalize(const Vec3 &v) {
  float len = length(v);
  return (len > 1e-7f) ? (v / len) : Vec3{0.0f, 1.0f, 0.0f};
}

inline Vec3 lerp(const Vec3 &a, const Vec3 &b, float t) {
  return a + (b - a) * t;
}

// Column-major 4x4 matrix matching GLSL mat4: element(row, col) = m[col * 4 + row]
struct Mat4 {
  float m[16]{};

  static Mat4 identity() {
    Mat4 r{};
    r.m[0] = 1.0f;
    r.m[5] = 1.0f;
    r.m[10] = 1.0f;
    r.m[15] = 1.0f;
    return r;
  }

  float &at(int row, int col) { return m[col * 4 + row]; }
  float at(int row, int col) const { return m[col * 4 + row]; }
};

inline Mat4 matMul(const Mat4 &a, const Mat4 &b) {
  Mat4 r{};
  for (int col = 0; col < 4; ++col) {
    for (int row = 0; row < 4; ++row) {
      r.at(row, col) = a.at(row, 0) * b.at(0, col) +
                       a.at(row, 1) * b.at(1, col) +
                       a.at(row, 2) * b.at(2, col) +
                       a.at(row, 3) * b.at(3, col);
    }
  }
  return r;
}

inline Vec4 matMulVec4(const Mat4 &m, const Vec4 &v) {
  return {
      m.at(0, 0) * v.x + m.at(0, 1) * v.y + m.at(0, 2) * v.z + m.at(0, 3) * v.w,
      m.at(1, 0) * v.x + m.at(1, 1) * v.y + m.at(1, 2) * v.z + m.at(1, 3) * v.w,
      m.at(2, 0) * v.x + m.at(2, 1) * v.y + m.at(2, 2) * v.z + m.at(2, 3) * v.w,
      m.at(3, 0) * v.x + m.at(3, 1) * v.y + m.at(3, 2) * v.z + m.at(3, 3) * v.w};
}

inline Vec3 transformPoint(const Mat4 &m, const Vec3 &p) {
  Vec4 v = matMulVec4(m, Vec4(p, 1.0f));
  return {v.x, v.y, v.z};
}

inline Vec3 transformVector(const Mat4 &m, const Vec3 &v) {
  Vec4 r = matMulVec4(m, Vec4(v, 0.0f));
  return {r.x, r.y, r.z};
}

inline Mat4 matTranslation(const Vec3 &t) {
  Mat4 r = Mat4::identity();
  r.at(0, 3) = t.x;
  r.at(1, 3) = t.y;
  r.at(2, 3) = t.z;
  return r;
}

inline Mat4 matScale(const Vec3 &s) {
  Mat4 r{};
  r.at(0, 0) = s.x;
  r.at(1, 1) = s.y;
  r.at(2, 2) = s.z;
  r.at(3, 3) = 1.0f;
  return r;
}

inline Mat4 matRotationY(float rad) {
  Mat4 r = Mat4::identity();
  float c = std::cos(rad);
  float s = std::sin(rad);
  r.at(0, 0) = c;
  r.at(0, 2) = s;
  r.at(2, 0) = -s;
  r.at(2, 2) = c;
  return r;
}

inline Mat4 matTRS(const Vec3 &t, float rotYRad, const Vec3 &s) {
  return matMul(matTranslation(t), matMul(matRotationY(rotYRad), matScale(s)));
}

// Right-handed look-at view matrix (+Y up, camera looks along -Z in view space)
inline Mat4 matLookAt(const Vec3 &eye, const Vec3 &target, const Vec3 &worldUp) {
  Vec3 f = normalize(target - eye);
  Vec3 r = normalize(cross(f, worldUp));
  Vec3 u = cross(r, f);

  Mat4 view = Mat4::identity();
  view.at(0, 0) = r.x;  view.at(0, 1) = r.y;  view.at(0, 2) = r.z;  view.at(0, 3) = -dot(r, eye);
  view.at(1, 0) = u.x;  view.at(1, 1) = u.y;  view.at(1, 2) = u.z;  view.at(1, 3) = -dot(u, eye);
  view.at(2, 0) = -f.x; view.at(2, 1) = -f.y; view.at(2, 2) = -f.z; view.at(2, 3) = dot(f, eye);
  return view;
}

// Vulkan Y-flipped Reversed-Z perspective projection: near -> 1.0, far -> 0.0
inline Mat4 matPerspectiveReversedZ(float fovYRad, float aspect, float nearZ, float farZ) {
  float f = 1.0f / std::tan(fovYRad * 0.5f);
  Mat4 proj{};
  proj.at(0, 0) = f / std::max(0.01f, aspect);
  proj.at(1, 1) = -f; // Vulkan clip-space +Y is downward
  proj.at(2, 2) = nearZ / (farZ - nearZ);
  proj.at(3, 2) = -1.0f;
  proj.at(2, 3) = (farZ * nearZ) / (farZ - nearZ);
  return proj;
}

// Vulkan Y-flipped standard [0, 1] orthographic matrix for directional shadow map
inline Mat4 matOrthoVulkan(float left, float right, float bottom, float top, float nearZ, float farZ) {
  Mat4 o{};
  o.at(0, 0) = 2.0f / (right - left);
  o.at(1, 1) = -2.0f / (top - bottom);
  o.at(2, 2) = -1.0f / (farZ - nearZ);
  o.at(0, 3) = -(right + left) / (right - left);
  o.at(1, 3) = (top + bottom) / (top - bottom); // Y-flipped
  o.at(2, 3) = -nearZ / (farZ - nearZ);
  o.at(3, 3) = 1.0f;
  return o;
}

// Extracts 6 normalized inward frustum planes from Reversed-Z Vulkan viewProj
inline void extractFrustumPlanesReversedZ(const Mat4 &vp, Vec4 planes[6]) {
  auto row = [&](int r) -> Vec4 {
    return {vp.at(r, 0), vp.at(r, 1), vp.at(r, 2), vp.at(r, 3)};
  };
  auto add4 = [](const Vec4 &a, const Vec4 &b) -> Vec4 {
    return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
  };
  auto sub4 = [](const Vec4 &a, const Vec4 &b) -> Vec4 {
    return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
  };
  auto normPlane = [](Vec4 p) -> Vec4 {
    float len = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
    if (len > 1e-7f) {
      p.x /= len;
      p.y /= len;
      p.z /= len;
      p.w /= len;
    }
    return p;
  };

  Vec4 r0 = row(0), r1 = row(1), r2 = row(2), r3 = row(3);
  planes[0] = normPlane(add4(r3, r0)); // Left:   x + w >= 0
  planes[1] = normPlane(sub4(r3, r0)); // Right:  w - x >= 0
  planes[2] = normPlane(add4(r3, r1)); // Top:    y + w >= 0
  planes[3] = normPlane(sub4(r3, r1)); // Bottom: w - y >= 0
  planes[4] = normPlane(r2);           // Far:    z >= 0 (Reversed-Z)
  planes[5] = normPlane(sub4(r3, r2)); // Near:   w - z >= 0 (Reversed-Z)
}

inline Vec4 transformBoundingSphere(const Mat4 &model, const Vec4 &localSphere) {
  Vec3 worldCenter = transformPoint(model, {localSphere.x, localSphere.y, localSphere.z});
  float sx = length({model.at(0, 0), model.at(1, 0), model.at(2, 0)});
  float sy = length({model.at(0, 1), model.at(1, 1), model.at(2, 1)});
  float sz = length({model.at(0, 2), model.at(1, 2), model.at(2, 2)});
  float maxScale = std::max({sx, sy, sz, 1e-4f});
  return {worldCenter.x, worldCenter.y, worldCenter.z, localSphere.w * maxScale};
}

struct CameraState {
  Vec3 pos{-1050.0f, 220.0f, -20.0f};
  float yaw{0.0f};      // 0 = +X down Sponza nave
  float pitch{-0.04f};
  float fovY{0.78f};    // ~45 deg
  float nearZ{10.0f};
  float farZ{12000.0f};

  Vec3 forward() const {
    float cp = std::cos(pitch);
    return normalize({std::cos(yaw) * cp, std::sin(pitch), std::sin(yaw) * cp});
  }

  Vec3 right() const {
    return normalize(cross(forward(), Vec3{0.0f, 1.0f, 0.0f}));
  }

  Vec3 up() const {
    return cross(right(), forward());
  }

  Mat4 viewMatrix() const {
    return matLookAt(pos, pos + forward(), {0.0f, 1.0f, 0.0f});
  }

  Mat4 projMatrix(float aspect) const {
    return matPerspectiveReversedZ(fovY, aspect, nearZ, farZ);
  }
};

inline Vec3 evalCatmullRom(const Vec3 &p0, const Vec3 &p1, const Vec3 &p2, const Vec3 &p3, float t) {
  float t2 = t * t;
  float t3 = t2 * t;
  return (p1 * 2.0f +
          (p2 - p0) * t +
          (p0 * 2.0f - p1 * 5.0f + p2 * 4.0f - p3) * t2 +
          (-p0 + p1 * 3.0f - p2 * 3.0f + p3) * t3) *
         0.5f;
}

} // namespace gpubench::showcase
