#version 460

layout(location = 0) out vec2 vNdc;

void main() {
  // Fullscreen triangle covering [-1, 1] x [-1, 1] at far plane z = 0.0 (Reversed-Z)
  vec2 pos = vec2((gl_VertexIndex == 1) ? 3.0 : -1.0,
                  (gl_VertexIndex == 2) ? 3.0 : -1.0);
  vNdc = pos;
  gl_Position = vec4(pos, 0.0, 1.0);
}
