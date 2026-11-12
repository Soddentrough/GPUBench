#!/usr/bin/env python3
"""
Deterministic zero-dependency generator for GPUBench's first-party MIT / CC0-1.0
indoor architectural ray-tracing benchmark asset (`assets/models/sponza.glb`).

Replaces third-party Crytek Sponza with a clean-room Renaissance Colonnade Atrium
in the exact centimeter coordinate frame (`X in [-1900, 1800]`, `Y in [0, 1340]`,
`Z in [-1100, 1100]`) expected by `GltfScene` and `RaySchedulingBench` (`-s indoor`),
featuring:
  - Open-air Renaissance central courtyard skylight (`X in [-1440, 1440]`, `Z in [-355, 355]`)
    framed by covered two-story perimeter colonnade galleries, rooftop marble balustrades,
    and pergola cross-beams for natural overhead sunlight and sky-silhouette divergence
  - 426,064 indexed triangles (248,427 vertices) with smooth/faceted normals, tangents, and UVs
  - 25 named glTF 2.0 PBR materials exercising all 8 `GltfScene` BSDF archetypes
    (Conductor Metal, SSS Marble/Stone, Dielectric Glass, Anisotropic Velvet/Fabric,
    Weathered Stone/Rust/Brick/Wood, Polished Terrazzo Floor, Clearcoat Lacquer/Paint,
    and Alpha-Cutout Foliage `alphaMode="MASK"`)
  - Embedded procedural RGBA8 PNG textures (baseColor, normal, metallicRoughness,
    and serrated alpha-masked 3D botanical leaf blades)

Uses only the Python 3 standard library (`array`, `json`, `math`, `random`, `struct`, `zlib`).
License of this generator and its output `assets/models/sponza.glb`:
  MIT License / CC0 1.0 Universal (Dual Licensed, Public Domain Dedication).
"""

import array
import json
import math
import os
import random
import struct
import zlib
from typing import Callable, Dict, List, Optional, Tuple


def clamp_u8(v: float) -> int:
    if v <= 0.0:
        return 0
    if v >= 255.0:
        return 255
    return int(v)


def encode_rgba8_png(w: int, h: int, pixel_fn: Callable[[float, float], Tuple[int, int, int, int]]) -> bytes:
    """Encode an (H, W) procedural RGBA8 PNG using stdlib zlib."""
    rows = bytearray()
    inv_w = 1.0 / float(w)
    inv_h = 1.0 / float(h)
    for py in range(h):
        rows.append(0)  # Filter type 0 (None)
        yf = py * inv_h
        for px in range(w):
            r, g, b, a = pixel_fn(px * inv_w, yf)
            rows.extend((r, g, b, a))
    compressed = zlib.compress(bytes(rows), level=9)

    def chunk(tag: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", compressed) + chunk(b"IEND", b"")


def build_procedural_textures() -> List[Tuple[str, bytes]]:
    size = 64
    tau = 2.0 * math.pi
    textures: List[Tuple[str, bytes]] = []

    def tex_terrazzo(x: float, y: float) -> Tuple[int, int, int, int]:
        checker = float((int(x * 8.0) + int(y * 8.0)) & 1)
        speck = 0.5 + 0.5 * math.sin(x * 37.0 * tau + math.cos(y * 29.0 * tau))
        r = clamp_u8((0.72 * checker + 0.34 * (1.0 - checker) + 0.06 * speck) * 255.0)
        g = clamp_u8((0.68 * checker + 0.31 * (1.0 - checker) + 0.05 * speck) * 255.0)
        b = clamp_u8((0.62 * checker + 0.29 * (1.0 - checker) + 0.05 * speck) * 255.0)
        return (r, g, b, 255)

    def tex_ashlar(x: float, y: float) -> Tuple[int, int, int, int]:
        row = int(y * 8.0)
        bx = (x * 4.0 + (0.5 if (row & 1) else 0.0)) % 1.0
        by = (y * 8.0) % 1.0
        mortar = 1.0 if (bx < 0.06 or by < 0.09) else 0.0
        grain = 0.5 + 0.5 * math.sin((x * 11.0 + y * 23.0) * tau)
        r = clamp_u8(((1.0 - mortar) * (0.68 + 0.08 * grain) + mortar * 0.45) * 255.0)
        g = clamp_u8(((1.0 - mortar) * (0.58 + 0.07 * grain) + mortar * 0.43) * 255.0)
        b = clamp_u8(((1.0 - mortar) * (0.46 + 0.06 * grain) + mortar * 0.40) * 255.0)
        return (r, g, b, 255)

    def tex_marble(x: float, y: float) -> Tuple[int, int, int, int]:
        vein = abs(math.sin((x * 3.0 + 0.35 * math.sin(y * 4.0 * tau)) * tau)) ** 0.35
        return (
            clamp_u8((0.55 + 0.40 * vein) * 255.0),
            clamp_u8((0.56 + 0.39 * vein) * 255.0),
            clamp_u8((0.59 + 0.37 * vein) * 255.0),
            255,
        )

    def tex_fabric(x: float, y: float) -> Tuple[int, int, int, int]:
        weave = 0.82 + 0.16 * (math.sin(x * 32.0 * tau) * math.cos(y * 32.0 * tau))
        border = 1.0 if (x < 0.08 or x > 0.92 or y < 0.06 or y > 0.94) else 0.0
        return (
            clamp_u8((weave * (1.0 - border) + border * 0.95) * 255.0),
            clamp_u8((weave * (1.0 - border) + border * 0.82) * 255.0),
            clamp_u8((weave * (1.0 - border) + border * 0.35) * 255.0),
            255,
        )

    def tex_foliage(x: float, y: float) -> Tuple[int, int, int, int]:
        u = x * 2.0 - 1.0
        vf = min(max(y, 0.0), 1.0)
        env = (math.sin(vf * math.pi) ** 0.72) * (0.86 + 0.14 * math.sin(vf * 11.0 * math.pi))
        dist = abs(u)
        inside = (dist <= env) and (vf > 0.02) and (vf < 0.98)
        midrib = math.exp(-16.0 * dist)
        vein = math.exp(-14.0 * abs(math.sin((vf * 7.0 - dist * 2.2) * math.pi)))
        shade = 0.72 + 0.28 * vf
        return (
            clamp_u8((0.13 + 0.12 * midrib + 0.05 * vein) * shade * 255.0),
            clamp_u8((0.40 + 0.22 * midrib + 0.10 * vein) * shade * 255.0),
            clamp_u8((0.11 + 0.07 * midrib + 0.03 * vein) * shade * 255.0),
            255 if inside else 0,
        )

    def tex_normal(x: float, y: float) -> Tuple[int, int, int, int]:
        nx = 0.18 * math.cos(x * 8.0 * tau)
        ny = 0.18 * math.cos(y * 8.0 * tau)
        nz = math.sqrt(max(0.0, 1.0 - nx * nx - ny * ny))
        return (
            clamp_u8((nx * 0.5 + 0.5) * 255.0),
            clamp_u8((ny * 0.5 + 0.5) * 255.0),
            clamp_u8((nz * 0.5 + 0.5) * 255.0),
            255,
        )

    def tex_mr(x: float, y: float) -> Tuple[int, int, int, int]:
        checker = float((int(x * 8.0) + int(y * 8.0)) & 1)
        rough = clamp_u8((0.35 + 0.35 * (1.0 - checker)) * 255.0)
        return (240, rough, 0, 255)

    textures.append(("terrazzo_albedo", encode_rgba8_png(size, size, tex_terrazzo)))
    textures.append(("ashlar_albedo", encode_rgba8_png(size, size, tex_ashlar)))
    textures.append(("marble_albedo", encode_rgba8_png(size, size, tex_marble)))
    textures.append(("fabric_albedo", encode_rgba8_png(size, size, tex_fabric)))
    textures.append(("foliage_cutout", encode_rgba8_png(size, size, tex_foliage)))
    textures.append(("stone_normal", encode_rgba8_png(size, size, tex_normal)))
    textures.append(("stone_mr", encode_rgba8_png(size, size, tex_mr)))
    return textures


def normalize3(x: float, y: float, z: float) -> Tuple[float, float, float]:
    l = math.sqrt(x * x + y * y + z * z)
    if l <= 1e-8:
        return (0.0, 1.0, 0.0)
    inv = 1.0 / l
    return (x * inv, y * inv, z * inv)


def cross3(ax: float, ay: float, az: float, bx: float, by: float, bz: float) -> Tuple[float, float, float]:
    return (ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx)


def tangent_from_normal(nx: float, ny: float, nz: float) -> Tuple[float, float, float, float]:
    if abs(ny) < 0.92:
        tx, ty, tz = cross3(0.0, 1.0, 0.0, nx, ny, nz)
    else:
        tx, ty, tz = cross3(1.0, 0.0, 0.0, nx, ny, nz)
    tx, ty, tz = normalize3(tx, ty, tz)
    return (tx, ty, tz, 1.0)


class MeshBuilder:
    """Accumulates indexed triangle primitives per material ID using flat C-typed arrays."""

    def __init__(self) -> None:
        self.pos = array.array("f")
        self.norm = array.array("f")
        self.tan = array.array("f")
        self.uv = array.array("f")
        self.idx = array.array("I")
        self.vcount = 0
        self.pmin = [1e9, 1e9, 1e9]
        self.pmax = [-1e9, -1e9, -1e9]

    def _push_vert(self, px: float, py: float, pz: float, nx: float, ny: float, nz: float, tx: float, ty: float, tz: float, tw: float, u: float, v: float) -> None:
        self.pos.extend((px, py, pz))
        self.norm.extend((nx, ny, nz))
        self.tan.extend((tx, ty, tz, tw))
        self.uv.extend((u, v))
        if px < self.pmin[0]:
            self.pmin[0] = px
        if py < self.pmin[1]:
            self.pmin[1] = py
        if pz < self.pmin[2]:
            self.pmin[2] = pz
        if px > self.pmax[0]:
            self.pmax[0] = px
        if py > self.pmax[1]:
            self.pmax[1] = py
        if pz > self.pmax[2]:
            self.pmax[2] = pz

    def _emit_grid_indices(self, base: int, nu: int, nv: int, flip: bool = False) -> None:
        stride = nu + 1
        ext = self.idx.extend
        for i in range(nv):
            row0 = base + i * stride
            row1 = row0 + stride
            if not flip:
                for j in range(nu):
                    v00 = row0 + j
                    v10 = v00 + 1
                    v01 = row1 + j
                    v11 = v01 + 1
                    ext((v00, v10, v11, v00, v11, v01))
            else:
                for j in range(nu):
                    v00 = row0 + j
                    v10 = v00 + 1
                    v01 = row1 + j
                    v11 = v01 + 1
                    ext((v00, v01, v11, v00, v11, v10))

    def add_grid_patch(
        self,
        origin: Tuple[float, float, float],
        axis_u: Tuple[float, float, float],
        axis_v: Tuple[float, float, float],
        nu: int,
        nv: int,
        uv_scale: Tuple[float, float] = (1.0, 1.0),
        wave_amp: float = 0.0,
        wave_freq: float = 2.0,
    ) -> None:
        ox, oy, oz = origin
        ux, uy, uz = axis_u
        vx, vy, vz = axis_v
        nx, ny, nz = normalize3(*cross3(ux, uy, uz, vx, vy, vz))
        tx, ty, tz, tw = tangent_from_normal(nx, ny, nz)

        base = self.vcount
        inv_u = 1.0 / float(nu)
        inv_v = 1.0 / float(nv)
        su, sv = uv_scale

        for i in range(nv + 1):
            vf = i * inv_v
            sin_v = math.sin(vf * wave_freq * math.pi) if wave_amp != 0.0 else 0.0
            bx = ox + vf * vx
            by = oy + vf * vy
            bz = oz + vf * vz
            for j in range(nu + 1):
                uf = j * inv_u
                px = bx + uf * ux
                py = by + uf * uy
                pz = bz + uf * uz
                if wave_amp != 0.0:
                    d = wave_amp * sin_v * math.sin(uf * 2.0 * math.pi)
                    px += nx * d
                    py += ny * d
                    pz += nz * d
                self._push_vert(px, py, pz, nx, ny, nz, tx, ty, tz, tw, uf * su, vf * sv)

        self.vcount += (nu + 1) * (nv + 1)
        self._emit_grid_indices(base, nu, nv, flip=False)

    def add_botanical_leaf(
        self,
        stem: Tuple[float, float, float],
        axis_u: Tuple[float, float, float],
        axis_v: Tuple[float, float, float],
        nu: int = 4,
        nv: int = 6,
    ) -> None:
        sx, sy, sz = stem
        ux, uy, uz = axis_u
        vx, vy, vz = axis_v
        nx, ny, nz = normalize3(*cross3(ux, uy, uz, vx, vy, vz))
        tx, ty, tz, tw = tangent_from_normal(nx, ny, nz)

        base = self.vcount
        inv_u = 1.0 / float(nu)
        inv_v = 1.0 / float(nv)
        len_v = math.sqrt(vx * vx + vy * vy + vz * vz)
        arch_scale = 0.14 * len_v
        fold_scale = 0.08 * len_v

        for i in range(nv + 1):
            vf = i * inv_v
            sin_v = math.sin(vf * math.pi)
            w_env = (sin_v ** 0.72) if sin_v > 0.0 else 0.0
            arch = arch_scale * sin_v
            bx = sx + vf * vx + nx * arch
            by = sy + vf * vy + ny * arch
            bz = sz + vf * vz + nz * arch
            for j in range(nu + 1):
                uf = j * inv_u
                us = 2.0 * uf - 1.0
                span = 0.5 * us * w_env
                fold = -fold_scale * abs(us) * w_env
                px = bx + span * ux + nx * fold
                py = by + span * uy + ny * fold
                pz = bz + span * uz + nz * fold
                lnx, lny, lnz = normalize3(
                    nx - 0.22 * us * ux / max(len_v, 1.0),
                    ny - 0.22 * us * uy / max(len_v, 1.0),
                    nz - 0.22 * us * uz / max(len_v, 1.0),
                )
                self._push_vert(px, py, pz, lnx, lny, lnz, tx, ty, tz, tw, 0.5 + 0.48 * us, vf)

        self.vcount += (nu + 1) * (nv + 1)
        self._emit_grid_indices(base, nu, nv, flip=False)

    def add_box(
        self,
        xmin: float,
        ymin: float,
        zmin: float,
        xmax: float,
        ymax: float,
        zmax: float,
        subdiv: Tuple[int, int, int] = (2, 2, 2),
        uv_tile: float = 0.01,
    ) -> None:
        nx, ny, nz = subdiv
        dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
        faces = (
            ((xmin, ymax, zmax), (dx, 0.0, 0.0), (0.0, 0.0, -dz), nx, nz, (dx * uv_tile, dz * uv_tile)),
            ((xmin, ymin, zmin), (dx, 0.0, 0.0), (0.0, 0.0, dz), nx, nz, (dx * uv_tile, dz * uv_tile)),
            ((xmin, ymin, zmax), (dx, 0.0, 0.0), (0.0, dy, 0.0), nx, ny, (dx * uv_tile, dy * uv_tile)),
            ((xmax, ymin, zmin), (-dx, 0.0, 0.0), (0.0, dy, 0.0), nx, ny, (dx * uv_tile, dy * uv_tile)),
            ((xmax, ymin, zmax), (0.0, 0.0, -dz), (0.0, dy, 0.0), nz, ny, (dz * uv_tile, dy * uv_tile)),
            ((xmin, ymin, zmin), (0.0, 0.0, dz), (0.0, dy, 0.0), nz, ny, (dz * uv_tile, dy * uv_tile)),
        )
        for orig, au, av, su, sv, uvs in faces:
            self.add_grid_patch(orig, au, av, su, sv, uvs)

    def add_lathe_column(
        self,
        cx: float,
        y0: float,
        y1: float,
        cz: float,
        r_base: float,
        radial_segs: int,
        height_segs: int,
        flutes: int = 16,
        flute_depth: float = 0.045,
        entasis: float = 0.06,
    ) -> None:
        base = self.vcount
        tau = 2.0 * math.pi
        inv_r = 1.0 / float(radial_segs)
        inv_h = 1.0 / float(height_segs)
        dy = y1 - y0

        cos_th = [math.cos(j * inv_r * tau) for j in range(radial_segs + 1)]
        sin_th = [math.sin(j * inv_r * tau) for j in range(radial_segs + 1)]
        flute_c = [0.5 + 0.5 * math.cos(flutes * j * inv_r * tau) for j in range(radial_segs + 1)]

        for i in range(height_segs + 1):
            tt = i * inv_h
            sin_t = math.sin(tt * math.pi)
            profile = 1.0 + entasis * sin_t - 0.10 * tt
            if tt < 0.07:
                flare = 1.18 - 2.5 * tt
            elif tt > 0.93:
                flare = 1.0 + 2.4 * (tt - 0.93)
            else:
                flare = 1.0
            py = y0 + tt * dy
            r_ring = r_base * profile * flare
            for j in range(radial_segs + 1):
                r = r_ring * (1.0 - flute_depth * flute_c[j] * sin_t)
                ct = cos_th[j]
                st = sin_th[j]
                self._push_vert(cx + r * ct, py, cz + r * st, ct, 0.0, st, -st, 0.0, ct, 1.0, j * inv_r * 2.0, tt * 4.0)

        self.vcount += (radial_segs + 1) * (height_segs + 1)
        self._emit_grid_indices(base, radial_segs, height_segs, flip=True)

    def add_arcade_arch(
        self,
        x0: float,
        x1: float,
        y_spring: float,
        z_center: float,
        thickness_z: float,
        ring_height: float,
        arch_segs: int,
        depth_segs: int,
    ) -> None:
        cx = 0.5 * (x0 + x1)
        r_in = 0.5 * abs(x1 - x0) - 24.0
        z0 = z_center - 0.5 * thickness_z

        # Intrados
        base = self.vcount
        inv_a = 1.0 / float(arch_segs)
        inv_d = 1.0 / float(depth_segs)
        for i in range(depth_segs + 1):
            zf = i * inv_d
            pz = z0 + zf * thickness_z
            for j in range(arch_segs + 1):
                af = j * inv_a
                phi = af * math.pi
                cp = math.cos(phi)
                sp = math.sin(phi)
                px = cx - r_in * cp
                py = y_spring + r_in * sp
                tx, ty, tz, tw = tangent_from_normal(cp, -sp, 0.0)
                self._push_vert(px, py, pz, cp, -sp, 0.0, tx, ty, tz, tw, af * 3.0, zf)
        self.vcount += (arch_segs + 1) * (depth_segs + 1)
        self._emit_grid_indices(base, arch_segs, depth_segs, flip=False)

        # Front & back voussoir rings
        ring_segs = 3
        for z_face, n_sign, flip in ((z0, -1.0, True), (z0 + thickness_z, 1.0, False)):
            fbase = self.vcount
            for i in range(ring_segs + 1):
                rf = i / float(ring_segs)
                rr = r_in + rf * ring_height
                for j in range(arch_segs + 1):
                    af = j * inv_a
                    phi = af * math.pi
                    px = cx - rr * math.cos(phi)
                    py = y_spring + rr * math.sin(phi)
                    self._push_vert(px, py, z_face, 0.0, 0.0, n_sign, n_sign, 0.0, 0.0, 1.0, af * 4.0, rf)
            self.vcount += (arch_segs + 1) * (ring_segs + 1)
            self._emit_grid_indices(fbase, arch_segs, ring_segs, flip=flip)

    def add_uv_sphere(
        self,
        cx: float,
        cy: float,
        cz: float,
        rx: float,
        ry: float,
        rz: float,
        u_segs: int,
        v_segs: int,
    ) -> None:
        base = self.vcount
        tau = 2.0 * math.pi
        inv_u = 1.0 / float(u_segs)
        inv_v = 1.0 / float(v_segs)
        for i in range(v_segs + 1):
            vf = i * inv_v
            sv = math.sin(vf * math.pi)
            cv = math.cos(vf * math.pi)
            for j in range(u_segs + 1):
                uf = j * inv_u
                cu = math.cos(uf * tau)
                su = math.sin(uf * tau)
                ux = sv * cu
                uy = cv
                uz = sv * su
                nx, ny, nz = normalize3(ux / max(rx, 1.0), uy / max(ry, 1.0), uz / max(rz, 1.0))
                self._push_vert(cx + rx * ux, cy + ry * uy, cz + rz * uz, nx, ny, nz, -su, 0.0, cu, 1.0, uf, vf)
        self.vcount += (u_segs + 1) * (v_segs + 1)
        self._emit_grid_indices(base, u_segs, v_segs, flip=True)


def build_materials() -> List[Dict]:
    def mat(
        name: str,
        color: Tuple[float, float, float, float],
        metallic: float,
        roughness: float,
        base_tex: Optional[int] = None,
        norm_tex: Optional[int] = None,
        mr_tex: Optional[int] = None,
        alpha_mask: bool = False,
        transmission: float = 0.0,
        ior: float = 1.5,
        emissive: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> Dict:
        pbr: Dict = {
            "baseColorFactor": list(color),
            "metallicFactor": metallic,
            "roughnessFactor": roughness,
        }
        if base_tex is not None:
            pbr["baseColorTexture"] = {"index": base_tex}
        if mr_tex is not None:
            pbr["metallicRoughnessTexture"] = {"index": mr_tex}
        m: Dict = {
            "name": name,
            "pbrMetallicRoughness": pbr,
            "doubleSided": True,
        }
        if norm_tex is not None:
            m["normalTexture"] = {"index": norm_tex, "scale": 1.0}
        if alpha_mask:
            m["alphaMode"] = "MASK"
            m["alphaCutoff"] = 0.5
        if any(c > 0.0 for c in emissive):
            m["emissiveFactor"] = list(emissive)
        if transmission > 0.0:
            m["extensions"] = {
                "KHR_materials_transmission": {"transmissionFactor": transmission},
                "KHR_materials_ior": {"ior": ior},
            }
        return m

    return [
        mat("terrazzo_courtyard_floor", (0.92, 0.90, 0.86, 1.0), 0.0, 0.28, base_tex=0, norm_tex=5, mr_tex=6),
        mat("gallery_limestone_pavement", (0.78, 0.75, 0.69, 1.0), 0.0, 0.42, base_tex=0, norm_tex=5, mr_tex=6),
        mat("travertine_nave_wall", (0.86, 0.78, 0.66, 1.0), 0.0, 0.68, base_tex=1, norm_tex=5, mr_tex=6),
        mat("renaissance_brick_wall", (0.74, 0.46, 0.36, 1.0), 0.0, 0.76, base_tex=1, norm_tex=5),
        mat("coffered_cedar_wood_roof", (0.48, 0.33, 0.22, 1.0), 0.0, 0.62, base_tex=1, norm_tex=5),
        mat("weathered_rust_ironwork", (0.44, 0.26, 0.19, 1.0), 0.35, 0.82, norm_tex=5),
        mat("carrara_marble_column", (0.95, 0.94, 0.92, 1.0), 0.0, 0.22, base_tex=2, norm_tex=5),
        mat("rosso_verona_pillar", (0.80, 0.48, 0.42, 1.0), 0.0, 0.26, base_tex=2, norm_tex=5),
        mat("carved_limestone_arch", (0.88, 0.84, 0.76, 1.0), 0.0, 0.48, base_tex=1, norm_tex=5),
        mat("alabaster_lion_statue", (0.94, 0.91, 0.85, 1.0), 0.0, 0.30, base_tex=2),
        mat("gilded_brass_capital", (0.94, 0.76, 0.34, 1.0), 0.95, 0.20),
        mat("brushed_bronze_metal", (0.72, 0.52, 0.32, 1.0), 0.92, 0.28),
        mat("forged_iron_chain", (0.32, 0.33, 0.35, 1.0), 0.88, 0.38),
        mat("burnished_gold_lantern", (0.98, 0.82, 0.38, 1.0), 0.96, 0.16, emissive=(0.35, 0.24, 0.08)),
        mat("crimson_velvet_banner", (0.68, 0.09, 0.14, 1.0), 0.0, 0.78, base_tex=3),
        mat("royal_blue_silk_curtain", (0.12, 0.24, 0.62, 1.0), 0.0, 0.72, base_tex=3),
        mat("emerald_damask_fabric", (0.11, 0.46, 0.28, 1.0), 0.0, 0.75, base_tex=3),
        mat("imperial_brocade_cloth", (0.78, 0.60, 0.18, 1.0), 0.0, 0.68, base_tex=3),
        mat("clerestory_stained_glass", (0.78, 0.88, 0.98, 0.35), 0.0, 0.06, transmission=0.88, ior=1.52, emissive=(0.12, 0.16, 0.22)),
        mat("fountain_crystal_glass", (0.72, 0.92, 0.95, 0.30), 0.0, 0.04, transmission=0.90, ior=1.333),
        mat("crimson_clearcoat_paint_body", (0.58, 0.06, 0.08, 1.0), 0.25, 0.10),
        mat("cobalt_enamel_paint_trim", (0.08, 0.18, 0.52, 1.0), 0.30, 0.12),
        mat("courtyard_ivy_leaf", (0.85, 0.95, 0.82, 1.0), 0.0, 0.54, base_tex=4, alpha_mask=True),
        mat("planter_laurel_leaves", (0.78, 0.92, 0.74, 1.0), 0.0, 0.58, base_tex=4, alpha_mask=True),
        mat("hanging_fern_flora", (0.82, 0.96, 0.78, 1.0), 0.0, 0.52, base_tex=4, alpha_mask=True),
    ]


def build_atrium_geometry() -> Dict[int, MeshBuilder]:
    builders: Dict[int, MeshBuilder] = {m: MeshBuilder() for m in range(25)}

    # 1. Outer Full Architectural Enclosure: X in [-1900, 1800], Y in [0, 1450], Z in [-1100, 1100]
    builders[0].add_grid_patch((-1900.0, 0.0, 380.0), (3700.0, 0.0, 0.0), (0.0, 0.0, -760.0), 84, 30, (37.0, 7.6))
    builders[1].add_grid_patch((-1900.0, 0.0, -380.0), (3700.0, 0.0, 0.0), (0.0, 0.0, -720.0), 70, 20, (37.0, 7.2))
    builders[1].add_grid_patch((-1900.0, 0.0, 1100.0), (3700.0, 0.0, 0.0), (0.0, 0.0, -720.0), 70, 20, (37.0, 7.2))

    for z_a, z_b in ((-1100.0, -360.0), (360.0, 1100.0)):
        builders[1].add_box(-1900.0, 635.0, z_a, 1800.0, 660.0, z_b, subdiv=(54, 1, 12))
    builders[1].add_box(-1900.0, 635.0, -360.0, -1480.0, 660.0, 360.0, subdiv=(8, 1, 12))
    builders[1].add_box(1440.0, 635.0, -360.0, 1800.0, 660.0, 360.0, subdiv=(8, 1, 12))

    builders[2].add_grid_patch((-1900.0, 0.0, -1100.0), (3700.0, 0.0, 0.0), (0.0, 660.0, 0.0), 72, 16, (37.0, 6.6))
    builders[2].add_grid_patch((1800.0, 0.0, 1100.0), (-3700.0, 0.0, 0.0), (0.0, 660.0, 0.0), 72, 16, (37.0, 6.6))
    builders[2].add_grid_patch((1800.0, 0.0, -1100.0), (0.0, 0.0, 2200.0), (0.0, 660.0, 0.0), 44, 16, (22.0, 6.6))
    builders[2].add_grid_patch((-1900.0, 0.0, 1100.0), (0.0, 0.0, -2200.0), (0.0, 660.0, 0.0), 44, 16, (22.0, 6.6))

    builders[3].add_grid_patch((-1900.0, 660.0, -1100.0), (3700.0, 0.0, 0.0), (0.0, 590.0, 0.0), 72, 16, (37.0, 5.9))
    builders[3].add_grid_patch((1800.0, 660.0, 1100.0), (-3700.0, 0.0, 0.0), (0.0, 590.0, 0.0), 72, 16, (37.0, 5.9))
    builders[3].add_grid_patch((1800.0, 660.0, -1100.0), (0.0, 0.0, 2200.0), (0.0, 590.0, 0.0), 44, 16, (22.0, 5.9))
    builders[3].add_grid_patch((-1900.0, 660.0, 1100.0), (0.0, 0.0, -2200.0), (0.0, 590.0, 0.0), 44, 16, (22.0, 5.9))

    # Covered perimeter gallery roofs framing the open-air central courtyard skylight (X in [-1440, 1440], Z in [-355, 355])
    for z_a, z_b in ((-1100.0, -355.0), (355.0, 1100.0)):
        builders[4].add_box(-1900.0, 1242.0, z_a, 1800.0, 1266.0, z_b, subdiv=(54, 1, 14))
    builders[4].add_box(-1900.0, 1242.0, -355.0, -1440.0, 1266.0, 355.0, subdiv=(8, 1, 12))
    builders[4].add_box(1440.0, 1242.0, -355.0, 1800.0, 1266.0, 355.0, subdiv=(8, 1, 12))

    # Coffered cedar beams beneath the North & South upper galleries
    for bx in range(-1600, 1601, 320):
        builders[4].add_box(bx - 16.0, 1205.0, -1095.0, bx + 16.0, 1242.0, -360.0, subdiv=(2, 2, 10))
        builders[4].add_box(bx - 16.0, 1205.0, 360.0, bx + 16.0, 1242.0, 1095.0, subdiv=(2, 2, 10))

    # Vertical stained-glass clerestory windows along the upper North & South gallery facades
    for bx in range(-1280, 1281, 320):
        builders[18].add_grid_patch((bx - 95.0, 780.0, -1092.0), (190.0, 0.0, 0.0), (0.0, 360.0, 0.0), 12, 14, (2.0, 3.0))
        builders[18].add_grid_patch((bx + 95.0, 780.0, 1092.0), (-190.0, 0.0, 0.0), (0.0, 360.0, 0.0), 12, 14, (2.0, 3.0))

    # Rooftop carved-limestone & Carrara-marble balustrade framing the open courtyard sky
    for side_z in (-355.0, 355.0):
        builders[8].add_box(-1456.0, 1322.0, side_z - 16.0, 1456.0, 1338.0, side_z + 16.0, subdiv=(40, 1, 2))
        for rx in range(-1400, 1401, 80):
            builders[6].add_lathe_column(float(rx), 1266.0, 1322.0, side_z, r_base=8.0, radial_segs=18, height_segs=8, flutes=6, flute_depth=0.03, entasis=0.22)
    for end_x in (-1440.0, 1440.0):
        builders[8].add_box(end_x - 16.0, 1322.0, -355.0, end_x + 16.0, 1338.0, 355.0, subdiv=(2, 1, 12))
        for rz in range(-312, 313, 48):
            builders[6].add_lathe_column(end_x, 1266.0, 1322.0, float(rz), r_base=8.0, radial_segs=18, height_segs=8, flutes=6, flute_depth=0.03, entasis=0.22)

    # 2. Two-Story Renaissance Colonnade Arcades along Z = -360 and Z = +360
    col_xs = [float(-1440 + i * 360) for i in range(9)]
    for side_z in (-360.0, 360.0):
        for cx in col_xs:
            builders[8].add_box(cx - 46.0, 0.0, side_z - 46.0, cx + 46.0, 52.0, side_z + 46.0, subdiv=(3, 2, 3))
            builders[6].add_lathe_column(cx, 52.0, 455.0, side_z, r_base=33.0, radial_segs=72, height_segs=32, flutes=20)
            builders[10].add_lathe_column(cx, 455.0, 505.0, side_z, r_base=40.0, radial_segs=36, height_segs=10, flutes=12, flute_depth=0.08, entasis=0.12)

            builders[8].add_box(cx - 36.0, 660.0, side_z - 36.0, cx + 36.0, 700.0, side_z + 36.0, subdiv=(2, 2, 2))
            builders[7].add_lathe_column(cx, 700.0, 1040.0, side_z, r_base=25.0, radial_segs=56, height_segs=26, flutes=16)
            builders[10].add_lathe_column(cx, 1040.0, 1082.0, side_z, r_base=31.0, radial_segs=32, height_segs=8, flutes=10, flute_depth=0.08)

        for b in range(8):
            x0, x1 = col_xs[b], col_xs[b + 1]
            builders[8].add_arcade_arch(x0, x1, y_spring=505.0, z_center=side_z, thickness_z=56.0, ring_height=48.0, arch_segs=42, depth_segs=8)
            builders[8].add_arcade_arch(x0, x1, y_spring=1082.0, z_center=side_z, thickness_z=44.0, ring_height=38.0, arch_segs=36, depth_segs=6)
            builders[5].add_box(x0 + 28.0, 500.0, side_z - 3.5, x1 - 28.0, 507.0, side_z + 3.5, subdiv=(6, 1, 1))

            builders[8].add_box(x0 + 36.0, 752.0, side_z - 18.0, x1 - 36.0, 768.0, side_z + 18.0, subdiv=(6, 1, 2))
            for k in range(5):
                bx = x0 + 68.0 + k * 56.0
                builders[6].add_lathe_column(bx, 660.0, 752.0, side_z, r_base=8.5, radial_segs=24, height_segs=10, flutes=8, flute_depth=0.03, entasis=0.22)

            banner_mat = 14 + ((b + (0 if side_z < 0 else 2)) % 4)
            z_banner = side_z + (34.0 if side_z < 0 else -34.0)
            bx_mid = 0.5 * (x0 + x1)
            builders[banner_mat].add_grid_patch(
                (bx_mid - 58.0, 530.0, z_banner),
                (116.0, 0.0, 0.0),
                (0.0, 225.0, 0.0),
                nu=22,
                nv=34,
                uv_scale=(1.0, 2.0),
                wave_amp=9.0 if side_z < 0 else -9.0,
                wave_freq=3.0,
            )

        builders[8].add_box(-1500.0, 610.0, side_z - 42.0, 1500.0, 642.0, side_z + 42.0, subdiv=(48, 2, 2))
        builders[8].add_box(-1500.0, 1215.0, side_z - 36.0, 1500.0, 1250.0, side_z + 36.0, subdiv=(48, 2, 2))

    # East Focal Apse Arcade
    for cz in (-240.0, -80.0, 80.0, 240.0):
        builders[6].add_lathe_column(1480.0, 0.0, 480.0, cz, r_base=30.0, radial_segs=52, height_segs=24, flutes=18)
        builders[10].add_lathe_column(1480.0, 480.0, 525.0, cz, r_base=36.0, radial_segs=28, height_segs=8, flutes=10)
    builders[18].add_grid_patch((1785.0, 760.0, 220.0), (0.0, 0.0, -440.0), (0.0, 420.0, 0.0), 24, 24, (3.0, 3.0))

    # 3. Central Courtyard Fountain, Clearcoat Urns, Lion Sculptures, Lantern Chandeliers & Foliage
    fx, fz = 260.0, 0.0
    builders[8].add_lathe_column(fx, 0.0, 68.0, fz, r_base=145.0, radial_segs=56, height_segs=10, flutes=8, flute_depth=0.02, entasis=-0.04)
    builders[19].add_grid_patch((fx - 115.0, 58.0, fz + 115.0), (230.0, 0.0, 0.0), (0.0, 0.0, -230.0), 24, 24, (3.0, 3.0), wave_amp=2.5, wave_freq=4.0)
    builders[6].add_lathe_column(fx, 50.0, 215.0, fz, r_base=28.0, radial_segs=44, height_segs=18, flutes=12, flute_depth=0.06, entasis=0.18)
    builders[11].add_uv_sphere(fx, 232.0, fz, 26.0, 26.0, 26.0, 32, 20)

    for ux, uz in ((-420.0, -210.0), (-420.0, 210.0), (760.0, -210.0), (760.0, 210.0)):
        builders[21].add_box(ux - 32.0, 0.0, uz - 32.0, ux + 32.0, 64.0, uz + 32.0, subdiv=(3, 3, 3))
        builders[20].add_lathe_column(ux, 64.0, 188.0, uz, r_base=28.0, radial_segs=48, height_segs=20, flutes=10, flute_depth=0.03, entasis=0.34)

    for sx, sz in ((1120.0, -190.0), (1120.0, 190.0)):
        builders[8].add_box(sx - 44.0, 0.0, sz - 34.0, sx + 44.0, 78.0, sz + 34.0, subdiv=(3, 2, 3))
        builders[9].add_uv_sphere(sx, 122.0, sz, 42.0, 48.0, 28.0, 40, 28)
        builders[9].add_uv_sphere(sx - 28.0, 162.0, sz, 24.0, 26.0, 22.0, 32, 20)

    # Open-courtyard wrought-iron & cedar pergola tie-beams supporting suspended chandeliers
    for lx, ly, lz in ((-650.0, 310.0, 0.0), (0.0, 490.0, 0.0), (650.0, 310.0, 0.0)):
        builders[5].add_box(lx - 10.0, 1224.0, -360.0, lx + 10.0, 1242.0, 360.0, subdiv=(2, 1, 14))
        builders[12].add_lathe_column(lx, ly + 35.0, 1226.0, lz, r_base=4.5, radial_segs=16, height_segs=18, flutes=4, flute_depth=0.12, entasis=0.0)
        builders[11].add_lathe_column(lx, ly, ly + 35.0, lz, r_base=48.0, radial_segs=40, height_segs=8, flutes=12, flute_depth=0.06, entasis=0.25)
        builders[13].add_uv_sphere(lx, ly - 10.0, lz, 18.0, 22.0, 18.0, 28, 18)

    for bx in (-900.0, -180.0, 540.0, 1260.0):
        for bz in (-285.0, 285.0):
            builders[11].add_lathe_column(bx, 0.0, 118.0, bz, r_base=20.0, radial_segs=32, height_segs=12, flutes=8, flute_depth=0.08, entasis=-0.18)

    rng = random.Random(1337)
    planter_spots = [
        (-720.0, -280.0), (-720.0, 280.0),
        (-60.0, -280.0), (-60.0, 280.0),
        (580.0, -280.0), (580.0, 280.0),
        (1220.0, -280.0), (1220.0, 280.0),
    ]
    for px, pz in planter_spots:
        builders[5].add_box(px - 36.0, 0.0, pz - 36.0, px + 36.0, 44.0, pz + 36.0, subdiv=(2, 2, 2))
        for c in range(42):
            ang = rng.uniform(0.0, 2.0 * math.pi)
            tilt = rng.uniform(0.32, 0.95)
            rad = rng.uniform(4.0, 26.0)
            ly = rng.uniform(40.0, 96.0)
            lx = px + rad * math.cos(ang)
            lz = pz + rad * math.sin(ang)
            w = rng.uniform(14.0, 22.0)
            l = rng.uniform(34.0, 56.0)
            au = (-w * math.sin(ang), 0.0, w * math.cos(ang))
            av = (l * tilt * math.cos(ang), l * rng.uniform(0.55, 0.95), l * tilt * math.sin(ang))
            mat_leaf = 22 + (c % 3)
            builders[mat_leaf].add_botanical_leaf((lx, ly, lz), au, av, 4, 6)

    for gx in range(-1260, 1261, 180):
        for gz in (-342.0, 342.0):
            inward = 1.0 if gz < 0.0 else -1.0
            for k in range(8):
                ox = gx + (k - 3.5) * 20.0
                splay = (k - 3.5) * 3.5
                oy = 626.0 + (k % 2) * 16.0
                builders[22 + (k % 3)].add_botanical_leaf(
                    (ox, oy, gz),
                    (18.0, 0.0, 0.0),
                    (splay, 36.0, 14.0 * inward),
                    4,
                    6,
                )
                oy_top = 1220.0 + (k % 2) * 18.0
                builders[22 + ((k + 1) % 3)].add_botanical_leaf(
                    (ox, oy_top, gz),
                    (18.0, 0.0, 0.0),
                    (splay, 38.0, 14.0 * inward),
                    4,
                    6,
                )

    return builders


def to_le_bytes(arr: array.array) -> bytes:
    if os.sys.byteorder != "little":
        cp = array.array(arr.typecode, arr)
        cp.byteswap()
        return cp.tobytes()
    return arr.tobytes()


def export_glb(output_path: str) -> Tuple[int, int, int]:
    textures = build_procedural_textures()
    materials = build_materials()
    builders = build_atrium_geometry()

    bin_chunks: List[bytes] = []
    buffer_views: List[Dict] = []
    accessors: List[Dict] = []
    images: List[Dict] = []
    gltf_textures: List[Dict] = []
    primitives: List[Dict] = []

    offset = 0

    def append_bytes(data: bytes, target: Optional[int] = None) -> int:
        nonlocal offset
        pad = (4 - (len(data) % 4)) % 4
        bv: Dict = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
        if target is not None:
            bv["target"] = target
        bv_idx = len(buffer_views)
        buffer_views.append(bv)
        bin_chunks.append(data + (b"\x00" * pad))
        offset += len(data) + pad
        return bv_idx

    for name, png_bytes in textures:
        bv_idx = append_bytes(png_bytes)
        img_idx = len(images)
        images.append({"name": name, "bufferView": bv_idx, "mimeType": "image/png"})
        gltf_textures.append({"sampler": 0, "source": img_idx})

    total_verts = 0
    total_tris = 0
    for mat_id in sorted(builders.keys()):
        b = builders[mat_id]
        if b.vcount == 0:
            continue
        vcount = b.vcount
        icount = len(b.idx)
        total_verts += vcount
        total_tris += icount // 3

        pos_bv = append_bytes(to_le_bytes(b.pos), target=34962)
        norm_bv = append_bytes(to_le_bytes(b.norm), target=34962)
        tan_bv = append_bytes(to_le_bytes(b.tan), target=34962)
        uv_bv = append_bytes(to_le_bytes(b.uv), target=34962)
        idx_bv = append_bytes(to_le_bytes(b.idx), target=34963)

        pos_acc = len(accessors)
        accessors.append({
            "bufferView": pos_bv,
            "componentType": 5126,
            "count": vcount,
            "type": "VEC3",
            "min": [float(v) for v in b.pmin],
            "max": [float(v) for v in b.pmax],
        })
        norm_acc = len(accessors)
        accessors.append({
            "bufferView": norm_bv,
            "componentType": 5126,
            "count": vcount,
            "type": "VEC3",
        })
        tan_acc = len(accessors)
        accessors.append({
            "bufferView": tan_bv,
            "componentType": 5126,
            "count": vcount,
            "type": "VEC4",
        })
        uv_acc = len(accessors)
        accessors.append({
            "bufferView": uv_bv,
            "componentType": 5126,
            "count": vcount,
            "type": "VEC2",
        })
        idx_acc = len(accessors)
        accessors.append({
            "bufferView": idx_bv,
            "componentType": 5125,
            "count": icount,
            "type": "SCALAR",
        })

        primitives.append({
            "attributes": {
                "POSITION": pos_acc,
                "NORMAL": norm_acc,
                "TANGENT": tan_acc,
                "TEXCOORD_0": uv_acc,
            },
            "indices": idx_acc,
            "material": mat_id,
            "mode": 4,
        })

    gltf_doc = {
        "asset": {
            "version": "2.0",
            "generator": "GPUBench First-Party Architectural Atrium Generator",
            "copyright": "Copyright (c) 2026 GPUBench Contributors. Dual-licensed under MIT and CC0-1.0.",
        },
        "extensionsUsed": [
            "KHR_materials_transmission",
            "KHR_materials_ior",
        ],
        "scene": 0,
        "scenes": [{"name": "RenaissanceColonnadeAtrium", "nodes": [0]}],
        "nodes": [{"name": "AtriumRoot", "mesh": 0}],
        "meshes": [{"name": "RenaissanceColonnadeAtriumMesh", "primitives": primitives}],
        "materials": materials,
        "textures": gltf_textures,
        "images": images,
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}],
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": offset}],
    }

    json_bytes = json.dumps(gltf_doc, separators=(",", ":")).encode("utf-8")
    json_pad = (4 - (len(json_bytes) % 4)) % 4
    json_chunk = json_bytes + (b" " * json_pad)
    bin_payload = b"".join(bin_chunks)
    assert len(bin_payload) == offset

    total_len = 12 + 8 + len(json_chunk) + 8 + len(bin_payload)
    header = struct.pack("<III", 0x46546C67, 2, total_len)
    chunk0 = struct.pack("<II", len(json_chunk), 0x4E4F534A) + json_chunk
    chunk1 = struct.pack("<II", len(bin_payload), 0x004E4942) + bin_payload

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(header)
        f.write(chunk0)
        f.write(chunk1)

    return total_verts, total_tris, total_len


if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_glb = os.path.join(repo_root, "assets", "models", "sponza.glb")
    verts, tris, nbytes = export_glb(out_glb)
    print(f"Generated {out_glb}: {verts:,} vertices, {tris:,} triangles, {nbytes / (1024 * 1024):.2f} MB")
