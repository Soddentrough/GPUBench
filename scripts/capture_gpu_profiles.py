#!/usr/bin/env python3
import os
import sys
import glob
import json
import shutil
import subprocess
import time
import re

AMDSMI_BIN = "/opt/rocm/core-10.0/bin/amd-smi"
RGA_BIN = "/opt/RadeonDeveloperToolSuite-2026-05-28-1806/rga"
GPUBENCH_BIN = "./build/gpubench"
PROFILES_DIR = "profiles"

TESTS = [
    # Indoor Atrium
    {"scene": "indoor", "config": 0, "name": "Material Shading", "paradigm": "Traditional Megakernel", "slug": "indoor_material_traditional", "shader": "rt_scheduling_traditional.comp"},
    {"scene": "indoor", "config": 2, "name": "Material Shading", "paradigm": "Work Lists (Material Sorting)", "slug": "indoor_material_worklist", "shader": "rt_scheduling_worklist_material.comp"},
    {"scene": "indoor", "config": 4, "name": "Path Tracing (4 Bounces)", "paradigm": "Traditional Megakernel", "slug": "indoor_pathtracing_traditional", "shader": "rt_scheduling_traditional.comp"},
    {"scene": "indoor", "config": 6, "name": "Path Tracing (4 Bounces)", "paradigm": "Work Lists (Active Ray Compaction)", "slug": "indoor_pathtracing_worklist", "shader": "rt_scheduling_worklist_bounce.comp"},
    {"scene": "indoor", "config": 8, "name": "Incoherent Secondary Rays", "paradigm": "Traditional Megakernel", "slug": "indoor_incoherent_traditional", "shader": "rt_scheduling_traditional.comp"},
    {"scene": "indoor", "config": 10, "name": "Incoherent Secondary Rays", "paradigm": "Work Lists (Directional Binning)", "slug": "indoor_incoherent_worklist", "shader": "rt_scheduling_worklist_classify.comp"},
    {"scene": "indoor", "config": 12, "name": "Primary Ray Tracing", "paradigm": "Traditional Megakernel", "slug": "indoor_primary_traditional", "shader": "rt_scheduling_traditional.comp"},
    {"scene": "indoor", "config": 14, "name": "Primary Ray Tracing", "paradigm": "Work Lists (Decoupled Micro-Kernels)", "slug": "indoor_primary_worklist", "shader": "rt_scheduling_worklist_material.comp"},

    # Outdoor Landscape
    {"scene": "outdoor", "config": 0, "name": "Material Shading", "paradigm": "Traditional Megakernel", "slug": "outdoor_material_traditional", "shader": "rt_scheduling_traditional.comp"},
    {"scene": "outdoor", "config": 2, "name": "Material Shading", "paradigm": "Work Lists (Material Sorting)", "slug": "outdoor_material_worklist", "shader": "rt_scheduling_worklist_material.comp"},
    {"scene": "outdoor", "config": 4, "name": "Path Tracing (4 Bounces)", "paradigm": "Traditional Megakernel", "slug": "outdoor_pathtracing_traditional", "shader": "rt_scheduling_traditional.comp"},
    {"scene": "outdoor", "config": 6, "name": "Path Tracing (4 Bounces)", "paradigm": "Work Lists (Active Ray Compaction)", "slug": "outdoor_pathtracing_worklist", "shader": "rt_scheduling_worklist_bounce.comp"},
    {"scene": "outdoor", "config": 8, "name": "Incoherent Secondary Rays", "paradigm": "Traditional Megakernel", "slug": "outdoor_incoherent_traditional", "shader": "rt_scheduling_traditional.comp"},
    {"scene": "outdoor", "config": 10, "name": "Incoherent Secondary Rays", "paradigm": "Work Lists (Directional Binning)", "slug": "outdoor_incoherent_worklist", "shader": "rt_scheduling_worklist_classify.comp"},
    {"scene": "outdoor", "config": 12, "name": "Primary Ray Tracing", "paradigm": "Traditional Megakernel", "slug": "outdoor_primary_traditional", "shader": "rt_scheduling_traditional.comp"},
    {"scene": "outdoor", "config": 14, "name": "Primary Ray Tracing", "paradigm": "Work Lists (Decoupled Micro-Kernels)", "slug": "outdoor_primary_worklist", "shader": "rt_scheduling_worklist_material.comp"},
]

def query_amd_smi():
    try:
        res = subprocess.run([AMDSMI_BIN, "metric", "-g", "1", "--json"], capture_output=True, text=True, check=True)
        data = json.loads(res.stdout)
        dev = {}
        if isinstance(data, dict) and "gpu_data" in data and len(data["gpu_data"]) > 0:
            dev = data["gpu_data"][0]
        elif isinstance(data, list) and len(data) > 0:
            dev = data[0]

        usage = dev.get("usage", {})
        power = dev.get("power", {})
        temp = dev.get("temperature", {})
        mem = dev.get("mem_usage", {})
        clock = dev.get("clock", {})

        gfx_clk = clock.get("gfx_0", {}).get("clk", {})
        gfx_mhz = gfx_clk.get("value") if isinstance(gfx_clk, dict) else None
        mem_clk = clock.get("mem_0", {}).get("clk", {})
        mem_mhz = mem_clk.get("value") if isinstance(mem_clk, dict) else None

        pwr_val = power.get("socket_power", {}).get("value") if isinstance(power.get("socket_power"), dict) else None
        if pwr_val is None:
            pwr_val = power.get("gfx_power", {}).get("value") if isinstance(power.get("gfx_power"), dict) else None

        edge_temp = temp.get("edge", {}).get("value") if isinstance(temp.get("edge"), dict) else None
        hotspot_temp = temp.get("hotspot", {}).get("value") if isinstance(temp.get("hotspot"), dict) else None
        used_vram = mem.get("used_vram", {}).get("value") if isinstance(mem.get("used_vram"), dict) else None
        gfx_util = usage.get("gfx_activity", {}).get("value") if isinstance(usage.get("gfx_activity"), dict) else None

        return {
            "gfx_clock_mhz": gfx_mhz,
            "mem_clock_mhz": mem_mhz,
            "power_watts": pwr_val,
            "edge_temp_c": edge_temp,
            "hotspot_temp_c": hotspot_temp,
            "used_vram_mb": used_vram,
            "gfx_util_pct": gfx_util,
        }
    except Exception as e:
        return {"error": str(e)}

def parse_gpubench_output(stdout_text, stderr_text):
    combined = stdout_text + "\n" + stderr_text
    throughput = None
    metric = None
    fps = None
    for raw_line in combined.splitlines():
        line = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', raw_line)
        if "Vulkan |" in line:
            parts = line.split("Vulkan |")
            if len(parts) > 1:
                val_str = parts[1].strip()
                m = re.search(r"([\d,.]+)\s+([A-Za-z/]+)(?:\s+\(([\d.]+)\s+FPS\))?", val_str)
                if m:
                    throughput = float(m.group(1).replace(",", ""))
                    metric = m.group(2)
                    if m.group(3):
                        fps = float(m.group(3))
                    elif metric == "MRays/s":
                        fps = (throughput * 1e6) / (1920.0 * 1080.0)
    return throughput, metric, fps

def run_test(test_info):
    os.makedirs(PROFILES_DIR, exist_ok=True)
    slug = test_info["slug"]
    rgp_dest = os.path.join(PROFILES_DIR, f"{slug}.rgp")

    # Clean previous /tmp/gpubench_*.rgp
    for f in glob.glob("/tmp/gpubench_*.rgp"):
        try:
            os.remove(f)
        except OSError:
            pass

    smi_before = query_amd_smi()

    env = os.environ.copy()
    env["MESA_VK_TRACE"] = "rgp"
    env["MESA_VK_TRACE_PER_SUBMIT"] = "1"

    cmd = [
        GPUBENCH_BIN,
        "-d", "1",
        "-b", "RayScheduling",
        "--scene", test_info["scene"],
        "-c", str(test_info["config"]),
        "-r", "1080p",
        "--profile-snapshot"
    ]

    t0 = time.time()
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    t1 = time.time()

    smi_after = query_amd_smi()

    # Find the largest .rgp file in /tmp/ (corresponding to the timed test dispatch)
    rgp_files = glob.glob("/tmp/gpubench_*.rgp")
    rgp_size = 0
    if rgp_files:
        rgp_files.sort(key=os.path.getsize, reverse=True)
        largest = rgp_files[0]
        rgp_size = os.path.getsize(largest)
        shutil.copyfile(largest, rgp_dest)

    # Clean up /tmp/*.rgp
    for f in rgp_files:
        try:
            os.remove(f)
        except OSError:
            pass

    throughput, metric, fps = parse_gpubench_output(res.stdout, res.stderr)

    return {
        "scene": test_info["scene"],
        "config": test_info["config"],
        "name": test_info["name"],
        "paradigm": test_info["paradigm"],
        "slug": slug,
        "rgp_path": rgp_dest,
        "rgp_size_bytes": rgp_size,
        "profiled_throughput": throughput,
        "metric": metric,
        "profiled_fps": round(fps, 1) if fps is not None else None,
        "elapsed_sec": round(t1 - t0, 3),
        "telemetry_pre": smi_before,
        "telemetry_post": smi_after,
        "shader": test_info["shader"]
    }

def analyze_shaders_rga():
    shaders = [
        "rt_scheduling_traditional.comp",
        "rt_scheduling_worklist_classify.comp",
        "rt_scheduling_worklist_material.comp",
        "rt_scheduling_worklist_bounce.comp"
    ]
    rga_results = {}
    for s in shaders:
        spath = os.path.join("shaders", s)
        base = os.path.splitext(s)[0]
        out_analysis_prefix = f"/tmp/rga_analysis_{base}"
        out_isa_prefix = f"/tmp/rga_isa_{base}"

        cmd = [
            RGA_BIN,
            "-s", "vk-spv-offline",
            "-c", "gfx1201",
            "--comp", spath,
            "--analysis", out_analysis_prefix + ".csv",
            "--isa", out_isa_prefix + ".txt"
        ]
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            gen_csv = f"/tmp/gfx1201_rga_analysis_{base}_comp.csv"
            gen_isa = f"/tmp/gfx1201_rga_isa_{base}_comp.txt"

            entry = {}
            if os.path.exists(gen_csv):
                with open(gen_csv, "r") as f:
                    lines = [l.strip() for l in f if l.strip()]
                if len(lines) >= 2:
                    header = [h.strip() for h in lines[0].split(",")]
                    vals = [v.strip() for v in lines[1].split(",")]
                    entry = dict(zip(header, vals))
                os.remove(gen_csv)

            if os.path.exists(gen_isa):
                valu = 0
                salu = 0
                vmem = 0
                smem = 0
                ds = 0
                branches = 0
                waits = 0
                total_instructions = 0

                with open(gen_isa, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("_") or line.startswith("//"):
                            continue
                        op = line.split()[0] if line.split() else ""
                        if op.startswith("v_"):
                            valu += 1
                            total_instructions += 1
                        elif op.startswith("s_cbranch") or op.startswith("s_branch"):
                            branches += 1
                            total_instructions += 1
                        elif op.startswith("s_wait") or op.startswith("s_delay"):
                            waits += 1
                            total_instructions += 1
                        elif op.startswith("s_"):
                            smem_ops = ["s_load", "s_buffer_load", "s_store"]
                            if any(op.startswith(so) for so in smem_ops):
                                smem += 1
                            else:
                                salu += 1
                            total_instructions += 1
                        elif op.startswith("global_") or op.startswith("buffer_") or op.startswith("flat_") or op.startswith("image_"):
                            vmem += 1
                            total_instructions += 1
                        elif op.startswith("ds_"):
                            ds += 1
                            total_instructions += 1

                entry["instruction_breakdown"] = {
                    "total": total_instructions,
                    "valu": valu,
                    "salu": salu,
                    "vmem": vmem,
                    "smem": smem,
                    "lds_ds": ds,
                    "branches": branches,
                    "waits_stalls": waits
                }
                os.remove(gen_isa)

            rga_results[s] = entry
        except Exception as e:
            rga_results[s] = {"error": str(e)}

    return rga_results

def main():
    print("=" * 80)
    print("  GPUBench Dual-Scenario GPU Profiling Capture & Architecture Analysis")
    print("  Hardware: AMD Radeon AI PRO R9700 (GFX1201, GPU 1)")
    print("=" * 80)

    results = []
    for idx, test_info in enumerate(TESTS):
        print(f"[{idx+1:02d}/{len(TESTS)}] Capturing snapshot: {test_info['scene'].upper():<7} | {test_info['name']} ({test_info['paradigm']})...", flush=True)
        res = run_test(test_info)
        results.append(res)
        tp_str = f"{res['profiled_throughput']:,.2f} {res['metric']}" if res['profiled_throughput'] else "N/A"
        fps_str = f"({res['profiled_fps']:.1f} FPS)" if res['profiled_fps'] else ""
        print(f"       -> Profile Throughput: {tp_str:<22} {fps_str:<12} | RGP Trace: {res['rgp_path']} ({res['rgp_size_bytes']:,} bytes)")

    print("\nRunning Radeon GPU Analyzer (RGA) ISA disassembly & occupancy evaluation...")
    shader_stats = analyze_shaders_rga()

    out_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu": "AMD Radeon AI PRO R9700 (GFX1201, ID 1)",
        "resolution": "1920x1080 (1080p)",
        "snapshots": results,
        "shader_architecture": shader_stats
    }

    out_file = os.path.join(PROFILES_DIR, "gpu_profiling_snapshots.json")
    with open(out_file, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nSaved consolidated profiling data to: {out_file}")

if __name__ == "__main__":
    main()
