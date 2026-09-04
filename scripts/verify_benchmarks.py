#!/usr/bin/env python3
"""
GPUBench Automated Benchmark Verification & Regression Detection Suite

Validates benchmark results against:
1. Published theoretical hardware specs and baseline expected ranges.
2. Cross-backend parity (OpenCL, ROCm, Vulkan within ~±10%).
3. Logical invariants (e.g. Work Lists > Megakernel in ray tracing).
"""

import sys
import os
import json
import argparse
import subprocess
from typing import Dict, List, Any, Tuple

# Baseline expected performance ranges for known architectures
# Format: (min_acceptable, expected_nominal, max_acceptable) in native metric units
HARDWARE_BASELINES = {
    "gfx1201": {
        "description": "AMD Radeon AI PRO R9700 / GFX1201 (RDNA4 / Navi 48)",
        "baselines": {
            ("FP64", "TFLOPS"): (0.70, 0.83, 0.95),
            ("FP32", "TFLOPS"): (35.0, 42.0, 48.0),
            ("FP16 (Vector)", "TFLOPS"): (42.0, 52.0, 60.0),
            ("FP16 (Matrix)", "TFLOPS"): (180.0, 205.0, 240.0),
            ("BF16 (Vector)", "TFLOPS"): (42.0, 52.0, 60.0),
            ("BF16 (Matrix)", "TFLOPS"): (180.0, 205.0, 240.0),
            ("FP8 (Matrix)", "TFLOPS"): (320.0, 400.0, 480.0),
            ("INT8 (Vector)", "TOPS"): (28.0, 36.0, 45.0),
            ("INT8 (Matrix)", "TOPS"): (320.0, 380.0, 450.0),
            ("Memory Bandwidth (Read)", "GB/s"): (550.0, 640.0, 750.0),
            ("Memory Bandwidth (Write)", "GB/s"): (420.0, 550.0, 650.0),
            ("Memory Bandwidth (R/W)", "GB/s"): (400.0, 450.0, 580.0),
            ("L1 Cache", "GB/s"): (2000.0, 3500.0, 6000.0),
        }
    }
}

class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def identify_arch(device_str: str) -> str:
    dev = device_str.lower()
    if "gfx1201" in dev or "r9700" in dev or "9070" in dev:
        return "gfx1201"
    return "gfx1201"  # Default to target system GPU


def run_gpubench(args: argparse.Namespace) -> List[Dict[str, Any]]:
    import tempfile
    binary = args.binary or "./build/gpubench"
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        cmd = [
            binary,
            "-d", str(args.device),
            "-k", args.backends,
            "-b", args.benchmarks,
            "--output", "json",
            "--output-file", tmp_path,
        ]
        print(f"{Colors.CYAN}{Colors.BOLD}Running benchmark command:{Colors.RESET} {' '.join(cmd)}")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if result.returncode != 0:
            print(f"{Colors.RED}GPUBench failed with return code {result.returncode}:{Colors.RESET}")
            print(result.stdout)
            sys.exit(result.returncode)

        if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
            with open(tmp_path, "r") as f:
                return json.load(f)
        else:
            print(f"{Colors.RED}Output JSON file was empty or not created.{Colors.RESET}")
            print(result.stdout)
            sys.exit(1)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def canonicalize_bench_name(bench: str, subcat: str) -> str:
    b = bench.lower()
    if "fp64" in b:
        return "FP64"
    if "fp32" in b:
        return "FP32"
    if "fp16" in b:
        if "matrix" in b or "wmma" in b or "tensor" in b:
            return "FP16 (Matrix)"
        return "FP16 (Vector)"
    if "int8" in b:
        if "matrix" in b or "wmma" in b or "tensor" in b:
            return "INT8 (Matrix)"
        return "INT8 (Vector)"
    if "fp8" in b:
        if "matrix" in b or "wmma" in b:
            return "FP8 (Matrix)"
        return "FP8 (Vector)"
    if "bf16" in b:
        if "matrix" in b or "wmma" in b:
            return "BF16 (Matrix)"
        return "BF16 (Vector)"
    if "memory bandwidth" in b:
        # Avoid matching 'read' inside 'threads'!
        mode_str = "READ"
        if "write" in b:
            mode_str = "WRITE"
        elif "r/w" in b or "readwrite" in b:
            mode_str = "R/W"
        elif "read " in b or b.startswith("read") or "(read" in b:
            mode_str = "READ"

        for sz in ["1024", "256", "128"]:
            if sz in b:
                return f"Memory Bandwidth ({mode_str} {sz})"
        return f"Memory Bandwidth ({mode_str})"
    if "l1 cache" in b:
        return "L1 Cache"
    return bench


def evaluate_results(data: List[Dict[str, Any]]) -> bool:
    arch = "gfx1201"
    for item in data:
        dev_name = item.get("device", "")
        arch = identify_arch(dev_name)
        break

    arch_profile = HARDWARE_BASELINES.get(arch)
    arch_desc = arch_profile["description"] if arch_profile else f"Unknown ({arch})"
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN} GPUBench Verification & Regression Analysis{Colors.RESET}")
    print(f" Target Architecture Profile: {Colors.BOLD}{arch_desc}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")

    all_passed = True
    regressions: List[str] = []
    warnings: List[str] = []

    # 1. Check against Baseline Expected Ranges
    print(f"{Colors.BOLD}1. Hardware Baseline Expected Ranges:{Colors.RESET}")
    print(f"{'Benchmark':<32} {'Backend':<10} {'Result':<16} {'Expected Range':<20} {'Status'}")
    print("-" * 88)

    # Group by benchmark subcategory and backend
    grouped: Dict[str, Dict[str, float]] = {}
    metric_map: Dict[str, str] = {}
    rt_scheduling: Dict[str, float] = {}

    for item in data:
        if item.get("unsupported", False):
            continue
        bench = item.get("benchmark", "")
        subcat = item.get("subcategory", "")
        metric = item.get("metric", "")
        backend = item.get("backend", "")
        val = item.get("value", 0.0)

        name = canonicalize_bench_name(bench, subcat)
        if name not in grouped:
            grouped[name] = {}
            metric_map[name] = metric
        grouped[name][backend] = val

        if "RayScheduling" in bench or "RayScheduling" in name:
            rt_scheduling[bench] = val

        # Check baseline
        key = (name, metric)
        baseline_found = None
        if arch_profile:
            if key in arch_profile["baselines"]:
                baseline_found = arch_profile["baselines"][key]
            elif "Memory Bandwidth" in name:
                if "READ" in name and ("Memory Bandwidth (Read)", metric) in arch_profile["baselines"]:
                    baseline_found = arch_profile["baselines"][("Memory Bandwidth (Read)", metric)]
                elif "WRITE" in name and ("Memory Bandwidth (Write)", metric) in arch_profile["baselines"]:
                    baseline_found = arch_profile["baselines"][("Memory Bandwidth (Write)", metric)]
                elif "R/W" in name and ("Memory Bandwidth (R/W)", metric) in arch_profile["baselines"]:
                    baseline_found = arch_profile["baselines"][("Memory Bandwidth (R/W)", metric)]

        if baseline_found:
            min_val, nom_val, max_val = baseline_found
            res_str = f"{val:.2f} {metric}"
            exp_str = f"[{min_val:.2f} - {max_val:.2f}] (nom {nom_val:.2f})"
            if val < min_val:
                status = f"{Colors.RED}REGRESSION (Too Low){Colors.RESET}"
                all_passed = False
                regressions.append(f"{name} on {backend} is {val:.2f} {metric} (below minimum expected {min_val:.2f} {metric})")
            elif val > max_val * 1.3:
                status = f"{Colors.YELLOW}ANOMALY (Too High - Check Math){Colors.RESET}"
                warnings.append(f"{name} on {backend} is {val:.2f} {metric} (exceeds theoretical max {max_val:.2f} {metric})")
            else:
                status = f"{Colors.GREEN}PASS{Colors.RESET}"
            print(f"{name:<32} {backend:<10} {res_str:<16} {exp_str:<20} {status}")
        else:
            print(f"{name:<32} {backend:<10} {val:.2f} {metric:<13} {'[No baseline]':<20} {Colors.GREEN}RECORDED{Colors.RESET}")

    # 2. Check Cross-Backend Parity (within ±10-15%)
    print(f"\n{Colors.BOLD}2. Cross-Backend Parity Check (±10% Target, ±15% Hard Limit):{Colors.RESET}")
    print(f"{'Benchmark':<24} {'Backends Compared':<22} {'Relative Diff':<16} {'Status'}")
    print("-" * 80)

    for name, backend_vals in grouped.items():
        backends = list(backend_vals.keys())
        if len(backends) < 2:
            continue
        vals = [backend_vals[b] for b in backends]
        min_v = min(vals)
        max_v = max(vals)
        if max_v > 0:
            diff_pct = ((max_v - min_v) / max_v) * 100.0
        else:
            diff_pct = 0.0

        b_str = ", ".join([f"{b}: {backend_vals[b]:.2f}" for b in backends])
        diff_str = f"{diff_pct:.1f}%"

        if diff_pct <= 10.0:
            status = f"{Colors.GREEN}PASS (≤10%){Colors.RESET}"
        elif diff_pct <= 15.0:
            status = f"{Colors.YELLOW}WARN (10-15% variance){Colors.RESET}"
            warnings.append(f"{name} has {diff_pct:.1f}% cross-backend variance ({b_str})")
        else:
            status = f"{Colors.RED}FAIL (>15% Parity Violation){Colors.RESET}"
            all_passed = False
            regressions.append(f"{name} parity violation: {diff_pct:.1f}% variance between backends ({b_str})")

        print(f"{name:<24} {', '.join(backends):<22} {diff_str:<16} {status}")

    # 3. Check Logical Invariants (e.g. Work Lists > Megakernel)
    if rt_scheduling:
        print(f"\n{Colors.BOLD}3. Logical Invariant Checks (Ray Tracing Scheduling):{Colors.RESET}")
        print(f"{'Workload Scenario':<32} {'Megakernel':<15} {'Work Lists':<15} {'Speedup':<12} {'Status'}")
        print("-" * 80)

        scenarios = [
            ("Primary Ray Tracing", "Primary Ray Tracing - Traditional Megakernel", "Primary Ray Tracing - Work Lists (Material Sorting)"),
            ("Material Shading", "Material Shading - Traditional Megakernel", "Material Shading - Work Lists (Material Sorting)"),
            ("Incoherent Ray Tracing", "Incoherent Ray Tracing - Traditional Megakernel", "Incoherent Ray Tracing - Work Lists (Directional Binning)"),
            ("Path Tracing", "Path Tracing - Traditional Megakernel", "Path Tracing - Work Lists (Active Ray Compaction)"),
        ]

        for sc_name, mega_key, wl_key in scenarios:
            mega_val = None
            wl_val = None
            for k, v in rt_scheduling.items():
                if mega_key in k:
                    mega_val = v
                if wl_key in k:
                    wl_val = v

            if mega_val is not None and wl_val is not None:
                speedup = (wl_val / mega_val) if mega_val > 0 else 0.0
                sp_str = f"{speedup:.2f}x"
                if speedup >= 1.10:
                    st = f"{Colors.GREEN}PASS (Faster){Colors.RESET}"
                elif speedup >= 1.0:
                    st = f"{Colors.YELLOW}WARN (Marginal){Colors.RESET}"
                    warnings.append(f"Work Lists was only {speedup:.2f}x of Megakernel for {sc_name}")
                else:
                    st = f"{Colors.RED}FAIL (Slower than Megakernel!){Colors.RESET}"
                    all_passed = False
                    regressions.append(f"Work Lists is SLOWER than Megakernel ({speedup:.2f}x) for {sc_name}")
                print(f"{sc_name:<32} {mega_val:.1f} {'':<8} {wl_val:.1f} {'':<8} {sp_str:<12} {st}")
    # 4. Check Unsupported Benchmark Diagnostic Explanations
    unsupported_items = [item for item in data if item.get("unsupported", False)]
    if unsupported_items:
        print(f"\n{Colors.BOLD}4. Unsupported Diagnostic Reason Verification:{Colors.RESET}")
        print(f"{'Benchmark':<32} {'Backend':<10} {'Diagnostic Reason':<55} {'Status'}")
        print("-" * 105)

        valid_markers = ["extension ", "bit not set", "No support for ", "lacks ", "not defined", "missing"]

        for item in unsupported_items:
            bench = item.get("benchmark", "")
            backend = item.get("backend", "")
            reason = item.get("unsupported_reason", "").strip()

            if not reason:
                status = f"{Colors.RED}FAIL (Missing Reason){Colors.RESET}"
                all_passed = False
                regressions.append(f"{bench} on {backend} marked UNSUPPORTED without diagnostic reason")
            elif not any(marker.lower() in reason.lower() for marker in valid_markers):
                status = f"{Colors.YELLOW}WARN (Non-standard explanation){Colors.RESET}"
                warnings.append(f"{bench} on {backend} reason lacks technical specificity: '{reason}'")
            else:
                status = f"{Colors.GREEN}PASS{Colors.RESET}"

            display_reason = (reason[:52] + "...") if len(reason) > 55 else reason
            print(f"{bench:<32} {backend:<10} {display_reason:<55} {status}")

    # Final Summary
    print(f"\n{Colors.BOLD}{'='*80}{Colors.RESET}")
    if all_passed:
        print(f"{Colors.GREEN}{Colors.BOLD}✔ ALL VERIFICATION CHECKS PASSED: Hardware baselines, cross-backend parity, invariants, and unsupported diagnostics satisfied.{Colors.RESET}")
        if warnings:
            print(f"\n{Colors.YELLOW}Warnings flagged for review:{Colors.RESET}")
            for w in warnings:
                print(f" - {w}")
    else:
        print(f"{Colors.RED}{Colors.BOLD}✘ REGRESSIONS DETECTED:{Colors.RESET}")
        for r in regressions:
            print(f" - {Colors.RED}{r}{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*80}{Colors.RESET}\n")

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="GPUBench Verification & Regression Detection Suite")
    parser.add_argument("-d", "--device", type=int, default=1, help="Target GPU device index (default: 1)")
    parser.add_argument("-k", "--backends", type=str, default="opencl,rocm,vulkan", help="Backends to benchmark")
    parser.add_argument("-b", "--benchmarks", type=str, default="fp64,fp32,fp16,int8,ray_scheduling", help="Benchmarks to run")
    parser.add_argument("--binary", type=str, default="./build/gpubench", help="Path to gpubench binary")
    parser.add_argument("--input", type=str, default=None, help="Evaluate pre-existing JSON result file instead of running")

    args = parser.parse_args()

    if args.input:
        with open(args.input, "r") as f:
            data = json.load(f)
    else:
        data = run_gpubench(args)

    passed = evaluate_results(data)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
