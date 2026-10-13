# Project Agent Guidelines & System Environment

## AMD GPU Tools & ROCm Paths
Do not search for SMI tool paths across runs; use these exact absolute paths:
- **`amd-smi`**:
  - `/opt/rocm/current/bin/amd-smi`
  - `~/.local/bin/amd-smi`
  - `/opt/rocm/core-10.1/bin/amd-smi`
  - `/opt/rocm/core-10.0/bin/amd-smi`
- **`rocm-smi`**:
  - `/opt/rocm/current/bin/rocm-smi`
  - `~/.local/bin/rocm-smi`
  - `/opt/rocm/core-10.1/bin/rocm-smi`
  - `/opt/rocm/core-10.0/bin/rocm-smi`

*Note*: Executing binaries located outside the repository workspace (such as in `/opt/rocm` or `~/.local/bin`) requires running with `BypassSandbox: true`.

## Hardware & Target GPU Discovery
- **Target GPU**: Dynamic discovery. Always check available devices with `./build/gpubench --list-devices` or `/opt/rocm/core-10.0/bin/amd-smi list` before executing.
  - On single-GPU systems, use **`-d 0`** (or omit `-d`, which defaults to 0). Never attempt `-d 1` on single-GPU nodes.
  - On multi-GPU systems, target `-d 0` or `-d 1` as enumerated by `--list-devices`.
- **Supported Workstation Profiles**:
  - **Dual RDNA 4 Workstation**: AMD Ryzen Threadripper 3970X (32C/64T), 64 GB RAM, 2x AMD Radeon AI PRO R9700 (32 GB GDDR6 each, `gfx1201` / RDNA 4 / Vulkan 1.4 / ROCm 10).
  - **Strix Halo APU Workstation**: AMD Ryzen AI MAX+ 395 (16C/32T), 128 GB Unified LPDDR5X, AMD Radeon 8060S Graphics (`gfx1151` / RDNA 3.5 / Vulkan 1.4).
- **Build Parallelism**: Use dynamic core formula: `cmake --build . --parallel $(( (n = $(nproc) - 4) > 8 ? n : 8 ))` (cap build jobs at 16–32 on high-core-count systems to avoid exhausting RAM).

## System & Execution Rules
- No sudo commands.
- No Python virtual environments or containers (system-wide packages only).
- Do not ignore warnings or deprecation notices.
- Rigorous end-to-end verification and visual parity testing before concluding tasks.
