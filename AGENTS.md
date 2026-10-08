# Project Agent Guidelines & System Environment

## AMD GPU Tools & ROCm Paths
Do not search for SMI tool paths across runs; use these exact absolute paths:
- **`amd-smi`**:
  - `~/.local/bin/amd-smi`
  - `/opt/rocm/core-10.1/bin/amd-smi`
  - `/opt/rocm/core-10.0/bin/amd-smi`
- **`rocm-smi`**:
  - `~/.local/bin/rocm-smi`
  - `/opt/rocm/core-10.1/bin/rocm-smi`
  - `/opt/rocm/core-10.0/bin/rocm-smi`

*Note*: Executing binaries located outside the repository workspace (such as in `/opt/rocm` or `~/.local/bin`) requires running with `BypassSandbox: true`.

## Hardware & Target GPU Discovery
- **Target GPU**: Dynamic discovery. Always check available devices with `./build/gpubench --list-devices` or ROCm tools before executing.
  - On this single-GPU system, use **`-d 0`** (or omit `-d`, which defaults to 0). Never attempt `-d 1` on single-GPU nodes.
  - On multi-GPU systems, target `-d 1` only if device index 1 is actively enumerated.
- **Current System Hardware**:
  - **CPU**: AMD Ryzen AI MAX+ 395 (16 Cores / 32 Threads).
  - **GPU**: AMD Radeon 8060S Graphics (RADV STRIX_HALO / RDNA 3.5 / Vulkan 1.4 / SPIR-V 1.4). Exactly 1 physical GPU device (Device 0).
  - **RAM**: 128 GB Unified LPDDR5X.
  - **Operating System**: Fedora 44.
- **Build Parallelism**: Use dynamic core formula: `cmake --build . --parallel $(( (n = $(nproc) - 4) > 8 ? n : 8 ))`.

## System & Execution Rules
- No sudo commands.
- No Python virtual environments or containers (system-wide packages only).
- Do not ignore warnings or deprecation notices.
- Rigorous end-to-end verification and visual parity testing before concluding tasks.
