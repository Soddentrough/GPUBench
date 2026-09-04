# Project Agent Guidelines & System Environment

## AMD GPU Tools & ROCm Paths
Do not search for SMI tool paths across runs; use these exact absolute paths:
- **`amd-smi`**:
  - `/home/naoki/.local/bin/amd-smi`
  - `/opt/rocm/core-10.0/bin/amd-smi`
- **`rocm-smi`**:
  - `/home/naoki/.local/bin/rocm-smi`
  - `/opt/rocm/core-10.0/bin/rocm-smi`

*Note*: Executing binaries located outside the repository workspace (such as in `/opt/rocm` or `~/.local/bin`) requires running with `BypassSandbox: true`.

## Hardware & Target GPU
- **Target GPU**: GPU 1 (`-d 1`). (GPU 0 is reserved for external workloads).
- **Architecture**: AMD Radeon AI PRO R9700 (GFX1201 / Vulkan 1.4 / SPIR-V 1.4).
- **CPU & RAM**: Threadripper 3750X with 64GB RAM. Limit parallel compilation jobs to `-j16`.
- **Operating System**: Fedora 44.

## System & Execution Rules
- No sudo commands.
- No Python virtual environments or containers (system-wide packages only).
- Do not ignore warnings or deprecation notices.
- Rigorous end-to-end verification and visual parity testing before concluding tasks.
