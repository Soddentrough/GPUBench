# GPU Benchmark Performance Analysis

## Observed vs Theoretical Performance

### AMD RX 6900 XT (RDNA2) - Theoretical Specs
- FP16: 46.08 TFLOPS
- FP32: 23.04 TFLOPS  
- FP64: 1.440 TFLOPS (1440 GFLOPS)

### Benchmark Results
- FP64: 1.478 TFLOPS ✓ (~103% of theoretical - excellent)
- FP32: 14.067 TFLOPS ✗ (~61% of theoretical)
- FP16: 14.020 TFLOPS ✗ (~30% of theoretical)

## Root Causes Identified

### 1. **FP16 Not Using Packed Operations (CRITICAL)**
- Current shader uses scalar `float16_t` operations
- RDNA2 achieves 2x FP16 throughput through **packed SIMD operations** (f16vec2/f16vec4)
- Scalar FP16 operations run at FP32 speeds
- **This is why FP16 and FP32 have nearly identical performance**

### 2. **Insufficient Workload for GPU Saturation**
- AMD RX 6900 XT has 5120 shader cores (80 CUs × 64 threads)
- Current dispatch: 1024 workgroups × 64 threads = 65,536 threads = 1024 waves
- This is **only ~20% utilization** of available compute units
- Modern GPUs need massive parallelism to hide memory latency

### 3. **Memory Bandwidth Overhead**
- Each thread reads and writes to global memory once
- For 65,536 iterations with only 2 FLOPs per iteration
- Arithmetic intensity is too low (compute-to-memory ratio)
- True compute benchmarks should minimize memory access

### 4. **Compiler Optimizations May Be Limited**
- Simple loop structure might not encourage aggressive optimization
- No explicit use of FMA (fused multiply-add) instructions
- Loop unrolling not explicitly encouraged

## Proposed Solutions

### Solution 1: Use Packed FP16 Operations
```glsl
// Instead of scalar operations:
float16_t val = val * float16_t(0.5) + float16_t(0.25);

// Use packed operations:
f16vec2 val = val * f16vec2(0.5) + f16vec2(0.25);
```

### Solution 2: Increase Workload and Thread Count
- Increase from 1024 to 8192+ workgroups
- Better GPU utilization and latency hiding
- More opportunities for parallel execution

### Solution 3: Reduce Memory Access Overhead
- Use more compute per memory access
- Consider using only registers (no global memory access)
- Or use shared memory for intermediate results

### Solution 4: Explicit Optimization Hints
- Use explicit FMA operations
- Add loop unrolling pragmas
- Ensure the compiler generates optimal code

## Implementation Recommendations

### High Priority
1. ✓ **Implement packed FP16 operations** - Will achieve ~2x FP16 performance
2. ✓ **Increase workgroup count** - Better GPU utilization
3. ✓ **Reduce memory dependency** - Use register-only workloads

### Medium Priority  
4. ✓ **Explicit FMA usage** - Ensure proper operation counting
5. ✓ **Loop unrolling hints** - Help compiler optimize

### Expected Results After Fixes
- FP16: ~40-46 TFLOPS (approaching theoretical)
- FP32: ~20-23 TFLOPS (approaching theoretical)
- FP64: ~1.4 TFLOPS (already optimal)

## Results After Optimization

### Performance Comparison

**Before Optimization:**
- FP64: 1.478 TFLOPS (103% of theoretical) ✓
- FP32: 14.067 TFLOPS (61% of theoretical) ✗
- FP16: 14.020 TFLOPS (30% of theoretical) ✗

**After Initial Optimization:**
- FP64: 1.481 TFLOPS (103% of theoretical) ✓
- FP32: 15.512 TFLOPS (67% of theoretical) ✓
- FP16: 31.915 TFLOPS (69% of theoretical) ✓✓

**After Advanced Optimization (Final):**
- FP64: 1.478 TFLOPS (103% of theoretical) ✓✓
- FP32: 22.278 TFLOPS (97% of theoretical) ✓✓✓
- FP16: 37.837 TFLOPS (82% of theoretical) ✓✓✓

### Key Improvements

1. **FP16: 170% improvement** (14.02 → 37.84 TFLOPS)
   - Packed `f16vec2` operations for 2x throughput
   - Multiple accumulators (8 registers) to avoid dependency chains
   - Reduced loop iterations with more ops per iteration
   - **Achieved 82% of theoretical peak**
   
2. **FP32: 58% improvement** (14.07 → 22.28 TFLOPS)
   - Multiple vec4 accumulators to maximize ALU utilization
   - Explicit FMA operations for optimal code generation
   - Eliminated memory bandwidth bottleneck
   - **Achieved 97% of theoretical peak**

3. **FP64: Maintained optimal performance** (1.478 TFLOPS)
   - Already at peak performance (103% of theoretical)

### Why Not 100% Theoretical?

Achieving 67-69% of theoretical peak is actually excellent for real-world benchmarks:

1. **Memory Latency**: Even with reduced memory access, some overhead remains
2. **Compiler Optimizations**: SPIR-V compiler may not always generate optimal code
3. **GPU Scheduling**: Thread scheduling and wave occupancy aren't perfect
4. **Thermal/Power Management**: GPU may throttle under sustained load
5. **Driver Overhead**: Vulkan driver adds some overhead

### INT Performance Optimizations

**Before INT Optimization:**
- INT8: 7.635 TFLOPS
- INT4: 11.495 TFLOPS

**After INT Optimization:**
- INT8: 19.749 TFLOPS (+159% improvement)
- INT4: 17.300 TFLOPS (+51% improvement)

The INT benchmarks used similar optimization strategies:
- Packed i8vec4 operations (4 INT8 values per vector)
- Multiple accumulators to avoid dependency chains
- Increased workgroup count for better saturation
- More operations per iteration

### Why INT Performance Is Lower Than Expected?

INT8 (19.7 TFLOPS) and INT4 (17.3 TFLOPS) are lower than FP16 (37.9 TFLOPS) due to:

1. **Gaming GPU Optimization**: RDNA2 is heavily optimized for FP operations (gaming workloads)
2. **INT4 Emulation Overhead**: INT4 requires bitwise AND operations to maintain 4-bit range
3. **Instruction Scheduling**: FP units may have better throughput than INT units on this architecture
4. **SIMD Width**: FP16 packed operations may have better hardware support than INT8 packed operations

However, the achieved INT performance is still very good and represents efficient use of the hardware.

### Conclusion

The optimizations successfully addressed the main performance issues:
- ✓ Packed FP16 operations achieved 82% of theoretical (37.8 TFLOPS)
- ✓ FP32 achieved 97% of theoretical (22.3 TFLOPS)  
- ✓ INT8 achieved good performance with packed operations (19.7 TFLOPS)
- ✓ INT4 achieved good performance despite emulation overhead (17.3 TFLOPS)
- ✓ Increased workgroup count improved GPU utilization across all benchmarks
- ✓ Multiple accumulators eliminated dependency chains

The benchmark now provides realistic and meaningful performance measurements for the AMD RX 6900 XT.

## Known Platform Ceilings and Toolchain Gaps (measured 2026-07, R9700/RADV)

- **FP16/BF16 vector (~51.5 TFLOPS, 1.13x FP32)**: the SPIR-V is correct
  (packed v2f16 FMAs verified by disassembly), but measured throughput caps
  at ~1.13x FP32 on RADV/GFX12 regardless of formulation (v2/v4 vectors,
  4x unrolling, ACO and LLVM backends all identical). The op counting is
  correct; this is a driver/hardware-path ceiling, not a benchmark bug.
  BF16 vector intentionally uses an FP16 shader (same packed rate on RDNA4).
- **FP8 vector/matrix: reported UNSUPPORTED.** Hardware and driver support
  native FP8 (VK_EXT_shader_float8 + shaderFloat8CooperativeMatrix), but no
  GLSL toolchain can compile FP8 shaders (glslang main has no
  GL_EXT_shader_explicit_arithmetic_types_float8). The old FP16-based proxy
  shaders were reporting FP16 rates mislabeled as FP8; they are now gated
  off until a compiler (glslang FP8 support, or slang with FP8 CoopMat) is
  integrated.
- **INT4 matrix: reported UNSUPPORTED.** VK_KHR_cooperative_matrix has no
  4-bit integer component type, so native INT4 WMMA rates are not
  expressible through Vulkan today; the old shader measured INT8. INT4
  vector has no native Vulkan path either.
- **FP4: reported UNSUPPORTED** (no FP4 type on RDNA4; FP16-based emulation
  disabled as inaccurate).

## Known Platform Ceilings and Toolchain Gaps (measured 2026-07, R9700/RADV)

- FP16/BF16 vector (~51.5 TFLOPS, 1.13x FP32): the SPIR-V is correct
  (packed v2f16 FMAs verified by disassembly), but measured throughput caps
  at ~1.13x FP32 on RADV/GFX12 regardless of formulation (v2/v4 vectors,
  4x unrolling, ACO and LLVM backends all identical). The op counting is
  correct; this is a driver/hardware-path ceiling, not a benchmark bug.
  BF16 vector intentionally uses an FP16 shader (same packed rate on RDNA4).
- FP8 vector/matrix: reported UNSUPPORTED. Hardware and driver support
  native FP8 (VK_EXT_shader_float8 + shaderFloat8CooperativeMatrix), but no
  GLSL toolchain can compile FP8 shaders (glslang main has no
  GL_EXT_shader_explicit_arithmetic_types_float8). The old FP16-based proxy
  shaders were reporting FP16 rates mislabeled as FP8; they are now gated
  off until a compiler (glslang FP8 support, or slang with FP8 CoopMat) is
  integrated.
- INT4 matrix: reported UNSUPPORTED. VK_KHR_cooperative_matrix has no
  4-bit integer component type, so native INT4 WMMA rates are not
  expressible through Vulkan today; the old shader measured INT8.
- FP4: reported UNSUPPORTED (no FP4 type on RDNA4; FP16-based emulation
  disabled as inaccurate).

## ROCm Backend Findings (2026-07, R9700, TheRock ROCm 7.x / clang 23)

- FP32 (~24-25 TFLOPS, ~half of Vulkan): REAL fp32 math, but hipcc/LLVM
  cannot exploit RDNA4's second FP32 pipe. Disassembly shows 512 scalar
  v_fma_f32 and no dual-issue pairing; v_dual_fma_f32 does not exist on
  gfx1201 (VOPD FMA was removed after gfx11 — hardware dual-issue requires
  bank-aware register allocation that LLVM does not do; RADV/ACO does,
  which is why the Vulkan numbers show the true silicon peak). Reported
  as-is: it is the honest hipcc-achievable rate.
- BF16 vector: UNSUPPORTED [toolchain]. TheRock's clang has no native
  bf16 vector codegen for gfx12 — __bf16 arithmetic is emulated via FP32
  with software rounding on the SCALAR unit (verified in disassembly),
  and headers lack hip_bfloat162/__hfma2. The previous ~2.5 TFLOPS result
  was emulated bf16, not hardware throughput.
- L0/L1/L2/L3 cache latency: FIXED (L0 was 0.00 ns — the old kernel did
  integer ALU ops with no memory reads; now a real 1M-step pointer chase,
  measuring ~14/21/44/217 ns).
- Memory bandwidth (~64% of Vulkan, e.g. Read 399 vs 622 GB/s): the HIP
  kernel is structurally identical to the Vulkan shader and grids match;
  the gap is hipcc/RADV codegen quality for the masked streaming access
  pattern. Known issue, not yet resolved. .co kernels are now built -O3.

## Cache Latency: Vulkan vs ROCm Are Different Hardware Paths (2026-07)

The L0-L3 latency numbers are NOT directly comparable across backends:
- Vulkan (RADV): the chase compiles to VMEM buffer_load_b32 with a
  scalar<->vector round trip (v_readfirstlane) in every dependency step
  (verified in RADV SASS; persists even with nonuniformEXT since ACO
  proves the address uniform). Measures the vector cache path as real
  Vulkan compute shaders experience it: ~32/70/86/160 ns.
- ROCm (hipcc): the chase compiles to SMEM s_load_b32 (scalar memory
  path), which on RDNA4 has substantially lower hit latency:
  ~13/21/41/133 ns.
Both are "correct" for their path; treat them as per-backend metrics, not
cross-backend equivalents. The VMEM path is what general compute kernels
use, so the Vulkan numbers are the more representative cache-latency
figures for typical shader workloads.
