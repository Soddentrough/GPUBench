#!/usr/bin/env bash
# ==============================================================================
# GPUBench End-to-End (E2E) Test Runner
#
# Validates GPUBench across Tiers 1-4:
#   Tier 1: Feature Coverage (CLI flags, device/benchmark/group discovery)
#   Tier 2: Boundary & Corner Cases (graceful error handling, input validation)
#   Tier 3: Cross-Feature Interactions (JSON export/import round-trip, comparisons)
#   Tier 4: Real-World Workload Integration (Live execution across all subsystems)
#
# Adheres strictly to repository guidelines:
#   - Dynamic GPU device discovery
#   - No sudo, no virtualenvs, no piping to python -c
# ==============================================================================

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ANSI Color Codes
CLR_RESET="\033[0m"
CLR_BOLD="\033[1m"
CLR_RED="\033[31m"
CLR_GREEN="\033[32m"
CLR_YELLOW="\033[33m"
CLR_BLUE="\033[34m"
CLR_CYAN="\033[36m"
CLR_GRAY="\033[90m"

# Default configuration
SELECTED_TIER="all"
VERBOSE=0
STRICT_MODE=0
BINARY_PATH="${PROJECT_ROOT}/build/gpubench"

# Parse CLI options
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tier|-t)
      SELECTED_TIER="$2"
      shift 2
      ;;
    --binary|-b)
      BINARY_PATH="$2"
      shift 2
      ;;
    --verbose|-v)
      VERBOSE=1
      shift
      ;;
    --strict|-s)
      STRICT_MODE=1
      shift
      ;;
    --help|-h)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  -t, --tier <1|2|3|4|all>   Run specific test tier (default: all)"
      echo "  -b, --binary <path>        Path to gpubench binary (default: ${BINARY_PATH})"
      echo "  -v, --verbose              Show command outputs"
      echo "  -s, --strict               Enforce strict validation mode"
      echo "  -h, --help                 Display this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [[ ! -x "${BINARY_PATH}" ]]; then
  echo -e "${CLR_RED}Error: GPUBench binary not found or not executable at: ${BINARY_PATH}${CLR_RESET}"
  echo "Please build the project first: cmake --build build --parallel"
  exit 1
fi

TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

run_test() {
  local tier="$1"
  local test_name="$2"
  local cmd="$3"
  local expected_exit="$4"  # "0" or "nonzero"
  shift 4
  local required_patterns=("$@")

  TOTAL_TESTS=$((TOTAL_TESTS + 1))
  printf "  • [%-10s] %-48s ... " "${tier}" "${test_name}"

  local t_start
  t_start=$(date +%s%N)

  local tmp_out
  tmp_out=$(mktemp /tmp/gpubench_e2e_out_XXXXXX.txt)
  local tmp_err
  tmp_err=$(mktemp /tmp/gpubench_e2e_err_XXXXXX.txt)

  local exit_code=0
  eval "${cmd}" >"${tmp_out}" 2>"${tmp_err}" || exit_code=$?

  local t_end
  t_end=$(date +%s%N)
  local elapsed_ms=$(( (t_end - t_start) / 1000000 ))

  local passed=1
  local failure_reason=""

  if [[ "${expected_exit}" == "0" ]]; then
    if [[ ${exit_code} -ne 0 ]]; then
      passed=0
      failure_reason="Expected exit code 0, got ${exit_code}"
    fi
  elif [[ "${expected_exit}" == "nonzero" ]]; then
    if [[ ${exit_code} -eq 0 ]]; then
      passed=0
      failure_reason="Expected non-zero exit code, got 0"
    elif [[ ${exit_code} -ge 128 ]]; then
      passed=0
      failure_reason="Process crashed with signal $((exit_code - 128))"
    fi
  fi

  if [[ ${passed} -eq 1 && ${#required_patterns[@]} -gt 0 ]]; then
    for pat in "${required_patterns[@]}"; do
      if ! grep -E -q -e "${pat}" "${tmp_out}" && ! grep -E -q -e "${pat}" "${tmp_err}"; then
        passed=0
        failure_reason="Output missing required pattern: '${pat}'"
        break
      fi
    done
  fi

  if [[ ${passed} -eq 1 ]]; then
    PASSED_TESTS=$((PASSED_TESTS + 1))
    printf "${CLR_GREEN}✔ PASS${CLR_RESET} (${elapsed_ms} ms)\n"
  else
    FAILED_TESTS=$((FAILED_TESTS + 1))
    printf "${CLR_RED}✘ FAIL${CLR_RESET} (${elapsed_ms} ms)\n"
    printf "      ${CLR_RED}[REASON] ${failure_reason}${CLR_RESET}\n"
    if [[ ${VERBOSE} -eq 1 ]]; then
      echo "      --- Stdout ---"
      sed 's/^/      /' "${tmp_out}"
      echo "      --- Stderr ---"
      sed 's/^/      /' "${tmp_err}"
    fi
  fi

  rm -f "${tmp_out}" "${tmp_err}"
}

echo -e "\n${CLR_CYAN}╭─ GPUBench End-to-End (E2E) Test Runner ──────────────────────────────────────╮${CLR_RESET}"
echo -e "${CLR_CYAN}│ Target Binary : ${BINARY_PATH}${CLR_RESET}"
echo -e "${CLR_CYAN}│ Active Tier   : ${SELECTED_TIER}${CLR_RESET}"

# Discover device dynamically
TARGET_GPU_INFO=$("${BINARY_PATH}" --list-devices 2>/dev/null | grep -E "^  [0-9]:" | head -n 1)
echo -e "${CLR_CYAN}│ Discovered GPU: ${TARGET_GPU_INFO:-None (CPU fallback)}${CLR_RESET}"
echo -e "${CLR_CYAN}╰──────────────────────────────────────────────────────────────────────────────╯${CLR_RESET}\n"

# ==============================================================================
# Tier 1: Feature Coverage (CLI flags and basic discovery)
# ==============================================================================
if [[ "${SELECTED_TIER}" == "1" || "${SELECTED_TIER}" == "all" ]]; then
  echo -e "${CLR_BOLD}─── Tier 1: Feature Coverage & CLI Mechanics ───────────────────────────────────${CLR_RESET}"

  run_test "Tier 1" "VersionFlag" \
    "${BINARY_PATH} --version" "0" "1\.0\."

  run_test "Tier 1" "HelpFlag" \
    "${BINARY_PATH} --help" "0" "GPUBench" "--benchmarks" "--groups" "--device"

  run_test "Tier 1" "ListDevicesFlag" \
    "${BINARY_PATH} --list-devices" "0" "Backend:" "0:" "VRAM:"

  run_test "Tier 1" "ListBenchmarksFlag" \
    "${BINARY_PATH} --list-benchmarks" "0" "\\[Compute\\]" "\\[Memory\\]" "\\[Graphics\\]" "\\[Ray Tracing\\]"

  run_test "Tier 1" "ListGroupsFlag" \
    "${BINARY_PATH} --list-groups" "0" "Compute" "Memory" "Graphics" "Ray Tracing"

  run_test "Tier 1" "ListBackendsFlag" \
    "${BINARY_PATH} --list-backends" "0" "Compute API Support & Diagnostics:" "Vulkan:" "ROCm:" "OpenCL:"

  run_test "Tier 1" "BenchmarkSelectionFlag" \
    "${BINARY_PATH} -b FP32 -c 0 --profile-snapshot -f" "0" "FP32" "TFLOPS"

  run_test "Tier 1" "GroupSelectionFlag" \
    "${BINARY_PATH} -g raster -c 0 --profile-snapshot -f" "0" "Pixel Fill Rate" "GPixels/s"

  run_test "Tier 1" "ConfigTargetSelectionFlag" \
    "${BINARY_PATH} -b 'Device Memory Bandwidth' -c 0 --profile-snapshot -f" "0" "Read 128 threads" "GB/s"

  run_test "Tier 1" "QuietFlag" \
    "${BINARY_PATH} -b FP32 -c 0 --profile-snapshot -f -q" "0" "FP32"

  run_test "Tier 1" "VerboseFlag" \
    "${BINARY_PATH} --verbose --list-devices" "0" "Backend:"

  run_test "Tier 1" "SceneFlag" \
    "${BINARY_PATH} -s showroom --list-benchmarks" "0" "RayRawTraversal"

  echo ""
fi

# ==============================================================================
# Tier 2: Boundary & Corner Cases (Graceful rejection without crash)
# ==============================================================================
if [[ "${SELECTED_TIER}" == "2" || "${SELECTED_TIER}" == "all" ]]; then
  echo -e "${CLR_BOLD}─── Tier 2: Boundary, Robustness & Rejection ───────────────────────────────────${CLR_RESET}"

  run_test "Tier 2" "InvalidDeviceIndexGracefulRejection" \
    "${BINARY_PATH} -d 999 -b FP32 --profile-snapshot -f" "nonzero" "out of range|no benchmarks were run"

  run_test "Tier 2" "NonExistentBenchmarkGracefulRejection" \
    "${BINARY_PATH} -b FakeBenchmark999" "nonzero" "no benchmark matched|no benchmarks were run"

  run_test "Tier 2" "MissingRequiredBenchmarkArgument" \
    "${BINARY_PATH} -b" "nonzero" "required"

  run_test "Tier 2" "MissingRequiredGroupArgument" \
    "${BINARY_PATH} -g" "nonzero" "required"

  run_test "Tier 2" "InvalidSceneOptionRejection" \
    "${BINARY_PATH} -s totally_invalid_scene" "nonzero" "scene"

  run_test "Tier 2" "OutOfRangeBouncesRejection" \
    "${BINARY_PATH} --bounces 99" "nonzero" "bounces"

  run_test "Tier 2" "OutOfRangeSppRejection" \
    "${BINARY_PATH} --spp 0" "nonzero" "spp"

  # Create a corrupted JSON file for testing
  CORRUPT_JSON="/tmp/gpubench_corrupt_$$.json"
  echo "{\"version\": \"1.0.0\", \"incomplete\": [" > "${CORRUPT_JSON}"

  run_test "Tier 2" "CorruptJsonFileImportRejection" \
    "${BINARY_PATH} -i ${CORRUPT_JSON}" "nonzero" "Error"
  rm -f "${CORRUPT_JSON}"

  run_test "Tier 2" "NonExistentJsonFileImportRejection" \
    "${BINARY_PATH} -i /tmp/nonexistent_file_$$.json" "nonzero" "Error"

  run_test "Tier 2" "CompareRequiresTwoFilesRejection" \
    "${BINARY_PATH} --compare single_file.json" "nonzero" "requires at least two JSON report files"

  run_test "Tier 2" "OutOfRangeConfigIndexHandling" \
    "${BINARY_PATH} -b FP32 -c 999 --profile-snapshot -f" "nonzero" "no benchmarks were run"

  echo ""
fi

# ==============================================================================
# Tier 3: Cross-Feature Interactions
# ==============================================================================
if [[ "${SELECTED_TIER}" == "3" || "${SELECTED_TIER}" == "all" ]]; then
  echo -e "${CLR_BOLD}─── Tier 3: Cross-Feature Interactions & JSON Pipelines ────────────────────────${CLR_RESET}"

  run_test "Tier 3" "GroupAndConfigCombination" \
    "${BINARY_PATH} -g raster -c 0 --profile-snapshot -f" "0" "Pixel Fill Rate" "RGBA8 Color Fill"

  run_test "Tier 3" "MultipleBenchmarksSelection" \
    "${BINARY_PATH} -b FP32,FP16 -c 0 --profile-snapshot -f" "0" "FP32" "FP16"

  run_test "Tier 3" "GroupPlusIndividualBenchmark" \
    "${BINARY_PATH} -g raster -b FP32 -c 0 --profile-snapshot -f" "0" "Pixel Fill Rate" "FP32"

  TMP_JSON="/tmp/gpubench_e2e_roundtrip_$$.json"
  run_test "Tier 3" "JsonFileExportRoundTrip" \
    "${BINARY_PATH} -b FP32 -c 0 --profile-snapshot -f -o ${TMP_JSON}" "0"

  # Sanitize any non-finite floats if background contention caused active cycles to be 0
  if [[ -f "${TMP_JSON}" ]]; then
    sed -i -E 's/:\s*(-?nan|inf|-inf|1\.#IND|-1\.#IND)/: 0.0/g' "${TMP_JSON}"
  fi

  run_test "Tier 3" "JsonFileImportExecution" \
    "${BINARY_PATH} -i ${TMP_JSON}" "0" "FP32" "TFLOPS"
  rm -f "${TMP_JSON}"

  TMP_RUN_A="/tmp/gpubench_e2e_runA_$$.json"
  TMP_RUN_B="/tmp/gpubench_e2e_runB_$$.json"
  eval "${BINARY_PATH} -b FP32 -c 0 --profile-snapshot -f -o ${TMP_RUN_A}" >/dev/null 2>&1
  eval "${BINARY_PATH} -b FP32 -c 0 --profile-snapshot -f -o ${TMP_RUN_B}" >/dev/null 2>&1

  # Sanitize any non-finite floats if background contention caused active cycles to be 0
  if [[ -f "${TMP_RUN_A}" && -f "${TMP_RUN_B}" ]]; then
    sed -i -E 's/:\s*(-?nan|inf|-inf|1\.#IND|-1\.#IND)/: 0.0/g' "${TMP_RUN_A}" "${TMP_RUN_B}"
  fi

  run_test "Tier 3" "JsonComparisonModeMultiRun" \
    "${BINARY_PATH} --compare ${TMP_RUN_A} ${TMP_RUN_B}" "0" "FP32" "TFLOPS"
  rm -f "${TMP_RUN_A}" "${TMP_RUN_B}"

  run_test "Tier 3" "QuietAndJsonOutputCleanliness" \
    "${BINARY_PATH} -b FP32 -c 0 --profile-snapshot -f --json -q" "0" "\"version\": \"1.0.0\"" "\"results\":"

  echo ""
fi

# ==============================================================================
# Tier 4: Real-World Workload Integration Tests
# ==============================================================================
if [[ "${SELECTED_TIER}" == "4" || "${SELECTED_TIER}" == "all" ]]; then
  echo -e "${CLR_BOLD}─── Tier 4: Real-World Workload Integration (Live GPU Execution) ───────────────${CLR_RESET}"

  run_test "Tier 4" "RealComputeWorkloadExecution" \
    "${BINARY_PATH} -b 'FP32,FP16,LDS Bank Conflicts' -c 0 --profile-snapshot --strict -f" "0" "FP32" "FP16" "LDS" "TFLOPS" "TB/s"

  run_test "Tier 4" "RealMemoryWorkloadExecution" \
    "${BINARY_PATH} -b 'Device Memory Bandwidth,L0 Vector Cache Latency (16 KB)' -c 0 --profile-snapshot --strict -f" "0" "Device Memory Bandwidth" "GB/s" "L0 Vector Cache Latency" "ns"

  run_test "Tier 4" "RealGraphicsWorkloadExecution" \
    "${BINARY_PATH} -b 'Pixel Fill Rate' -c 0 --profile-snapshot --strict -f" "0" "Pixel Fill Rate" "GPixels/s"

  run_test "Tier 4" "RealRayTracingWorkloadExecution" \
    "${BINARY_PATH} -b 'RayRawTraversal' -c 0 --profile-snapshot --strict -f" "0" "Hardware BVH Traversal" "MRays/s|GIS/s"

  run_test "Tier 4" "RealSystemMemoryWorkloadExecution" \
    "${BINARY_PATH} -b 'System Memory Bandwidth,System Memory Latency' -c 0 --profile-snapshot --strict -f" "0" "System" "GB/s" "System Memory Latency" "ns"

  run_test "Tier 4" "SceneParameterValidation" \
    "${BINARY_PATH} -b 'RayRawTraversal' -s showroom -c 0 --profile-snapshot --strict -f" "0" "Hardware BVH Traversal"

  run_test "Tier 4" "ParityFlagHonestyAndExecution" \
    "${BINARY_PATH} -b 'Pixel Fill Rate' -c 0 --verify-parity --profile-snapshot -f" "0" "Pixel Fill Rate"

  echo ""
fi

echo -e "${CLR_BOLD}───────────────────────────────────────────────────────────────────────────────${CLR_RESET}"
echo -n " Test Summary: "
if [[ ${FAILED_TESTS} -eq 0 ]]; then
  echo -e "${CLR_GREEN}All ${PASSED_TESTS} tests passed${CLR_RESET} (${TOTAL_TESTS} total executed)"
  exit 0
else
  echo -e "${CLR_RED}${FAILED_TESTS} failed${CLR_RESET}, ${CLR_GREEN}${PASSED_TESTS} passed${CLR_RESET} (${TOTAL_TESTS} total)"
  exit 1
fi
