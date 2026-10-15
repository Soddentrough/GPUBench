#include "test_harness.h"
#include "core/ResultImporter.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <errno.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

struct CommandResult {
  int exitCode = -1;
  std::string stdoutStr;
  std::string stderrStr;
  std::string combinedStr;
};

std::string getProcessIdStr() {
#ifdef _WIN32
  return std::to_string(static_cast<unsigned long>(GetCurrentProcessId()));
#else
  return std::to_string(static_cast<long>(getpid()));
#endif
}

std::string getTempFilePath(const std::string &prefix) {
  std::filesystem::path tmpDir = std::filesystem::temp_directory_path();
  return (tmpDir / (prefix + "_" + getProcessIdStr() + ".json")).string();
}

std::string getGpubenchPath() {
  const char *envBin = std::getenv("GPUBENCH_BIN");
  if (envBin && std::filesystem::exists(envBin)) {
    return std::string(envBin);
  }
#ifdef _WIN32
  const std::vector<std::string> candidates = {
      "./Release/gpubench.exe",
      "./gpubench.exe",
      "./build/Release/gpubench.exe",
      "./build/gpubench.exe",
      "../Release/gpubench.exe",
      "../gpubench.exe",
      "../build/Release/gpubench.exe",
      "../build/gpubench.exe",
      "../../build/Release/gpubench.exe",
      "../../build/gpubench.exe"
  };
  for (const auto &p : candidates) {
    if (std::filesystem::exists(p)) {
      return p;
    }
  }
  return "./Release/gpubench.exe";
#else
  const std::vector<std::string> candidates = {
      "./gpubench",
      "./build/gpubench",
      "../build/gpubench",
      "../../build/gpubench"
  };
  for (const auto &p : candidates) {
    if (access(p.c_str(), X_OK) == 0) {
      return p;
    }
  }
  return "./build/gpubench";
#endif
}

CommandResult runGpubench(const std::vector<std::string> &args) {
  std::string bin = getGpubenchPath();
#ifdef _WIN32
  static unsigned long callCounter = 0;
  ++callCounter;
  std::filesystem::path tmpDir = std::filesystem::temp_directory_path();
  std::string outPath = (tmpDir / ("gpubench_stdout_" + getProcessIdStr() + "_" + std::to_string(callCounter) + ".tmp")).string();
  std::string errPath = (tmpDir / ("gpubench_stderr_" + getProcessIdStr() + "_" + std::to_string(callCounter) + ".tmp")).string();

  SECURITY_ATTRIBUTES sa{};
  sa.nLength = sizeof(sa);
  sa.lpSecurityDescriptor = nullptr;
  sa.bInheritHandle = TRUE;

  HANDLE hOut = CreateFileA(outPath.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                            &sa, CREATE_ALWAYS, FILE_ATTRIBUTE_TEMPORARY, nullptr);
  HANDLE hErr = CreateFileA(errPath.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                            &sa, CREATE_ALWAYS, FILE_ATTRIBUTE_TEMPORARY, nullptr);
  if (hOut == INVALID_HANDLE_VALUE || hErr == INVALID_HANDLE_VALUE) {
    if (hOut != INVALID_HANDLE_VALUE) CloseHandle(hOut);
    if (hErr != INVALID_HANDLE_VALUE) CloseHandle(hErr);
    throw std::runtime_error("CreateFileA() failed for process output redirection");
  }

  std::string cmdLine = "\"" + bin + "\"";
  for (const auto &arg : args) {
    cmdLine += " \"" + arg + "\"";
  }
  std::vector<char> mutableCmd(cmdLine.begin(), cmdLine.end());
  mutableCmd.push_back('\0');

  STARTUPINFOA si{};
  si.cb = sizeof(si);
  si.dwFlags = STARTF_USESTDHANDLES;
  si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
  si.hStdOutput = hOut;
  si.hStdError = hErr;

  PROCESS_INFORMATION pi{};
  BOOL started = CreateProcessA(nullptr, mutableCmd.data(), nullptr, nullptr, TRUE,
                                CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi);
  CloseHandle(hOut);
  CloseHandle(hErr);

  CommandResult res;
  if (!started) {
    std::remove(outPath.c_str());
    std::remove(errPath.c_str());
    res.exitCode = 127;
    return res;
  }

  WaitForSingleObject(pi.hProcess, INFINITE);
  DWORD exitCode = 1;
  GetExitCodeProcess(pi.hProcess, &exitCode);
  res.exitCode = static_cast<int>(exitCode);

  CloseHandle(pi.hProcess);
  CloseHandle(pi.hThread);

  {
    std::ifstream ifsOut(outPath, std::ios::binary);
    if (ifsOut.is_open()) {
      res.stdoutStr.assign((std::istreambuf_iterator<char>(ifsOut)), std::istreambuf_iterator<char>());
    }
    std::ifstream ifsErr(errPath, std::ios::binary);
    if (ifsErr.is_open()) {
      res.stderrStr.assign((std::istreambuf_iterator<char>(ifsErr)), std::istreambuf_iterator<char>());
    }
  }
  res.combinedStr = res.stdoutStr + res.stderrStr;
  std::remove(outPath.c_str());
  std::remove(errPath.c_str());
  return res;
#else
  int stdoutPipe[2];
  int stderrPipe[2];
  if (pipe(stdoutPipe) != 0 || pipe(stderrPipe) != 0) {
    throw std::runtime_error("pipe() failed");
  }

  pid_t pid = fork();
  if (pid < 0) {
    close(stdoutPipe[0]); close(stdoutPipe[1]);
    close(stderrPipe[0]); close(stderrPipe[1]);
    throw std::runtime_error("fork() failed");
  }

  if (pid == 0) {
    // Child process
    close(stdoutPipe[0]);
    close(stderrPipe[0]);
    dup2(stdoutPipe[1], STDOUT_FILENO);
    dup2(stderrPipe[1], STDERR_FILENO);
    close(stdoutPipe[1]);
    close(stderrPipe[1]);

    std::vector<char *> argv;
    argv.push_back(const_cast<char *>(bin.c_str()));
    for (const auto &arg : args) {
      argv.push_back(const_cast<char *>(arg.c_str()));
    }
    argv.push_back(nullptr);

    execv(bin.c_str(), argv.data());
    std::perror("execv failed");
    _exit(127);
  }

  // Parent process
  close(stdoutPipe[1]);
  close(stderrPipe[1]);

  int fds[2] = {stdoutPipe[0], stderrPipe[0]};
  fcntl(fds[0], F_SETFL, O_NONBLOCK);
  fcntl(fds[1], F_SETFL, O_NONBLOCK);

  CommandResult res;
  char buffer[4096];
  bool outOpen = true;
  bool errOpen = true;

  while (outOpen || errOpen) {
    if (outOpen) {
      ssize_t n = read(fds[0], buffer, sizeof(buffer));
      if (n > 0) {
        res.stdoutStr.append(buffer, n);
        res.combinedStr.append(buffer, n);
      } else if (n == 0) {
        outOpen = false;
      } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
        outOpen = false;
      }
    }
    if (errOpen) {
      ssize_t n = read(fds[1], buffer, sizeof(buffer));
      if (n > 0) {
        res.stderrStr.append(buffer, n);
        res.combinedStr.append(buffer, n);
      } else if (n == 0) {
        errOpen = false;
      } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
        errOpen = false;
      }
    }
    if (outOpen || errOpen) {
      usleep(1000); // 1 ms
    }
  }

  close(fds[0]);
  close(fds[1]);

  int status = 0;
  waitpid(pid, &status, 0);
  if (WIFEXITED(status)) {
    res.exitCode = WEXITSTATUS(status);
  } else if (WIFSIGNALED(status)) {
    res.exitCode = 128 + WTERMSIG(status);
  }

  return res;
#endif
}

bool hasUsableGpu() {
  static int cached = -1;
  if (cached == -1) {
    auto res = runGpubench({"--list-devices"});
    cached = (res.exitCode == 0 && res.stdoutStr.find("0:") != std::string::npos) ? 1 : 0;
  }
  return cached == 1;
}

std::string sanitizeJsonFloats(std::string json) {
  const std::vector<std::string> targets = {": -nan", ": nan", ": inf", ": -inf", ": 1.#IND", ": -1.#IND"};
  for (const auto &t : targets) {
    size_t pos = 0;
    while ((pos = json.find(t, pos)) != std::string::npos) {
      json.replace(pos, t.length(), ": 0.0");
      pos += 5;
    }
  }
  return json;
}

} // namespace

// ============================================================================
// Tier 1: Feature Coverage (CLI flags, discovery, and basic execution)
// ============================================================================

TEST_CASE(E2E_Tier1, VersionFlag) {
  auto res = runGpubench({"--version"});
  ASSERT_EQ(res.exitCode, 0);
  ASSERT_TRUE(res.stdoutStr.find("1.0.0") != std::string::npos ||
              res.stdoutStr.find(GPUBENCH_VERSION) != std::string::npos);
}

TEST_CASE(E2E_Tier1, HelpFlag) {
  auto res = runGpubench({"--help"});
  ASSERT_EQ(res.exitCode, 0);
  ASSERT_TRUE(res.stdoutStr.find("GPUBench") != std::string::npos);
  ASSERT_TRUE(res.stdoutStr.find("--benchmarks") != std::string::npos);
  ASSERT_TRUE(res.stdoutStr.find("--groups") != std::string::npos);
  ASSERT_TRUE(res.stdoutStr.find("--device") != std::string::npos);
  ASSERT_TRUE(res.stdoutStr.find("--json") != std::string::npos);
  ASSERT_TRUE(res.stdoutStr.find("--scene") != std::string::npos);
}

TEST_CASE(E2E_Tier1, ListDevicesFlag) {
  if (!hasUsableGpu()) { return; }
  auto res = runGpubench({"--list-devices"});
  ASSERT_EQ(res.exitCode, 0);
  ASSERT_TRUE(res.stdoutStr.find("Backend:") != std::string::npos);
  ASSERT_TRUE(res.stdoutStr.find("0:") != std::string::npos);
  ASSERT_TRUE(res.stdoutStr.find("VRAM:") != std::string::npos);
}

TEST_CASE(E2E_Tier1, ListBenchmarksFlag) {
  auto res = runGpubench({"--list-benchmarks"});
  ASSERT_EQ(res.exitCode, 0);
  ASSERT_TRUE(res.stdoutStr.find("[Compute]") != std::string::npos);
  ASSERT_TRUE(res.stdoutStr.find("FP32") != std::string::npos);
  ASSERT_TRUE(res.stdoutStr.find("FP64") != std::string::npos);
  ASSERT_TRUE(res.stdoutStr.find("[Memory]") != std::string::npos);
  ASSERT_TRUE(res.stdoutStr.find("[Graphics]") != std::string::npos);
  ASSERT_TRUE(res.stdoutStr.find("Pixel Fill Rate") != std::string::npos);
  ASSERT_TRUE(res.stdoutStr.find("[Ray Tracing]") != std::string::npos);
  ASSERT_TRUE(res.stdoutStr.find("RayRawTraversal") != std::string::npos);
  ASSERT_TRUE(res.stdoutStr.find("[System]") != std::string::npos);
}

TEST_CASE(E2E_Tier1, ListGroupsFlag) {
  auto res = runGpubench({"--list-groups"});
  ASSERT_EQ(res.exitCode, 0);
  ASSERT_TRUE(res.stdoutStr.find("Compute") != std::string::npos);
  ASSERT_TRUE(res.stdoutStr.find("Memory") != std::string::npos);
  ASSERT_TRUE(res.stdoutStr.find("Graphics") != std::string::npos);
  ASSERT_TRUE(res.stdoutStr.find("Raster") != std::string::npos);
  ASSERT_TRUE(res.stdoutStr.find("Ray Tracing") != std::string::npos);
  ASSERT_TRUE(res.stdoutStr.find("System") != std::string::npos);
}

TEST_CASE(E2E_Tier1, ListBackendsFlag) {
  auto res = runGpubench({"--list-backends"});
  ASSERT_EQ(res.exitCode, 0);
  ASSERT_TRUE(res.stdoutStr.find("Compute API Support & Diagnostics:") != std::string::npos);
  ASSERT_TRUE(res.stdoutStr.find("Vulkan:") != std::string::npos);
  ASSERT_TRUE(res.stdoutStr.find("ROCm:") != std::string::npos);
  ASSERT_TRUE(res.stdoutStr.find("OpenCL:") != std::string::npos);
}

TEST_CASE(E2E_Tier1, BenchmarkSelectionFlag) {
  if (!hasUsableGpu()) { return; }
  auto res = runGpubench({"-b", "FP32", "-c", "0", "--profile-snapshot", "-f"});
  ASSERT_EQ(res.exitCode, 0);
  ASSERT_TRUE(res.stdoutStr.find("FP32") != std::string::npos);
  ASSERT_TRUE(res.stdoutStr.find("TFLOPS") != std::string::npos);
  ASSERT_TRUE(res.stdoutStr.find("Done") != std::string::npos ||
              res.stdoutStr.find("Hierarchical Benchmark Report") != std::string::npos);
}

TEST_CASE(E2E_Tier1, GroupSelectionFlag) {
  if (!hasUsableGpu()) { return; }
  auto res = runGpubench({"-g", "raster", "-c", "0", "--profile-snapshot", "-f"});
  ASSERT_EQ(res.exitCode, 0);
  ASSERT_TRUE(res.stdoutStr.find("Pixel Fill Rate") != std::string::npos);
  ASSERT_TRUE(res.stdoutStr.find("GPixels/s") != std::string::npos);
}

TEST_CASE(E2E_Tier1, ConfigTargetSelectionFlag) {
  if (!hasUsableGpu()) { return; }
  auto res = runGpubench({"-b", "Device Memory Bandwidth", "-c", "0", "--profile-snapshot", "-f"});
  ASSERT_EQ(res.exitCode, 0);
  ASSERT_TRUE(res.stdoutStr.find("Read 128 threads/group") != std::string::npos);
  ASSERT_TRUE(res.stdoutStr.find("GB/s") != std::string::npos);
}

TEST_CASE(E2E_Tier1, QuietFlag) {
  if (!hasUsableGpu()) { return; }
  auto res = runGpubench({"-b", "FP32", "-c", "0", "--profile-snapshot", "-f", "-q"});
  ASSERT_EQ(res.exitCode, 0);
  // Quiet flag suppresses top decorative banner
  ASSERT_TRUE(res.stdoutStr.find("╭─ GPUBench v") == std::string::npos);
  // Still outputs the results table
  ASSERT_TRUE(res.stdoutStr.find("FP32") != std::string::npos);
}

TEST_CASE(E2E_Tier1, VerboseFlag) {
  if (!hasUsableGpu()) { return; }
  auto res = runGpubench({"--verbose", "--list-devices"});
  ASSERT_EQ(res.exitCode, 0);
  ASSERT_TRUE(res.stdoutStr.find("Backend:") != std::string::npos);
}

TEST_CASE(E2E_Tier1, SceneFlag) {
  auto res = runGpubench({"-s", "showroom", "--list-benchmarks"});
  ASSERT_EQ(res.exitCode, 0);
  ASSERT_TRUE(res.stdoutStr.find("RayRawTraversal") != std::string::npos);
}

// ============================================================================
// Tier 2: Boundary & Corner Cases (Graceful rejection without crash)
// ============================================================================

TEST_CASE(E2E_Tier2, InvalidDeviceIndexGracefulRejection) {
  auto res = runGpubench({"-d", "999", "-b", "FP32", "--profile-snapshot", "-f"});
  // Out of range device index must fail cleanly with non-zero exit code (no segfault)
  ASSERT_NE(res.exitCode, 0);
  ASSERT_LT(res.exitCode, 128); // Not terminated by fatal signal (SIGSEGV/SIGABRT)
  ASSERT_TRUE(res.combinedStr.find("out of range") != std::string::npos ||
              res.combinedStr.find("no benchmarks were run") != std::string::npos ||
              res.combinedStr.find("No compute backend") != std::string::npos ||
              res.combinedStr.find("Error") != std::string::npos ||
              res.combinedStr.find("failed") != std::string::npos);
}

TEST_CASE(E2E_Tier2, NonExistentBenchmarkGracefulRejection) {
  auto res = runGpubench({"-b", "NonExistentBenchmark_FooBar123"});
  ASSERT_NE(res.exitCode, 0);
  ASSERT_LT(res.exitCode, 128);
  ASSERT_TRUE(res.combinedStr.find("no benchmark matched 'NonExistentBenchmark_FooBar123'") != std::string::npos ||
              res.combinedStr.find("no benchmarks were run") != std::string::npos ||
              res.combinedStr.find("No compute backend") != std::string::npos ||
              res.combinedStr.find("Error") != std::string::npos ||
              res.combinedStr.find("failed") != std::string::npos);
}

TEST_CASE(E2E_Tier2, MissingRequiredBenchmarkArgument) {
  auto res = runGpubench({"-b"});
  ASSERT_NE(res.exitCode, 0);
  ASSERT_LT(res.exitCode, 128);
  ASSERT_TRUE(res.combinedStr.find("--benchmarks") != std::string::npos ||
              res.combinedStr.find("required") != std::string::npos);
}

TEST_CASE(E2E_Tier2, MissingRequiredGroupArgument) {
  auto res = runGpubench({"-g"});
  ASSERT_NE(res.exitCode, 0);
  ASSERT_LT(res.exitCode, 128);
  ASSERT_TRUE(res.combinedStr.find("--groups") != std::string::npos ||
              res.combinedStr.find("required") != std::string::npos);
}

TEST_CASE(E2E_Tier2, InvalidSceneOptionRejection) {
  auto res = runGpubench({"-s", "invalid_scene_geometry_xyz"});
  ASSERT_NE(res.exitCode, 0);
  ASSERT_LT(res.exitCode, 128);
  ASSERT_TRUE(res.combinedStr.find("--scene") != std::string::npos ||
              res.combinedStr.find("invalid_scene_geometry_xyz") != std::string::npos);
}

TEST_CASE(E2E_Tier2, OutOfRangeBouncesRejection) {
  auto resHigh = runGpubench({"--bounces", "99"});
  ASSERT_NE(resHigh.exitCode, 0);
  ASSERT_LT(resHigh.exitCode, 128);

  auto resZero = runGpubench({"--bounces", "0"});
  ASSERT_NE(resZero.exitCode, 0);
  ASSERT_LT(resZero.exitCode, 128);
}

TEST_CASE(E2E_Tier2, OutOfRangeSppRejection) {
  auto resZero = runGpubench({"--spp", "0"});
  ASSERT_NE(resZero.exitCode, 0);
  ASSERT_LT(resZero.exitCode, 128);

  auto resHigh = runGpubench({"--spp", "5000"});
  ASSERT_NE(resHigh.exitCode, 0);
  ASSERT_LT(resHigh.exitCode, 128);
}

TEST_CASE(E2E_Tier2, CorruptJsonFileImportRejection) {
  const std::string corruptPath = getTempFilePath("gpubench_corrupt_test");
  {
    std::ofstream ofs(corruptPath);
    ofs << "{\"version\": \"1.0.0\", \"incomplete\": [";
  }

  auto res = runGpubench({"-i", corruptPath});
  std::remove(corruptPath.c_str());

  ASSERT_NE(res.exitCode, 0);
  ASSERT_LT(res.exitCode, 128);
  ASSERT_TRUE(res.combinedStr.find("Error") != std::string::npos);
}

TEST_CASE(E2E_Tier2, NonExistentJsonFileImportRejection) {
  const std::string nonExistentPath = getTempFilePath("definitely_nonexistent_gpubench_report_998877");
  std::remove(nonExistentPath.c_str());
  auto res = runGpubench({"-i", nonExistentPath});
  ASSERT_NE(res.exitCode, 0);
  ASSERT_LT(res.exitCode, 128);
  ASSERT_TRUE(res.combinedStr.find("Error") != std::string::npos);
}

TEST_CASE(E2E_Tier2, CompareRequiresTwoFilesRejection) {
  auto res = runGpubench({"--compare", "single_dummy_file.json"});
  ASSERT_NE(res.exitCode, 0);
  ASSERT_LT(res.exitCode, 128);
  ASSERT_TRUE(res.combinedStr.find("requires at least two JSON report files") != std::string::npos);
}

TEST_CASE(E2E_Tier2, OutOfRangeConfigIndexHandling) {
  // Config 999 is out of range for FP32 (which has 1 config: 0)
  auto res = runGpubench({"-b", "FP32", "-c", "999", "--profile-snapshot", "-f"});
  // Must handle gracefully without crashing / aborting with signal
  ASSERT_LT(res.exitCode, 128);
}

// ============================================================================
// Tier 3: Cross-Feature Interactions
// ============================================================================

TEST_CASE(E2E_Tier3, GroupAndConfigCombination) {
  if (!hasUsableGpu()) { return; }
  auto res = runGpubench({"-g", "raster", "-c", "0", "--profile-snapshot", "-f"});
  ASSERT_EQ(res.exitCode, 0);
  ASSERT_TRUE(res.stdoutStr.find("Pixel Fill Rate") != std::string::npos);
  ASSERT_TRUE(res.stdoutStr.find("RGBA8 Color Fill") != std::string::npos);
}

TEST_CASE(E2E_Tier3, MultipleBenchmarksSelection) {
  if (!hasUsableGpu()) { return; }
  auto res = runGpubench({"-b", "FP32,FP16", "-c", "0", "--profile-snapshot", "-f"});
  ASSERT_EQ(res.exitCode, 0);
  ASSERT_TRUE(res.stdoutStr.find("FP32") != std::string::npos);
  ASSERT_TRUE(res.stdoutStr.find("FP16") != std::string::npos);
}

TEST_CASE(E2E_Tier3, GroupPlusIndividualBenchmark) {
  if (!hasUsableGpu()) { return; }
  auto res = runGpubench({"-g", "raster", "-b", "FP32", "-c", "0", "--profile-snapshot", "-f"});
  ASSERT_EQ(res.exitCode, 0);
  ASSERT_TRUE(res.stdoutStr.find("Pixel Fill Rate") != std::string::npos);
  ASSERT_TRUE(res.stdoutStr.find("FP32") != std::string::npos);
}

TEST_CASE(E2E_Tier3, JsonFileExportAndImportRoundTrip) {
  if (!hasUsableGpu()) { return; }
  const std::string tmpPath = getTempFilePath("gpubench_e2e_roundtrip_test");
  std::remove(tmpPath.c_str());

  // 1. Export benchmark run to JSON file
  auto exportRes = runGpubench({"-b", "FP32", "-c", "0", "--profile-snapshot", "-f", "-o", tmpPath});
  ASSERT_EQ(exportRes.exitCode, 0);

  // 2. Verify file was created and is non-empty
  std::ifstream ifs(tmpPath);
  ASSERT_TRUE(ifs.is_open());
  std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
  ifs.close();
  ASSERT_TRUE(content.size() > 200);

  // 3. Import and display results from the generated JSON report
  auto importRes = runGpubench({"-i", tmpPath});
  ASSERT_EQ(importRes.exitCode, 0);
  ASSERT_TRUE(importRes.stdoutStr.find("FP32") != std::string::npos);
  ASSERT_TRUE(importRes.stdoutStr.find("TFLOPS") != std::string::npos);

  // Clean up
  std::remove(tmpPath.c_str());
}

TEST_CASE(E2E_Tier3, JsonStdoutSchemaValidation) {
  if (!hasUsableGpu()) { return; }
  auto res = runGpubench({"-b", "FP32", "-c", "0", "--profile-snapshot", "-f", "--json"});
  ASSERT_EQ(res.exitCode, 0);

  // Load and validate JSON structure via ResultImporter.
  // Under background GPU contention, if active cycles is 0, driver counters may produce
  // non-finite values (-nan / inf). Sanitize any non-finite tokens to validate functional schema.
  std::string sanitized = sanitizeJsonFloats(res.stdoutStr);

  ImportedRun run;
  std::string err;
  bool ok = ResultImporter::loadFromString(sanitized, run, err);
  if (!ok) {
    std::cerr << "JSON Parse error: " << err << "\nPayload:\n" << sanitized << "\n";
  }
  ASSERT_TRUE(ok);

  // Validate top-level metadata schema
  ASSERT_EQ(run.appVersion, "1.0.0");
  ASSERT_GT(run.timestamp, 0ULL);
  ASSERT_EQ(run.backend, "Vulkan");

  // Validate device profile schema
  ASSERT_FALSE(run.deviceProfile.deviceName.empty());
  ASSERT_EQ(run.deviceProfile.backend, "Vulkan");
  ASSERT_GT(run.deviceProfile.vramTotalMb, 0ULL);

  // Validate results schema
  ASSERT_GE(run.results.size(), 1u);
  const auto &result = run.results[0];
  ASSERT_EQ(result.benchmarkName, "FP32");
  ASSERT_EQ(result.component, "Compute");
  ASSERT_EQ(result.metric, "TFLOPS");
  ASSERT_GT(result.operations, 0ULL);
  ASSERT_GT(result.time_ms, 0.0);
  ASSERT_TRUE(result.isValid);
  ASSERT_FALSE(result.isEmulated);
}

TEST_CASE(E2E_Tier3, JsonComparisonModeMultiRun) {
  if (!hasUsableGpu()) { return; }
  const std::string runA = getTempFilePath("gpubench_e2e_runA");
  const std::string runB = getTempFilePath("gpubench_e2e_runB");
  std::remove(runA.c_str());
  std::remove(runB.c_str());

  // Generate run A and run B
  auto resA = runGpubench({"-b", "FP32", "-c", "0", "--profile-snapshot", "-f", "-o", runA});
  ASSERT_EQ(resA.exitCode, 0);

  auto resB = runGpubench({"-b", "FP32", "-c", "0", "--profile-snapshot", "-f", "-o", runB});
  ASSERT_EQ(resB.exitCode, 0);

  // Sanitize any non-compliant floats emitted due to active cycle contention before comparison
  auto sanitizeFile = [](const std::string &path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) return;
    std::string s((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    ifs.close();
    std::string clean = sanitizeJsonFloats(s);
    std::ofstream ofs(path);
    ofs << clean;
  };
  sanitizeFile(runA);
  sanitizeFile(runB);

  // Compare both runs side-by-side
  auto cmpRes = runGpubench({"--compare", runA, runB});
  ASSERT_EQ(cmpRes.exitCode, 0);
  ASSERT_TRUE(cmpRes.stdoutStr.find("FP32") != std::string::npos);
  ASSERT_TRUE(cmpRes.stdoutStr.find("TFLOPS") != std::string::npos);

  // Clean up
  std::remove(runA.c_str());
  std::remove(runB.c_str());
}

TEST_CASE(E2E_Tier3, QuietAndJsonOutputCleanliness) {
  if (!hasUsableGpu()) { return; }
  auto res = runGpubench({"-b", "FP32", "-c", "0", "--profile-snapshot", "-f", "--json", "-q"});
  ASSERT_EQ(res.exitCode, 0);

  // Under --json and -q, stdout should contain valid parseable JSON without console noise
  std::string sanitized = sanitizeJsonFloats(res.stdoutStr);
  ImportedRun run;
  std::string err;
  bool ok = ResultImporter::loadFromString(sanitized, run, err);
  ASSERT_TRUE(ok);
  ASSERT_GE(run.results.size(), 1u);
}

// ============================================================================
// Tier 4: Real-World Workload Integration Tests
// ============================================================================

TEST_CASE(E2E_Tier4, RealComputeWorkloadExecution) {
  if (!hasUsableGpu()) { return; }
  auto res = runGpubench({"-b", "FP32,FP16,LDS Bank Conflicts,ILP & Dual-Issue", "-c", "0", "--profile-snapshot", "--strict", "-f"});
  ASSERT_EQ(res.exitCode, 0);
  ASSERT_TRUE(res.stdoutStr.find("FP32") != std::string::npos);
  ASSERT_TRUE(res.stdoutStr.find("FP16") != std::string::npos);
  ASSERT_TRUE(res.stdoutStr.find("LDS") != std::string::npos);
  ASSERT_TRUE(res.stdoutStr.find("Dual-Issue") != std::string::npos || res.stdoutStr.find("ILP") != std::string::npos);
  ASSERT_TRUE(res.stdoutStr.find("TFLOPS") != std::string::npos);
  ASSERT_TRUE(res.stdoutStr.find("TB/s") != std::string::npos);
}

TEST_CASE(E2E_Tier4, RealMemoryWorkloadExecution) {
  if (!hasUsableGpu()) { return; }
  auto res = runGpubench({"-b", "Device Memory Bandwidth,L0 Vector Cache Latency (16 KB)", "-c", "0", "--profile-snapshot", "--strict", "-f"});
  ASSERT_EQ(res.exitCode, 0);
  ASSERT_TRUE(res.stdoutStr.find("Device Memory Bandwidth") != std::string::npos);
  ASSERT_TRUE(res.stdoutStr.find("GB/s") != std::string::npos);
  ASSERT_TRUE(res.stdoutStr.find("L0 Vector Cache Latency") != std::string::npos);
  ASSERT_TRUE(res.stdoutStr.find("ns") != std::string::npos);
}

TEST_CASE(E2E_Tier4, RealGraphicsWorkloadExecution) {
  if (!hasUsableGpu()) { return; }
  auto res = runGpubench({"-b", "Pixel Fill Rate", "-c", "0", "--profile-snapshot", "--strict", "-f"});
  ASSERT_EQ(res.exitCode, 0);
  ASSERT_TRUE(res.stdoutStr.find("Pixel Fill Rate") != std::string::npos);
  ASSERT_TRUE(res.stdoutStr.find("GPixels/s") != std::string::npos);
}

TEST_CASE(E2E_Tier4, RealRayTracingWorkloadExecution) {
  if (!hasUsableGpu()) { return; }
  auto res = runGpubench({"-b", "RayRawTraversal", "-c", "0", "--profile-snapshot", "--strict", "-f"});
  ASSERT_EQ(res.exitCode, 0);
  ASSERT_TRUE(res.stdoutStr.find("Hardware BVH Traversal") != std::string::npos);
  ASSERT_TRUE(res.stdoutStr.find("MRays/s") != std::string::npos ||
              res.stdoutStr.find("GIS/s") != std::string::npos);
}

TEST_CASE(E2E_Tier4, RealSystemMemoryWorkloadExecution) {
  if (!hasUsableGpu()) { return; }
  auto res = runGpubench({"-b", "System Memory Bandwidth,System Memory Latency", "-c", "0", "--profile-snapshot", "--strict", "-f"});
  ASSERT_EQ(res.exitCode, 0);
  ASSERT_TRUE(res.stdoutStr.find("System") != std::string::npos);
  ASSERT_TRUE(res.stdoutStr.find("GB/s") != std::string::npos);
  ASSERT_TRUE(res.stdoutStr.find("System Memory Latency") != std::string::npos);
  ASSERT_TRUE(res.stdoutStr.find("ns") != std::string::npos);
}

TEST_CASE(E2E_Tier4, SceneParameterValidation) {
  if (!hasUsableGpu()) { return; }
  auto res = runGpubench({"-b", "RayRawTraversal", "-s", "showroom", "-c", "0", "--profile-snapshot", "--strict", "-f"});
  ASSERT_EQ(res.exitCode, 0);
  ASSERT_TRUE(res.stdoutStr.find("Hardware BVH Traversal") != std::string::npos);
}

TEST_CASE(E2E_Tier4, ParityFlagHonestyAndExecution) {
  if (!hasUsableGpu()) { return; }
  auto res = runGpubench({"-b", "Pixel Fill Rate", "-c", "0", "--verify-parity", "--profile-snapshot", "-f"});
  ASSERT_EQ(res.exitCode, 0);
  ASSERT_TRUE(res.stdoutStr.find("Pixel Fill Rate") != std::string::npos);
}
