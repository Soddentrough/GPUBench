#include "core/RooflineModel.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>

namespace gpubench {

namespace {
std::string toLowerCopy(const std::string &s) {
  std::string out = s;
  std::transform(out.begin(), out.end(), out.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return out;
}
} // namespace

double DeviceRoofline::primaryVramGBps() const {
  for (const auto &bw : bandwidthCeilings) {
    if (bw.level == "VRAM" && bw.bandwidthGBps > 0.0) {
      return bw.bandwidthGBps;
    }
  }
  return 0.0;
}

double DeviceRoofline::primaryFp32Gops() const {
  for (const auto &c : computeCeilings) {
    if (c.name == "FP32 Vector" && c.peakGops > 0.0) {
      return c.peakGops;
    }
  }
  return 0.0;
}

double DeviceRoofline::highestComputeGops() const {
  double best = 0.0;
  for (const auto &c : computeCeilings) {
    best = std::max(best, c.peakGops);
  }
  return best;
}

double RooflineModel::inferOperationalIntensity(const ResultData &r) {
  if (r.operationalIntensity > 0.0 && std::isfinite(r.operationalIntensity)) {
    return r.operationalIntensity;
  }

  const std::string lowName = toLowerCopy(r.benchmarkName);
  const std::string lowSub = toLowerCopy(r.subcategory);

  if (r.metric == "GB/s") {
    // Copy/streaming benchmarks execute ~1 address/element op per 8-16 bytes transferred
    if (lowName.find("vram bandwidth") != std::string::npos ||
        lowName.find("device memory bandwidth") != std::string::npos ||
        lowSub == "vram bandwidth") {
      if (lowName.find("1024") != std::string::npos) return 0.0625;
      if (lowName.find("256") != std::string::npos) return 0.125;
      return 0.25;
    }
    if (lowName.find("l0") != std::string::npos) return 0.50;
    if (lowName.find("l1") != std::string::npos) return 0.35;
    if (lowName.find("l2") != std::string::npos) return 0.25;
    if (lowName.find("l3") != std::string::npos) return 0.18;
    if (lowName.find("interconnect") != std::string::npos ||
        lowSub.find("interconnect") != std::string::npos ||
        lowName.find("h2d") != std::string::npos ||
        lowName.find("d2h") != std::string::npos ||
        lowName.find("rebar") != std::string::npos) {
      return 0.03125;
    }
    return 0.0;
  }

  if (r.metric != "TFLOPS" && r.metric != "TOPS") {
    return 0.0;
  }

  // Matrix / Tensor / Cooperative Vector workloads
  const bool isMatrix = (lowName.find("matrix") != std::string::npos) ||
                        (lowName.find("fp8") != std::string::npos) ||
                        (lowName.find("mxfp6") != std::string::npos) ||
                        (lowName.find("mxfp4") != std::string::npos) ||
                        (lowName.find("int4") != std::string::npos) ||
                        (lowName.find("sparse") != std::string::npos) ||
                        (lowName.find("coop") != std::string::npos);
  if (isMatrix) {
    if (lowName.find("sparse") != std::string::npos) return 8192.0;
    if (lowName.find("mxfp4") != std::string::npos || lowName.find("int4") != std::string::npos)
      return 8192.0;
    if (lowName.find("mxfp6") != std::string::npos) return 6826.0;
    if (lowName.find("fp8") != std::string::npos || lowName.find("int8") != std::string::npos)
      return 6144.0;
    if (lowName.find("coop") != std::string::npos || lowName.find("neural") != std::string::npos)
      return 512.0;
    return 4096.0;
  }

  // Vector ALU FMA loops
  if (lowName.find("fp64") != std::string::npos) return 1024.0;
  if (lowName.find("dual-issue") != std::string::npos) return 4096.0;
  if (lowName.find("fp16") != std::string::npos || lowName.find("bf16") != std::string::npos)
    return 4096.0;
  if (lowName.find("int8") != std::string::npos) return 4096.0;
  if (lowName.find("fp32") != std::string::npos) return 2048.0;

  return 0.0;
}

std::vector<DeviceRoofline>
RooflineModel::analyze(const std::vector<ResultData> &results) {
  struct GroupKey {
    std::string backend;
    std::string device;
    uint32_t devIdx = 0;
    bool operator<(const GroupKey &o) const {
      if (devIdx != o.devIdx) return devIdx < o.devIdx;
      if (device != o.device) return device < o.device;
      return backend < o.backend;
    }
  };

  std::map<GroupKey, std::vector<const ResultData *>> grouped;
  for (const auto &r : results) {
    if (r.isUnsupported || !r.isValid || !r.errorString.empty()) continue;
    if (r.backendName == "Host" || r.deviceName.find("System") == 0) continue;
    if (r.time_ms <= 0.0) continue;
    grouped[{r.backendName, r.deviceName, r.deviceIndex}].push_back(&r);
  }

  std::vector<DeviceRoofline> out;
  for (const auto &[key, group] : grouped) {
    DeviceRoofline dr;
    dr.backendName = key.backend;
    dr.deviceName = key.device;
    dr.deviceIndex = key.devIdx;

    struct PeakTrack {
      double val = 0.0;
      std::string src;
      std::string metric;
    };

    PeakTrack bwVram, bwL3, bwL2, bwL1, bwL0, bwPcie;
    PeakTrack cFp64, cFp32, cFp16Vec, cBf16Vec, cFp16Mat, cLowPrecMat;

    auto updatePeak = [](PeakTrack &p, double v, const std::string &src,
                         const std::string &metric = "") {
      if (v > p.val && std::isfinite(v)) {
        p.val = v;
        p.src = src;
        if (!metric.empty()) p.metric = metric;
      }
    };

    for (const ResultData *rp : group) {
      const ResultData &r = *rp;
      const double val = computeResultValue(r);
      if (val <= 0.0 || !std::isfinite(val)) continue;
      const std::string lowName = toLowerCopy(r.benchmarkName);
      const std::string lowSub = toLowerCopy(r.subcategory);

      if (r.metric == "GB/s") {
        if (lowName.find("vram bandwidth") != std::string::npos ||
            lowName.find("device memory bandwidth") != std::string::npos ||
            lowSub == "vram bandwidth") {
          updatePeak(bwVram, val, r.benchmarkName);
        } else if (lowName.find("l3") != std::string::npos &&
                   lowName.find("bandwidth") != std::string::npos) {
          updatePeak(bwL3, val, r.benchmarkName);
        } else if (lowName.find("l2") != std::string::npos &&
                   lowName.find("bandwidth") != std::string::npos) {
          updatePeak(bwL2, val, r.benchmarkName);
        } else if (lowName.find("l1") != std::string::npos &&
                   lowName.find("bandwidth") != std::string::npos) {
          updatePeak(bwL1, val, r.benchmarkName);
        } else if (lowName.find("l0") != std::string::npos &&
                   lowName.find("bandwidth") != std::string::npos) {
          updatePeak(bwL0, val, r.benchmarkName);
        } else if (lowSub.find("interconnect") != std::string::npos ||
                   lowName.find("h2d") != std::string::npos ||
                   lowName.find("d2h") != std::string::npos ||
                   lowName.find("rebar") != std::string::npos) {
          updatePeak(bwPcie, val, r.benchmarkName);
        }
      } else if (r.metric == "TFLOPS" || r.metric == "TOPS") {
        const bool isMat = (lowName.find("matrix") != std::string::npos) ||
                           (lowName.find("fp8") != std::string::npos) ||
                           (lowName.find("mxfp6") != std::string::npos) ||
                           (lowName.find("mxfp4") != std::string::npos) ||
                           (lowName.find("int4") != std::string::npos) ||
                           (lowName.find("sparse") != std::string::npos);

        if (lowName.find("fp64") != std::string::npos) {
          updatePeak(cFp64, val, r.benchmarkName, r.metric);
        } else if (!isMat && (lowName.find("fp32") != std::string::npos ||
                              lowName.find("dual-issue") != std::string::npos)) {
          // Only float32 / mixed dual-issue counts for FP32 ceiling (exclude pure INT32 dual-issue)
          if (lowName.find("int32") == std::string::npos) {
            updatePeak(cFp32, val, r.benchmarkName, "TFLOPS");
          }
        } else if (!isMat && lowName.find("fp16") != std::string::npos) {
          updatePeak(cFp16Vec, val, r.benchmarkName, r.metric);
        } else if (!isMat && lowName.find("bf16") != std::string::npos && !r.isEmulated) {
          updatePeak(cBf16Vec, val, r.benchmarkName, r.metric);
        } else if (isMat && (lowName.find("fp16") != std::string::npos ||
                             lowName.find("bf16") != std::string::npos) &&
                   lowName.find("sparse") == std::string::npos) {
          updatePeak(cFp16Mat, val, r.benchmarkName, r.metric);
        } else if (isMat) {
          updatePeak(cLowPrecMat, val, r.benchmarkName, r.metric);
        }
      }
    }

    auto pushBw = [&](const char *lvl, const PeakTrack &pt) {
      if (pt.val > 0.0) {
        dr.bandwidthCeilings.push_back({lvl, pt.src, pt.val});
      }
    };
    pushBw("PCIe", bwPcie);
    pushBw("VRAM", bwVram);
    pushBw("L3", bwL3);
    pushBw("L2", bwL2);
    pushBw("L1", bwL1);
    pushBw("L0", bwL0);

    auto pushComp = [&](const char *name, const PeakTrack &pt) {
      if (pt.val > 0.0) {
        dr.computeCeilings.push_back(
            {name, pt.metric.empty() ? "TFLOPS" : pt.metric, pt.src, pt.val,
             pt.val * 1000.0});
      }
    };
    pushComp("FP64 Vector", cFp64);
    pushComp("FP32 Vector", cFp32);
    pushComp("BF16 Vector", cBf16Vec);
    pushComp("FP16 Vector", cFp16Vec);
    pushComp("FP16/BF16 Matrix", cFp16Mat);
    pushComp("Low-Precision/Sparse Matrix", cLowPrecMat);

    // Compute Ridge Points across measured memory levels & key compute ceilings
    for (const auto &bw : dr.bandwidthCeilings) {
      if (bw.bandwidthGBps <= 0.0) continue;
      for (const auto &cc : dr.computeCeilings) {
        if (cc.peakGops <= 0.0) continue;
        RooflineRidgePoint rp;
        rp.memoryLevel = bw.level;
        rp.computeCeiling = cc.name;
        rp.bandwidthGBps = bw.bandwidthGBps;
        rp.peakTops = cc.peakTops;
        rp.ridgeOpsPerByte = cc.peakGops / bw.bandwidthGBps;
        dr.ridgePoints.push_back(rp);
      }
    }

    const double refBwGBps =
        bwVram.val > 0.0
            ? bwVram.val
            : (!dr.bandwidthCeilings.empty()
                   ? dr.bandwidthCeilings.front().bandwidthGBps
                   : 0.0);

    // Map executed workloads onto Roofline points
    for (const ResultData *rp : group) {
      const ResultData &r = *rp;
      const double oi = inferOperationalIntensity(r);
      if (oi <= 0.0) continue;

      const double val = computeResultValue(r);
      if (val <= 0.0 || !std::isfinite(val)) continue;

      RooflineWorkloadPoint wp;
      wp.label = r.benchmarkName;
      wp.subcategory = r.subcategory;
      wp.metric = r.metric;
      wp.operationalIntensity = oi;

      const std::string lowName = toLowerCopy(r.benchmarkName);
      wp.isMatrix = (lowName.find("matrix") != std::string::npos) ||
                    (lowName.find("fp8") != std::string::npos) ||
                    (lowName.find("mxfp6") != std::string::npos) ||
                    (lowName.find("mxfp4") != std::string::npos) ||
                    (lowName.find("int4") != std::string::npos) ||
                    (lowName.find("sparse") != std::string::npos) ||
                    (lowName.find("coop") != std::string::npos);

      if (r.metric == "GB/s") {
        // Achieved effective GOPS at streaming operational intensity: GB/s * OP/Byte
        wp.achievedGops = val * oi;
      } else {
        wp.achievedGops = val * 1000.0;
      }

      // Match applicable compute ceiling
      double targetCompGops = dr.highestComputeGops();
      wp.targetCeilingName = "Peak Compute";
      auto selectCeiling = [&](const char *cname) {
        for (const auto &cc : dr.computeCeilings) {
          if (cc.name == cname && cc.peakGops > 0.0) {
            targetCompGops = cc.peakGops;
            wp.targetCeilingName = cc.name;
            return true;
          }
        }
        return false;
      };

      if (lowName.find("fp64") != std::string::npos) {
        selectCeiling("FP64 Vector");
      } else if (wp.isMatrix) {
        if (lowName.find("fp16") != std::string::npos ||
            lowName.find("bf16") != std::string::npos) {
          if (!selectCeiling("FP16/BF16 Matrix")) selectCeiling("FP16 Vector");
        } else {
          if (!selectCeiling("Low-Precision/Sparse Matrix"))
            selectCeiling("FP16/BF16 Matrix");
        }
      } else if (lowName.find("fp16") != std::string::npos) {
        if (!selectCeiling("FP16 Vector")) selectCeiling("FP32 Vector");
      } else if (lowName.find("bf16") != std::string::npos && !r.isEmulated) {
        if (!selectCeiling("BF16 Vector")) selectCeiling("FP16 Vector");
      } else {
        selectCeiling("FP32 Vector");
      }

      const std::string lowSub = toLowerCopy(r.subcategory);
      double applicableBwGBps = refBwGBps;
      if (lowSub.find("interconnect") != std::string::npos ||
          lowName.find("interconnect") != std::string::npos ||
          lowName.find("h2d") != std::string::npos ||
          lowName.find("d2h") != std::string::npos ||
          lowName.find("rebar") != std::string::npos) {
        if (bwPcie.val > 0.0) applicableBwGBps = bwPcie.val;
      } else if (lowName.find("l0") != std::string::npos && bwL0.val > 0.0) {
        applicableBwGBps = bwL0.val;
      } else if (lowName.find("l1") != std::string::npos && bwL1.val > 0.0) {
        applicableBwGBps = bwL1.val;
      } else if (lowName.find("l2") != std::string::npos && bwL2.val > 0.0) {
        applicableBwGBps = bwL2.val;
      } else if (lowName.find("l3") != std::string::npos && bwL3.val > 0.0) {
        applicableBwGBps = bwL3.val;
      }

      if (applicableBwGBps > 0.0 && targetCompGops > 0.0) {
        const double memBoundGops = oi * applicableBwGBps;
        wp.isMemoryBound = (memBoundGops < targetCompGops);
        wp.rooflineBoundGops = std::min(targetCompGops, memBoundGops);
        if (wp.rooflineBoundGops > 0.0) {
          wp.attainmentPct =
              std::min(100.0, (wp.achievedGops / wp.rooflineBoundGops) * 100.0);
        }
      } else if (targetCompGops > 0.0) {
        wp.isMemoryBound = (r.metric == "GB/s");
        wp.rooflineBoundGops = targetCompGops;
        wp.attainmentPct =
            std::min(100.0, (wp.achievedGops / targetCompGops) * 100.0);
      }

      dr.workloadPoints.push_back(wp);
    }

    if (!dr.bandwidthCeilings.empty() || !dr.computeCeilings.empty()) {
      out.push_back(std::move(dr));
    }
  }
  return out;
}

void RooflineModel::printReport(const std::vector<DeviceRoofline> &rooflines) {
  if (rooflines.empty()) {
    std::cout << "\n[Empirical Roofline] Insufficient bandwidth/compute measurements "
                 "to construct roofline (run -b memory,cache,compute).\n";
    return;
  }

  for (const auto &dr : rooflines) {
    std::cout << "\n========================================================================================\n";
    std::cout << "  EMPIRICAL ROOFLINE MODEL: " << dr.deviceName << " ["
              << dr.backendName << "]\n";
    std::cout << "========================================================================================\n";

    if (!dr.bandwidthCeilings.empty()) {
      std::cout << "\n  Measured Memory Hierarchy Ceilings:\n";
      for (const auto &bw : dr.bandwidthCeilings) {
        std::cout << "    - " << std::left << std::setw(8) << bw.level << ": "
                  << std::right << std::setw(10) << std::fixed
                  << std::setprecision(2) << bw.bandwidthGBps << " GB/s  ("
                  << bw.sourceLabel << ")\n";
      }
    }

    if (!dr.computeCeilings.empty()) {
      std::cout << "\n  Measured Compute Ceilings:\n";
      for (const auto &cc : dr.computeCeilings) {
        std::cout << "    - " << std::left << std::setw(28) << cc.name << ": "
                  << std::right << std::setw(9) << std::fixed
                  << std::setprecision(2) << cc.peakTops << " " << std::left
                  << std::setw(6) << cc.metric << " (" << cc.sourceLabel << ")\n";
      }
    }

    if (!dr.ridgePoints.empty()) {
      std::cout << "\n  Empirical Ridge Points (Arithmetic Intensity Required to Saturate Compute):\n";
      std::cout << "    " << std::left << std::setw(10) << "Memory"
                << std::setw(30) << "Compute Ceiling"
                << std::right << std::setw(18) << "Ridge (OP/Byte)" << "\n";
      std::cout << "    ----------------------------------------------------------\n";
      for (const auto &rp : dr.ridgePoints) {
        std::cout << "    " << std::left << std::setw(10) << rp.memoryLevel
                  << std::setw(30) << rp.computeCeiling << std::right
                  << std::setw(15) << std::fixed << std::setprecision(2)
                  << rp.ridgeOpsPerByte << " OP/B\n";
      }
    }

    if (!dr.workloadPoints.empty()) {
      std::cout << "\n  Workload Roofline Placement:\n";
      std::cout << "    " << std::left << std::setw(36) << "Workload"
                << std::right << std::setw(12) << "OI (OP/B)"
                << std::setw(14) << "Achieved"
                << "   " << std::left << std::setw(14) << "Regime"
                << std::right << std::setw(10) << "Roof Eff\n";
      std::cout << "    ----------------------------------------------------------------------------------------\n";
      for (const auto &wp : dr.workloadPoints) {
        std::ostringstream ach;
        if (wp.achievedGops >= 1000.0) {
          ach << std::fixed << std::setprecision(2) << (wp.achievedGops / 1000.0)
              << (wp.metric == "TOPS" ? " TOPS" : " TFLOPS");
        } else {
          ach << std::fixed << std::setprecision(1) << wp.achievedGops
              << (wp.metric == "TOPS" ? " GOPS" : " GFLOPS");
        }
        std::string shortLabel = wp.label;
        if (shortLabel.size() > 34) shortLabel = shortLabel.substr(0, 31) + "...";
        std::cout << "    " << std::left << std::setw(36) << shortLabel
                  << std::right << std::setw(12) << std::fixed
                  << std::setprecision(2) << wp.operationalIntensity
                  << std::setw(14) << ach.str() << "   " << std::left
                  << std::setw(14)
                  << (wp.isMemoryBound ? "Memory-Bound" : "Compute-Bound");
        if (wp.attainmentPct > 0.0) {
          std::cout << std::right << std::setw(8) << std::fixed
                    << std::setprecision(1) << wp.attainmentPct << "%\n";
        } else {
          std::cout << std::right << std::setw(9) << "-\n";
        }
      }
    }
    std::cout << "========================================================================================\n";
  }
}

} // namespace gpubench
