#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace gpubench {

struct ImageMetrics {
  float maxDelta = 0.0f;         // Max 8-bit sRGB channel delta [0, 1]
  float mae = 0.0f;              // 8-bit sRGB Mean Absolute Error [0, 1]
  float rmse = 0.0f;             // 8-bit sRGB Root Mean Square Error [0, 1]
  float psnr = 0.0f;             // 8-bit sRGB Peak Signal-to-Noise Ratio (dB)
  uint32_t exactPixels = 0;      // Count of 8-bit sRGB exact pixels (0 LSB delta)
  uint32_t ldr1LsbPixels = 0;    // Count of 8-bit sRGB pixels differing by exactly 1 LSB
  uint32_t diffPixels = 0;       // Count of 8-bit sRGB pixels differing by > 1 LSB
  uint32_t totalPixels = 0;
  float hdrRmse = 0.0f;          // Raw linear FP32 HDR Root Mean Square Error
  float hdrMaxAbsDelta = 0.0f;   // Raw linear FP32 HDR Max Absolute Delta
  float hdrMaxRelDelta = 0.0f;   // Raw linear FP32 HDR Max Relative Delta
  uint32_t hdrExactPixels = 0;   // Count of bit-identical linear FP32 HDR pixels
};

class ImageExport {
public:
  static uint8_t floatToSrgb(float linear) {
    linear = std::clamp(linear, 0.0f, 1.0f);
    if (linear <= 0.0031308f) {
      return static_cast<uint8_t>(linear * 12.92f * 255.0f + 0.5f);
    }
    return static_cast<uint8_t>((1.055f * std::pow(linear, 1.0f / 2.4f) - 0.055f) * 255.0f + 0.5f);
  }

  static bool writePPM(const std::string &path, uint32_t width, uint32_t height,
                       const std::vector<uint8_t> &rgbData) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) return false;

    file << "P6\n" << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<const char *>(rgbData.data()), rgbData.size());
    return file.good();
  }

  static bool writePNG(const std::string &path, uint32_t width, uint32_t height,
                       const std::vector<uint8_t> &rgbData);

  static void convertPPMtoPNG(const std::string &ppmPath, const std::string &pngPath,
                              const std::string &profileJsonPath = "", const std::string &type = "traditional",
                              bool annotate = false);

  static ImageMetrics compareAndTonemap(
      const float *hdrA, const float *hdrB,
      uint32_t width, uint32_t height,
      std::vector<uint8_t> &outLdrA,
      std::vector<uint8_t> &outLdrB,
      std::vector<uint8_t> &outDiffHeatmap) {

    uint32_t numPixels = width * height;
    outLdrA.resize(numPixels * 3);
    outLdrB.resize(numPixels * 3);
    outDiffHeatmap.resize(numPixels * 3);

    ImageMetrics metrics;
    metrics.totalPixels = numPixels;

    double sumSqDiff = 0.0;
    double sumAbsDiff = 0.0;
    double sumHdrSqDiff = 0.0;

    for (uint32_t i = 0; i < numPixels; ++i) {
      float rA = hdrA[i * 4 + 0];
      float gA = hdrA[i * 4 + 1];
      float bA = hdrA[i * 4 + 2];

      float rB = hdrB[i * 4 + 0];
      float gB = hdrB[i * 4 + 1];
      float bB = hdrB[i * 4 + 2];

      // Raw linear FP32 HDR comparison prior to 8-bit sRGB quantization
      double hdrDr = static_cast<double>(rA) - static_cast<double>(rB);
      double hdrDg = static_cast<double>(gA) - static_cast<double>(gB);
      double hdrDb = static_cast<double>(bA) - static_cast<double>(bB);
      if (std::isfinite(rA) && std::isfinite(gA) && std::isfinite(bA) &&
          std::isfinite(rB) && std::isfinite(gB) && std::isfinite(bB)) {
        sumHdrSqDiff += (hdrDr * hdrDr + hdrDg * hdrDg + hdrDb * hdrDb) / 3.0;
        float absPix = std::max({std::abs(rA - rB), std::abs(gA - gB), std::abs(bA - bB)});
        float denom = std::max({std::abs(rA), std::abs(gA), std::abs(bA),
                                std::abs(rB), std::abs(gB), std::abs(bB), 1e-4f});
        metrics.hdrMaxAbsDelta = std::max(metrics.hdrMaxAbsDelta, absPix);
        metrics.hdrMaxRelDelta = std::max(metrics.hdrMaxRelDelta, absPix / denom);
      }
      if (rA == rB && gA == gB && bA == bB) {
        metrics.hdrExactPixels++;
      }

      uint8_t uR_A = floatToSrgb(rA);
      uint8_t uG_A = floatToSrgb(gA);
      uint8_t uB_A = floatToSrgb(bA);

      uint8_t uR_B = floatToSrgb(rB);
      uint8_t uG_B = floatToSrgb(gB);
      uint8_t uB_B = floatToSrgb(bB);

      outLdrA[i * 3 + 0] = uR_A;
      outLdrA[i * 3 + 1] = uG_A;
      outLdrA[i * 3 + 2] = uB_A;

      outLdrB[i * 3 + 0] = uR_B;
      outLdrB[i * 3 + 1] = uG_B;
      outLdrB[i * 3 + 2] = uB_B;

      // Differences in 8-bit sRGB quantized [0, 1] domain
      int dR_i = std::abs(int(uR_A) - int(uR_B));
      int dG_i = std::abs(int(uG_A) - int(uG_B));
      int dB_i = std::abs(int(uB_A) - int(uB_B));
      int maxDiffLdr = std::max({dR_i, dG_i, dB_i});

      float dR = float(dR_i) / 255.0f;
      float dG = float(dG_i) / 255.0f;
      float dB = float(dB_i) / 255.0f;

      float maxD = std::max({dR, dG, dB});
      metrics.maxDelta = std::max(metrics.maxDelta, maxD);

      float pixAbs = (dR + dG + dB) / 3.0f;
      sumAbsDiff += pixAbs;

      float pixSq = (dR * dR + dG * dG + dB * dB) / 3.0f;
      sumSqDiff += pixSq;

      if (maxDiffLdr == 0) {
        metrics.exactPixels++;
      } else if (maxDiffLdr == 1) {
        metrics.ldr1LsbPixels++;
      } else {
        metrics.diffPixels++;
      }

      // Generate 10x amplified difference heatmap (Pure black for zero difference, heat color for deviations)
      float heat = std::min(1.0f, maxD * 10.0f);
      if (heat == 0.0f) {
        // Zero difference: pure black
        outDiffHeatmap[i * 3 + 0] = 0;
        outDiffHeatmap[i * 3 + 1] = 0;
        outDiffHeatmap[i * 3 + 2] = 0;
      } else {
        // False color heatmap for differences:
        // heat in (0, 1]: Blue (subtle) -> Cyan -> Yellow -> Red (large deviation)
        outDiffHeatmap[i * 3 + 0] = static_cast<uint8_t>(std::clamp(heat * 2.0f, 0.0f, 1.0f) * 255.0f);
        outDiffHeatmap[i * 3 + 1] = static_cast<uint8_t>(std::clamp(2.0f - heat * 2.0f, 0.0f, 1.0f) * 255.0f);
        outDiffHeatmap[i * 3 + 2] = static_cast<uint8_t>((1.0f - heat) * 255.0f);
      }
    }

    const double safePixels = (numPixels > 0u) ? static_cast<double>(numPixels) : 1.0;
    metrics.mae = static_cast<float>(sumAbsDiff / safePixels);
    metrics.rmse = static_cast<float>(std::sqrt(sumSqDiff / safePixels));
    metrics.hdrRmse = static_cast<float>(std::sqrt(sumHdrSqDiff / safePixels));

    if (metrics.rmse < 1e-7f) {
      metrics.psnr = 120.0f; // Quantized 8-bit sRGB zero-delta ceiling
    } else {
      metrics.psnr = static_cast<float>(20.0 * std::log10(1.0 / metrics.rmse));
    }

    return metrics;
  }
};

} // namespace gpubench
