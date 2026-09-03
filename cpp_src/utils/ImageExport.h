#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace gpubench {

struct ImageMetrics {
  float maxDelta = 0.0f;     // Max channel delta [0, 1]
  float mae = 0.0f;          // Mean Absolute Error
  float rmse = 0.0f;         // Root Mean Square Error
  float psnr = 0.0f;         // Peak Signal-to-Noise Ratio (dB)
  uint32_t exactPixels = 0;  // Count of 100% bit-exact pixels
  uint32_t diffPixels = 0;   // Count of pixels with delta > 1/255
  uint32_t totalPixels = 0;
};

class ImageExport {
public:
  static inline uint8_t floatToSrgb(float x) {
    if (x <= 0.0f) return 0;
    // Simple exposure & Reinhard tonemapping
    float mapped = x / (1.0f + x);
    // Gamma 2.2
    float gamma = std::pow(mapped, 1.0f / 2.2f);
    int val = static_cast<int>(gamma * 255.0f + 0.5f);
    return static_cast<uint8_t>(std::clamp(val, 0, 255));
  }

  static bool writePPM(const std::string &path, uint32_t width, uint32_t height,
                       const std::vector<uint8_t> &rgbData) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) return false;

    file << "P6\n" << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<const char *>(rgbData.data()), rgbData.size());
    return file.good();
  }

  static void convertPPMtoPNG(const std::string &ppmPath, const std::string &pngPath) {
    std::string cmd = "python3 -c \"from PIL import Image; Image.open('" + ppmPath + "').save('" + pngPath + "')\" 2>/dev/null";
    int ret = std::system(cmd.c_str());
    (void)ret;
  }

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

    for (uint32_t i = 0; i < numPixels; ++i) {
      float rA = hdrA[i * 4 + 0];
      float gA = hdrA[i * 4 + 1];
      float bA = hdrA[i * 4 + 2];

      float rB = hdrB[i * 4 + 0];
      float gB = hdrB[i * 4 + 1];
      float bB = hdrB[i * 4 + 2];

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

      // Differences in tonemapped [0, 1] domain
      float dR = std::abs(float(uR_A) - float(uR_B)) / 255.0f;
      float dG = std::abs(float(uG_A) - float(uG_B)) / 255.0f;
      float dB = std::abs(float(uB_A) - float(uB_B)) / 255.0f;

      float maxD = std::max({dR, dG, dB});
      metrics.maxDelta = std::max(metrics.maxDelta, maxD);

      float pixAbs = (dR + dG + dB) / 3.0f;
      sumAbsDiff += pixAbs;

      float pixSq = (dR * dR + dG * dG + dB * dB) / 3.0f;
      sumSqDiff += pixSq;

      if (maxD == 0.0f) {
        metrics.exactPixels++;
      }
      if (maxD > (1.0f / 255.0f)) {
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

    metrics.mae = static_cast<float>(sumAbsDiff / numPixels);
    metrics.rmse = static_cast<float>(std::sqrt(sumSqDiff / numPixels));

    if (metrics.rmse < 1e-7f) {
      metrics.psnr = 120.0f; // Perfect floating point match
    } else {
      metrics.psnr = static_cast<float>(20.0 * std::log10(1.0 / metrics.rmse));
    }

    return metrics;
  }
};

} // namespace gpubench
