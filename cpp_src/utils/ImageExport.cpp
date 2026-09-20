#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third_party/stb_image_write.h"
#include "utils/ImageExport.h"
#include <fstream>
#include <vector>
#include <cstdio>

namespace gpubench {

bool ImageExport::writePNG(const std::string &path, uint32_t width, uint32_t height,
                           const std::vector<uint8_t> &rgbData) {
  if (rgbData.size() < width * height * 3) return false;
  return stbi_write_png(path.c_str(), static_cast<int>(width), static_cast<int>(height),
                        3, rgbData.data(), static_cast<int>(width * 3)) != 0;
}

void ImageExport::convertPPMtoPNG(const std::string &ppmPath, const std::string &pngPath,
                                  const std::string &profileJsonPath, const std::string &type,
                                  bool annotate) {
  (void)profileJsonPath;
  (void)type;
  (void)annotate;
  std::ifstream file(ppmPath, std::ios::binary);
  if (!file.is_open()) return;
  std::string magic;
  uint32_t width = 0, height = 0, maxVal = 0;
  file >> magic >> width >> height >> maxVal;
  file.get(); // skip trailing newline / whitespace
  if (magic == "P6" && width > 0 && height > 0 && maxVal == 255) {
    std::vector<uint8_t> buffer(width * height * 3);
    file.read(reinterpret_cast<char *>(buffer.data()), buffer.size());
    writePNG(pngPath, width, height, buffer);
  }
  file.close();
  std::remove(ppmPath.c_str());
}

} // namespace gpubench
