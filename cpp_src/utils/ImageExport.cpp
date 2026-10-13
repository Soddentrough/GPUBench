#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third_party/stb_image_write.h"
#include "utils/ImageExport.h"
#include <fstream>
#include <vector>
#include <cstdio>

namespace gpubench {

bool ImageExport::writePNG(const std::string &path, uint32_t width, uint32_t height,
                           const std::vector<uint8_t> &rgbData) {
  size_t expectedBytes = static_cast<size_t>(width) * static_cast<size_t>(height) * 3;
  if (width == 0 || height == 0 || rgbData.size() < expectedBytes) return false;
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

  auto skipWhitespaceAndComments = [](std::ifstream &in) {
    while (in.good()) {
      int ch = in.peek();
      if (ch == EOF) break;
      if (ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n') {
        in.get();
      } else if (ch == '#') {
        std::string dummy;
        std::getline(in, dummy);
      } else {
        break;
      }
    }
  };

  std::string magic;
  uint32_t width = 0, height = 0, maxVal = 0;
  skipWhitespaceAndComments(file);
  file >> magic;
  skipWhitespaceAndComments(file);
  file >> width;
  skipWhitespaceAndComments(file);
  file >> height;
  skipWhitespaceAndComments(file);
  file >> maxVal;
  file.get(); // skip single trailing whitespace character after maxVal

  bool wrotePng = false;
  if (magic == "P6" && width > 0 && height > 0 && maxVal == 255) {
    size_t numBytes = static_cast<size_t>(width) * static_cast<size_t>(height) * 3;
    std::vector<uint8_t> buffer(numBytes);
    file.read(reinterpret_cast<char *>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
    if (static_cast<size_t>(file.gcount()) == numBytes) {
      wrotePng = writePNG(pngPath, width, height, buffer);
    }
  }
  file.close();
  if (wrotePng) {
    std::remove(ppmPath.c_str());
  }
}

} // namespace gpubench
