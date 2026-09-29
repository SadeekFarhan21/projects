#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace gb::png {

struct Gray {
  int width = 0, height = 0;
  std::vector<std::uint8_t> pixels;  // 8-bit luminance, row major
};

// Minimal PNG reader: non-interlaced, bit depth <= 8, color types gray,
// RGB, RGBA, gray+alpha and palette. Enough for the test-suite reference
// images. Throws std::runtime_error on anything else.
Gray read_gray(const std::string& path);

// Writes an 8-bit grayscale PNG.
void write_gray(const std::string& path, int width, int height, const std::uint8_t* pixels);

// Shade 0..3 (0 = white) to the gray levels used by the dmg-acid2 reference.
inline std::uint8_t shade_to_gray(std::uint8_t s) {
  static constexpr std::uint8_t k[4] = {0xFF, 0xAA, 0x55, 0x00};
  return k[s & 3];
}

}  // namespace gb::png
