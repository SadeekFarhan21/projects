#pragma once
#include <array>
#include <string>

#include "core/types.h"

namespace gb {

// Runners for the public hardware test suites. Each returns a verdict plus
// enough detail to debug a failure.
struct TestVerdict {
  bool passed = false;
  bool timed_out = false;
  double emu_seconds = 0;
  std::string detail;
  std::array<u8, kScreenW * kScreenH> frame{};  // last frame (for screenshot tests)
};

// Blargg: the ROM prints its result over the serial port; we look for
// "Passed" or "Failed" in that text.
TestVerdict run_blargg(const std::string& rom_path, double max_seconds = 120);

// Mooneye: the ROM executes LD B,B when done; success leaves the Fibonacci
// numbers 3 5 8 13 21 34 in B C D E H L.
TestVerdict run_mooneye(const std::string& rom_path, double max_seconds = 60);

// Screenshot tests (dmg-acid2): run until LD B,B, then a few more frames so
// the final image is on screen. `passed` is left false; the caller compares
// `frame` with the reference image.
TestVerdict run_until_ld_b_b(const std::string& rom_path, double max_seconds = 20, int extra_frames = 3);

}  // namespace gb
