#pragma once
#include <cstdint>

namespace gb {

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using i8 = std::int8_t;

inline constexpr int kScreenW = 160;
inline constexpr int kScreenH = 144;
inline constexpr int kCyclesPerFrame = 70224;  // T-cycles (dots) per frame
inline constexpr int kClockHz = 4194304;       // T-cycles per second
inline constexpr double kFramesPerSecond = double(kClockHz) / kCyclesPerFrame;

// Interrupt bits in IF / IE.
enum Interrupt : u8 {
  kIntVBlank = 0x01,
  kIntStat = 0x02,
  kIntTimer = 0x04,
  kIntSerial = 0x08,
  kIntJoypad = 0x10,
};

// Button bitmask used everywhere (joypad, logger actions). 1 = pressed.
enum Button : u8 {
  kBtnA = 0x01,
  kBtnB = 0x02,
  kBtnSelect = 0x04,
  kBtnStart = 0x08,
  kBtnRight = 0x10,
  kBtnLeft = 0x20,
  kBtnUp = 0x40,
  kBtnDown = 0x80,
};

}  // namespace gb
