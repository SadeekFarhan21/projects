#pragma once
#include <memory>
#include <string>

#include "core/bus.h"
#include "core/cartridge.h"
#include "core/cpu.h"

namespace gb {

// Owns the whole machine. Starts in the documented DMG post-boot state
// (PC = 0x0100) because the boot ROM is not redistributable.
class GameBoy {
 public:
  explicit GameBoy(std::unique_ptr<Cartridge> cart) : bus(std::move(cart)), cpu(bus) {}
  static std::unique_ptr<GameBoy> from_file(const std::string& path) {
    return std::make_unique<GameBoy>(Cartridge::from_file(path));
  }

  // Run until the PPU enters VBlank (one video frame). With the LCD off it
  // returns after one frame's worth of cycles instead so callers never stall.
  void run_frame();

  // Run for at least `mcycles` M-cycles.
  void run_mcycles(u64 mcycles);

  void set_buttons(u8 buttons) { bus.set_buttons(buttons); }
  const std::array<u8, kScreenW * kScreenH>& framebuffer() const { return bus.ppu().framebuffer(); }
  double emulated_seconds() const { return double(bus.mcycles()) * 4.0 / kClockHz; }

  Bus bus;
  Cpu cpu;
};

}  // namespace gb
