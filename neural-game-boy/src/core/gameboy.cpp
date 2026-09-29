#include "core/gameboy.h"

namespace gb {

void GameBoy::run_frame() {
  Ppu& ppu = bus.ppu();
  ppu.frame_ready = false;
  const u64 limit = bus.mcycles() + kCyclesPerFrame / 4;
  while (!ppu.frame_ready && bus.mcycles() < limit) cpu.step();
}

void GameBoy::run_mcycles(u64 mcycles) {
  const u64 limit = bus.mcycles() + mcycles;
  while (bus.mcycles() < limit) cpu.step();
}

}  // namespace gb
