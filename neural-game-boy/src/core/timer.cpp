#include "core/timer.h"

namespace gb {

u8 Timer::read(u16 addr) const {
  switch (addr) {
    case 0xFF04: return u8(counter_ >> 8);
    case 0xFF05: return tima_;
    case 0xFF06: return tma_;
    case 0xFF07: return u8(tac_ | 0xF8);
  }
  return 0xFF;
}

void Timer::write(u16 addr, u8 value) {
  switch (addr) {
    case 0xFF04: {
      // Resetting the divider can itself produce a falling edge.
      bool before = signal();
      counter_ = 0;
      if (before) increment();
      break;
    }
    case 0xFF05:
      if (reloading_) break;         // TMA wins during the reload cycle
      tima_ = value;
      overflow_pending_ = false;      // cancels a pending reload
      break;
    case 0xFF06:
      tma_ = value;
      if (reloading_) tima_ = value;
      break;
    case 0xFF07: {
      bool before = signal();
      tac_ = u8(value | 0xF8);
      if (before && !signal()) increment();  // DMG glitch on TAC change
      break;
    }
  }
}

}  // namespace gb
