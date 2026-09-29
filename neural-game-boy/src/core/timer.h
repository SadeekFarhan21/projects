#pragma once
#include "core/types.h"

namespace gb {

// DIV/TIMA/TMA/TAC.
//
// The hardware keeps a 16-bit counter that increments every T-cycle; DIV is
// its upper byte. TIMA increments on the falling edge of
// (TAC.enable AND counter bit selected by TAC). Because the emulator steps
// in M-cycles (4 T-cycles) and every selectable bit is >= bit 3, one M-cycle
// can produce at most one falling edge, so a single before/after comparison
// per M-cycle is exact.
//
// Overflow: TIMA reads 0x00 for one M-cycle, then TMA is loaded and the timer
// interrupt is requested. A TIMA write during the zero cycle cancels the
// reload; a TIMA write during the reload cycle is ignored; a TMA write during
// the reload cycle also lands in TIMA.
class Timer {
 public:
  explicit Timer(u8& if_reg) : if_(if_reg) {}

  void reset(u16 div_counter) {
    counter_ = div_counter;
    tima_ = tma_ = 0;
    tac_ = 0xF8;
    overflow_pending_ = reloading_ = false;
  }

  // Advance one M-cycle. Returns the counter before the step (used by the
  // serial port, which is clocked from the same divider).
  u16 tick() {
    reloading_ = false;
    if (overflow_pending_) {
      overflow_pending_ = false;
      tima_ = tma_;
      if_ |= kIntTimer;
      reloading_ = true;
    }
    u16 old = counter_;
    counter_ = u16(counter_ + 4);
    if ((tac_ & 0x04) && (old & mask()) && !(counter_ & mask())) increment();
    return old;
  }

  u8 read(u16 addr) const;
  void write(u16 addr, u8 value);

  u16 counter() const { return counter_; }

 private:
  u16 mask() const {
    static constexpr u16 kBits[4] = {1u << 9, 1u << 3, 1u << 5, 1u << 7};
    return kBits[tac_ & 3];
  }
  bool signal() const { return (tac_ & 0x04) && (counter_ & mask()); }
  void increment() {
    if (tima_ == 0xFF) {
      tima_ = 0;
      overflow_pending_ = true;
    } else {
      ++tima_;
    }
  }

  u8& if_;
  u16 counter_ = 0;
  u8 tima_ = 0, tma_ = 0, tac_ = 0xF8;
  bool overflow_pending_ = false;  // TIMA overflowed this M-cycle
  bool reloading_ = false;         // TMA was copied to TIMA this M-cycle
};

}  // namespace gb
