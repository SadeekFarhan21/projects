#pragma once
#include <array>
#include <memory>
#include <string>

#include "core/cartridge.h"
#include "core/ppu.h"
#include "core/timer.h"
#include "core/types.h"

namespace gb {

// The memory bus and everything that hangs off it except the CPU:
// cartridge, work RAM, HRAM, IO registers, timer, PPU, OAM DMA, joypad,
// serial port and an APU register file (no sound synthesis in v0).
//
// Clocking contract: the CPU calls tick() exactly once per M-cycle, before
// performing that cycle's memory access (if any). Every component therefore
// sees time advance in lock step with CPU bus activity, which is what the
// Mooneye timing tests measure.
class Bus {
 public:
  explicit Bus(std::unique_ptr<Cartridge> cart);

  void reset_post_boot();

  // One M-cycle of every non-CPU component.
  inline void tick() {
    ++mcycles_;
    u16 old = timer_.tick();
    if (serial_active_ && (old & 0x100) && !(timer_.counter() & 0x100)) serial_shift();
    u8 before = if_;
    ppu_.tick();
    ppu_late_if_ = u8(if_ & ~before);
    if (dma_active_ || dma_start_delay_) dma_step();
  }

  // Untimed access (the CPU wraps these with tick()).
  inline u8 read(u16 addr) const {
    switch (addr >> 12) {
      case 0x0: case 0x1: case 0x2: case 0x3:
      case 0x4: case 0x5: case 0x6: case 0x7:
        return cart_->read(addr);
      case 0x8: case 0x9:
        return ppu_.vram_accessible() ? ppu_.vram[addr & 0x1FFF] : 0xFF;
      case 0xA: case 0xB:
        return cart_->read(addr);
      case 0xC: case 0xD:
        return wram_[addr & 0x1FFF];
      case 0xE:
        return wram_[addr & 0x1FFF];
      default:
        if (addr < 0xFE00) return wram_[addr & 0x1FFF];
        return read_high(addr);
    }
  }
  inline void write(u16 addr, u8 v) {
    switch (addr >> 12) {
      case 0x0: case 0x1: case 0x2: case 0x3:
      case 0x4: case 0x5: case 0x6: case 0x7:
        cart_->write(addr, v);
        return;
      case 0x8: case 0x9:
        if (ppu_.vram_accessible()) ppu_.vram[addr & 0x1FFF] = v;
        return;
      case 0xA: case 0xB:
        cart_->write(addr, v);
        return;
      case 0xC: case 0xD: case 0xE:
        wram_[addr & 0x1FFF] = v;
        return;
      default:
        if (addr < 0xFE00) {
          wram_[addr & 0x1FFF] = v;
          return;
        }
        write_high(addr, v);
    }
  }

  u8 ie() const { return ie_; }
  u8& if_reg() { return if_; }
  u8 pending_interrupts() const { return u8(ie_ & if_ & 0x1F); }
  // What the CPU's interrupt sampler sees during the current M-cycle. The
  // PPU raises its requests at the end of the 4 dots it just ran, which is
  // after the sample point, so those bits only count from the next cycle.
  u8 sampled_interrupts() const { return u8(ie_ & if_ & ~ppu_late_if_ & 0x1F); }

  void set_buttons(u8 buttons);
  u8 buttons() const { return buttons_; }

  Ppu& ppu() { return ppu_; }
  const Ppu& ppu() const { return ppu_; }
  Timer& timer() { return timer_; }
  Cartridge& cart() { return *cart_; }
  u64 mcycles() const { return mcycles_; }

  // Bytes the program sent over the serial port (Blargg tests print here).
  std::string serial_output;

 private:
  u8 read_high(u16 addr) const;
  void write_high(u16 addr, u8 v);
  u8 read_io(u16 addr) const;
  void write_io(u16 addr, u8 v);
  void dma_step();
  void serial_shift();

  std::unique_ptr<Cartridge> cart_;
  u8 if_ = 0xE1;
  u8 ie_ = 0x00;
  Timer timer_{if_};
  Ppu ppu_{if_};
  std::array<u8, 0x2000> wram_{};
  std::array<u8, 0x7F> hram_{};
  std::array<u8, 0x30> apu_{};  // FF10-FF3F, raw values
  u64 mcycles_ = 0;
  u8 ppu_late_if_ = 0;  // IF bits the PPU raised during the latest tick

  // Joypad.
  u8 p1_select_ = 0x30;
  u8 buttons_ = 0;

  // Serial.
  u8 sb_ = 0, sc_ = 0;
  bool serial_active_ = false;
  int serial_bits_ = 0;

  // OAM DMA.
  u8 dma_reg_ = 0xFF;
  bool dma_active_ = false;
  int dma_start_delay_ = 0;
  u16 dma_src_ = 0, dma_pending_src_ = 0;
  int dma_index_ = 0;
};

}  // namespace gb
