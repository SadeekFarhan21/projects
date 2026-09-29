#include "core/bus.h"

#include <cstdlib>

namespace gb {

namespace {
// Bits that always read as 1 in FF10-FF3F (sound registers). v0 has no
// synthesis, but the register file behaves like hardware so software that
// polls it (and the Mooneye register tests) sees the right values.
constexpr u8 kApuReadMask[0x30] = {
    0x80, 0x3F, 0x00, 0xFF, 0xBF,                    // NR10-NR14
    0xFF, 0x3F, 0x00, 0xFF, 0xBF,                    // (FF15) NR21-NR24
    0x7F, 0xFF, 0x9F, 0xFF, 0xBF,                    // NR30-NR34
    0xFF, 0xFF, 0x00, 0x00, 0xBF,                    // (FF1F) NR41-NR44
    0x00, 0x00, 0x70,                                // NR50 NR51 NR52
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,  // FF27-FF2F
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // wave RAM
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
};
constexpr u8 kApuBoot[0x17] = {0x80, 0xBF, 0xF3, 0xFF, 0xBF, 0xFF, 0x3F, 0x00,
                               0xFF, 0xBF, 0x7F, 0xFF, 0x9F, 0xFF, 0xBF, 0xFF,
                               0xFF, 0x00, 0x00, 0xBF, 0x77, 0xF3, 0xF1};
}  // namespace

Bus::Bus(std::unique_ptr<Cartridge> cart) : cart_(std::move(cart)) { reset_post_boot(); }

void Bus::reset_post_boot() {
  // State of a DMG (CPU revision A/B/C) right after the boot ROM, since we do
  // not ship the copyrighted boot ROM.
  if_ = 0xE1;
  ie_ = 0x00;
  timer_.reset(0xABCC);
  if (const char* e = std::getenv("GB_DIV_INIT")) timer_.reset(u16(std::strtol(e, nullptr, 0)));
  ppu_.reset_post_boot();
  wram_.fill(0);
  hram_.fill(0);
  apu_.fill(0);
  for (int i = 0; i < 0x17; ++i) apu_[i] = kApuBoot[i];
  p1_select_ = 0x00;
  sb_ = 0;
  sc_ = 0;
  serial_active_ = false;
  serial_bits_ = 0;
  dma_reg_ = 0xFF;
  dma_active_ = false;
  dma_start_delay_ = 0;
  mcycles_ = 0;
}

void Bus::set_buttons(u8 buttons) {
  // A 1->0 transition on any selected P1 input line requests the joypad IRQ.
  u8 before = read_io(0xFF00) & 0x0F;
  buttons_ = buttons;
  u8 after = read_io(0xFF00) & 0x0F;
  if (before & ~after) if_ |= kIntJoypad;
}

u8 Bus::read_high(u16 addr) const {
  if (addr < 0xFEA0) {
    if (dma_active_ || !ppu_.oam_accessible()) return 0xFF;
    return ppu_.oam[addr - 0xFE00];
  }
  if (addr < 0xFF00) return (dma_active_ || !ppu_.oam_accessible()) ? 0xFF : 0x00;
  if (addr < 0xFF80) return read_io(addr);
  if (addr < 0xFFFF) return hram_[addr - 0xFF80];
  return ie_;
}

void Bus::write_high(u16 addr, u8 v) {
  if (addr < 0xFEA0) {
    if (dma_active_ || !ppu_.oam_accessible()) return;
    ppu_.oam[addr - 0xFE00] = v;
    return;
  }
  if (addr < 0xFF00) return;
  if (addr < 0xFF80) {
    write_io(addr, v);
    return;
  }
  if (addr < 0xFFFF) {
    hram_[addr - 0xFF80] = v;
    return;
  }
  ie_ = v;
}

u8 Bus::read_io(u16 addr) const {
  switch (addr) {
    case 0xFF00: {
      u8 r = u8(0xC0 | (p1_select_ & 0x30) | 0x0F);
      if (!(p1_select_ & 0x10)) r &= u8(~((buttons_ >> 4) & 0x0F));  // d-pad
      if (!(p1_select_ & 0x20)) r &= u8(~(buttons_ & 0x0F));         // buttons
      return r;
    }
    case 0xFF01: return sb_;
    case 0xFF02: return u8(sc_ | 0x7E);
    case 0xFF04: case 0xFF05: case 0xFF06: case 0xFF07: return timer_.read(addr);
    case 0xFF0F: return u8(if_ | 0xE0);
    case 0xFF46: return dma_reg_;
    default: break;
  }
  if (addr >= 0xFF10 && addr < 0xFF40) {
    u8 i = u8(addr - 0xFF10);
    return u8(apu_[i] | kApuReadMask[i]);
  }
  if (addr >= 0xFF40 && addr <= 0xFF4B) return ppu_.read_reg(addr);
  return 0xFF;
}

void Bus::write_io(u16 addr, u8 v) {
  switch (addr) {
    case 0xFF00: p1_select_ = v & 0x30; return;
    case 0xFF01: sb_ = v; return;
    case 0xFF02:
      sc_ = v & 0x81;
      if ((v & 0x81) == 0x81) {
        // Internal clock transfer. No link partner: 0xFF shifts in.
        serial_output.push_back(char(sb_));
        serial_active_ = true;
        serial_bits_ = 0;
      } else {
        serial_active_ = false;
      }
      return;
    case 0xFF04: case 0xFF05: case 0xFF06: case 0xFF07: timer_.write(addr, v); return;
    case 0xFF0F: if_ = v & 0x1F; return;
    case 0xFF46:
      dma_reg_ = v;
      dma_pending_src_ = u16(v << 8);
      dma_start_delay_ = 2;
      return;
    default: break;
  }
  if (addr >= 0xFF10 && addr < 0xFF40) {
    u8 i = u8(addr - 0xFF10);
    if (addr == 0xFF26) {
      if (!(v & 0x80)) {
        for (int k = 0; k < 0x16; ++k) apu_[k] = 0;  // power off clears registers
        apu_[0x16] = 0;
      } else {
        apu_[0x16] = 0x80 | (apu_[0x16] & 0x0F);
      }
      return;
    }
    if (addr < 0xFF30 && !(apu_[0x16] & 0x80)) return;  // ignored while off
    apu_[i] = v;
    return;
  }
  if (addr >= 0xFF40 && addr <= 0xFF4B) ppu_.write_reg(addr, v);
}

void Bus::dma_step() {
  // Timeline for a write to FF46 in M-cycle N: N+1 is a setup cycle (OAM
  // still readable), bytes 0..159 move in cycles N+2..N+161 while OAM is
  // blocked, and OAM is readable again in N+162. A restart keeps the old
  // transfer running (and OAM blocked) through the new setup cycle.
  if (dma_active_ && dma_index_ == 0xA0) dma_active_ = false;
  if (dma_start_delay_ > 0 && --dma_start_delay_ == 0) {
    dma_active_ = true;
    dma_src_ = dma_pending_src_;
    dma_index_ = 0;
  }
  if (dma_active_ && dma_index_ < 0xA0) {
    u16 src = u16(dma_src_ + dma_index_);
    if (src >= 0xE000) src = u16(src - 0x2000);  // echo region for FE/FF sources
    u8 byte;
    switch (src >> 12) {
      case 0x8: case 0x9: byte = ppu_.vram[src & 0x1FFF]; break;
      case 0xC: case 0xD: byte = wram_[src & 0x1FFF]; break;
      default: byte = (src < 0x8000 || (src >= 0xA000 && src < 0xC000)) ? cart_->read(src) : 0xFF;
    }
    ppu_.oam[dma_index_] = byte;
    ++dma_index_;
  }
}

void Bus::serial_shift() {
  sb_ = u8((sb_ << 1) | 1);
  if (++serial_bits_ == 8) {
    serial_active_ = false;
    sc_ &= 0x7F;
    if_ |= kIntSerial;
  }
}

}  // namespace gb
