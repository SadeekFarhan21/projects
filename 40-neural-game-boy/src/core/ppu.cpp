#include "core/ppu.h"

#include <algorithm>
#include <cstdlib>

namespace gb {

namespace {
constexpr int kBootDot = 400;
}

namespace {
struct SpriteList {
  int idx[10];
  int n = 0;
};

// OAM scan: the first 10 objects (in OAM order) that overlap line `ly`,
// then ordered by DMG drawing priority (lower X first, ties by OAM index).
SpriteList select_sprites(const std::array<u8, 0xA0>& oam, int ly, int height) {
  SpriteList s;
  for (int i = 0; i < 40 && s.n < 10; ++i) {
    int sy = oam[i * 4];
    if (ly + 16 >= sy && ly + 16 < sy + height) s.idx[s.n++] = i;
  }
  std::stable_sort(s.idx, s.idx + s.n, [&](int a, int b) { return oam[a * 4 + 1] < oam[b * 4 + 1]; });
  return s;
}
}  // namespace

void Ppu::reset_post_boot() {
  lcdc_ = 0x91;
  stat_ = 0x00;
  scy_ = scx_ = 0;
  lyc_ = 0;
  bgp_ = 0xFC;
  obp0_ = obp1_ = 0xFF;
  wy_ = wx_ = 0;
  // The DMG boot ROM hands over during line 153 after the LY=0 quirk has
  // kicked in: LY reads 0 and STAT reads 0x85 (VBlank, LY=LYC).
  line_ = 153;
  ly_ = 0;
  dot_ = kBootDot;
  state_ = State::VBlank;
  mode_ = 1;
  next_event_ = 456;
  if (const char* e = std::getenv("GB_PPU_DOT")) dot_ = std::atoi(e);
  stat_line_ = false;
  first_line_ = false;
  wy_hit_ = false;
  window_line_ = 0;
  update_coincidence();
  fb_.fill(0);
}

u8 Ppu::read_reg(u16 addr) const {
  switch (addr) {
    case 0xFF40: return lcdc_;
    case 0xFF41: return u8(0x80 | (stat_ & 0x78) | (coincidence_ ? 0x04 : 0) | mode());
    case 0xFF42: return scy_;
    case 0xFF43: return scx_;
    case 0xFF44: return ly_;
    case 0xFF45: return lyc_;
    case 0xFF47: return bgp_;
    case 0xFF48: return obp0_;
    case 0xFF49: return obp1_;
    case 0xFF4A: return wy_;
    case 0xFF4B: return wx_;
  }
  return 0xFF;
}

void Ppu::write_reg(u16 addr, u8 v) {
  switch (addr) {
    case 0xFF40: {
      bool was_on = lcdc_ & 0x80;
      lcdc_ = v;
      if (was_on && !(v & 0x80)) {
        // LCD off: LY resets, mode reads 0, the screen goes blank (white).
        line_ = 0;
        ly_ = 0;
        dot_ = 0;
        mode_ = 0;
        state_ = State::HBlank;
        stat_line_ = false;
        fb_.fill(0);
      } else if (!was_on && (v & 0x80)) {
        // LCD on: restart at line 0. The first line reports mode 0 where
        // mode 2 would be.
        line_ = 0;
        ly_ = 0;
        dot_ = 0;
        state_ = State::OamScan;
        mode_ = 0;
        first_line_ = true;
        next_event_ = 80;
        wy_hit_ = false;
        window_line_ = 0;
        update_coincidence();
        update_stat_line();
      }
      break;
    }
    case 0xFF41:
      stat_ = v & 0x78;
      if (lcd_on()) update_stat_line();
      break;
    case 0xFF42: scy_ = v; break;
    case 0xFF43: scx_ = v; break;
    case 0xFF44: break;  // read only
    case 0xFF45:
      lyc_ = v;
      if (lcd_on()) {
        update_coincidence();
        update_stat_line();
      }
      break;
    case 0xFF47: bgp_ = v; break;
    case 0xFF48: obp0_ = v; break;
    case 0xFF49: obp1_ = v; break;
    case 0xFF4A: wy_ = v; break;
    case 0xFF4B: wx_ = v; break;
  }
}

void Ppu::update_coincidence() { coincidence_ = (ly_ == lyc_); }

void Ppu::update_stat_line(bool vblank_oam_quirk) {
  // All enabled sources are ORed into one line; only a rising edge requests
  // the interrupt ("STAT blocking").
  bool line = ((stat_ & 0x40) && coincidence_) || ((stat_ & 0x08) && mode_ == 0) ||
              ((stat_ & 0x10) && mode_ == 1) ||
              ((stat_ & 0x20) && (mode_ == 2 || vblank_oam_quirk));
  if (line && !stat_line_) if_ |= kIntStat;
  stat_line_ = line;
}

void Ppu::start_line() {
  state_ = State::OamScan;
  mode_ = 2;
  next_event_ = 80;
  if (ly_ == wy_) wy_hit_ = true;
  update_coincidence();
  update_stat_line();
}

void Ppu::run_events() {
  while (dot_ >= next_event_) {
    switch (state_) {
      case State::OamScan:
        state_ = State::Drawing;
        mode_ = 3;
        first_line_ = false;
        if (ly_ == wy_) wy_hit_ = true;
        mode3_len_ = compute_mode3_length(line_);
        render_line(line_);
        next_event_ = 80 + mode3_len_;
        update_stat_line();
        break;
      case State::Drawing:
        state_ = State::HBlank;
        mode_ = 0;
        next_event_ = 456;
        update_stat_line();
        break;
      case State::HBlank:
        dot_ -= 456;
        ++line_;
        ly_ = u8(line_);
        if (line_ == 144) {
          state_ = State::VBlank;
          mode_ = 1;
          next_event_ = 456;
          if_ |= kIntVBlank;
          frame_ready = true;
          ++frames;
          update_coincidence();
          update_stat_line(/*vblank_oam_quirk=*/true);
          update_stat_line();
        } else {
          start_line();
        }
        break;
      case State::VBlank:
        if (line_ == 153 && ly_ != 0) {
          // Line 153 quirk: LY reads 0 after the first M-cycle.
          ly_ = 0;
          update_coincidence();
          update_stat_line();
          next_event_ = 456;
          break;
        }
        dot_ -= 456;
        ++line_;
        if (line_ > 153) {
          line_ = 0;
          ly_ = 0;
          wy_hit_ = false;
          window_line_ = 0;
          start_line();
        } else {
          ly_ = u8(line_);
          next_event_ = (line_ == 153) ? 4 : 456;
          update_coincidence();
          update_stat_line();
        }
        break;
    }
  }
}

int Ppu::compute_mode3_length(int ly) const {
  int len = 172 + (scx_ & 7);
  if ((lcdc_ & 0x20) && wy_hit_ && wx_ <= 166) len += 6;
  if (lcdc_ & 0x02) {
    SpriteList s = select_sprites(oam, ly, (lcdc_ & 0x04) ? 16 : 8);
    for (int k = 0; k < s.n; ++k) {
      int sx = oam[s.idx[k] * 4 + 1];
      len += 11 - std::min(5, (sx + scx_) & 7);
    }
  }
  return len;
}

void Ppu::render_line(int ly) {
  if (ly < 0 || ly >= kScreenH) return;
  u8* out = &fb_[size_t(ly) * kScreenW];
  u8 bgi[kScreenW];  // raw background/window color index, for OBJ priority

  auto tile_row = [&](u8 tile, int row) -> u16 {
    if (lcdc_ & 0x10) return u16(tile * 16 + row * 2);
    return u16(0x1000 + i8(tile) * 16 + row * 2);
  };

  // Background.
  {
    u16 map = (lcdc_ & 0x08) ? 0x1C00 : 0x1800;
    u8 y = u8(scy_ + ly);
    int x = 0;
    while (x < kScreenW) {
      u8 px = u8(scx_ + x);
      u8 tile = vram[map + (y >> 3) * 32 + (px >> 3)];
      u16 a = tile_row(tile, y & 7);
      u8 lo = vram[a], hi = vram[a + 1];
      for (int b = 7 - (px & 7); b >= 0 && x < kScreenW; --b, ++x)
        bgi[x] = u8((((hi >> b) & 1) << 1) | ((lo >> b) & 1));
    }
  }

  // Window. Its line counter only advances on lines where it is drawn.
  if ((lcdc_ & 0x20) && wy_hit_ && wx_ <= 166) {
    int wx0 = int(wx_) - 7;
    u16 map = (lcdc_ & 0x40) ? 0x1C00 : 0x1800;
    int wy = window_line_;
    for (int x = std::max(0, wx0); x < kScreenW; ++x) {
      int wxp = x - wx0;
      u8 tile = vram[map + (wy >> 3) * 32 + (wxp >> 3)];
      u16 a = tile_row(tile, wy & 7);
      int b = 7 - (wxp & 7);
      bgi[x] = u8((((vram[a + 1] >> b) & 1) << 1) | ((vram[a] >> b) & 1));
    }
    ++window_line_;
  }

  if (lcdc_ & 0x01) {
    for (int x = 0; x < kScreenW; ++x) out[x] = (bgp_ >> (bgi[x] * 2)) & 3;
  } else {
    // DMG: LCDC.0 = 0 blanks both background and window to white.
    for (int x = 0; x < kScreenW; ++x) {
      bgi[x] = 0;
      out[x] = 0;
    }
  }

  // Objects.
  if (lcdc_ & 0x02) {
    int h = (lcdc_ & 0x04) ? 16 : 8;
    SpriteList s = select_sprites(oam, ly, h);
    bool taken[kScreenW] = {};
    for (int k = 0; k < s.n; ++k) {
      const u8* o = &oam[s.idx[k] * 4];
      int sy = o[0], sx = o[1];
      u8 tile = o[2], attr = o[3];
      int row = ly + 16 - sy;
      if (attr & 0x40) row = h - 1 - row;
      if (h == 16) tile &= 0xFE;
      u16 a = u16(tile * 16 + row * 2);
      u8 lo = vram[a], hi = vram[a + 1];
      u8 pal = (attr & 0x10) ? obp1_ : obp0_;
      for (int p = 0; p < 8; ++p) {
        int x = sx - 8 + p;
        if (x < 0 || x >= kScreenW || taken[x]) continue;
        int b = (attr & 0x20) ? p : 7 - p;
        u8 ci = u8((((hi >> b) & 1) << 1) | ((lo >> b) & 1));
        if (ci == 0) continue;  // transparent: a lower priority object may show
        taken[x] = true;
        if ((attr & 0x80) && bgi[x] != 0) continue;  // behind BG colors 1-3
        out[x] = (pal >> (ci * 2)) & 3;
      }
    }
  }
}

}  // namespace gb
