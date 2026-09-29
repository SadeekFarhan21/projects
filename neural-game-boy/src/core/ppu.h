#pragma once
#include <array>

#include "core/types.h"

namespace gb {

// Pixel processing unit.
//
// Timing is modeled per scanline with an event schedule instead of per dot:
// the PPU advances 4 dots per M-cycle and only does work when the dot counter
// crosses the next mode boundary. Each visible line is
//   mode 2 (OAM scan, 80 dots) -> mode 3 (drawing, 172 + penalties dots)
//   -> mode 0 (HBlank, rest of the 456 dot line)
// and lines 144..153 are mode 1 (VBlank). The pixels of a line are produced
// all at once when mode 3 starts. This is a scanline renderer, so effects
// that change registers in the middle of mode 3 are not reproduced; that is
// a documented trade-off (see DESIGN.md).
class Ppu {
 public:
  explicit Ppu(u8& if_reg) : if_(if_reg) {}

  void reset_post_boot();

  // Advance one M-cycle (4 dots).
  void tick() {
    if (!(lcdc_ & 0x80)) return;
    dot_ += 4;
    if (dot_ >= next_event_) run_events();
  }

  u8 read_reg(u16 addr) const;
  void write_reg(u16 addr, u8 value);

  u8 mode() const { return (lcdc_ & 0x80) ? mode_ : 0; }
  bool lcd_on() const { return lcdc_ & 0x80; }
  bool vram_accessible() const { return !(lcd_on() && state_ == State::Drawing); }
  bool oam_accessible() const {
    return !(lcd_on() && (state_ == State::OamScan || state_ == State::Drawing));
  }

  std::array<u8, 0x2000> vram{};
  std::array<u8, 0xA0> oam{};

  // Frame buffer of shades 0..3 (0 = white, 3 = black), row-major 160x144.
  const std::array<u8, kScreenW * kScreenH>& framebuffer() const { return fb_; }
  // Set when the PPU enters VBlank; the caller clears it.
  bool frame_ready = false;
  u64 frames = 0;

  // Test hooks.
  int dot() const { return dot_; }
  int line() const { return line_; }
  int mode3_length() const { return mode3_len_; }
  void render_line_for_test(int ly) { render_line(ly); }

 private:
  enum class State : u8 { OamScan, Drawing, HBlank, VBlank };

  void run_events();
  void start_line();
  void update_coincidence();
  void update_stat_line(bool vblank_oam_quirk = false);
  void render_line(int ly);
  int compute_mode3_length(int ly) const;

  u8& if_;
  std::array<u8, kScreenW * kScreenH> fb_{};

  u8 lcdc_ = 0x91, stat_ = 0x80, scy_ = 0, scx_ = 0, ly_ = 0, lyc_ = 0;
  u8 bgp_ = 0xFC, obp0_ = 0xFF, obp1_ = 0xFF, wy_ = 0, wx_ = 0;

  State state_ = State::OamScan;
  u8 mode_ = 2;       // mode reported in STAT
  int line_ = 0;      // internal line counter 0..153
  int dot_ = 0;       // dot within the line, 0..455
  int next_event_ = 80;
  int mode3_len_ = 172;
  bool stat_line_ = false;     // level of the combined STAT interrupt line
  bool coincidence_ = false;   // LY == LYC
  bool first_line_ = false;    // first line after LCD enable reports mode 0
  bool wy_hit_ = false;        // WY matched LY at some line this frame
  int window_line_ = 0;        // internal window line counter
};

}  // namespace gb
