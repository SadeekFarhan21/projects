#include "core/testrom.h"

#include <cstdio>

#include "core/gameboy.h"

namespace gb {

namespace {
std::string hex_regs(const Registers& r) {
  char buf[96];
  std::snprintf(buf, sizeof buf, "A=%02X B=%02X C=%02X D=%02X E=%02X H=%02X L=%02X PC=%04X", r.a,
                r.b, r.c, r.d, r.e, r.h, r.l, r.pc);
  return buf;
}
}  // namespace

TestVerdict run_blargg(const std::string& rom_path, double max_seconds) {
  TestVerdict v;
  auto gb = GameBoy::from_file(rom_path);
  while (gb->emulated_seconds() < max_seconds) {
    gb->run_frame();
    const std::string& out = gb->bus.serial_output;
    if (out.find("Passed") != std::string::npos) {
      v.passed = true;
      break;
    }
    if (out.find("Failed") != std::string::npos) {
      // Let it finish printing the failure details.
      gb->run_mcycles(u64(kClockHz / 4) / 2);
      break;
    }
    if (gb->cpu.locked) break;
  }
  v.emu_seconds = gb->emulated_seconds();
  v.timed_out = !v.passed && gb->bus.serial_output.find("Failed") == std::string::npos;
  v.detail = gb->bus.serial_output;
  v.frame = gb->framebuffer();
  return v;
}

TestVerdict run_mooneye(const std::string& rom_path, double max_seconds) {
  TestVerdict v;
  auto gb = GameBoy::from_file(rom_path);
  const u64 limit = u64(max_seconds * kClockHz / 4);
  Cpu& cpu = gb->cpu;
  while (!cpu.ld_b_b_hit && !cpu.locked && gb->bus.mcycles() < limit) cpu.step();
  const Registers& r = cpu.r;
  v.passed = cpu.ld_b_b_hit && r.b == 3 && r.c == 5 && r.d == 8 && r.e == 13 && r.h == 21 && r.l == 34;
  v.timed_out = !cpu.ld_b_b_hit;
  v.emu_seconds = gb->emulated_seconds();
  v.detail = hex_regs(r) + (cpu.locked ? " (locked)" : "");
  v.frame = gb->framebuffer();
  return v;
}

TestVerdict run_until_ld_b_b(const std::string& rom_path, double max_seconds, int extra_frames) {
  TestVerdict v;
  auto gb = GameBoy::from_file(rom_path);
  const u64 limit = u64(max_seconds * kClockHz / 4);
  Cpu& cpu = gb->cpu;
  while (!cpu.ld_b_b_hit && !cpu.locked && gb->bus.mcycles() < limit) cpu.step();
  v.timed_out = !cpu.ld_b_b_hit;
  for (int i = 0; i < extra_frames; ++i) gb->run_frame();
  v.emu_seconds = gb->emulated_seconds();
  v.detail = hex_regs(cpu.r);
  v.frame = gb->framebuffer();
  return v;
}

}  // namespace gb
