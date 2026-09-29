#pragma once
#include "core/bus.h"
#include "core/types.h"

namespace gb {

struct Registers {
  u8 a = 0x01, f = 0xB0, b = 0x00, c = 0x13, d = 0x00, e = 0xD8, h = 0x01, l = 0x4D;
  u16 sp = 0xFFFE, pc = 0x0100;

  u16 af() const { return u16((a << 8) | f); }
  u16 bc() const { return u16((b << 8) | c); }
  u16 de() const { return u16((d << 8) | e); }
  u16 hl() const { return u16((h << 8) | l); }
  void set_af(u16 v) { a = u8(v >> 8); f = u8(v & 0xF0); }
  void set_bc(u16 v) { b = u8(v >> 8); c = u8(v); }
  void set_de(u16 v) { d = u8(v >> 8); e = u8(v); }
  void set_hl(u16 v) { h = u8(v >> 8); l = u8(v); }
};

// Flag bits in F.
inline constexpr u8 kFlagZ = 0x80, kFlagN = 0x40, kFlagH = 0x20, kFlagC = 0x10;

// Sharp SM83 core. Every memory access costs one M-cycle and advances the
// rest of the system through Bus::tick() before the access happens; internal
// delay cycles call idle(). Instruction timing therefore falls out of the
// access pattern instead of a cycle table.
class Cpu {
 public:
  explicit Cpu(Bus& bus) : bus_(bus) {}

  // Execute one instruction, one HALT M-cycle, or one interrupt dispatch.
  void step();

  Registers r;
  bool ime = false;
  bool halted = false;
  bool locked = false;       // executed an illegal opcode; the CPU hangs
  bool ld_b_b_hit = false;   // LD B,B executed (Mooneye / acid2 exit marker)
  u64 instructions = 0;

 private:
  // Timed bus access.
  u8 read(u16 addr) { bus_.tick(); return bus_.read(addr); }
  void write(u16 addr, u8 v) { bus_.tick(); bus_.write(addr, v); }
  void idle() { bus_.tick(); }
  u8 fetch() {
    u8 v = read(r.pc);
    if (halt_bug_) halt_bug_ = false;  // PC fails to increment once
    else ++r.pc;
    return v;
  }
  u16 fetch16() { u8 lo = fetch(); u8 hi = fetch(); return u16(lo | (hi << 8)); }
  void push(u16 v) {
    --r.sp; write(r.sp, u8(v >> 8));
    --r.sp; write(r.sp, u8(v));
  }
  u16 pop() {
    u8 lo = read(r.sp++);
    u8 hi = read(r.sp++);
    return u16(lo | (hi << 8));
  }

  void dispatch_interrupt();
  void execute(u8 op);
  void execute_cb(u8 op);

  // Operand helpers indexed like the opcode encoding: B C D E H L (HL) A.
  u8 get_r(int i);
  void set_r(int i, u8 v);
  u16 get_rp(int p) const;  // BC DE HL SP
  void set_rp(int p, u16 v);
  bool cond(int cc) const;  // NZ Z NC C

  void alu(int op, u8 v);  // ADD ADC SUB SBC AND XOR OR CP
  u8 inc8(u8 v);
  u8 dec8(u8 v);
  void add_hl(u16 v);
  u16 add_sp_e(u8 e);  // flags as for ADD SP,e / LD HL,SP+e
  void daa();

  bool flag(u8 m) const { return r.f & m; }
  void set_flags(bool z, bool n, bool h, bool c) {
    r.f = u8((z ? kFlagZ : 0) | (n ? kFlagN : 0) | (h ? kFlagH : 0) | (c ? kFlagC : 0));
  }

  Bus& bus_;
  bool ei_pending_ = false;  // EI takes effect after the next instruction
  bool halt_bug_ = false;
};

}  // namespace gb
