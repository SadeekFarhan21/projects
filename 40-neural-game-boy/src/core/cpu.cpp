#include "core/cpu.h"

namespace gb {

void Cpu::step() {
  if (locked) {
    idle();
    return;
  }
  if (halted) {
    // HALT burns M-cycles until IE & IF != 0, regardless of IME.
    idle();
    if (!bus_.pending_interrupts()) return;
    halted = false;
  }
  // The opcode fetch M-cycle is also when the CPU samples interrupts, so an
  // IF bit raised during this very cycle (for example by a timer reload)
  // still wins over the instruction being fetched.
  bus_.tick();
  if (ime && bus_.sampled_interrupts()) {
    dispatch_interrupt();
    return;
  }
  bool apply_ei = ei_pending_;
  ei_pending_ = false;
  u8 op = bus_.read(r.pc);
  if (halt_bug_) halt_bug_ = false;  // PC fails to increment once
  else ++r.pc;
  execute(op);
  ++instructions;
  if (apply_ei && op != 0xF3) ime = true;  // EI then DI leaves IME clear
}

void Cpu::dispatch_interrupt() {
  // 5 M-cycles in total: the discarded opcode fetch (already spent by the
  // caller), an internal cycle, push PC high, push PC low, and the jump.
  // The vector is chosen after the high byte is pushed, so a push that
  // overwrites IE (SP = 0x0000) can cancel the dispatch and jump to 0x0000.
  ime = false;
  idle();
  --r.sp;
  write(r.sp, u8(r.pc >> 8));
  u8 pending = bus_.pending_interrupts();
  --r.sp;
  write(r.sp, u8(r.pc));
  if (pending == 0) {
    r.pc = 0x0000;
  } else {
    int bit = __builtin_ctz(pending);
    bus_.if_reg() &= u8(~(1u << bit));
    r.pc = u16(0x40 + 8 * bit);
  }
  idle();
}

u8 Cpu::get_r(int i) {
  switch (i) {
    case 0: return r.b;
    case 1: return r.c;
    case 2: return r.d;
    case 3: return r.e;
    case 4: return r.h;
    case 5: return r.l;
    case 6: return read(r.hl());
    default: return r.a;
  }
}

void Cpu::set_r(int i, u8 v) {
  switch (i) {
    case 0: r.b = v; break;
    case 1: r.c = v; break;
    case 2: r.d = v; break;
    case 3: r.e = v; break;
    case 4: r.h = v; break;
    case 5: r.l = v; break;
    case 6: write(r.hl(), v); break;
    default: r.a = v; break;
  }
}

u16 Cpu::get_rp(int p) const {
  switch (p) {
    case 0: return r.bc();
    case 1: return r.de();
    case 2: return r.hl();
    default: return r.sp;
  }
}

void Cpu::set_rp(int p, u16 v) {
  switch (p) {
    case 0: r.set_bc(v); break;
    case 1: r.set_de(v); break;
    case 2: r.set_hl(v); break;
    default: r.sp = v; break;
  }
}

bool Cpu::cond(int cc) const {
  switch (cc) {
    case 0: return !flag(kFlagZ);
    case 1: return flag(kFlagZ);
    case 2: return !flag(kFlagC);
    default: return flag(kFlagC);
  }
}

void Cpu::alu(int op, u8 v) {
  u8 a = r.a;
  switch (op) {
    case 0: {  // ADD
      unsigned res = unsigned(a) + v;
      set_flags(u8(res) == 0, false, ((a & 0xF) + (v & 0xF)) > 0xF, res > 0xFF);
      r.a = u8(res);
      break;
    }
    case 1: {  // ADC
      unsigned c = flag(kFlagC) ? 1 : 0;
      unsigned res = unsigned(a) + v + c;
      set_flags(u8(res) == 0, false, ((a & 0xF) + (v & 0xF) + c) > 0xF, res > 0xFF);
      r.a = u8(res);
      break;
    }
    case 2:    // SUB
    case 7: {  // CP
      u8 res = u8(a - v);
      set_flags(res == 0, true, (a & 0xF) < (v & 0xF), a < v);
      if (op == 2) r.a = res;
      break;
    }
    case 3: {  // SBC
      int c = flag(kFlagC) ? 1 : 0;
      int res = int(a) - int(v) - c;
      set_flags(u8(res) == 0, true, (int(a & 0xF) - int(v & 0xF) - c) < 0, res < 0);
      r.a = u8(res);
      break;
    }
    case 4: r.a = a & v; set_flags(r.a == 0, false, true, false); break;
    case 5: r.a = a ^ v; set_flags(r.a == 0, false, false, false); break;
    case 6: r.a = a | v; set_flags(r.a == 0, false, false, false); break;
  }
}

u8 Cpu::inc8(u8 v) {
  u8 res = u8(v + 1);
  r.f = u8((res == 0 ? kFlagZ : 0) | ((v & 0xF) == 0xF ? kFlagH : 0) | (r.f & kFlagC));
  return res;
}

u8 Cpu::dec8(u8 v) {
  u8 res = u8(v - 1);
  r.f = u8((res == 0 ? kFlagZ : 0) | kFlagN | ((v & 0xF) == 0 ? kFlagH : 0) | (r.f & kFlagC));
  return res;
}

void Cpu::add_hl(u16 v) {
  u16 hl = r.hl();
  unsigned res = unsigned(hl) + v;
  r.f = u8((r.f & kFlagZ) | (((hl & 0xFFF) + (v & 0xFFF)) > 0xFFF ? kFlagH : 0) |
           (res > 0xFFFF ? kFlagC : 0));
  r.set_hl(u16(res));
}

u16 Cpu::add_sp_e(u8 e) {
  // H and C come from the unsigned add of the low byte.
  set_flags(false, false, ((r.sp & 0xF) + (e & 0xF)) > 0xF, ((r.sp & 0xFF) + e) > 0xFF);
  return u16(r.sp + i8(e));
}

void Cpu::daa() {
  u8 a = r.a;
  bool c = flag(kFlagC);
  if (!flag(kFlagN)) {
    if (c || a > 0x99) {
      a = u8(a + 0x60);
      c = true;
    }
    if (flag(kFlagH) || (a & 0x0F) > 0x09) a = u8(a + 0x06);
  } else {
    if (c) a = u8(a - 0x60);
    if (flag(kFlagH)) a = u8(a - 0x06);
  }
  r.a = a;
  r.f = u8((a == 0 ? kFlagZ : 0) | (r.f & kFlagN) | (c ? kFlagC : 0));
}

void Cpu::execute(u8 op) {
  const int x = op >> 6, y = (op >> 3) & 7, z = op & 7, p = y >> 1, q = y & 1;
  switch (x) {
    case 0:
      switch (z) {
        case 0:
          switch (y) {
            case 0: return;  // NOP
            case 1: {        // LD (nn),SP
              u16 a = fetch16();
              write(a, u8(r.sp));
              write(u16(a + 1), u8(r.sp >> 8));
              return;
            }
            case 2:  // STOP: 2-byte opcode; resets DIV. Low power mode is not modeled.
              fetch();
              bus_.timer().write(0xFF04, 0);
              return;
            case 3: {  // JR e
              i8 e = i8(fetch());
              idle();
              r.pc = u16(r.pc + e);
              return;
            }
            default: {  // JR cc,e
              i8 e = i8(fetch());
              if (cond(y - 4)) {
                idle();
                r.pc = u16(r.pc + e);
              }
              return;
            }
          }
        case 1:
          if (!q) {
            set_rp(p, fetch16());  // LD rr,nn
          } else {
            idle();  // ADD HL,rr
            add_hl(get_rp(p));
          }
          return;
        case 2: {
          u16 addr;
          switch (p) {
            case 0: addr = r.bc(); break;
            case 1: addr = r.de(); break;
            case 2: addr = r.hl(); r.set_hl(u16(addr + 1)); break;
            default: addr = r.hl(); r.set_hl(u16(addr - 1)); break;
          }
          if (!q) write(addr, r.a);
          else r.a = read(addr);
          return;
        }
        case 3:  // INC rr / DEC rr
          idle();
          set_rp(p, u16(get_rp(p) + (q ? -1 : 1)));
          return;
        case 4:
          if (y == 6) {
            u16 hl = r.hl();
            write(hl, inc8(read(hl)));
          } else {
            set_r(y, inc8(get_r(y)));
          }
          return;
        case 5:
          if (y == 6) {
            u16 hl = r.hl();
            write(hl, dec8(read(hl)));
          } else {
            set_r(y, dec8(get_r(y)));
          }
          return;
        case 6: {  // LD r,n
          u8 n = fetch();
          set_r(y, n);
          return;
        }
        default: {
          u8 a = r.a;
          switch (y) {
            case 0: r.a = u8((a << 1) | (a >> 7)); set_flags(false, false, false, a & 0x80); break;
            case 1: r.a = u8((a >> 1) | (a << 7)); set_flags(false, false, false, a & 0x01); break;
            case 2: r.a = u8((a << 1) | (flag(kFlagC) ? 1 : 0)); set_flags(false, false, false, a & 0x80); break;
            case 3: r.a = u8((a >> 1) | (flag(kFlagC) ? 0x80 : 0)); set_flags(false, false, false, a & 0x01); break;
            case 4: daa(); break;
            case 5: r.a = u8(~a); r.f |= kFlagN | kFlagH; break;
            case 6: r.f = u8((r.f & kFlagZ) | kFlagC); break;
            case 7: r.f = u8((r.f & kFlagZ) | ((r.f & kFlagC) ^ kFlagC)); break;
          }
          return;
        }
      }
    case 1:
      if (op == 0x76) {  // HALT
        if (!ime && bus_.pending_interrupts()) halt_bug_ = true;
        else halted = true;
        return;
      }
      if (op == 0x40) ld_b_b_hit = true;
      set_r(y, get_r(z));
      return;
    case 2:
      alu(y, get_r(z));
      return;
    default:
      switch (z) {
        case 0:
          switch (y) {
            case 4: {  // LDH (n),A
              u8 n = fetch();
              write(u16(0xFF00 | n), r.a);
              return;
            }
            case 5: {  // ADD SP,e
              u8 e = fetch();
              u16 res = add_sp_e(e);
              idle();
              idle();
              r.sp = res;
              return;
            }
            case 6: {  // LDH A,(n)
              u8 n = fetch();
              r.a = read(u16(0xFF00 | n));
              return;
            }
            case 7: {  // LD HL,SP+e
              u8 e = fetch();
              r.set_hl(add_sp_e(e));
              idle();
              return;
            }
            default:  // RET cc
              idle();
              if (cond(y)) {
                r.pc = pop();
                idle();
              }
              return;
          }
        case 1:
          if (!q) {  // POP rr
            u16 v = pop();
            switch (p) {
              case 0: r.set_bc(v); break;
              case 1: r.set_de(v); break;
              case 2: r.set_hl(v); break;
              default: r.set_af(v); break;
            }
            return;
          }
          switch (p) {
            case 0: r.pc = pop(); idle(); return;             // RET
            case 1: r.pc = pop(); idle(); ime = true; return;  // RETI
            case 2: r.pc = r.hl(); return;                     // JP HL
            default: idle(); r.sp = r.hl(); return;            // LD SP,HL
          }
        case 2:
          switch (y) {
            case 4: write(u16(0xFF00 | r.c), r.a); return;
            case 5: write(fetch16(), r.a); return;
            case 6: r.a = read(u16(0xFF00 | r.c)); return;
            case 7: r.a = read(fetch16()); return;
            default: {  // JP cc,nn
              u16 a = fetch16();
              if (cond(y)) {
                idle();
                r.pc = a;
              }
              return;
            }
          }
        case 3:
          switch (y) {
            case 0: {  // JP nn
              u16 a = fetch16();
              idle();
              r.pc = a;
              return;
            }
            case 1: execute_cb(fetch()); return;
            case 6: ime = false; ei_pending_ = false; return;  // DI
            case 7: ei_pending_ = true; return;                // EI
            default: locked = true; return;                    // D3 DB E3 EB
          }
        case 4:
          if (y < 4) {  // CALL cc,nn
            u16 a = fetch16();
            if (cond(y)) {
              idle();
              push(r.pc);
              r.pc = a;
            }
          } else {
            locked = true;  // E4 EC F4 FC
          }
          return;
        case 5:
          if (!q) {  // PUSH rr
            u16 v;
            switch (p) {
              case 0: v = r.bc(); break;
              case 1: v = r.de(); break;
              case 2: v = r.hl(); break;
              default: v = r.af(); break;
            }
            idle();
            push(v);
          } else if (p == 0) {  // CALL nn
            u16 a = fetch16();
            idle();
            push(r.pc);
            r.pc = a;
          } else {
            locked = true;  // DD ED FD
          }
          return;
        case 6:
          alu(y, fetch());
          return;
        default:  // RST
          idle();
          push(r.pc);
          r.pc = u16(y * 8);
          return;
      }
  }
}

void Cpu::execute_cb(u8 op) {
  const int x = op >> 6, y = (op >> 3) & 7, z = op & 7;
  u8 v = get_r(z);
  switch (x) {
    case 0: {
      u8 res = 0;
      bool c = false;
      switch (y) {
        case 0: c = v & 0x80; res = u8((v << 1) | (v >> 7)); break;             // RLC
        case 1: c = v & 0x01; res = u8((v >> 1) | (v << 7)); break;             // RRC
        case 2: c = v & 0x80; res = u8((v << 1) | (flag(kFlagC) ? 1 : 0)); break;    // RL
        case 3: c = v & 0x01; res = u8((v >> 1) | (flag(kFlagC) ? 0x80 : 0)); break; // RR
        case 4: c = v & 0x80; res = u8(v << 1); break;                          // SLA
        case 5: c = v & 0x01; res = u8((v >> 1) | (v & 0x80)); break;           // SRA
        case 6: c = false; res = u8((v << 4) | (v >> 4)); break;                // SWAP
        case 7: c = v & 0x01; res = u8(v >> 1); break;                          // SRL
      }
      set_flags(res == 0, false, false, c);
      set_r(z, res);
      return;
    }
    case 1:  // BIT
      r.f = u8((v & (1 << y) ? 0 : kFlagZ) | kFlagH | (r.f & kFlagC));
      return;
    case 2: set_r(z, u8(v & ~(1 << y))); return;  // RES
    default: set_r(z, u8(v | (1 << y))); return;  // SET
  }
}

}  // namespace gb
