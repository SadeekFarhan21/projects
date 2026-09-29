#include "core/cartridge.h"

#include <fstream>
#include <iterator>
#include <stdexcept>

namespace gb {

namespace {
size_t ram_size_from_code(u8 code) {
  switch (code) {
    case 0x00: return 0;
    case 0x01: return 2 * 1024;
    case 0x02: return 8 * 1024;
    case 0x03: return 32 * 1024;
    case 0x04: return 128 * 1024;
    case 0x05: return 64 * 1024;
    default: return 0;
  }
}
}  // namespace

CartHeader parse_header(const std::vector<u8>& rom) {
  if (rom.size() < 0x150) throw std::runtime_error("ROM too small for a header");
  CartHeader h;
  for (int i = 0x134; i < 0x144; ++i) {
    char c = static_cast<char>(rom[i]);
    if (c == 0) break;
    if (c >= 32 && c < 127) h.title.push_back(c);
  }
  h.cart_type = rom[0x147];
  h.rom_size_code = rom[0x148];
  h.ram_size_code = rom[0x149];
  h.ram_bytes = ram_size_from_code(h.ram_size_code);
  switch (h.cart_type) {
    case 0x00: h.mapper = MapperKind::RomOnly; break;
    case 0x08: h.mapper = MapperKind::RomOnly; break;
    case 0x09: h.mapper = MapperKind::RomOnly; h.has_battery = true; break;
    case 0x01: case 0x02: h.mapper = MapperKind::Mbc1; break;
    case 0x03: h.mapper = MapperKind::Mbc1; h.has_battery = true; break;
    case 0x19: case 0x1A: case 0x1C: case 0x1D: h.mapper = MapperKind::Mbc5; break;
    case 0x1B: case 0x1E: h.mapper = MapperKind::Mbc5; h.has_battery = true; break;
    default:
      throw std::runtime_error("unsupported cartridge type 0x" +
                               std::to_string(static_cast<int>(h.cart_type)));
  }
  return h;
}

Cartridge::Cartridge(std::vector<u8> rom) : rom_(std::move(rom)) {
  header_ = parse_header(rom_);
  // Round the ROM up to a power-of-two number of 16 KiB banks so bank
  // arithmetic can use a mask. The header size wins if it is larger.
  size_t header_bytes = size_t(32 * 1024) << (header_.rom_size_code & 0x0F);
  size_t want = std::max(header_bytes, rom_.size());
  size_t pow2 = 32 * 1024;
  while (pow2 < want) pow2 <<= 1;
  rom_.resize(pow2, 0xFF);
  rom_banks_ = static_cast<unsigned>(pow2 / 0x4000);
  ram_.assign(header_.ram_bytes, 0x00);
}

std::unique_ptr<Cartridge> Cartridge::from_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open ROM: " + path);
  std::vector<u8> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  return std::make_unique<Cartridge>(std::move(data));
}

u64 Cartridge::rom_hash() const {
  u64 h = 1469598103934665603ULL;
  for (u8 b : rom_) {
    h ^= b;
    h *= 1099511628211ULL;
  }
  return h;
}

unsigned Cartridge::rom_bank_low() const {
  if (header_.mapper == MapperKind::Mbc1 && mode_ == 1) return (unsigned(bank2_) << 5) & (rom_banks_ - 1);
  return 0;
}

unsigned Cartridge::rom_bank_high() const {
  switch (header_.mapper) {
    case MapperKind::RomOnly: return 1;
    case MapperKind::Mbc1: return ((unsigned(bank2_) << 5) | bank1_) & (rom_banks_ - 1);
    case MapperKind::Mbc5: return mbc5_rom_bank_ & (rom_banks_ - 1);
  }
  return 1;
}

unsigned Cartridge::ram_bank() const {
  switch (header_.mapper) {
    case MapperKind::RomOnly: return 0;
    case MapperKind::Mbc1: return mode_ == 1 ? bank2_ : 0;
    case MapperKind::Mbc5: return bank2_ & 0x0F;
  }
  return 0;
}

u8 Cartridge::read(u16 addr) const {
  if (addr < 0x4000) return rom_[size_t(rom_bank_low()) * 0x4000 + addr];
  if (addr < 0x8000) return rom_[size_t(rom_bank_high()) * 0x4000 + (addr - 0x4000)];
  // External RAM window 0xA000-0xBFFF.
  if (ram_.empty()) return 0xFF;
  if (header_.mapper != MapperKind::RomOnly && !ram_enabled_) return 0xFF;
  size_t off = (size_t(ram_bank()) * 0x2000 + (addr - 0xA000)) & (ram_.size() - 1);
  return ram_[off];
}

void Cartridge::write(u16 addr, u8 value) {
  if (addr >= 0xA000) {
    if (ram_.empty()) return;
    if (header_.mapper != MapperKind::RomOnly && !ram_enabled_) return;
    size_t off = (size_t(ram_bank()) * 0x2000 + (addr - 0xA000)) & (ram_.size() - 1);
    ram_[off] = value;
    return;
  }
  switch (header_.mapper) {
    case MapperKind::RomOnly: return;
    case MapperKind::Mbc1:
      if (addr < 0x2000) {
        ram_enabled_ = (value & 0x0F) == 0x0A;
      } else if (addr < 0x4000) {
        bank1_ = value & 0x1F;
        if (bank1_ == 0) bank1_ = 1;  // the zero check happens before masking
      } else if (addr < 0x6000) {
        bank2_ = value & 0x03;
      } else {
        mode_ = value & 0x01;
      }
      return;
    case MapperKind::Mbc5:
      if (addr < 0x2000) {
        ram_enabled_ = value == 0x0A;
      } else if (addr < 0x3000) {
        mbc5_rom_bank_ = (mbc5_rom_bank_ & 0x100) | value;
      } else if (addr < 0x4000) {
        mbc5_rom_bank_ = u16((mbc5_rom_bank_ & 0xFF) | ((value & 1) << 8));
      } else if (addr < 0x6000) {
        bank2_ = value & 0x0F;
      }
      return;
  }
}

}  // namespace gb
