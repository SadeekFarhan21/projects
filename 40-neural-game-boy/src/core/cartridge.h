#pragma once
#include <memory>
#include <string>
#include <vector>

#include "core/types.h"

namespace gb {

enum class MapperKind { RomOnly, Mbc1, Mbc5 };

struct CartHeader {
  std::string title;
  u8 cart_type = 0;
  u8 rom_size_code = 0;
  u8 ram_size_code = 0;
  MapperKind mapper = MapperKind::RomOnly;
  bool has_battery = false;
  size_t ram_bytes = 0;
};

// Parses the header at 0x0134..0x014F. Throws std::runtime_error on
// unsupported cartridge types.
CartHeader parse_header(const std::vector<u8>& rom);

// Cartridge = ROM + optional external RAM + mapper registers.
// The mapper is a small tagged switch rather than a class hierarchy so the
// hot read path stays a single non-virtual call.
class Cartridge {
 public:
  explicit Cartridge(std::vector<u8> rom);
  static std::unique_ptr<Cartridge> from_file(const std::string& path);

  u8 read(u16 addr) const;         // 0x0000-0x7FFF and 0xA000-0xBFFF
  void write(u16 addr, u8 value);  // mapper registers and external RAM

  const CartHeader& header() const { return header_; }
  const std::vector<u8>& rom() const { return rom_; }
  std::vector<u8>& ram() { return ram_; }
  u64 rom_hash() const;  // FNV-1a 64 of the ROM image

  // Exposed for unit tests.
  unsigned rom_bank_low() const;   // bank mapped at 0x0000-0x3FFF
  unsigned rom_bank_high() const;  // bank mapped at 0x4000-0x7FFF
  unsigned ram_bank() const;
  bool ram_enabled() const { return ram_enabled_; }

 private:
  std::vector<u8> rom_;
  std::vector<u8> ram_;
  CartHeader header_;
  unsigned rom_banks_ = 2;  // number of 16 KiB banks (power of two)

  bool ram_enabled_ = false;
  u8 bank1_ = 1;  // MBC1: 5-bit register at 0x2000. MBC5: low 8 bits.
  u8 bank2_ = 0;  // MBC1: 2-bit register at 0x4000. MBC5: RAM bank / bit 8.
  u8 mode_ = 0;   // MBC1 banking mode.
  u16 mbc5_rom_bank_ = 1;
};

}  // namespace gb
