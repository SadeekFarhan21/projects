#pragma once

#include <cstddef>
#include <cstdint>

namespace mk {

// CRC-32C (Castagnoli), the checksum Kafka uses for record batches.
// Uses the ARMv8 CRC32 instructions when available, else a table-driven fallback.
uint32_t crc32c(const void* data, size_t n, uint32_t seed = 0);

// Table-driven version, exposed so tests can check both paths agree.
uint32_t crc32c_sw(const void* data, size_t n, uint32_t seed = 0);

}  // namespace mk
