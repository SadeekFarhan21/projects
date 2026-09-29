// Fixed-width output records. The decoder writes these as packed little-endian
// rows to <table>.bin; python/mdq/schema.py holds the matching numpy dtypes and
// converts them to Parquet. Every row carries the event timestamp (ns UTC) and
// the IEX-TP sequence number of the message that produced it.
#pragma once

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace mdp {

#pragma pack(push, 1)
struct QuoteRec {  // TOPS Quote Update
  int64_t ts;
  int64_t seq;
  uint64_t symbol;
  int64_t bid_price;
  int64_t ask_price;
  uint32_t bid_size;
  uint32_t ask_size;
  uint8_t flags;
};
struct LevelRec {  // DEEP Price Level Update
  int64_t ts;
  int64_t seq;
  uint64_t symbol;
  int64_t price;
  uint32_t size;
  uint8_t side;   // 'B' or 'S'
  uint8_t flags;  // bit 0 = event processing complete
};
struct BboRec {  // consistent DEEP BBO, emitted when it changes at an event boundary
  int64_t ts;
  int64_t seq;
  uint64_t symbol;
  int64_t bid_price;
  int64_t ask_price;
  uint32_t bid_size;
  uint32_t ask_size;
  uint16_t bid_levels;
  uint16_t ask_levels;
};
struct TradeRec {
  int64_t ts;
  int64_t seq;
  uint64_t symbol;
  int64_t price;
  int64_t trade_id;
  uint32_t size;
  uint8_t flags;
  uint8_t msg_type;  // 'T' report or 'B' break
};
struct SystemEventRec {
  int64_t ts;
  int64_t seq;
  uint8_t event;
};
struct SecurityDirectoryRec {
  int64_t ts;
  int64_t seq;
  uint64_t symbol;
  int64_t adjusted_poc_price;
  uint32_t round_lot;
  uint8_t flags;
  uint8_t luld_tier;
};
struct TradingStatusRec {
  int64_t ts;
  int64_t seq;
  uint64_t symbol;
  uint32_t reason;
  uint8_t status;
};
struct SecurityStatusRec {  // I, O, P, E messages
  int64_t ts;
  int64_t seq;
  uint64_t symbol;
  uint8_t msg_type;
  uint8_t status;
  uint8_t detail;
};
struct OfficialPriceRec {
  int64_t ts;
  int64_t seq;
  uint64_t symbol;
  int64_t price;
  uint8_t price_type;
};
struct AuctionRec {
  int64_t ts;
  int64_t seq;
  uint64_t symbol;
  int64_t reference_price;
  int64_t indicative_clearing_price;
  int64_t auction_book_clearing_price;
  int64_t collar_reference_price;
  int64_t lower_auction_collar;
  int64_t upper_auction_collar;
  uint32_t paired_shares;
  uint32_t imbalance_shares;
  uint32_t scheduled_auction_time;
  uint8_t auction_type;
  uint8_t imbalance_side;
  uint8_t extension_number;
};
struct UnknownRec {
  int64_t send_time;
  int64_t seq;
  uint16_t length;
  uint8_t msg_type;
};
struct SegmentRec {
  int64_t capture_ts;
  int64_t send_time;
  int64_t first_seq;
  int64_t stream_offset;
  uint32_t channel_id;
  uint32_t session_id;
  uint16_t protocol_id;
  uint16_t message_count;
  uint16_t payload_length;
  uint16_t skipped;  // leading messages dropped as duplicates
};
#pragma pack(pop)

static_assert(sizeof(QuoteRec) == 49);
static_assert(sizeof(LevelRec) == 38);
static_assert(sizeof(BboRec) == 52);
static_assert(sizeof(TradeRec) == 46);
static_assert(sizeof(SystemEventRec) == 17);
static_assert(sizeof(SecurityDirectoryRec) == 38);
static_assert(sizeof(TradingStatusRec) == 29);
static_assert(sizeof(SecurityStatusRec) == 27);
static_assert(sizeof(OfficialPriceRec) == 33);
static_assert(sizeof(AuctionRec) == 87);
static_assert(sizeof(UnknownRec) == 19);
static_assert(sizeof(SegmentRec) == 48);

// Buffered append-only writer for one table. A null writer (empty path)
// counts rows without touching disk, which the benchmark uses.
template <class R>
class TableWriter {
 public:
  TableWriter() = default;
  explicit TableWriter(const std::string& path) { open(path); }
  ~TableWriter() { close(); }
  TableWriter(const TableWriter&) = delete;
  TableWriter& operator=(const TableWriter&) = delete;

  void open(const std::string& path) {
    path_ = path;
    if (!path.empty()) {
      f_ = std::fopen(path.c_str(), "wb");
      if (!f_) throw std::runtime_error("cannot open " + path);
      buf_.reserve(kBufRows);
    }
  }
  void push(const R& r) {
    ++rows_;
    if (!f_) return;
    buf_.push_back(r);
    if (buf_.size() == kBufRows) flush();
  }
  void flush() {
    if (f_ && !buf_.empty()) {
      if (std::fwrite(buf_.data(), sizeof(R), buf_.size(), f_) != buf_.size())
        throw std::runtime_error("short write to " + path_);
      buf_.clear();
    }
  }
  void close() {
    flush();
    if (f_) std::fclose(f_);
    f_ = nullptr;
  }
  uint64_t rows() const { return rows_; }
  static constexpr size_t record_size() { return sizeof(R); }

 private:
  static constexpr size_t kBufRows = (1u << 20) / sizeof(R) + 1;
  std::FILE* f_ = nullptr;
  std::string path_;
  std::vector<R> buf_;
  uint64_t rows_ = 0;
};

}  // namespace mdp
