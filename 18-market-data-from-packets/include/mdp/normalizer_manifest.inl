// Manifest writer for Normalizer. Kept out of the main header for readability.
#pragma once

#include <cstdio>
#include <fstream>
#include <sstream>

namespace mdp {

inline Normalizer::Normalizer(const std::string& out_dir, bool build_book) : dir_(out_dir), build_book_(build_book) {
  if (dir_.empty()) return;
  auto p = [&](const char* name) { return dir_ + "/" + name + ".bin"; };
  quotes_.open(p("quotes"));
  deep_levels_.open(p("deep_levels"));
  if (build_book_) bbo_.open(p("bbo_from_deep"));
  trades_.open(p("trades"));
  system_events_.open(p("system_events"));
  security_directory_.open(p("security_directory"));
  trading_status_.open(p("trading_status"));
  security_status_.open(p("security_status"));
  official_prices_.open(p("official_prices"));
  auctions_.open(p("auctions"));
  unknown_.open(p("unknown_messages"));
  segments_.open(p("segments"));
}

inline void Normalizer::close() {
  quotes_.close();
  deep_levels_.close();
  bbo_.close();
  trades_.close();
  system_events_.close();
  security_directory_.close();
  trading_status_.close();
  security_status_.close();
  official_prices_.close();
  auctions_.close();
  unknown_.close();
  segments_.close();
}

template <class Dec>
void Normalizer::write_manifest(const std::string& path, const Dec& dec, const std::string& feed,
                                const std::string& input, double seconds, uint64_t bytes_in, bool truncated,
                                const std::string& capture_format) const {
  std::ostringstream o;
  const DecodeStats& s = dec.stats();
  o << "{\n";
  o << "  \"feed\": \"" << feed << "\",\n";
  o << "  \"input\": \"" << input << "\",\n";
  o << "  \"capture_format\": \"" << capture_format << "\",\n";
  o << "  \"input_truncated\": " << (truncated ? "true" : "false") << ",\n";
  o << "  \"decode_seconds\": " << seconds << ",\n";
  o << "  \"capture_bytes\": " << bytes_in << ",\n";
  o << "  \"min_event_ts\": " << (min_ts_ == INT64_MAX ? 0 : min_ts_) << ",\n";
  o << "  \"max_event_ts\": " << (max_ts_ == INT64_MIN ? 0 : max_ts_) << ",\n";
  o << "  \"tables\": {\n";
  auto t = [&](const char* name, uint64_t rows, size_t rs, bool last = false) {
    o << "    \"" << name << "\": {\"rows\": " << rows << ", \"record_size\": " << rs << "}" << (last ? "\n" : ",\n");
  };
  t("quotes", quotes_.rows(), sizeof(QuoteRec));
  t("deep_levels", deep_levels_.rows(), sizeof(LevelRec));
  t("bbo_from_deep", bbo_.rows(), sizeof(BboRec));
  t("trades", trades_.rows(), sizeof(TradeRec));
  t("system_events", system_events_.rows(), sizeof(SystemEventRec));
  t("security_directory", security_directory_.rows(), sizeof(SecurityDirectoryRec));
  t("trading_status", trading_status_.rows(), sizeof(TradingStatusRec));
  t("security_status", security_status_.rows(), sizeof(SecurityStatusRec));
  t("official_prices", official_prices_.rows(), sizeof(OfficialPriceRec));
  t("auctions", auctions_.rows(), sizeof(AuctionRec));
  t("unknown_messages", unknown_.rows(), sizeof(UnknownRec));
  t("segments", segments_.rows(), sizeof(SegmentRec), true);
  o << "  },\n";
  o << "  \"stats\": {\n";
  o << "    \"packets\": " << s.packets << ",\n";
  o << "    \"captured_bytes\": " << s.captured_bytes << ",\n";
  o << "    \"udp_payload_bytes\": " << s.udp_payload_bytes << ",\n";
  o << "    \"non_iextp_packets\": " << s.non_iextp << ",\n";
  o << "    \"net_errors\": {";
  for (int i = 1; i < 7; ++i) {
    o << "\"" << to_string(static_cast<NetStatus>(i)) << "\": " << s.net_errors[i] << (i < 6 ? ", " : "");
  }
  o << "},\n";
  o << "    \"segment_errors\": {";
  for (int i = 1; i < 4; ++i) {
    o << "\"" << to_string(static_cast<SegStatus>(i)) << "\": " << s.seg_errors[i] << (i < 3 ? ", " : "");
  }
  o << "},\n";
  o << "    \"segments\": " << s.segments << ",\n";
  o << "    \"heartbeats\": " << s.heartbeats << ",\n";
  o << "    \"messages\": " << s.messages << ",\n";
  o << "    \"messages_skipped_duplicate\": " << s.messages_skipped_dup << ",\n";
  o << "    \"framing_errors\": " << s.framing_errors << ",\n";
  o << "    \"trailing_bytes\": " << s.trailing_bytes << ",\n";
  o << "    \"malformed_messages\": " << malformed_ << ",\n";
  o << "    \"messages_by_type\": {";
  bool first = true;
  for (int i = 0; i < 256; ++i) {
    if (!s.by_type[i]) continue;
    o << (first ? "" : ", ") << "\"";
    if (i >= 32 && i < 127 && i != '"' && i != '\\') {
      o << static_cast<char>(i);
    } else {
      o << "0x" << std::hex << i << std::dec;
    }
    o << "\": " << s.by_type[i];
    first = false;
  }
  o << "}\n";
  o << "  },\n";
  // Book summary (DEEP only).
  uint64_t upd = 0, ev = 0, delmiss = 0, crossed = 0, locked = 0, transition_at_end = 0;
  for (uint32_t i = 0; i < books_.size(); ++i) {
    const auto& b = books_.book_by_id(i);
    upd += b.counters().updates;
    ev += b.counters().events_completed;
    delmiss += b.counters().delete_missing;
    crossed += b.counters().crossed_at_complete;
    locked += b.counters().locked_at_complete;
    transition_at_end += b.in_transition() ? 1 : 0;
  }
  o << "  \"book\": {\"symbols\": " << books_.size() << ", \"updates\": " << upd << ", \"events_completed\": " << ev
    << ", \"delete_missing_level\": " << delmiss << ", \"crossed_at_event_end\": " << crossed
    << ", \"locked_at_event_end\": " << locked << ", \"symbols_in_transition_at_end\": " << transition_at_end
    << "},\n";
  // Sequence tracking.
  const auto& seq = dec.seq();
  o << "  \"sessions\": [\n";
  size_t n = 0;
  for (const auto& [k, v] : seq.sessions()) {
    o << "    {\"protocol_id\": " << v.protocol_id << ", \"channel_id\": " << v.channel_id
      << ", \"session_id\": " << v.session_id << ", \"start_seq\": " << v.start_seq
      << ", \"next_seq\": " << v.next_seq << ", \"segments\": " << v.segments << ", \"heartbeats\": " << v.heartbeats
      << ", \"messages\": " << v.messages << ", \"gaps\": " << v.gaps << ", \"gap_messages\": " << v.gap_messages
      << ", \"duplicates\": " << v.duplicates << ", \"duplicate_messages\": " << v.duplicate_messages
      << ", \"overlaps\": " << v.overlaps << ", \"offset_mismatches\": " << v.offset_mismatches
      << ", \"send_time_regressions\": " << v.send_time_regressions
      << ", \"first_send_time\": " << v.first_send_time << ", \"last_send_time\": " << v.last_send_time << "}"
      << (++n < seq.sessions().size() ? ",\n" : "\n");
  }
  o << "  ],\n";
  o << "  \"seq_anomaly_count\": " << seq.anomaly_count() << ",\n";
  o << "  \"seq_anomalies\": [\n";
  const auto& an = seq.anomalies();
  for (size_t i = 0; i < an.size(); ++i) {
    const auto& a = an[i];
    o << "    {\"kind\": \"" << to_string(a.kind) << "\", \"session_id\": " << a.session_id
      << ", \"channel_id\": " << a.channel_id << ", \"expected_seq\": " << a.expected_seq
      << ", \"first_seq\": " << a.first_seq << ", \"message_count\": " << a.message_count
      << ", \"expected_offset\": " << a.expected_offset << ", \"stream_offset\": " << a.stream_offset
      << ", \"send_time\": " << a.send_time << ", \"capture_ts\": " << a.capture_ts << "}"
      << (i + 1 < an.size() ? ",\n" : "\n");
  }
  o << "  ]\n";
  o << "}\n";
  std::ofstream f(path);
  f << o.str();
}

}  // namespace mdp
