// Packet to message pipeline: capture packet -> UDP -> IEX-TP segment ->
// sequence tracking -> per-message dispatch.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "mdp/iex_messages.hpp"
#include "mdp/iextp.hpp"
#include "mdp/net.hpp"
#include "mdp/pcap.hpp"

namespace mdp {

enum class SeqKind : uint8_t {
  InOrder = 0,
  Gap,        // first_seq beyond the next expected sequence: messages missing
  Duplicate,  // entire segment already seen: dropped
  Overlap,    // segment partially seen: the seen prefix is dropped
  OffsetMismatch,  // sequence in order but stream offset disagrees
  SessionStart,    // first segment seen for a (protocol, channel, session)
};

inline const char* to_string(SeqKind k) {
  switch (k) {
    case SeqKind::InOrder: return "in_order";
    case SeqKind::Gap: return "gap";
    case SeqKind::Duplicate: return "duplicate";
    case SeqKind::Overlap: return "overlap";
    case SeqKind::OffsetMismatch: return "offset_mismatch";
    case SeqKind::SessionStart: return "session_start";
  }
  return "?";
}

struct SeqAnomaly {
  SeqKind kind;
  uint16_t protocol_id;
  uint32_t channel_id;
  uint32_t session_id;
  int64_t expected_seq;
  int64_t first_seq;
  uint16_t message_count;
  int64_t expected_offset;
  int64_t stream_offset;
  int64_t send_time;
  int64_t capture_ts;
};

struct SessionStats {
  uint16_t protocol_id = 0;
  uint32_t channel_id = 0;
  uint32_t session_id = 0;
  bool started = false;
  int64_t start_seq = 0;
  int64_t next_seq = 0;
  int64_t next_offset = 0;
  uint64_t segments = 0;
  uint64_t heartbeats = 0;
  uint64_t messages = 0;      // processed (not duplicate)
  uint64_t gaps = 0;
  uint64_t gap_messages = 0;  // messages missing
  uint64_t duplicates = 0;
  uint64_t duplicate_messages = 0;
  uint64_t overlaps = 0;
  uint64_t offset_mismatches = 0;
  int64_t first_send_time = 0;
  int64_t last_send_time = 0;
  uint64_t send_time_regressions = 0;
};

// Tracks sequence and stream-offset continuity for each IEX-TP session.
class SeqTracker {
 public:
  // Decides how many leading messages of the segment to skip (already seen).
  // Records anomalies. Returns the number of messages to skip.
  uint16_t observe(const SegmentHeader& h, int64_t capture_ts);
  const std::map<std::tuple<uint16_t, uint32_t, uint32_t>, SessionStats>& sessions() const { return sessions_; }
  const std::vector<SeqAnomaly>& anomalies() const { return anomalies_; }
  uint64_t anomaly_count() const { return anomaly_count_; }
  size_t max_anomalies = 10000;

 private:
  void record(const SeqAnomaly& a) {
    ++anomaly_count_;
    if (anomalies_.size() < max_anomalies) anomalies_.push_back(a);
  }
  std::map<std::tuple<uint16_t, uint32_t, uint32_t>, SessionStats> sessions_;
  std::vector<SeqAnomaly> anomalies_;
  uint64_t anomaly_count_ = 0;
};

inline uint16_t SeqTracker::observe(const SegmentHeader& h, int64_t capture_ts) {
  auto& s = sessions_[{h.protocol_id, h.channel_id, h.session_id}];
  ++s.segments;
  if (h.message_count == 0) ++s.heartbeats;
  if (!s.started) {
    s.started = true;
    s.protocol_id = h.protocol_id;
    s.channel_id = h.channel_id;
    s.session_id = h.session_id;
    s.start_seq = h.first_seq;
    s.next_seq = h.first_seq;
    s.next_offset = h.stream_offset;
    s.first_send_time = h.send_time;
    s.last_send_time = h.send_time;
    record({SeqKind::SessionStart, h.protocol_id, h.channel_id, h.session_id, h.first_seq, h.first_seq,
            h.message_count, h.stream_offset, h.stream_offset, h.send_time, capture_ts});
  }
  if (h.send_time < s.last_send_time) ++s.send_time_regressions;
  if (h.send_time > s.last_send_time) s.last_send_time = h.send_time;

  const int64_t end = h.first_seq + h.message_count;
  uint16_t skip = 0;
  if (h.first_seq == s.next_seq) {
    if (h.stream_offset != s.next_offset) {
      ++s.offset_mismatches;
      record({SeqKind::OffsetMismatch, h.protocol_id, h.channel_id, h.session_id, s.next_seq, h.first_seq,
              h.message_count, s.next_offset, h.stream_offset, h.send_time, capture_ts});
    }
  } else if (h.first_seq > s.next_seq) {
    ++s.gaps;
    s.gap_messages += static_cast<uint64_t>(h.first_seq - s.next_seq);
    record({SeqKind::Gap, h.protocol_id, h.channel_id, h.session_id, s.next_seq, h.first_seq, h.message_count,
            s.next_offset, h.stream_offset, h.send_time, capture_ts});
  } else {
    // first_seq < next_seq
    if (end <= s.next_seq) {
      if (h.message_count > 0) {
        ++s.duplicates;
        s.duplicate_messages += h.message_count;
        record({SeqKind::Duplicate, h.protocol_id, h.channel_id, h.session_id, s.next_seq, h.first_seq,
                h.message_count, s.next_offset, h.stream_offset, h.send_time, capture_ts});
      }
      return h.message_count;  // nothing new; do not move next_seq backwards
    }
    ++s.overlaps;
    skip = static_cast<uint16_t>(s.next_seq - h.first_seq);
    s.duplicate_messages += skip;
    record({SeqKind::Overlap, h.protocol_id, h.channel_id, h.session_id, s.next_seq, h.first_seq,
            h.message_count, s.next_offset, h.stream_offset, h.send_time, capture_ts});
  }
  s.messages += static_cast<uint64_t>(h.message_count - skip);
  s.next_seq = end;
  s.next_offset = h.stream_offset + h.payload_length;
  return skip;
}

struct DecodeStats {
  uint64_t packets = 0;
  uint64_t captured_bytes = 0;
  uint64_t udp_payload_bytes = 0;
  uint64_t net_errors[8] = {};
  uint64_t non_iextp = 0;
  uint64_t seg_errors[4] = {};
  uint64_t segments = 0;
  uint64_t heartbeats = 0;
  uint64_t messages = 0;
  uint64_t messages_skipped_dup = 0;
  uint64_t framing_errors = 0;
  uint64_t trailing_bytes = 0;
  uint64_t by_type[256] = {};
};

// Observer for segments, called before messages are dispatched.
struct NullSegmentObserver {
  void on_segment(const SegmentHeader&, int64_t /*capture_ts*/, uint16_t /*skip*/) {}
};

// Decodes one captured packet end to end. Returns false if the packet did not
// carry an IEX-TP segment.
template <class H, class SegObs = NullSegmentObserver>
class FeedDecoder {
 public:
  FeedDecoder(H& handler, SegObs* seg_obs = nullptr) : h_(handler), obs_(seg_obs) {}

  bool on_packet(const Packet& pkt) {
    ++stats_.packets;
    stats_.captured_bytes += pkt.caplen;
    UdpDatagram udp;
    NetStatus ns = parse_udp(pkt.link_type, pkt.data, pkt.caplen, udp);
    if (ns != NetStatus::Ok) {
      ++stats_.net_errors[static_cast<int>(ns)];
      return false;
    }
    return on_udp(udp.payload, udp.len, pkt.ts_ns);
  }

  bool on_udp(const uint8_t* payload, uint32_t len, int64_t capture_ts) {
    stats_.udp_payload_bytes += len;
    SegmentHeader h;
    const uint8_t* body = nullptr;
    SegStatus st = parse_segment(payload, len, h, body);
    if (st != SegStatus::Ok) {
      ++stats_.seg_errors[static_cast<int>(st)];
      ++stats_.non_iextp;
      return false;
    }
    ++stats_.segments;
    if (h.message_count == 0) ++stats_.heartbeats;
    uint16_t skip = seq_.observe(h, capture_ts);
    if (obs_) obs_->on_segment(h, capture_ts, skip);
    MessageIterator it(body, h.payload_length, h.message_count);
    const uint8_t* m;
    uint16_t mlen;
    int64_t seq = h.first_seq;
    MsgContext ctx{0, h.send_time, h.protocol_id};
    uint16_t i = 0;
    while (it.next(m, mlen)) {
      if (i++ < skip) {
        ++stats_.messages_skipped_dup;
        ++seq;
        continue;
      }
      ctx.seq = seq++;
      ++stats_.messages;
      if (mlen > 0) ++stats_.by_type[m[0]];
      dispatch(ctx, m, mlen, h_);
    }
    if (it.framing_error()) ++stats_.framing_errors;
    if (it.trailing_bytes()) ++stats_.trailing_bytes;
    return true;
  }

  const DecodeStats& stats() const { return stats_; }
  const SeqTracker& seq() const { return seq_; }
  SeqTracker& seq() { return seq_; }

 private:
  H& h_;
  SegObs* obs_;
  DecodeStats stats_;
  SeqTracker seq_;
};

}  // namespace mdp
