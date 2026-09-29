#include <gtest/gtest.h>

#include "mdp/feed.hpp"

using namespace mdp;

namespace {
SegmentHeader seg(int64_t first, uint16_t count, int64_t offset, uint16_t payload, uint32_t session = 7) {
  SegmentHeader h;
  h.version = 1;
  h.protocol_id = kProtoTops16;
  h.channel_id = 1;
  h.session_id = session;
  h.first_seq = first;
  h.message_count = count;
  h.stream_offset = offset;
  h.payload_length = payload;
  h.send_time = first * 10;
  return h;
}
}  // namespace

TEST(SeqTracker, InOrderWithHeartbeats) {
  SeqTracker t;
  EXPECT_EQ(t.observe(seg(1, 0, 0, 0), 0), 0);   // heartbeat before first message
  EXPECT_EQ(t.observe(seg(1, 3, 0, 60), 0), 0);
  EXPECT_EQ(t.observe(seg(4, 2, 60, 40), 0), 0);
  EXPECT_EQ(t.observe(seg(6, 0, 100, 0), 0), 0);  // heartbeat carries the next seq and offset
  const auto& s = t.sessions().begin()->second;
  EXPECT_EQ(s.messages, 5u);
  EXPECT_EQ(s.heartbeats, 2u);
  EXPECT_EQ(s.gaps + s.duplicates + s.overlaps + s.offset_mismatches, 0u);
  EXPECT_EQ(s.next_seq, 6);
  EXPECT_EQ(t.anomaly_count(), 1u);  // only the session_start marker
}

TEST(SeqTracker, GapDuplicateOverlapAndOffset) {
  SeqTracker t;
  t.observe(seg(1, 3, 0, 60), 0);
  // gap: 4..9 missing
  EXPECT_EQ(t.observe(seg(10, 2, 200, 40), 0), 0);
  // exact duplicate of the previous segment
  EXPECT_EQ(t.observe(seg(10, 2, 200, 40), 0), 2);
  // overlap: 11 seen, 12 and 13 new
  EXPECT_EQ(t.observe(seg(11, 3, 220, 60), 0), 1);
  // in order sequence but wrong offset
  EXPECT_EQ(t.observe(seg(14, 1, 999, 20), 0), 0);
  const auto& s = t.sessions().begin()->second;
  EXPECT_EQ(s.gaps, 1u);
  EXPECT_EQ(s.gap_messages, 6u);
  EXPECT_EQ(s.duplicates, 1u);
  EXPECT_EQ(s.overlaps, 1u);
  EXPECT_EQ(s.duplicate_messages, 3u);
  EXPECT_EQ(s.offset_mismatches, 1u);
  EXPECT_EQ(s.messages, 3u + 2u + 2u + 1u);
  EXPECT_EQ(s.next_seq, 15);
}

TEST(SeqTracker, SessionsAreIndependent) {
  SeqTracker t;
  t.observe(seg(1, 2, 0, 40, 1), 0);
  t.observe(seg(1, 2, 0, 40, 2), 0);
  t.observe(seg(3, 2, 40, 40, 1), 0);
  EXPECT_EQ(t.sessions().size(), 2u);
  for (const auto& [k, s] : t.sessions()) EXPECT_EQ(s.gaps, 0u);
}
