#include <gtest/gtest.h>

#include "se/json.hpp"

using namespace se;

TEST(Json, ExtractsTopLevelStrings) {
  JsonFields f;
  std::string err;
  ASSERT_TRUE(parse_json_string_fields(
      R"({"_id": "42", "title": "A title", "text": "Body", "metadata": {"x": [1, 2, {"y": "z"}]}, "n": -1.5e3, "t": true})",
      f, &err))
      << err;
  EXPECT_EQ(json_get(f, "_id"), "42");
  EXPECT_EQ(json_get(f, "title"), "A title");
  EXPECT_EQ(json_get(f, "text"), "Body");
  EXPECT_EQ(json_get(f, "y"), "");  // nested strings are not promoted
  EXPECT_EQ(json_get(f, "missing"), "");
}

TEST(Json, DecodesEscapes) {
  JsonFields f;
  ASSERT_TRUE(parse_json_string_fields(R"({"a": "q\"b\\s\/n\nt\tu\u00e9x\ud83d\ude00"})", f));
  EXPECT_EQ(json_get(f, "a"), "q\"b\\s/n\nt\tu\xC3\xA9x\xF0\x9F\x98\x80");
}

TEST(Json, LoneSurrogateBecomesReplacementChar) {
  JsonFields f;
  ASSERT_TRUE(parse_json_string_fields(R"({"a": "x\ud800y"})", f));
  EXPECT_EQ(json_get(f, "a"), "x\xEF\xBF\xBDy");
}

TEST(Json, RejectsMalformed) {
  JsonFields f;
  for (const char* bad : {"", "[]", "\"str\"", "{", "{\"a\"}", "{\"a\": \"x}", "{\"a\": 1,}",
                          "{\"a\": \"x\"} trailing", "{\"a\": \"\\q\"}", "{\"a\": tru}"}) {
    std::string err;
    EXPECT_FALSE(parse_json_string_fields(bad, f, &err)) << bad;
    EXPECT_FALSE(err.empty()) << bad;
  }
}
