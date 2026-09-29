#include "exchange/types.hpp"

#include <gtest/gtest.h>

using namespace exch;

TEST(FixedPoint, ParsesExactDecimals) {
    EXPECT_EQ(parse_fixed("101.25", 2), 10125);
    EXPECT_EQ(parse_fixed("101.2", 2), 10120);
    EXPECT_EQ(parse_fixed("101", 2), 10100);
    EXPECT_EQ(parse_fixed("0.01", 2), 1);
    EXPECT_EQ(parse_fixed("-3.5", 1), -35);
    EXPECT_EQ(parse_fixed("7", 0), 7);
}

TEST(FixedPoint, RejectsInexactOrMalformed) {
    EXPECT_FALSE(parse_fixed("101.255", 2).has_value()); // would need rounding
    EXPECT_FALSE(parse_fixed("", 2).has_value());
    EXPECT_FALSE(parse_fixed(".", 2).has_value());
    EXPECT_FALSE(parse_fixed("1.2.3", 2).has_value());
    EXPECT_FALSE(parse_fixed("12a", 2).has_value());
    EXPECT_FALSE(parse_fixed("99999999999999999999", 2).has_value()); // overflow
}

TEST(FixedPoint, FormatRoundTrips) {
    EXPECT_EQ(format_fixed(10125, 2), "101.25");
    EXPECT_EQ(format_fixed(1, 2), "0.01");
    EXPECT_EQ(format_fixed(-35, 1), "-3.5");
    EXPECT_EQ(format_fixed(42, 0), "42");
    for (std::int64_t v : {0LL, 1LL, 99LL, 100LL, 123456789LL, -7LL})
        EXPECT_EQ(parse_fixed(format_fixed(v, 4), 4), v);
}
