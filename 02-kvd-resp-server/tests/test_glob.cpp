#include <gtest/gtest.h>

#include <string>

#include "glob.h"

using kv::glob_match;

TEST(Glob, Basics) {
    EXPECT_TRUE(glob_match("*", ""));
    EXPECT_TRUE(glob_match("*", "anything"));
    EXPECT_TRUE(glob_match("h?llo", "hello"));
    EXPECT_TRUE(glob_match("h?llo", "hallo"));
    EXPECT_FALSE(glob_match("h?llo", "hllo"));
    EXPECT_TRUE(glob_match("h*llo", "hllo"));
    EXPECT_TRUE(glob_match("h*llo", "heeeello"));
    EXPECT_TRUE(glob_match("user:*:name", "user:42:name"));
    EXPECT_FALSE(glob_match("user:*:name", "user:42:email"));
    EXPECT_FALSE(glob_match("abc", "abcd"));
    EXPECT_FALSE(glob_match("abcd", "abc"));
}

TEST(Glob, Classes) {
    EXPECT_TRUE(glob_match("h[ae]llo", "hello"));
    EXPECT_TRUE(glob_match("h[ae]llo", "hallo"));
    EXPECT_FALSE(glob_match("h[ae]llo", "hillo"));
    EXPECT_TRUE(glob_match("h[^e]llo", "hallo"));
    EXPECT_FALSE(glob_match("h[^e]llo", "hello"));
    EXPECT_TRUE(glob_match("h[a-b]llo", "hbllo"));
    EXPECT_FALSE(glob_match("h[a-b]llo", "hcllo"));
    EXPECT_TRUE(glob_match("k[0-9][0-9]", "k42"));
}

TEST(Glob, Escapes) {
    EXPECT_TRUE(glob_match("a\\*b", "a*b"));
    EXPECT_FALSE(glob_match("a\\*b", "axb"));
    EXPECT_TRUE(glob_match("a\\?", "a?"));
}

TEST(Glob, BacktrackingIsPolynomial) {
    // A naive recursive matcher takes exponential time here.
    std::string s(200, 'a');
    EXPECT_FALSE(glob_match("*a*a*a*a*a*a*a*a*a*a*b", s));
    EXPECT_TRUE(glob_match("*a*a*a*a*a*a*a*a*a*a", s));
}
