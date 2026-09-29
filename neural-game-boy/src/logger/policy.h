#pragma once
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "core/types.h"

namespace gb::rec {

// A policy maps a step index (and, later, an observation) to a button mask.
class Policy {
 public:
  virtual ~Policy() = default;
  virtual u8 act(u64 step) = 0;
  virtual std::string name() const = 0;
};

// Weighted random choice over a small discrete action set, holding each
// choice for a random number of steps ("sticky" actions). Holding matters:
// a platformer only reveals jump arcs if A stays down for a while.
class RandomPolicy : public Policy {
 public:
  explicit RandomPolicy(u64 seed, int min_hold = 1, int max_hold = 12);
  u8 act(u64 step) override;
  std::string name() const override { return "random"; }

  static const std::vector<std::pair<u8, double>>& action_table();

 private:
  std::mt19937_64 rng_;
  std::discrete_distribution<int> pick_;
  std::uniform_int_distribution<int> hold_;
  u8 current_ = 0;
  int remaining_ = 0;
};

// Replays "<steps> <buttons>" lines, e.g. "30 -", "4 START", "20 A+RIGHT".
// When the script ends it hands over to `after` (or holds no buttons).
class ScriptPolicy : public Policy {
 public:
  ScriptPolicy(std::vector<std::pair<u64, u8>> segments, std::unique_ptr<Policy> after);
  static std::unique_ptr<ScriptPolicy> from_file(const std::string& path, std::unique_ptr<Policy> after);
  u8 act(u64 step) override;
  std::string name() const override { return after_ ? "script+" + after_->name() : "script"; }
  u64 script_steps() const { return total_; }

 private:
  std::vector<std::pair<u64, u8>> segments_;
  std::unique_ptr<Policy> after_;
  u64 total_ = 0;
};

// "A+RIGHT" -> kBtnA | kBtnRight; "-" -> 0. Throws on unknown names.
u8 parse_buttons(const std::string& spec);
std::string buttons_to_string(u8 mask);

}  // namespace gb::rec
