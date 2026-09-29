#include "logger/policy.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace gb::rec {

const std::vector<std::pair<u8, double>>& RandomPolicy::action_table() {
  // Start and Select are rare so random play does not sit in pause menus.
  static const std::vector<std::pair<u8, double>> table = {
      {0, 1.0},
      {kBtnA, 1.0},
      {kBtnB, 0.5},
      {kBtnUp, 0.5},
      {kBtnDown, 0.5},
      {kBtnLeft, 1.0},
      {kBtnRight, 1.0},
      {u8(kBtnA | kBtnLeft), 1.0},
      {u8(kBtnA | kBtnRight), 1.0},
      {u8(kBtnB | kBtnLeft), 0.25},
      {u8(kBtnB | kBtnRight), 0.25},
      {kBtnStart, 0.05},
      {kBtnSelect, 0.02},
  };
  return table;
}

RandomPolicy::RandomPolicy(u64 seed, int min_hold, int max_hold)
    : rng_(seed), hold_(min_hold, max_hold) {
  std::vector<double> w;
  for (auto& [mask, weight] : action_table()) w.push_back(weight);
  pick_ = std::discrete_distribution<int>(w.begin(), w.end());
}

u8 RandomPolicy::act(u64) {
  if (remaining_ <= 0) {
    current_ = action_table()[size_t(pick_(rng_))].first;
    remaining_ = hold_(rng_);
  }
  --remaining_;
  return current_;
}

ScriptPolicy::ScriptPolicy(std::vector<std::pair<u64, u8>> segments, std::unique_ptr<Policy> after)
    : segments_(std::move(segments)), after_(std::move(after)) {
  for (auto& s : segments_) total_ += s.first;
}

std::unique_ptr<ScriptPolicy> ScriptPolicy::from_file(const std::string& path, std::unique_ptr<Policy> after) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open script " + path);
  std::vector<std::pair<u64, u8>> segs;
  std::string line;
  while (std::getline(f, line)) {
    auto hash = line.find('#');
    if (hash != std::string::npos) line.resize(hash);
    std::istringstream is(line);
    u64 n;
    std::string buttons;
    if (!(is >> n)) continue;
    if (!(is >> buttons)) buttons = "-";
    segs.emplace_back(n, parse_buttons(buttons));
  }
  return std::make_unique<ScriptPolicy>(std::move(segs), std::move(after));
}

u8 ScriptPolicy::act(u64 step) {
  if (step >= total_) return after_ ? after_->act(step - total_) : 0;
  u64 acc = 0;
  for (auto& [n, mask] : segments_) {
    acc += n;
    if (step < acc) return mask;
  }
  return 0;
}

u8 parse_buttons(const std::string& spec) {
  if (spec.empty() || spec == "-") return 0;
  u8 mask = 0;
  std::stringstream ss(spec);
  std::string tok;
  while (std::getline(ss, tok, '+')) {
    for (auto& c : tok) c = char(std::toupper(static_cast<unsigned char>(c)));
    if (tok == "A") mask |= kBtnA;
    else if (tok == "B") mask |= kBtnB;
    else if (tok == "SELECT") mask |= kBtnSelect;
    else if (tok == "START") mask |= kBtnStart;
    else if (tok == "RIGHT") mask |= kBtnRight;
    else if (tok == "LEFT") mask |= kBtnLeft;
    else if (tok == "UP") mask |= kBtnUp;
    else if (tok == "DOWN") mask |= kBtnDown;
    else throw std::runtime_error("unknown button: " + tok);
  }
  return mask;
}

std::string buttons_to_string(u8 mask) {
  static const char* names[8] = {"A", "B", "SELECT", "START", "RIGHT", "LEFT", "UP", "DOWN"};
  std::string s;
  for (int i = 0; i < 8; ++i)
    if (mask & (1 << i)) s += (s.empty() ? "" : "+") + std::string(names[i]);
  return s.empty() ? "-" : s;
}

}  // namespace gb::rec
