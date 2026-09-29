#include "c4/puct.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace c4 {

// ---------------------------------------------------------------- PuctTree

void PuctTree::reset(const Position& root) {
  if (root.is_terminal()) throw std::invalid_argument("search on terminal position");
  nodes_.clear();
  nodes_.reserve(1024);
  nodes_.push_back({root, -1, -1, 0, -1, 0, 0.0f, 1.0f, 0, 0.0f});
  sims_ = 0;
}

int PuctTree::select_child(int idx, const PuctConfig& cfg) const {
  const Node& n = nodes_[idx];
  const float sqrt_n = std::sqrt(static_cast<float>(std::max<std::uint32_t>(n.visits, 1)));
  int best = -1;
  float best_score = -1e30f;
  for (int k = 0; k < n.num_children; ++k) {
    const int ci = n.first_child + k;
    const Node& ch = nodes_[ci];
    const float q = ch.visits ? ch.value_sum / static_cast<float>(ch.visits) : 0.0f;
    const float u = cfg.c_puct * ch.prior * sqrt_n / (1.0f + static_cast<float>(ch.visits));
    const float score = q + u;
    if (score > best_score) {
      best_score = score;
      best = ci;
    }
  }
  return best;
}

void PuctTree::backup(int idx, float value_for_mover) {
  float v = value_for_mover;
  while (idx >= 0) {
    nodes_[idx].visits += 1;
    nodes_[idx].value_sum += v;
    v = -v;
    idx = nodes_[idx].parent;
  }
  ++sims_;
}

int PuctTree::select_leaf(int target_sims, const PuctConfig& cfg) {
  while (sims_ < target_sims) {
    int idx = 0;
    while (nodes_[idx].first_child >= 0 && !nodes_[idx].terminal) idx = select_child(idx, cfg);
    if (nodes_[idx].terminal) {
      backup(idx, nodes_[idx].terminal_value);
      continue;
    }
    return idx;
  }
  return -1;
}

void PuctTree::expand_and_backup(int leaf, const float* policy, float value, const PuctConfig& cfg,
                                 std::mt19937_64& rng) {
  if (nodes_[leaf].first_child >= 0) throw std::logic_error("leaf already expanded");
  const Position pos = nodes_[leaf].pos;
  float total = 0.0f;
  for (int c = 0; c < kWidth; ++c)
    if (pos.can_play(c)) total += std::max(policy[c], 0.0f);

  std::array<float, kWidth> priors{};
  int legal = 0;
  for (int c = 0; c < kWidth; ++c) {
    if (!pos.can_play(c)) continue;
    ++legal;
    priors[c] = total > 0.0f ? std::max(policy[c], 0.0f) / total : 0.0f;
  }
  if (total <= 0.0f)
    for (int c = 0; c < kWidth; ++c)
      if (pos.can_play(c)) priors[c] = 1.0f / static_cast<float>(legal);

  if (leaf == 0 && cfg.dirichlet_eps > 0.0f) {
    std::gamma_distribution<float> gamma(cfg.dirichlet_alpha, 1.0f);
    std::array<float, kWidth> noise{};
    float nsum = 0.0f;
    for (int c = 0; c < kWidth; ++c)
      if (pos.can_play(c)) nsum += (noise[c] = gamma(rng));
    if (nsum > 0.0f)
      for (int c = 0; c < kWidth; ++c)
        if (pos.can_play(c))
          priors[c] = (1.0f - cfg.dirichlet_eps) * priors[c] + cfg.dirichlet_eps * noise[c] / nsum;
  }

  const int first = static_cast<int>(nodes_.size());
  for (int c = 0; c < kWidth; ++c) {
    if (!pos.can_play(c)) continue;
    Position q = pos;
    q.play(c);
    Node ch{q, leaf, -1, 0, static_cast<std::int8_t>(c), 0, 0.0f, priors[c], 0, 0.0f};
    if (q.last_mover_won()) {
      ch.terminal = 1;
      ch.terminal_value = 1.0f;
    } else if (q.is_full()) {
      ch.terminal = 1;
      ch.terminal_value = 0.0f;
    }
    nodes_.push_back(ch);
  }
  nodes_[leaf].first_child = first;
  nodes_[leaf].num_children = static_cast<std::int8_t>(legal);
  // value is for the side to move at the leaf; the node stores the view of
  // the player who moved into it.
  backup(leaf, -value);
}

std::array<float, kWidth> PuctTree::policy_target() const {
  std::array<float, kWidth> pi{};
  const Node& r = nodes_[0];
  float total = 0.0f;
  for (int k = 0; k < r.num_children; ++k) total += static_cast<float>(nodes_[r.first_child + k].visits);
  for (int k = 0; k < r.num_children; ++k) {
    const Node& ch = nodes_[r.first_child + k];
    pi[ch.move] = total > 0.0f ? static_cast<float>(ch.visits) / total : ch.prior;
  }
  return pi;
}

int PuctTree::choose_move(bool sample, std::mt19937_64& rng) const {
  const Node& r = nodes_[0];
  if (r.first_child < 0) throw std::logic_error("choose_move before root expansion");
  if (sample) {
    const auto pi = policy_target();
    std::discrete_distribution<int> dist(pi.begin(), pi.end());
    return dist(rng);
  }
  int best = -1;
  for (int k = 0; k < r.num_children; ++k) {
    const Node& ch = nodes_[r.first_child + k];
    if (best < 0) {
      best = r.first_child + k;
      continue;
    }
    const Node& b = nodes_[best];
    if (ch.visits > b.visits || (ch.visits == b.visits && ch.prior > b.prior)) best = r.first_child + k;
  }
  return nodes_[best].move;
}

float PuctTree::root_value() const {
  const Node& r = nodes_[0];
  return r.visits ? -r.value_sum / static_cast<float>(r.visits) : 0.0f;
}

// ------------------------------------------------------------ BatchedGames

BatchedGames::BatchedGames(int num_slots, int max_games, PuctConfig cfg, std::uint64_t seed,
                           int opponent_rollouts, std::vector<std::string> openings)
    : cfg_(cfg),
      max_games_(max_games),
      opponent_rollouts_(opponent_rollouts),
      openings_(std::move(openings)),
      rng_(seed),
      slots_(static_cast<std::size_t>(num_slots)) {
  if (num_slots <= 0) throw std::invalid_argument("num_slots must be positive");
  for (auto& s : slots_) start_game(s);
}

void BatchedGames::start_game(Slot& s) {
  s.active = false;
  s.pending = -1;
  if (started_games_ >= max_games_) return;
  s.game_index = started_games_++;
  s.active = true;
  s.pos = Position();
  s.moves.clear();
  s.obs.clear();
  s.pi.clear();
  s.side.clear();
  if (opponent_rollouts_ > 0) {
    s.net_side = s.game_index % 2;
    s.opponent = std::make_unique<PureMCTS>(opponent_rollouts_, rng_());
  } else {
    s.net_side = -1;
  }
  if (!openings_.empty()) {
    const std::string& op = openings_[static_cast<std::size_t>(s.game_index / 2) % openings_.size()];
    for (char ch : op) {
      const int col = ch - '1';
      if (col < 0 || col >= kWidth || !s.pos.can_play(col)) throw std::invalid_argument("bad opening");
      s.pos.play(col);
      s.moves += ch;
      if (s.pos.is_terminal()) throw std::invalid_argument("opening ends the game");
    }
  }
  s.tree.reset(s.pos);
}

void BatchedGames::finish_game(Slot& s, int winner) {
  games_.push_back({winner, s.net_side, s.moves});
  if (s.net_side < 0) {
    const std::size_t n = s.side.size();
    for (std::size_t i = 0; i < n; ++i) {
      samples_.obs.insert(samples_.obs.end(), s.obs.begin() + static_cast<long>(i * kObsSize),
                          s.obs.begin() + static_cast<long>((i + 1) * kObsSize));
      samples_.pi.insert(samples_.pi.end(), s.pi.begin() + static_cast<long>(i * kWidth),
                         s.pi.begin() + static_cast<long>((i + 1) * kWidth));
      samples_.z.push_back(winner < 0 ? 0.0f : (winner == s.side[i] ? 1.0f : -1.0f));
    }
  }
  ++finished_games_;
  start_game(s);
}

bool BatchedGames::apply_move(Slot& s, int col) {
  const int mover = s.pos.side_to_move();
  s.pos.play(col);
  s.moves += static_cast<char>('1' + col);
  ++total_moves_;
  if (s.pos.last_mover_won()) {
    finish_game(s, mover);
    return true;
  }
  if (s.pos.is_full()) {
    finish_game(s, -1);
    return true;
  }
  s.tree.reset(s.pos);
  return false;
}

int BatchedGames::advance(Slot& s) {
  while (s.active) {
    if (s.net_side >= 0 && s.pos.side_to_move() != s.net_side) {
      apply_move(s, s.opponent->choose_move(s.pos));
      continue;
    }
    const int leaf = s.tree.select_leaf(cfg_.num_sims, cfg_);
    if (leaf >= 0) return leaf;
    // Search finished: record and play.
    if (s.net_side < 0) {
      std::vector<float> buf(kObsSize);
      s.pos.encode(buf.data());
      s.obs.insert(s.obs.end(), buf.begin(), buf.end());
      const auto pi = s.tree.policy_target();
      s.pi.insert(s.pi.end(), pi.begin(), pi.end());
      s.side.push_back(s.pos.side_to_move());
    }
    const bool sample = s.net_side < 0 && s.pos.moves() < cfg_.temp_moves;
    apply_move(s, s.tree.choose_move(sample, rng_));
  }
  return -1;
}

int BatchedGames::gather(float* obs_out) {
  row_slot_.clear();
  int n = 0;
  for (std::size_t i = 0; i < slots_.size(); ++i) {
    Slot& s = slots_[i];
    if (!s.active) continue;
    const int leaf = advance(s);
    if (leaf < 0) continue;
    s.pending = leaf;
    s.tree.node(leaf).pos.encode(obs_out + static_cast<std::ptrdiff_t>(n) * kObsSize);
    row_slot_.push_back(static_cast<int>(i));
    ++n;
  }
  return n;
}

void BatchedGames::scatter(const float* policy, const float* value) {
  for (std::size_t r = 0; r < row_slot_.size(); ++r) {
    Slot& s = slots_[static_cast<std::size_t>(row_slot_[r])];
    s.tree.expand_and_backup(s.pending, policy + r * kWidth, value[r], cfg_, rng_);
    s.pending = -1;
    ++total_evals_;
  }
  row_slot_.clear();
}

BatchedGames::Samples BatchedGames::drain_samples() {
  Samples out = std::move(samples_);
  samples_ = Samples{};
  return out;
}

std::vector<GameRecord> BatchedGames::drain_games() {
  std::vector<GameRecord> out = std::move(games_);
  games_.clear();
  return out;
}

// --------------------------------------------------------- BatchedAnalysis

BatchedAnalysis::BatchedAnalysis(std::vector<Position> positions, PuctConfig cfg, std::uint64_t seed)
    : cfg_(cfg), rng_(seed), trees_(positions.size()), pending_(positions.size(), -1) {
  for (std::size_t i = 0; i < positions.size(); ++i) trees_[i].reset(positions[i]);
}

int BatchedAnalysis::gather(float* obs_out) {
  row_tree_.clear();
  int n = 0;
  for (std::size_t i = 0; i < trees_.size(); ++i) {
    const int leaf = trees_[i].select_leaf(cfg_.num_sims, cfg_);
    pending_[i] = leaf;
    if (leaf < 0) continue;
    trees_[i].node(leaf).pos.encode(obs_out + static_cast<std::ptrdiff_t>(n) * kObsSize);
    row_tree_.push_back(static_cast<int>(i));
    ++n;
  }
  return n;
}

void BatchedAnalysis::scatter(const float* policy, const float* value) {
  for (std::size_t r = 0; r < row_tree_.size(); ++r) {
    const auto t = static_cast<std::size_t>(row_tree_[r]);
    trees_[t].expand_and_backup(pending_[t], policy + r * kWidth, value[r], cfg_, rng_);
    pending_[t] = -1;
  }
  row_tree_.clear();
}

bool BatchedAnalysis::done() const {
  return std::all_of(trees_.begin(), trees_.end(),
                     [&](const PuctTree& t) { return t.sims() >= cfg_.num_sims; });
}

std::vector<std::array<float, kWidth>> BatchedAnalysis::policies() const {
  std::vector<std::array<float, kWidth>> out;
  for (const auto& t : trees_) out.push_back(t.policy_target());
  return out;
}

std::vector<int> BatchedAnalysis::best_moves() const {
  std::vector<int> out;
  std::mt19937_64 dummy(0);
  for (const auto& t : trees_) out.push_back(t.choose_move(false, dummy));
  return out;
}

std::vector<float> BatchedAnalysis::root_values() const {
  std::vector<float> out;
  for (const auto& t : trees_) out.push_back(t.root_value());
  return out;
}

}  // namespace c4
