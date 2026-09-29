#include "figgie/arena.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace figgie {

namespace {

double brier(const std::array<double, kSuits>& belief, int goal) {
  double b = 0.0;
  for (int s = 0; s < kSuits; ++s) {
    double y = s == goal ? 1.0 : 0.0;
    b += (belief[s] - y) * (belief[s] - y);
  }
  return b;
}

template <class F>
void parallel_for(int n, int threads, F&& f) {
  threads = std::max(1, std::min(threads, n));
  std::atomic<int> next{0};
  auto worker = [&] {
    for (int i = next.fetch_add(1); i < n; i = next.fetch_add(1)) f(i);
  };
  if (threads == 1) {
    worker();
    return;
  }
  std::vector<std::thread> pool;
  for (int t = 0; t < threads; ++t) pool.emplace_back(worker);
  for (auto& th : pool) th.join();
}

}  // namespace

MatchResults run_games(const std::vector<std::string>& lineup, int n_deals, uint64_t seed, const Config& cfg,
                       int n_checkpoints, bool rotate, bool check, int threads) {
  const int n = cfg.n_players;
  if (static_cast<int>(lineup.size()) != n) throw std::invalid_argument("lineup size must equal n_players");
  if (n_checkpoints < 1) n_checkpoints = 1;
  const int R = rotate ? n : 1;
  MatchResults out;
  out.n_games = n_deals * R;
  out.n_bots = n;
  out.n_checkpoints = n_checkpoints;
  out.deal.resize(out.n_games);
  out.rotation.resize(out.n_games);
  out.pnl.resize(static_cast<size_t>(out.n_games) * n);
  out.brier.resize(static_cast<size_t>(out.n_games) * n * n_checkpoints);
  out.goal_cards_end.resize(static_cast<size_t>(out.n_games) * n);
  out.n_trades.resize(out.n_games);
  std::atomic<uint64_t> violations{0};
  std::mutex mu;

  // Checkpoint c is taken at the end of tick ceil(T * (c + 1) / C).
  std::vector<int> cp_tick(n_checkpoints);
  for (int c = 0; c < n_checkpoints; ++c)
    cp_tick[c] = static_cast<int>(std::ceil(static_cast<double>(cfg.ticks) * (c + 1) / n_checkpoints));

  parallel_for(out.n_games, threads, [&](int gi) {
    const int d = gi / R, r = gi % R;
    const uint64_t deal_seed = mix_seed(seed, static_cast<uint64_t>(d));
    Game g(cfg, deal_seed);
    // Lineup position i sits in seat (i + r) mod n.
    std::vector<std::unique_ptr<Bot>> by_seat(n);
    std::vector<int> bot_of_seat(n);
    for (int i = 0; i < n; ++i) {
      int seat = (i + r) % n;
      bot_of_seat[seat] = i;
      by_seat[seat] = make_bot(lineup[i]);
      by_seat[seat]->reset(View(g, seat), mix_seed(deal_seed, 1000 + i));
    }
    int cp = 0;
    while (!g.done()) {
      auto order = g.action_order();
      for (int k = 0; k < n; ++k) {
        int p = order[k];
        Action a = by_seat[p]->act(View(g, p));
        g.apply(p, a);
        if (check) {
          std::string e = g.check_invariants();
          if (!e.empty()) {
            if (violations.fetch_add(1) == 0) {
              std::lock_guard<std::mutex> lk(mu);
              out.first_violation = e;
            }
          }
        }
      }
      int t_end = g.tick() + 1;
      while (cp < n_checkpoints && t_end >= cp_tick[cp]) {
        // Record beliefs before settlement, while the goal is still hidden from bots.
        for (int p = 0; p < n; ++p) {
          size_t idx = (static_cast<size_t>(gi) * n + bot_of_seat[p]) * n_checkpoints + cp;
          out.brier[idx] = brier(by_seat[p]->goal_belief(View(g, p)), g.goal_suit());
        }
        ++cp;
      }
      g.end_tick();
    }
    if (check) {
      std::string e = g.check_invariants();
      if (!e.empty() && violations.fetch_add(1) == 0) {
        std::lock_guard<std::mutex> lk(mu);
        out.first_violation = e;
      }
    }
    out.deal[gi] = d;
    out.rotation[gi] = r;
    out.n_trades[gi] = static_cast<int>(g.trades().size());
    for (int p = 0; p < n; ++p) {
      out.pnl[static_cast<size_t>(gi) * n + bot_of_seat[p]] = g.pnl(p);
      out.goal_cards_end[static_cast<size_t>(gi) * n + bot_of_seat[p]] = g.hand(p, g.goal_suit());
    }
  });
  out.invariant_violations = violations.load();
  return out;
}

FuzzResults fuzz(uint64_t n_games, uint64_t seed, int threads, bool allow_bayes) {
  FuzzResults total;
  std::mutex mu;
  const int chunks = std::max(1, threads) * 8;
  parallel_for(chunks, threads, [&](int chunk) {
    FuzzResults local;
    Rng rng(mix_seed(seed, 0xF022 + chunk));
    uint64_t lo = n_games * chunk / chunks, hi = n_games * (chunk + 1) / chunks;
    auto names = bot_names();
    if (!allow_bayes) names.erase(std::remove(names.begin(), names.end(), std::string("bayes")), names.end());
    for (uint64_t gi = lo; gi < hi; ++gi) {
      Config cfg;
      cfg.n_players = rng.below(4) == 0 ? 5 : 4;
      cfg.clear_on_trade = rng.below(4) != 0;
      cfg.ticks = 1 + static_cast<int>(rng.below(60));
      cfg.max_price = 20 + static_cast<int>(rng.below(81));
      cfg.start_cash = cfg.ante() + static_cast<int>(rng.below(400));
      Game g(cfg, rng.next());
      // Seat types: -1 = chaos (uniform random, often invalid), else a bot.
      std::vector<int> kind(cfg.n_players);
      std::vector<std::unique_ptr<Bot>> bots(cfg.n_players);
      for (int p = 0; p < cfg.n_players; ++p) {
        kind[p] = static_cast<int>(rng.below(names.size() + 2)) - 2;
        if (kind[p] >= 0) {
          bots[p] = make_bot(names[kind[p]]);
          bots[p]->reset(View(g, p), rng.next());
        }
      }
      while (!g.done()) {
        auto order = g.action_order();
        for (int k = 0; k < cfg.n_players; ++k) {
          int p = order[k];
          Action a;
          if (kind[p] < 0) {
            a.type = static_cast<ActType>(rng.below(kNumActTypes));
            a.suit = static_cast<int>(rng.below(kSuits + 2)) - 1;
            a.price = static_cast<int>(rng.below(cfg.max_price + 10)) - 4;
            if (rng.below(2)) a.price = 1 + static_cast<int>(rng.below(25));
          } else {
            a = bots[p]->act(View(g, p));
          }
          Status st = g.apply(p, a);
          ++local.actions;
          if (st == Status::kRejected) ++local.rejected;
          std::string e = g.check_invariants();
          if (!e.empty() && local.violations++ == 0) local.first_violation = e;
        }
        g.end_tick();
      }
      std::string e = g.check_invariants();
      if (!e.empty() && local.violations++ == 0) local.first_violation = e;
      local.trades += g.trades().size();
      ++local.games;
    }
    std::lock_guard<std::mutex> lk(mu);
    total.games += local.games;
    total.actions += local.actions;
    total.trades += local.trades;
    total.rejected += local.rejected;
    if (local.violations && total.violations == 0) total.first_violation = local.first_violation;
    total.violations += local.violations;
  });
  return total;
}

// ------------------------------------------------------------------ VecEnv

VecEnv::VecEnv(int num_envs, int n_learners, const std::vector<std::string>& opponents, const Config& cfg,
               uint64_t seed, bool check)
    : num_envs_(num_envs), n_learners_(n_learners), cfg_(cfg), seed_(seed), check_(check), opponents_(opponents) {
  if (num_envs < 1) throw std::invalid_argument("num_envs must be >= 1");
  if (n_learners < 1 || n_learners > cfg.n_players) throw std::invalid_argument("bad n_learners");
  if (static_cast<int>(opponents.size()) != cfg.n_players - n_learners)
    throw std::invalid_argument("need exactly n_players - n_learners opponents");
  for (int e = 0; e < num_envs; ++e) {
    games_.push_back(std::make_unique<Game>(cfg_, 1));
    bots_.emplace_back();
    for (const auto& name : opponents_) bots_.back().push_back(make_bot(name));
  }
  for (int e = 0; e < num_envs; ++e) reset_env(e);
}

void VecEnv::reset_env(int e) {
  uint64_t s = mix_seed(seed_, episode_counter_++);
  games_[e]->reset(s);
  for (size_t i = 0; i < bots_[e].size(); ++i)
    bots_[e][i]->reset(View(*games_[e], n_learners_ + static_cast<int>(i)), mix_seed(s, 77 + i));
}

void VecEnv::reset(float* obs) {
  for (int e = 0; e < num_envs_; ++e) {
    reset_env(e);
    write_obs(e, obs);
  }
}

void VecEnv::step(const int32_t* actions, float* obs, float* reward, uint8_t* done) {
  for (int e = 0; e < num_envs_; ++e) {
    Game& g = *games_[e];
    auto order = g.action_order();
    for (int k = 0; k < cfg_.n_players; ++k) {
      int p = order[k];
      Action a;
      if (p < n_learners_) {
        const int32_t* x = actions + (static_cast<size_t>(e) * n_learners_ + p) * 3;
        int t = x[0];
        a.type = (t >= 0 && t < kNumActTypes) ? static_cast<ActType>(t) : ActType::kPass;
        a.suit = x[1];
        a.price = x[2];
        if (g.apply(p, a) == Status::kRejected) ++rejected_;
      } else {
        a = bots_[e][p - n_learners_]->act(View(g, p));
        g.apply(p, a);
      }
      if (check_ && !g.check_invariants().empty()) ++violations_;
    }
    g.end_tick();
    ++steps_;
    for (int l = 0; l < n_learners_; ++l) reward[static_cast<size_t>(e) * n_learners_ + l] = 0.0f;
    done[e] = g.done() ? 1 : 0;
    if (g.done()) {
      if (check_ && !g.check_invariants().empty()) ++violations_;
      for (int l = 0; l < n_learners_; ++l)
        reward[static_cast<size_t>(e) * n_learners_ + l] = static_cast<float>(g.pnl(l));
      ++episodes_;
      reset_env(e);
    }
    write_obs(e, obs);
  }
}

std::array<double, kSuits> VecEnv::bayes_goal_probs(int env, int learner) const {
  const Game& g = *games_[env];
  TapeTracker tt;
  tt.reset(cfg_.n_players, learner);
  tt.update(g.trades());
  return goal_probs(config_posterior(g.initial_hand(learner), tt.others_min(), cfg_.hand_size()));
}

void VecEnv::write_obs(int e, float* obs) const {
  const Game& g = *games_[e];
  const float mp = static_cast<float>(cfg_.max_price);
  // Public tape summaries shared by all learners.
  std::array<int, kSuits> last_px{}, n_tr{};
  std::array<Counts, kMaxPlayers> net{};
  for (const Trade& t : g.trades()) {
    last_px[t.suit] = t.price;
    n_tr[t.suit]++;
    net[t.buyer][t.suit]++;
    net[t.seller][t.suit]--;
  }
  for (int p = 0; p < n_learners_; ++p) {
    float* o = obs + (static_cast<size_t>(e) * n_learners_ + p) * kObsDim;
    int k = 0;
    for (int s = 0; s < kSuits; ++s) o[k++] = g.hand(p, s) / 10.0f;
    o[k++] = g.cash(p) / 100.0f;
    o[k++] = static_cast<float>(g.tick()) / cfg_.ticks;
    for (int s = 0; s < kSuits; ++s) {
      const SuitBook& b = g.book(s);
      auto bb = b.best(SuitBook::kBidSide);
      auto ba = b.best(SuitBook::kAskSide);
      o[k++] = bb ? bb->price / mp : 0.0f;
      o[k++] = bb ? 1.0f : 0.0f;
      o[k++] = ba ? ba->price / mp : 0.0f;
      o[k++] = ba ? 1.0f : 0.0f;
      o[k++] = b.find(SuitBook::kBidSide, p) >= 0 ? 1.0f : 0.0f;
      o[k++] = b.find(SuitBook::kAskSide, p) >= 0 ? 1.0f : 0.0f;
      o[k++] = last_px[s] / mp;
      o[k++] = n_tr[s] / 10.0f;
    }
    // Net bought per player, seats relative to the observer (self first).
    for (int j = 0; j < kMaxPlayers; ++j) {
      int q = (p + j) % cfg_.n_players;
      for (int s = 0; s < kSuits; ++s) o[k++] = j < cfg_.n_players ? net[q][s] / 10.0f : 0.0f;
    }
    for (int s = 0; s < kSuits; ++s) o[k++] = g.initial_hand(p)[s] / 10.0f;
  }
}

}  // namespace figgie
