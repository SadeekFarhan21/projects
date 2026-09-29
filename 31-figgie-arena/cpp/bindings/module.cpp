// pybind11 bindings: Game (for tests and tools), posterior functions, the
// tournament runner, the fuzzer and the vectorised environment.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "figgie/arena.hpp"
#include "figgie/game.hpp"
#include "figgie/posterior.hpp"

namespace py = pybind11;
using namespace figgie;

namespace {

Counts to_counts(const std::vector<int>& v) {
  if (v.size() != kSuits) throw std::invalid_argument("expected 4 suit counts");
  return Counts{v[0], v[1], v[2], v[3]};
}
std::vector<Counts> to_counts_list(const std::vector<std::vector<int>>& v) {
  std::vector<Counts> out;
  for (const auto& x : v) out.push_back(to_counts(x));
  return out;
}

template <class T>
py::array_t<T> to_array(const std::vector<T>& v, std::vector<py::ssize_t> shape) {
  py::array_t<T> a(shape);
  std::copy(v.begin(), v.end(), a.mutable_data());
  return a;
}

}  // namespace

PYBIND11_MODULE(_figgie, m) {
  m.doc() = "Figgie rules engine and arena (C++20)";

  py::class_<Config>(m, "Config")
      .def(py::init<>())
      .def(py::init([](int n_players, int ticks, int start_cash, int max_price, bool clear_on_trade) {
             Config c;
             c.n_players = n_players;
             c.ticks = ticks;
             c.start_cash = start_cash;
             c.max_price = max_price;
             c.clear_on_trade = clear_on_trade;
             return c;
           }),
           py::arg("n_players") = 4, py::arg("ticks") = 200, py::arg("start_cash") = 350,
           py::arg("max_price") = 100, py::arg("clear_on_trade") = true)
      .def_readwrite("n_players", &Config::n_players)
      .def_readwrite("ticks", &Config::ticks)
      .def_readwrite("start_cash", &Config::start_cash)
      .def_readwrite("max_price", &Config::max_price)
      .def_readwrite("clear_on_trade", &Config::clear_on_trade)
      .def_property_readonly("ante", &Config::ante)
      .def_property_readonly("pot", &Config::pot)
      .def_property_readonly("hand_size", &Config::hand_size);

  py::enum_<ActType>(m, "ActType")
      .value("PASS", ActType::kPass)
      .value("BID", ActType::kBid)
      .value("ASK", ActType::kAsk)
      .value("BUY", ActType::kBuy)
      .value("SELL", ActType::kSell)
      .value("CANCEL_BID", ActType::kCancelBid)
      .value("CANCEL_ASK", ActType::kCancelAsk)
      .value("CANCEL_ALL", ActType::kCancelAll);

  py::enum_<Status>(m, "Status")
      .value("OK", Status::kOk)
      .value("TRADED", Status::kTraded)
      .value("REJECTED", Status::kRejected)
      .value("NOOP", Status::kNoop);

  py::class_<Trade>(m, "Trade")
      .def_readonly("tick", &Trade::tick)
      .def_readonly("suit", &Trade::suit)
      .def_readonly("price", &Trade::price)
      .def_readonly("buyer", &Trade::buyer)
      .def_readonly("seller", &Trade::seller);

  py::class_<Game>(m, "Game")
      .def(py::init<Config, uint64_t>(), py::arg("config") = Config{}, py::arg("seed") = 1)
      .def("reset", &Game::reset)
      .def("reset_with_deal",
           [](Game& g, const std::vector<int>& sizes, const std::vector<std::vector<int>>& hands, uint64_t seed) {
             g.reset_with_deal(to_counts(sizes), to_counts_list(hands), seed);
           },
           py::arg("suit_counts"), py::arg("hands"), py::arg("seed") = 1)
      .def("apply",
           [](Game& g, int player, ActType t, int suit, int price) { return g.apply(player, Action{t, suit, price}); },
           py::arg("player"), py::arg("type"), py::arg("suit") = 0, py::arg("price") = 0)
      .def("end_tick", &Game::end_tick)
      .def("action_order",
           [](Game& g) {
             auto o = g.action_order();
             return std::vector<int>(o.begin(), o.begin() + g.n_players());
           })
      .def_property_readonly("done", &Game::done)
      .def_property_readonly("tick", &Game::tick)
      .def_property_readonly("n_players", &Game::n_players)
      .def_property_readonly("goal_suit", &Game::goal_suit)
      .def_property_readonly("pot", &Game::pot)
      .def_property_readonly("suit_counts",
                             [](const Game& g) { return std::vector<int>(g.suit_counts().begin(), g.suit_counts().end()); })
      .def("hand", [](const Game& g, int p) { auto h = g.hand_counts(p); return std::vector<int>(h.begin(), h.end()); })
      .def("initial_hand",
           [](const Game& g, int p) { auto h = g.initial_hand(p); return std::vector<int>(h.begin(), h.end()); })
      .def("cash", &Game::cash)
      .def("pnl", &Game::pnl)
      .def("payout_x60", [](const Game& g, int p) { return g.settlement().payout_x60[p]; })
      .def("best_bid",
           [](const Game& g, int s) -> py::object {
             auto o = g.book(s).best(SuitBook::kBidSide);
             if (!o) return py::none();
             return py::make_tuple(o->price, o->player);
           })
      .def("best_ask",
           [](const Game& g, int s) -> py::object {
             auto o = g.book(s).best(SuitBook::kAskSide);
             if (!o) return py::none();
             return py::make_tuple(o->price, o->player);
           })
      .def("book_depth",
           [](const Game& g, int s) {
             return py::make_tuple(g.book(s).size(SuitBook::kBidSide), g.book(s).size(SuitBook::kAskSide));
           })
      .def_property_readonly("trades", &Game::trades)
      .def("check_invariants", &Game::check_invariants)
      .def_static(
          "settle",
          [](const Config& cfg, int goal, const std::vector<std::vector<int>>& hands) {
            Settlement st = Game::settle(cfg, goal, to_counts_list(hands));
            return std::vector<int64_t>(st.payout_x60.begin(), st.payout_x60.begin() + hands.size());
          },
          "Payouts in 1/60 chip units");

  m.def("all_configs", [] {
    py::list out;
    for (const auto& c : all_configs())
      out.append(py::make_tuple(std::vector<int>(c.sizes.begin(), c.sizes.end()), c.goal));
    return out;
  });
  m.def("config_posterior",
        [](const std::vector<int>& hand, const std::vector<std::vector<int>>& mins, int hand_size) {
          auto p = config_posterior(to_counts(hand), to_counts_list(mins), hand_size);
          return std::vector<double>(p.begin(), p.end());
        },
        py::arg("hand"), py::arg("others_min"), py::arg("hand_size") = 10);
  m.def("goal_probs", [](const std::vector<double>& post) {
    ConfigProbs p{};
    std::copy(post.begin(), post.end(), p.begin());
    auto g = goal_probs(p);
    return std::vector<double>(g.begin(), g.end());
  });
  m.def("fair_values", [](const std::vector<double>& post, int pot) {
    ConfigProbs p{};
    std::copy(post.begin(), post.end(), p.begin());
    auto v = fair_values(p, pot);
    return std::vector<double>(v.begin(), v.end());
  });
  m.def("constraint_probability",
        [](const std::vector<int>& remaining, const std::vector<std::vector<int>>& mins, int hand_size) {
          return constraint_probability(to_counts(remaining), to_counts_list(mins), hand_size);
        });
  m.def("bot_names", &bot_names);

  m.def(
      "run_games",
      [](const std::vector<std::string>& lineup, int n_deals, uint64_t seed, const Config& cfg, int n_checkpoints,
         bool rotate, bool check, int threads) {
        MatchResults r;
        {
          py::gil_scoped_release release;
          r = run_games(lineup, n_deals, seed, cfg, n_checkpoints, rotate, check, threads);
        }
        py::dict d;
        const py::ssize_t G = r.n_games, B = r.n_bots, C = r.n_checkpoints;
        d["deal"] = to_array(r.deal, {G});
        d["rotation"] = to_array(r.rotation, {G});
        d["pnl"] = to_array(r.pnl, {G, B});
        d["brier"] = to_array(r.brier, {G, B, C});
        d["goal_cards_end"] = to_array(r.goal_cards_end, {G, B});
        d["n_trades"] = to_array(r.n_trades, {G});
        d["invariant_violations"] = r.invariant_violations;
        d["first_violation"] = r.first_violation;
        return d;
      },
      py::arg("lineup"), py::arg("n_deals"), py::arg("seed") = 0, py::arg("config") = Config{},
      py::arg("n_checkpoints") = 4, py::arg("rotate") = true, py::arg("check_invariants") = false,
      py::arg("threads") = 1);

  m.def(
      "fuzz",
      [](uint64_t n_games, uint64_t seed, int threads, bool allow_bayes) {
        FuzzResults r;
        {
          py::gil_scoped_release release;
          r = fuzz(n_games, seed, threads, allow_bayes);
        }
        py::dict d;
        d["games"] = r.games;
        d["actions"] = r.actions;
        d["trades"] = r.trades;
        d["rejected"] = r.rejected;
        d["violations"] = r.violations;
        d["first_violation"] = r.first_violation;
        return d;
      },
      py::arg("n_games"), py::arg("seed") = 0, py::arg("threads") = 1, py::arg("allow_bayes") = true);

  py::class_<VecEnv>(m, "VecEnv")
      .def(py::init<int, int, const std::vector<std::string>&, const Config&, uint64_t, bool>(),
           py::arg("num_envs"), py::arg("n_learners") = 1,
           py::arg("opponents") = std::vector<std::string>{"random", "random", "random"},
           py::arg("config") = Config{}, py::arg("seed") = 0, py::arg("check_invariants") = false)
      .def_property_readonly("num_envs", &VecEnv::num_envs)
      .def_property_readonly("n_learners", &VecEnv::n_learners)
      .def_property_readonly("obs_dim", &VecEnv::obs_dim)
      .def_property_readonly("steps", &VecEnv::steps)
      .def_property_readonly("episodes", &VecEnv::episodes)
      .def_property_readonly("violations", &VecEnv::violations)
      .def_property_readonly("rejected_learner_actions", &VecEnv::rejected_learner_actions)
      .def("reset",
           [](VecEnv& env) {
             py::array_t<float> obs({env.num_envs(), env.n_learners(), env.obs_dim()});
             env.reset(obs.mutable_data());
             return obs;
           })
      .def("step",
           [](VecEnv& env, py::array_t<int32_t, py::array::c_style | py::array::forcecast> actions) {
             if (actions.ndim() != 3 || actions.shape(0) != env.num_envs() || actions.shape(1) != env.n_learners() ||
                 actions.shape(2) != 3)
               throw std::invalid_argument("actions must have shape [num_envs, n_learners, 3]");
             py::array_t<float> obs({env.num_envs(), env.n_learners(), env.obs_dim()});
             py::array_t<float> rew({env.num_envs(), env.n_learners()});
             py::array_t<bool> done(env.num_envs());
             {
               py::gil_scoped_release release;
               env.step(actions.data(), obs.mutable_data(), rew.mutable_data(),
                        reinterpret_cast<uint8_t*>(done.mutable_data()));
             }
             return py::make_tuple(obs, rew, done);
           })
      .def("bayes_goal_probs",
           [](const VecEnv& env, int e, int l) {
             auto g = env.bayes_goal_probs(e, l);
             return std::vector<double>(g.begin(), g.end());
           })
      .def("goal_suit", [](const VecEnv& env, int e) { return env.game(e).goal_suit(); });
}
