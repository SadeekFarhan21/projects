// pybind11 module `c4core`. Numpy arrays cross the boundary as float32 and
// are copied once in each direction; the search itself never touches Python.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <random>

#include "c4/position.hpp"
#include "c4/puct.hpp"
#include "c4/pure_mcts.hpp"
#include "c4/solver.hpp"

namespace py = pybind11;
using namespace c4;

namespace {

using FArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

FArray encode_positions(const std::vector<Position>& ps) {
  FArray out({static_cast<py::ssize_t>(ps.size()), py::ssize_t{3}, py::ssize_t{kHeight}, py::ssize_t{kWidth}});
  float* p = out.mutable_data();
  for (std::size_t i = 0; i < ps.size(); ++i) ps[i].encode(p + i * kObsSize);
  return out;
}

template <class Engine>
FArray gather(Engine& e) {
  std::vector<float> buf(static_cast<std::size_t>(e.capacity()) * kObsSize);
  int n;
  {
    py::gil_scoped_release release;
    n = e.gather(buf.data());
  }
  FArray out({static_cast<py::ssize_t>(n), py::ssize_t{3}, py::ssize_t{kHeight}, py::ssize_t{kWidth}});
  std::copy(buf.begin(), buf.begin() + static_cast<long>(n) * kObsSize, out.mutable_data());
  return out;
}

template <class Engine>
void scatter(Engine& e, const FArray& policy, const FArray& value) {
  if (policy.ndim() != 2 || policy.shape(1) != kWidth) throw std::invalid_argument("policy must be (n, 7)");
  if (value.ndim() != 1 || value.shape(0) != policy.shape(0)) throw std::invalid_argument("value must be (n,)");
  py::gil_scoped_release release;
  e.scatter(policy.data(), value.data());
}

}  // namespace

PYBIND11_MODULE(c4core, m) {
  m.doc() = "Connect Four bitboards, perfect solver, pure MCTS and batched PUCT";
  m.attr("WIDTH") = kWidth;
  m.attr("HEIGHT") = kHeight;
  m.attr("INVALID_SCORE") = kInvalidScore;

  py::class_<Position>(m, "Position")
      .def(py::init<>())
      .def_static("from_moves", &Position::from_moves)
      .def("can_play", &Position::can_play)
      .def("play", [](Position& p, int c) {
        if (c < 0 || c >= kWidth || !p.can_play(c)) throw std::invalid_argument("illegal move");
        if (p.is_terminal()) throw std::invalid_argument("game is over");
        p.play(c);
      })
      .def("is_winning_move", &Position::is_winning_move)
      .def("last_mover_won", &Position::last_mover_won)
      .def("is_full", &Position::is_full)
      .def("is_terminal", &Position::is_terminal)
      .def("key", &Position::key)
      .def("mirrored", &Position::mirrored)
      .def("moves", &Position::moves)
      .def("side_to_move", &Position::side_to_move)
      .def("cell", &Position::cell)
      .def("legal_moves",
           [](const Position& p) {
             std::vector<int> v;
             for (int c = 0; c < kWidth; ++c)
               if (p.can_play(c)) v.push_back(c);
             return v;
           })
      .def("encode", [](const Position& p) { return encode_positions({p}).attr("__getitem__")(0); })
      .def("copy", [](const Position& p) { return Position(p); })
      .def("__str__", &Position::to_string);

  m.def("encode_batch", &encode_positions);

  py::class_<Solver>(m, "Solver")
      .def(py::init<int>(), py::arg("tt_log2_size") = 23)
      .def("solve", &Solver::solve, py::call_guard<py::gil_scoped_release>())
      .def("analyze", &Solver::analyze, py::call_guard<py::gil_scoped_release>())
      .def("best_move", &Solver::best_move, py::call_guard<py::gil_scoped_release>())
      .def("node_count", &Solver::node_count)
      .def("reset", &Solver::reset);
  m.def("brute_force_score", &brute_force_score);

  py::class_<PureMCTS>(m, "PureMCTS")
      .def(py::init<int, std::uint64_t, double>(), py::arg("rollouts"), py::arg("seed"),
           py::arg("c_uct") = 1.41421356)
      .def("choose_move", &PureMCTS::choose_move, py::call_guard<py::gil_scoped_release>());

  py::class_<PuctConfig>(m, "PuctConfig")
      .def(py::init<>())
      .def(py::init([](int num_sims, float c_puct, float alpha, float eps, int temp_moves) {
             return PuctConfig{num_sims, c_puct, alpha, eps, temp_moves};
           }),
           py::arg("num_sims") = 100, py::arg("c_puct") = 1.5f, py::arg("dirichlet_alpha") = 1.0f,
           py::arg("dirichlet_eps") = 0.25f, py::arg("temp_moves") = 10)
      .def_readwrite("num_sims", &PuctConfig::num_sims)
      .def_readwrite("c_puct", &PuctConfig::c_puct)
      .def_readwrite("dirichlet_alpha", &PuctConfig::dirichlet_alpha)
      .def_readwrite("dirichlet_eps", &PuctConfig::dirichlet_eps)
      .def_readwrite("temp_moves", &PuctConfig::temp_moves);

  py::class_<GameRecord>(m, "GameRecord")
      .def_readonly("winner", &GameRecord::winner)
      .def_readonly("net_side", &GameRecord::net_side)
      .def_readonly("moves", &GameRecord::moves);

  py::class_<BatchedGames>(m, "BatchedGames")
      .def(py::init<int, int, PuctConfig, std::uint64_t, int, std::vector<std::string>>(),
           py::arg("num_slots"), py::arg("max_games"), py::arg("config"), py::arg("seed"),
           py::arg("opponent_rollouts") = 0, py::arg("openings") = std::vector<std::string>{})
      .def("gather", [](BatchedGames& e) { return gather(e); })
      .def("scatter", [](BatchedGames& e, const FArray& p, const FArray& v) { scatter(e, p, v); })
      .def("done", &BatchedGames::done)
      .def("finished_games", &BatchedGames::finished_games)
      .def("total_moves", &BatchedGames::total_moves)
      .def("total_evals", &BatchedGames::total_evals)
      .def("drain_samples",
           [](BatchedGames& e) {
             auto s = e.drain_samples();
             const auto n = static_cast<py::ssize_t>(s.z.size());
             FArray obs({n, py::ssize_t{3}, py::ssize_t{kHeight}, py::ssize_t{kWidth}});
             FArray pi({n, py::ssize_t{kWidth}});
             FArray z(std::vector<py::ssize_t>{n});
             std::copy(s.obs.begin(), s.obs.end(), obs.mutable_data());
             std::copy(s.pi.begin(), s.pi.end(), pi.mutable_data());
             std::copy(s.z.begin(), s.z.end(), z.mutable_data());
             return py::make_tuple(obs, pi, z);
           })
      .def("drain_games", &BatchedGames::drain_games);

  py::class_<BatchedAnalysis>(m, "BatchedAnalysis")
      .def(py::init<std::vector<Position>, PuctConfig, std::uint64_t>(), py::arg("positions"),
           py::arg("config"), py::arg("seed"))
      .def("gather", [](BatchedAnalysis& e) { return gather(e); })
      .def("scatter", [](BatchedAnalysis& e, const FArray& p, const FArray& v) { scatter(e, p, v); })
      .def("done", &BatchedAnalysis::done)
      .def("policies", &BatchedAnalysis::policies)
      .def("best_moves", &BatchedAnalysis::best_moves)
      .def("root_values", &BatchedAnalysis::root_values);
}
