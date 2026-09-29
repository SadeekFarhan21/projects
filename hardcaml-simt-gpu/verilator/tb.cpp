// Verilator testbench for the Hardcaml generated GPU.
//
// For every vector file NAME.in given on the command line it drives the same
// host protocol as the OCaml harness (clear, load imem, load dmem, start,
// wait for done, read back dmem) and writes NAME.verilator.out in the same
// format as NAME.hardcaml.out, so the two can be compared byte for byte.

#include "Vgpu.h"
#include "verilated.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

namespace {

struct Vector {
  int threads = 0;
  std::array<uint32_t, 256> imem{};
  std::array<uint32_t, 256> dmem{};
};

bool read_vector(const std::string& path, Vector& v) {
  std::ifstream in(path);
  if (!in) return false;
  in >> v.threads;
  for (auto& w : v.imem) in >> std::hex >> w;
  for (auto& b : v.dmem) in >> std::hex >> b;
  return static_cast<bool>(in);
}

class Bench {
 public:
  Bench() : ctx_(std::make_unique<VerilatedContext>()), top_(std::make_unique<Vgpu>(ctx_.get())) {
    top_->clock = 0;
    top_->eval();
  }

  // One rising edge with the currently applied inputs, then settle.
  void cycle() {
    top_->clock = 1;
    top_->eval();
    top_->clock = 0;
    top_->eval();
    ++cycles_;
  }

  std::string run(const Vector& v, uint64_t max_cycles) {
    auto& t = *top_;
    t.clear = 1;
    cycle();
    t.clear = 0;
    t.host_imem_we = 1;
    for (int a = 0; a < 256; ++a) {
      t.host_imem_addr = a;
      t.host_imem_data = v.imem[a];
      cycle();
    }
    t.host_imem_we = 0;
    t.host_dmem_we = 1;
    for (int a = 0; a < 256; ++a) {
      t.host_dmem_addr = a;
      t.host_dmem_wdata = v.dmem[a];
      cycle();
    }
    t.host_dmem_we = 0;
    t.thread_count = v.threads;
    t.start = 1;
    cycle();
    t.start = 0;
    uint64_t n = 0;
    while (!t.done_) {
      if (n++ >= max_cycles) return "TIMEOUT\n";
      cycle();
    }
    std::ostringstream os;
    os << t.cycles << ' ' << t.stall_cycles << ' ' << t.instrs << ' ' << t.mem_reqs << ' '
       << t.divergent << ' ' << t.queue_full_cycles << '\n';
    for (int a = 0; a < 256; ++a) {
      t.host_dmem_raddr = a;
      cycle();
      char buf[4];
      std::snprintf(buf, sizeof buf, "%02x", static_cast<unsigned>(t.host_dmem_rdata));
      os << buf << (a == 255 ? '\n' : ' ');
    }
    return os.str();
  }

  uint64_t total_cycles() const { return cycles_; }

 private:
  std::unique_ptr<VerilatedContext> ctx_;
  std::unique_ptr<Vgpu> top_;
  uint64_t cycles_ = 0;
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: gpu_tb VECTOR.in...\n";
    return 2;
  }
  Bench bench;
  int ok = 0;
  for (int i = 1; i < argc; ++i) {
    std::string path = argv[i];
    Vector v;
    if (!read_vector(path, v)) {
      std::cerr << "cannot read " << path << '\n';
      return 1;
    }
    std::string out = bench.run(v, 200000);
    std::string base = path.substr(0, path.size() - 3);  // strip ".in"
    std::ofstream(base + ".verilator.out") << out;
    ++ok;
  }
  std::cout << "ran " << ok << " vectors, " << bench.total_cycles() << " simulated cycles\n";
  return 0;
}
