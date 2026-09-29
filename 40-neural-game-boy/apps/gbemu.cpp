// gbemu: command line driver for the emulator core.
//
//   gbemu info <rom>
//   gbemu play <rom> [--scale 4]                     (needs SDL2)
//   gbemu headless <rom> [--frames N] [--screenshot out.png] [--script f] [--random SEED]
//   gbemu bench <rom> [--frames N] [--repeat R]      JSON with frames per second
//   gbemu suite blargg|mooneye <rom>...              CSV rows on stdout
//   gbemu acid2 <rom> <reference.png> [--out out.png]
//   gbemu record <rom> --out f.gbrec [--steps N] [--seed S] [--frameskip K]
//                [--script f] [--chunk 256] [--level 6] [--policy random|noop]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "core/gameboy.h"
#include "core/testrom.h"
#include "logger/policy.h"
#include "logger/recorder.h"
#include "util/png.h"

#ifdef GB_HAVE_SDL
int run_sdl(const std::string& rom_path, int scale);
#endif

namespace {

struct Args {
  std::vector<std::string> pos;
  std::map<std::string, std::string> opt;
  std::string get(const std::string& k, const std::string& def = "") const {
    auto it = opt.find(k);
    return it == opt.end() ? def : it->second;
  }
  long long num(const std::string& k, long long def) const {
    auto it = opt.find(k);
    return it == opt.end() ? def : std::atoll(it->second.c_str());
  }
};

Args parse(int argc, char** argv, int start) {
  Args a;
  for (int i = start; i < argc; ++i) {
    std::string s = argv[i];
    if (s.rfind("--", 0) == 0) {
      std::string key = s.substr(2);
      if (i + 1 < argc && std::strncmp(argv[i + 1], "--", 2) != 0) a.opt[key] = argv[++i];
      else a.opt[key] = "1";
    } else {
      a.pos.push_back(s);
    }
  }
  return a;
}

double now_s() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}

void save_screenshot(const gb::GameBoy& g, const std::string& path) {
  std::vector<uint8_t> px(gb::kScreenW * gb::kScreenH);
  const auto& fb = g.framebuffer();
  for (size_t i = 0; i < px.size(); ++i) px[i] = gb::png::shade_to_gray(fb[i]);
  gb::png::write_gray(path, gb::kScreenW, gb::kScreenH, px.data());
}

std::unique_ptr<gb::rec::Policy> make_policy(const Args& a, uint64_t seed) {
  std::unique_ptr<gb::rec::Policy> base;
  std::string kind = a.get("policy", "random");
  if (kind == "random") base = std::make_unique<gb::rec::RandomPolicy>(seed);
  else if (kind != "noop") throw std::runtime_error("unknown policy " + kind);
  if (a.opt.count("script")) return gb::rec::ScriptPolicy::from_file(a.get("script"), std::move(base));
  if (!base) return std::make_unique<gb::rec::ScriptPolicy>(std::vector<std::pair<gb::u64, gb::u8>>{}, nullptr);
  return base;
}

int cmd_info(const Args& a) {
  auto cart = gb::Cartridge::from_file(a.pos.at(0));
  const auto& h = cart->header();
  std::printf("title=%s type=0x%02X rom_bytes=%zu ram_bytes=%zu hash=%016llx\n", h.title.c_str(), h.cart_type,
              cart->rom().size(), h.ram_bytes, (unsigned long long)cart->rom_hash());
  return 0;
}

int cmd_headless(const Args& a) {
  auto g = gb::GameBoy::from_file(a.pos.at(0));
  long long frames = a.num("frames", 600);
  std::unique_ptr<gb::rec::Policy> pol;
  if (a.opt.count("script") || a.opt.count("random")) {
    Args b = a;
    if (!a.opt.count("random")) b.opt["policy"] = "noop";
    pol = make_policy(b, uint64_t(a.num("random", 1)));
  }
  double t0 = now_s();
  for (long long i = 0; i < frames; ++i) {
    if (pol) g->set_buttons(pol->act(uint64_t(i)));
    g->run_frame();
  }
  double dt = now_s() - t0;
  if (a.opt.count("screenshot")) save_screenshot(*g, a.get("screenshot"));
  std::printf("{\"frames\": %lld, \"seconds\": %.4f, \"fps\": %.1f, \"speedup\": %.1f, \"serial\": \"%zu bytes\"}\n",
              frames, dt, frames / dt, frames / dt / gb::kFramesPerSecond, g->bus.serial_output.size());
  return 0;
}

int cmd_bench(const Args& a) {
  long long frames = a.num("frames", 3000);
  long long repeat = a.num("repeat", 3);
  std::printf("{\"rom\": \"%s\", \"frames\": %lld, \"fps\": [", a.pos.at(0).c_str(), frames);
  for (long long r = 0; r < repeat; ++r) {
    auto g = gb::GameBoy::from_file(a.pos.at(0));
    gb::rec::RandomPolicy pol(uint64_t(r + 1));
    double t0 = now_s();
    for (long long i = 0; i < frames; ++i) {
      g->set_buttons(pol.act(uint64_t(i)));
      g->run_frame();
    }
    double dt = now_s() - t0;
    std::printf("%s%.1f", r ? ", " : "", frames / dt);
    std::fflush(stdout);
  }
  std::printf("]}\n");
  return 0;
}

std::string csv_escape(const std::string& s) {
  std::string o = "\"";
  for (char c : s) {
    if (c == '"') o += "\"\"";
    else if (c == '\n' || c == '\r') o += ' ';
    else if (static_cast<unsigned char>(c) >= 32) o += c;
  }
  return o + "\"";
}

int cmd_suite(const Args& a) {
  const std::string kind = a.pos.at(0);
  std::printf("suite,rom,passed,timed_out,emu_seconds,wall_seconds,detail\n");
  int pass = 0, total = 0;
  for (size_t i = 1; i < a.pos.size(); ++i) {
    double t0 = now_s();
    gb::TestVerdict v;
    try {
      v = kind == "blargg" ? gb::run_blargg(a.pos[i]) : gb::run_mooneye(a.pos[i]);
    } catch (const std::exception& e) {
      v.detail = std::string("error: ") + e.what();
    }
    double dt = now_s() - t0;
    std::string detail = v.detail;
    if (kind == "blargg" && detail.size() > 200) detail = detail.substr(detail.size() - 200);
    std::printf("%s,%s,%d,%d,%.3f,%.3f,%s\n", kind.c_str(), a.pos[i].c_str(), v.passed, v.timed_out,
                v.emu_seconds, dt, csv_escape(detail).c_str());
    std::fflush(stdout);
    pass += v.passed;
    ++total;
  }
  std::fprintf(stderr, "%s: %d/%d passed\n", kind.c_str(), pass, total);
  return 0;
}

int cmd_acid2(const Args& a) {
  gb::TestVerdict v = gb::run_until_ld_b_b(a.pos.at(0));
  auto ref = gb::png::read_gray(a.pos.at(1));
  int mismatched = 0;
  std::vector<uint8_t> px(gb::kScreenW * gb::kScreenH);
  for (size_t i = 0; i < px.size(); ++i) {
    px[i] = gb::png::shade_to_gray(v.frame[i]);
    if (px[i] != ref.pixels[i]) ++mismatched;
  }
  if (a.opt.count("out")) gb::png::write_gray(a.get("out"), gb::kScreenW, gb::kScreenH, px.data());
  if (a.opt.count("diff")) {
    // White where pixels match, black where they differ.
    std::vector<uint8_t> d(px.size());
    for (size_t i = 0; i < px.size(); ++i) d[i] = px[i] == ref.pixels[i] ? 0xFF : 0x00;
    gb::png::write_gray(a.get("diff"), gb::kScreenW, gb::kScreenH, d.data());
  }
  std::printf("{\"rom\": \"%s\", \"reached_ld_b_b\": %s, \"pixels\": %zu, \"mismatched\": %d, \"match\": %.6f}\n",
              a.pos.at(0).c_str(), v.timed_out ? "false" : "true", px.size(), mismatched,
              1.0 - double(mismatched) / double(px.size()));
  return mismatched == 0 ? 0 : 2;
}

int cmd_record(const Args& a) {
  auto g = gb::GameBoy::from_file(a.pos.at(0));
  const std::string out = a.get("out");
  if (out.empty()) throw std::runtime_error("record needs --out");
  const uint64_t steps = uint64_t(a.num("steps", 10000));
  const uint64_t seed = uint64_t(a.num("seed", 1));
  const int frameskip = int(a.num("frameskip", 1));
  auto pol = make_policy(a, seed);
  gb::rec::RecorderOptions o;
  o.frameskip = uint32_t(frameskip);
  o.chunk_steps = uint32_t(a.num("chunk", 256));
  o.zlib_level = int(a.num("level", 6));
  o.rom_hash = g->bus.cart().rom_hash();
  o.seed = seed;
  o.policy = pol->name();
  gb::rec::Recorder rec(out, o);
  double t0 = now_s();
  double emu_time = 0;
  for (uint64_t t = 0; t < steps; ++t) {
    uint8_t act = pol->act(t);
    rec.add(g->framebuffer().data(), act);
    g->set_buttons(act);
    double e0 = now_s();
    for (int k = 0; k < frameskip; ++k) g->run_frame();
    emu_time += now_s() - e0;
  }
  rec.close();
  double dt = now_s() - t0;
  std::printf(
      "{\"out\": \"%s\", \"steps\": %llu, \"frameskip\": %d, \"seconds\": %.3f, \"steps_per_s\": %.1f, "
      "\"emu_seconds\": %.3f, \"bytes\": %llu, \"bytes_per_step\": %.1f, \"raw_bytes_per_step\": %d, "
      "\"policy\": \"%s\"}\n",
      out.c_str(), (unsigned long long)steps, frameskip, dt, steps / dt, emu_time,
      (unsigned long long)rec.bytes_written(), double(rec.bytes_written()) / double(steps),
      gb::rec::kPackedFrameBytes + 1, pol->name().c_str());
  return 0;
}

void usage() {
  std::fprintf(stderr,
               "usage: gbemu <info|play|headless|bench|suite|acid2|record> ...\n"
               "see the header of apps/gbemu.cpp or README.md for options\n");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    usage();
    return 1;
  }
  std::string cmd = argv[1];
  Args a = parse(argc, argv, 2);
  try {
    if (cmd == "info") return cmd_info(a);
    if (cmd == "headless") return cmd_headless(a);
    if (cmd == "bench") return cmd_bench(a);
    if (cmd == "suite") return cmd_suite(a);
    if (cmd == "acid2") return cmd_acid2(a);
    if (cmd == "record") return cmd_record(a);
    if (cmd == "play") {
#ifdef GB_HAVE_SDL
      return run_sdl(a.pos.at(0), int(a.num("scale", 4)));
#else
      std::fprintf(stderr, "built without SDL2; use headless\n");
      return 1;
#endif
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  usage();
  return 1;
}
