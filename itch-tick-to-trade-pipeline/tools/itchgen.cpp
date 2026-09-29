// itchgen: writes a synthetic ITCH-like capture file.
//
//   itchgen --out data/capture.itch [--messages N] [--symbols S] [--seed K] [--mean-gap-ns G]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "llt/capture.hpp"
#include "llt/synth.hpp"

int main(int argc, char** argv) {
    llt::SynthParams p;
    std::string out = "data/capture.itch";
    for (int i = 1; i < argc; ++i) {
        auto arg = [&](const char* name) { return std::strcmp(argv[i], name) == 0 && i + 1 < argc; };
        if (arg("--out")) out = argv[++i];
        else if (arg("--messages")) p.messages = std::strtoull(argv[++i], nullptr, 10);
        else if (arg("--symbols")) p.symbols = static_cast<uint32_t>(std::atoi(argv[++i]));
        else if (arg("--seed")) p.seed = std::strtoull(argv[++i], nullptr, 10);
        else if (arg("--mean-gap-ns")) p.mean_gap_ns = std::atof(argv[++i]);
        else {
            std::fprintf(stderr,
                         "usage: itchgen --out FILE [--messages N] [--symbols S] [--seed K] "
                         "[--mean-gap-ns G]\n");
            return 2;
        }
    }
    llt::SynthStats st;
    const auto bytes = llt::generate_capture(p, &st);
    llt::save_capture(out, bytes);
    std::printf(
        "wrote %s: %llu bytes, adds=%llu execs=%llu cancels=%llu deletes=%llu dir=%llu, "
        "span=%.3f s of exchange time\n",
        out.c_str(), (unsigned long long)st.bytes, (unsigned long long)st.adds,
        (unsigned long long)st.execs, (unsigned long long)st.cancels, (unsigned long long)st.deletes,
        (unsigned long long)st.directory, (st.last_ts - st.first_ts) / 1e9);
    return 0;
}
