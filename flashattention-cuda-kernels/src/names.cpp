#include "gkl/kernels.h"

namespace gkl {
const char* to_string(SgemmAlgo a) {
  switch (a) {
    case SgemmAlgo::Naive: return "naive";
    case SgemmAlgo::Tiled: return "tiled";
    case SgemmAlgo::RegBlocked: return "regblock";
  }
  return "?";
}
const char* to_string(SoftmaxAlgo a) { return a == SoftmaxAlgo::Naive ? "naive" : "online"; }
const char* to_string(AttnAlgo a) { return a == AttnAlgo::Naive ? "naive" : "flash"; }
}  // namespace gkl
