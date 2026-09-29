// Emulated launchers: namespace gkl::emu. The kernels in kernels/*.cuh are
// compiled as plain C++ against include/gkl/simt_emu.h and run on CPU threads.
#define GKL_BACKEND_NS emu
#include "../launchers.inc"
