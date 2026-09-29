// Thread placement. macOS has no hard CPU affinity: THREAD_AFFINITY_POLICY is
// only a grouping hint, and on Apple Silicon thread_policy_set returns
// KERN_NOT_SUPPORTED for it. The only lever that works is the QoS class, which
// steers a thread toward performance (USER_INTERACTIVE) or efficiency
// (BACKGROUND) cores. On Linux this uses pthread_setaffinity_np.
#pragma once

#include <string>

namespace llt {

// Default:     leave the thread alone
// Interactive: QOS_CLASS_USER_INTERACTIVE (prefers performance cores)
// Background:  QOS_CLASS_BACKGROUND (confined to efficiency cores)
// Fixed:       Interactive, plus THREAD_EXTENDED_POLICY timeshare=false and
//              THREAD_PRECEDENCE_POLICY, which stop the scheduler from
//              decaying the priority of a thread that spins at 100% CPU
// Realtime:    THREAD_TIME_CONSTRAINT_POLICY (the audio-thread policy)
enum class Qos { Default, Interactive, Background, Fixed, Realtime };

struct PlacementResult {
    int qos_rc{0};       // 0 on success
    int policy_rc{0};    // kern_return_t of the extended/precedence/time-constraint call
    int affinity_rc{0};  // kern_return_t on macOS (46 = KERN_NOT_SUPPORTED), errno on Linux
    bool affinity_supported{false};
};

// core: on Linux, the CPU to pin to (-1 = no pin). On macOS, used as the
// affinity tag for the (unsupported) hint.
PlacementResult place_current_thread(Qos qos, int core);

void set_thread_name(const char* name);

std::string describe(const PlacementResult& r);

const char* to_string(Qos q);

}  // namespace llt
