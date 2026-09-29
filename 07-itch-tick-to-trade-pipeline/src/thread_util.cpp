#include "llt/thread_util.hpp"

#include <pthread.h>

#include <cstdio>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <mach/thread_policy.h>
#include <pthread/qos.h>
#elif defined(__linux__)
#include <sched.h>
#endif

namespace llt {

PlacementResult place_current_thread(Qos qos, int core) {
    PlacementResult r;
#if defined(__APPLE__)
    const thread_port_t self = pthread_mach_thread_np(pthread_self());
    if (qos == Qos::Interactive || qos == Qos::Fixed) {
        r.qos_rc = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    } else if (qos == Qos::Background) {
        r.qos_rc = pthread_set_qos_class_self_np(QOS_CLASS_BACKGROUND, 0);
    }
    if (qos == Qos::Fixed) {
        // Timeshare threads lose priority as they accumulate CPU time. A spin
        // loop is 100% CPU, so on a busy machine it sinks below everything
        // else. A non-timeshare (fixed) thread keeps its priority.
        thread_extended_policy_data_t ext{FALSE};
        r.policy_rc = thread_policy_set(self, THREAD_EXTENDED_POLICY,
                                        reinterpret_cast<thread_policy_t>(&ext),
                                        THREAD_EXTENDED_POLICY_COUNT);
        thread_precedence_policy_data_t prec{63};
        const kern_return_t kr2 = thread_policy_set(self, THREAD_PRECEDENCE_POLICY,
                                                    reinterpret_cast<thread_policy_t>(&prec),
                                                    THREAD_PRECEDENCE_POLICY_COUNT);
        if (r.policy_rc == 0) r.policy_rc = kr2;
    } else if (qos == Qos::Realtime) {
        mach_timebase_info_data_t tb;
        mach_timebase_info(&tb);
        auto abs_from_ns = [&](double ns) {
            return static_cast<uint32_t>(ns * tb.denom / tb.numer);
        };
        thread_time_constraint_policy_data_t tc;
        tc.period = abs_from_ns(1'000'000);       // 1 ms
        tc.computation = abs_from_ns(500'000);    // 0.5 ms of every period
        tc.constraint = abs_from_ns(1'000'000);
        tc.preemptible = TRUE;
        r.policy_rc = thread_policy_set(self, THREAD_TIME_CONSTRAINT_POLICY,
                                        reinterpret_cast<thread_policy_t>(&tc),
                                        THREAD_TIME_CONSTRAINT_POLICY_COUNT);
    }
    if (core >= 0) {
        thread_affinity_policy_data_t pol{core + 1};  // tag 0 means "no affinity"
        const kern_return_t kr =
            thread_policy_set(self, THREAD_AFFINITY_POLICY,
                              reinterpret_cast<thread_policy_t>(&pol), THREAD_AFFINITY_POLICY_COUNT);
        r.affinity_rc = kr;
        r.affinity_supported = kr == KERN_SUCCESS;
    }
#elif defined(__linux__)
    (void)qos;
    if (core >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(core, &set);
        r.affinity_rc = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
        r.affinity_supported = r.affinity_rc == 0;
    }
#else
    (void)qos;
    (void)core;
#endif
    return r;
}

void set_thread_name(const char* name) {
#if defined(__APPLE__)
    pthread_setname_np(name);
#elif defined(__linux__)
    pthread_setname_np(pthread_self(), name);
#else
    (void)name;
#endif
}

std::string describe(const PlacementResult& r) {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "qos_rc=%d policy_rc=%d affinity_rc=%d affinity_supported=%s",
                  r.qos_rc, r.policy_rc, r.affinity_rc, r.affinity_supported ? "yes" : "no");
    return buf;
}

const char* to_string(Qos q) {
    switch (q) {
        case Qos::Default: return "default";
        case Qos::Interactive: return "interactive";
        case Qos::Background: return "background";
        case Qos::Fixed: return "fixed";
        case Qos::Realtime: return "realtime";
    }
    return "?";
}

}  // namespace llt
