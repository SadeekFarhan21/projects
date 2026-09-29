/* Kernel threads and the scheduler. */
#pragma once
#include "types.h"
#include "spinlock.h"

enum tstate { T_UNUSED, T_EMBRYO, T_RUNNABLE, T_RUNNING, T_SLEEPING, T_ZOMBIE };

/* Callee-saved registers; everything else is saved by the C caller of swtch. */
struct context {
    u64 ra, sp;
    u64 s[12];
};

struct thread {
    struct context ctx; /* must stay first */
    enum tstate state;
    int tid;
    int slot; /* index in threads[], also selects the kernel stack VA */
    char name[16];
    void *chan;         /* what we are sleeping on */
    void (*fn)(void *);
    void *arg;
    struct thread *rq_next;
    u64 kstack_lo, kstack_hi;
    int exit_code;
    bool is_idle;
    u32 slice;          /* ticks used in the current quantum */
    u64 nswitch;        /* times switched in */
    u64 npreempt;       /* times preempted by the timer */
};

extern struct spinlock sched_lock;
extern volatile u64 ticks;

void thread_init(void);
struct thread *thread_create(const char *name, void (*fn)(void *), void *arg);
NORETURN void thread_exit(int code);
int thread_join(struct thread *t);
struct thread *thread_current(void);
void yield(void);
void sched(void);
void sleep_on(void *chan, struct spinlock *lk);
void wakeup(void *chan);
void thread_sleep_ticks(u64 n);
void preempt_tick(void);
NORETURN void idle_loop(void);
void thread_dump(void);
u64 sched_nswitch(void);
void swtch(struct context *old, struct context *new);
