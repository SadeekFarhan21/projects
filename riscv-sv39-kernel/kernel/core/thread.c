/* Kernel threads and a round-robin preemptive scheduler.
 *
 * Design: switches go directly from the old thread to the next one
 * (one swtch per switch). xv6 instead switches to a per-CPU scheduler
 * thread and from there to the next thread (two swtch per switch). The
 * price of the direct approach is that sched_lock is acquired by the old
 * thread and released by the new one, so every path that resumes after
 * swtch (sched's caller, or thread_start for a fresh thread) must release it.
 *
 * Invariants (all checked in sched()):
 *   - sched_lock is held, and it is the only spinlock held (noff == 1)
 *   - interrupts are off
 *   - the current thread's state is no longer T_RUNNING
 *   - the idle thread is never on the run queue */
#include "kernel.h"
#include "riscv.h"
#include "cpu.h"
#include "mm.h"
#include "thread.h"

struct spinlock sched_lock = {.name = "sched"};
volatile u64 ticks;
static struct thread threads[NTHREAD];
static struct thread idle_threads[NCPU];
static struct thread *rq_head, *rq_tail;
static int next_tid = 1;
static u64 total_switches;

static void rq_push(struct thread *t) {
    t->rq_next = NULL;
    if (rq_tail) rq_tail->rq_next = t;
    else rq_head = t;
    rq_tail = t;
}

static struct thread *rq_pop(void) {
    struct thread *t = rq_head;
    if (t) {
        rq_head = t->rq_next;
        if (!rq_head) rq_tail = NULL;
        t->rq_next = NULL;
    }
    return t;
}

struct thread *thread_current(void) {
    push_off();
    struct thread *t = mycpu()->cur;
    pop_off();
    return t;
}

u64 sched_nswitch(void) { return total_switches; }

/* The boot flow becomes this hart's idle thread, running on the boot stack. */
void thread_init(void) {
    struct cpu *c = mycpu();
    struct thread *idle = &idle_threads[c->hartid];
    idle->state = T_RUNNING;
    idle->tid = 0;
    idle->slot = -1;
    strlcpy(idle->name, "idle", sizeof(idle->name));
    idle->is_idle = true;
    idle->kstack_lo = c->kstack_lo;
    idle->kstack_hi = c->kstack_hi;
    c->cur = idle;
    c->idle = idle;
}

/* First code a new thread runs, entered from swtch() inside sched(). */
void thread_start(void);
extern char thread_trampoline[];
void thread_start(void) {
    spin_unlock(&sched_lock); /* acquired by the thread that switched to us */
    intr_on();
    struct thread *t = thread_current();
    t->fn(t->arg);
    thread_exit(0);
}

struct thread *thread_create(const char *name, void (*fn)(void *), void *arg) {
    struct thread *t = NULL;
    spin_lock(&sched_lock);
    for (int i = 0; i < NTHREAD; i++) {
        if (threads[i].state == T_UNUSED) {
            t = &threads[i];
            t->slot = i;
            t->state = T_EMBRYO;
            t->tid = next_tid++;
            break;
        }
    }
    spin_unlock(&sched_lock);
    if (!t) return NULL;

    if (kvm_map_kstack(t->slot)) {
        spin_lock(&sched_lock);
        t->state = T_UNUSED;
        spin_unlock(&sched_lock);
        return NULL;
    }
    strlcpy(t->name, name, sizeof(t->name));
    t->fn = fn;
    t->arg = arg;
    t->chan = NULL;
    t->exit_code = 0;
    t->is_idle = false;
    t->slice = 0;
    t->nswitch = 0;
    t->npreempt = 0;
    t->kstack_lo = KSTACK_LO(t->slot);
    t->kstack_hi = KSTACK_HI(t->slot);
    memset(&t->ctx, 0, sizeof(t->ctx));
    t->ctx.ra = (u64)thread_trampoline;
    t->ctx.sp = t->kstack_hi;

    spin_lock(&sched_lock);
    t->state = T_RUNNABLE;
    rq_push(t);
    spin_unlock(&sched_lock);
    return t;
}

void sched(void) {
    struct cpu *c = mycpu();
    struct thread *cur = c->cur;
    if (!spin_holding(&sched_lock)) panic("sched: sched_lock not held");
    if (c->noff != 1) panic("sched: %d spinlocks held, expected only sched_lock", c->noff);
    if (cur->state == T_RUNNING) panic("sched: current thread still RUNNING");
    if (intr_get()) panic("sched: interrupts enabled");

    struct thread *next = rq_pop();
    if (!next) next = c->idle;
    if (next == cur) {
        cur->state = T_RUNNING;
        return;
    }
    next->state = T_RUNNING;
    next->slice = 0;
    next->nswitch++;
    total_switches++;
    c->cur = next;
    c->kstack_lo = next->kstack_lo;
    c->kstack_hi = next->kstack_hi;

    int intena = c->intena; /* belongs to this thread, not to the hart */
    swtch(&cur->ctx, &next->ctx);
    mycpu()->intena = intena;
}

void yield(void) {
    spin_lock(&sched_lock);
    struct thread *t = mycpu()->cur;
    t->state = T_RUNNABLE;
    if (!t->is_idle) rq_push(t);
    sched();
    spin_unlock(&sched_lock);
}

/* Timer interrupt path: preempt the running thread when its quantum is used. */
void preempt_tick(void) {
    struct thread *t = mycpu()->cur;
    if (t->is_idle) return; /* idle_loop checks the run queue itself */
    if (++t->slice >= QUANTUM_TICKS) {
        t->npreempt++;
        yield();
    }
}

/* Atomically release lk and sleep on chan; reacquire lk before returning.
 * Holding sched_lock across the state change is what prevents a lost wakeup:
 * wakeup() needs sched_lock too, so it cannot run between our check of the
 * condition (under lk) and our state becoming T_SLEEPING. */
void sleep_on(void *chan, struct spinlock *lk) {
    struct thread *t = mycpu()->cur;
    if (t->is_idle) panic("idle thread must not sleep");
    if (lk != &sched_lock) {
        spin_lock(&sched_lock);
        spin_unlock(lk);
    }
    t->chan = chan;
    t->state = T_SLEEPING;
    sched();
    t->chan = NULL;
    if (lk != &sched_lock) {
        spin_unlock(&sched_lock);
        spin_lock(lk);
    }
}

static void wakeup_locked(void *chan) {
    for (int i = 0; i < NTHREAD; i++) {
        struct thread *t = &threads[i];
        if (t->state == T_SLEEPING && t->chan == chan) {
            t->state = T_RUNNABLE;
            rq_push(t);
        }
    }
}

void wakeup(void *chan) {
    spin_lock(&sched_lock);
    wakeup_locked(chan);
    spin_unlock(&sched_lock);
}

void thread_sleep_ticks(u64 n) {
    spin_lock(&sched_lock);
    u64 until = ticks + n;
    while (ticks < until) sleep_on((void *)&ticks, &sched_lock);
    spin_unlock(&sched_lock);
}

void thread_exit(int code) {
    spin_lock(&sched_lock);
    struct thread *t = mycpu()->cur;
    if (t->is_idle) panic("idle thread exited");
    t->exit_code = code;
    t->state = T_ZOMBIE;
    wakeup_locked(t); /* joiners sleep on the thread pointer */
    sched();
    panic("zombie thread %s was scheduled", t->name);
}

/* Wait for t to exit, free its stack and slot, return its exit code.
 * Single hart: once we observe T_ZOMBIE the zombie has already switched off
 * its stack. With SMP this needs an "on_cpu" flag before freeing the stack. */
int thread_join(struct thread *t) {
    spin_lock(&sched_lock);
    while (t->state != T_ZOMBIE) sleep_on(t, &sched_lock);
    int code = t->exit_code;
    spin_unlock(&sched_lock);
    kvm_unmap_kstack(t->slot);
    spin_lock(&sched_lock);
    t->state = T_UNUSED;
    spin_unlock(&sched_lock);
    return code;
}

void idle_loop(void) {
    struct cpu *c = mycpu();
    for (;;) {
        intr_off();
        spin_lock(&sched_lock);
        if (rq_head) {
            c->idle->state = T_RUNNABLE;
            sched();
        }
        spin_unlock(&sched_lock);
        /* Interrupts are still off: wfi returns as soon as one is pending,
         * so a wakeup that raced with the check above is not missed. */
        if (!rq_head) wfi();
        intr_on();
    }
}

void thread_dump(void) {
    static const char *names[] = {"unused", "embryo", "runnable", "running", "sleeping", "zombie"};
    kprintf("  tid  name             state     switches  preempts  stack\n");
    spin_lock(&sched_lock);
    for (int i = -1; i < NTHREAD; i++) {
        struct thread *t = i < 0 ? mycpu()->idle : &threads[i];
        if (t->state == T_UNUSED) continue;
        kprintf("  %-4d %-16s %-9s %8lu  %8lu  %p..%p\n", t->tid, t->name, names[t->state],
                t->nswitch, t->npreempt, (void *)t->kstack_lo, (void *)t->kstack_hi);
    }
    spin_unlock(&sched_lock);
}
