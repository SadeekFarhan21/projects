/* Spinlocks and sleep locks.
 * Rule: a spinlock is only held with interrupts off on this hart, otherwise
 * an interrupt handler that wants the same lock would spin forever.
 * push_off/pop_off nest, and remember whether interrupts were on before the
 * outermost push_off so the last pop_off restores that state. */
#include "kernel.h"
#include "riscv.h"
#include "cpu.h"
#include "spinlock.h"
#include "thread.h"

void push_off(void) {
    bool old = intr_get();
    intr_off();
    struct cpu *c = mycpu();
    if (c->noff == 0) c->intena = old;
    c->noff++;
}

void pop_off(void) {
    struct cpu *c = mycpu();
    if (intr_get()) panic("pop_off: interrupts enabled");
    if (c->noff < 1) panic("pop_off: unbalanced");
    c->noff--;
    if (c->noff == 0 && c->intena) intr_on();
}

void spin_init(struct spinlock *lk, const char *name) {
    lk->locked = 0;
    lk->name = name;
    lk->owner = NULL;
}

bool spin_holding(struct spinlock *lk) { return lk->locked && lk->owner == mycpu(); }

void spin_lock(struct spinlock *lk) {
    push_off();
    if (spin_holding(lk)) panic("spin_lock: %s already held by this hart", lk->name);
    while (__atomic_exchange_n(&lk->locked, 1, __ATOMIC_ACQUIRE))
        ;
    lk->owner = mycpu();
}

void spin_unlock(struct spinlock *lk) {
    if (!spin_holding(lk)) panic("spin_unlock: %s not held", lk->name);
    lk->owner = NULL;
    __atomic_store_n(&lk->locked, 0, __ATOMIC_RELEASE);
    pop_off();
}

/* Sleep lock: may be held across sleeps, only usable from thread context. */
void sleeplock_init(struct sleeplock *sl, const char *name) {
    spin_init(&sl->lk, "sleeplock");
    sl->locked = false;
    sl->owner = NULL;
    sl->name = name;
}

void sleeplock_acquire(struct sleeplock *sl) {
    spin_lock(&sl->lk);
    while (sl->locked) sleep_on(sl, &sl->lk);
    sl->locked = true;
    sl->owner = thread_current();
    spin_unlock(&sl->lk);
}

void sleeplock_release(struct sleeplock *sl) {
    spin_lock(&sl->lk);
    if (sl->owner != thread_current()) panic("sleeplock_release: %s not owner", sl->name);
    sl->locked = false;
    sl->owner = NULL;
    wakeup(sl);
    spin_unlock(&sl->lk);
}

bool sleeplock_holding(struct sleeplock *sl) {
    spin_lock(&sl->lk);
    bool r = sl->locked && sl->owner == thread_current();
    spin_unlock(&sl->lk);
    return r;
}
