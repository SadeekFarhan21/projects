/* Spinlocks (interrupts disabled while held) and sleep locks. */
#pragma once
#include "types.h"

struct cpu;
struct thread;

struct spinlock {
    volatile u32 locked;
    const char *name;
    struct cpu *owner;
};

void spin_init(struct spinlock *lk, const char *name);
void spin_lock(struct spinlock *lk);
void spin_unlock(struct spinlock *lk);
bool spin_holding(struct spinlock *lk);
void push_off(void);
void pop_off(void);

struct sleeplock {
    struct spinlock lk;
    bool locked;
    struct thread *owner;
    const char *name;
};

void sleeplock_init(struct sleeplock *sl, const char *name);
void sleeplock_acquire(struct sleeplock *sl);
void sleeplock_release(struct sleeplock *sl);
bool sleeplock_holding(struct sleeplock *sl);
