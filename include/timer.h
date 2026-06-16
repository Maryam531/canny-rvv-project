#ifndef TIMER_H
#define TIMER_H

#include <stdio.h>

#if defined(__x86_64__) || defined(__i386__)
#include <time.h>

static inline double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

#elif defined(__riscv)
// ── RISC-V (QEMU) ── read the cycle CSR register ─────────────────────────
// rdcycle reads the hardware cycle counter.
// QEMU simulates roughly 1 GHz, so cycles / 1,000,000 gives milliseconds.
// The relative comparisons between O0/O2/O3/RVV are valid even if the
// absolute values are not cycle-accurate.

static inline double get_time_ms() {
    uint64_t cycles;
    __asm__ volatile ("rdcycle %0" : "=r"(cycles));
    return (double)cycles / 1.0e6;
}

#else
// ── Unknown platform fallback───
#include <time.h>

static inline double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

#endif

#endif // TIMER_H
