#ifndef TIMER_H
#define TIMER_H

#include <stdint.h>

#if defined(__riscv)
// ── RISC-V bare-metal: read the cycle CSR ────────────────────────────────
// rdcycle is available in QEMU user-mode and gives a consistent counter.
// Dividing by 1e6 gives approximate milliseconds at ~1GHz simulated clock.
// Relative comparisons (O0 vs O3 vs RVV) are valid.
static inline double get_time_ms() {
    uint64_t cycles;
    __asm__ volatile ("rdcycle %0" : "=r"(cycles));
    return (double)cycles / 1.0e6;
}

#elif defined(__x86_64__) || defined(__i386__)
#include <time.h>
static inline double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

#else
#include <time.h>
static inline double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
#endif

#endif // TIMER_H

