#ifndef TIMER_H
#define TIMER_H

#include <time.h>
#include <stdio.h>

static inline double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

#endif
