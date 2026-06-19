#include <time.h>
#include <cstdint>
#include <stddef.h>

#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#endif

#define SYS_clock_gettime 113
#define SYS_openat        56
#define SYS_close         57
#define SYS_read          63
#define SYS_write         64
#define SYS_lseek         62
#define SYS_fstat         80
#define AT_FDCWD         -100

extern "C" {

static long issue_qemu_syscall(long num, long a0, long a1, long a2, long a3 = 0)
{
    register long r_num asm("a7") = num;
    register long r_a0  asm("a0") = a0;
    register long r_a1  asm("a1") = a1;
    register long r_a2  asm("a2") = a2;
    register long r_a3  asm("a3") = a3;
    asm volatile(
        "ecall"
        : "+r"(r_a0)
        : "r"(r_num), "r"(r_a1), "r"(r_a2), "r"(r_a3)
        : "memory"
    );
    return r_a0;
}

int clock_gettime(clockid_t clk_id, struct timespec* tp)
{
    if (!tp) return -1;
    return (int)issue_qemu_syscall(SYS_clock_gettime, (long)clk_id, (long)tp, 0);
}

int _open(const char* path, int flags, int mode)
{
    return (int)issue_qemu_syscall(SYS_openat, AT_FDCWD, (long)path, flags, mode);
}

int _close(int fd)
{
    return (int)issue_qemu_syscall(SYS_close, fd, 0, 0);
}

int _read(int fd, void* buf, size_t count)
{
    return (int)issue_qemu_syscall(SYS_read, fd, (long)buf, (long)count);
}

int _write(int fd, const void* buf, size_t count)
{
    return (int)issue_qemu_syscall(SYS_write, fd, (long)buf, (long)count);
}

long _lseek(int fd, long offset, int whence)
{
    return issue_qemu_syscall(SYS_lseek, fd, offset, whence);
}

int _fstat(int fd, void* st)
{
    return (int)issue_qemu_syscall(SYS_fstat, fd, (long)st, 0);
}

int _isatty(int fd)
{
    return (fd >= 0 && fd <= 2) ? 1 : 0;
}

} // extern "C"
