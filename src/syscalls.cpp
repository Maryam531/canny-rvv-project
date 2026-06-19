// syscalls.cpp
// Low-level OS stubs for riscv64-unknown-elf + Newlib running under
// QEMU user-mode. QEMU intercepts every ecall and forwards it to the
// host Linux kernel, so printf / fopen / malloc all work normally.

#include <time.h>
#include <cstdint>
#include <stddef.h>
#include <sys/stat.h>
#include <errno.h>

#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#endif

// RISC-V 64-bit Linux syscall numbers
#define SYS_clock_gettime 113
#define SYS_openat        56
#define SYS_close         57
#define SYS_read          63
#define SYS_write         64
#define SYS_lseek         62
#define SYS_fstat         80
#define SYS_exit          93
#define AT_FDCWD         -100

extern "C" {

// ── Core ecall helper ─────────────────────────────────────────────────────
// All syscalls go through here. QEMU user-mode sees the ecall instruction
// and translates it into a real host kernel call automatically.
static long do_syscall(long num, long a0, long a1 = 0, long a2 = 0, long a3 = 0)
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

// ── clock_gettime ─────────────────────────────────────────────────────────
// Used by timer.h (get_time_ms) on the RISC-V path.
// QEMU forwards this to the host's real CLOCK_MONOTONIC — accurate timing.
int clock_gettime(clockid_t clk_id, struct timespec* tp)
{
    if (!tp) return -1;
    return (int)do_syscall(SYS_clock_gettime, (long)clk_id, (long)tp, 0);
}

// ── File I/O ──────────────────────────────────────────────────────────────
// Used by image_io.cpp (fopen/fread/fwrite map to these internally).
// QEMU forwards to host filesystem so your .raw image files are accessible.

int _open(const char* path, int flags, int mode)
{
    return (int)do_syscall(SYS_openat, AT_FDCWD, (long)path, flags, mode);
}

int _close(int fd)
{
    return (int)do_syscall(SYS_close, fd, 0, 0);
}

int _read(int fd, void* buf, size_t count)
{
    return (int)do_syscall(SYS_read, fd, (long)buf, (long)count);
}

int _write(int fd, const void* buf, size_t count)
{
    return (int)do_syscall(SYS_write, fd, (long)buf, (long)count);
}

long _lseek(int fd, long offset, int whence)
{
    return do_syscall(SYS_lseek, fd, offset, whence);
}

int _fstat(int fd, struct stat* st)
{
    return (int)do_syscall(SYS_fstat, fd, (long)st, 0);
}

int _isatty(int fd)
{
    return (fd >= 0 && fd <= 2) ? 1 : 0;
}

// ── Heap ──────────────────────────────────────────────────────────────────
// _sbrk is called by malloc/aligned_alloc to grow the heap.
// _end is defined by the linker script as the first free address after BSS.
extern char _end[];
static char* heap_ptr = 0;

void* _sbrk(ptrdiff_t incr)
{
    if (heap_ptr == 0) heap_ptr = _end;
    char* prev = heap_ptr;
    heap_ptr  += incr;
    return (void*)prev;
}

// ── Process ───────────────────────────────────────────────────────────────
void _exit(int status)
{
    do_syscall(SYS_exit, status);
    while (1); // never reached
}

int _getpid(void) { return 1; }

int _kill(int pid, int sig)
{
    (void)pid; (void)sig;
    errno = EINVAL;
    return -1;
}

} // extern "C"
