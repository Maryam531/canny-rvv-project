// syscalls.cpp
// Newlib syscall stubs for riscv64-unknown-elf running under QEMU user-mode.
//
// QEMU user-mode intercepts every ecall and forwards it to the host Linux
// kernel. So _write → printf works, _open → fopen works, etc.
// We do NOT reference _end or any linker symbols because QEMU user-mode
// manages the heap itself through the brk syscall.

#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <stddef.h>
#include <stdint.h>

// Linux RISC-V syscall numbers (same ones QEMU user-mode intercepts)
#define SYS_exit      93
#define SYS_read      63
#define SYS_write     64
#define SYS_openat    56
#define SYS_close     57
#define SYS_lseek     62
#define SYS_fstat     80
#define SYS_brk       214
#define AT_FDCWD     -100

extern "C" {

// ── Core ecall: all syscalls go through here ──────────────────────────────
static inline long _syscall(long num,
                             long a0 = 0, long a1 = 0,
                             long a2 = 0, long a3 = 0)
{
    register long r7 asm("a7") = num;
    register long r0 asm("a0") = a0;
    register long r1 asm("a1") = a1;
    register long r2 asm("a2") = a2;
    register long r3 asm("a3") = a3;
    asm volatile ("ecall"
        : "+r"(r0)
        : "r"(r7), "r"(r1), "r"(r2), "r"(r3)
        : "memory");
    return r0;
}

// ── _sbrk: heap growth via brk syscall ───────────────────────────────────
// Do NOT use _end symbol — that's a bare-metal linker symbol that doesn't
// exist in QEMU user-mode. Use the brk syscall instead, which QEMU handles.
void* _sbrk(ptrdiff_t incr)
{
    // brk(0) returns current break address
    static char* heap_end = nullptr;
    if (heap_end == nullptr)
        heap_end = (char*)_syscall(SYS_brk, 0);

    char* prev = heap_end;
    char* next = heap_end + incr;

    if (_syscall(SYS_brk, (long)next) != (long)next) {
        errno = ENOMEM;
        return (void*)-1;
    }
    heap_end = next;
    return (void*)prev;
}

// ── File I/O ──────────────────────────────────────────────────────────────
int _write(int fd, const void* buf, size_t count)
{
    return (int)_syscall(SYS_write, fd, (long)buf, (long)count);
}

int _read(int fd, void* buf, size_t count)
{
    return (int)_syscall(SYS_read, fd, (long)buf, (long)count);
}

int _open(const char* path, int flags, int mode)
{
    return (int)_syscall(SYS_openat, AT_FDCWD, (long)path, flags, mode);
}

int _close(int fd)
{
    return (int)_syscall(SYS_close, fd, 0, 0);
}

long _lseek(int fd, long offset, int whence)
{
    return _syscall(SYS_lseek, fd, offset, whence);
}

int _fstat(int fd, struct stat* st)
{
    return (int)_syscall(SYS_fstat, fd, (long)st, 0);
}

int _isatty(int fd)
{
    return (fd >= 0 && fd <= 2) ? 1 : 0;
}

// ── Process ───────────────────────────────────────────────────────────────
void __attribute__((noreturn)) _exit(int status)
{
    _syscall(SYS_exit, status);
    while (1);
}

int _getpid(void) { return 1; }

int _kill(int pid, int sig)
{
    (void)pid; (void)sig;
    errno = EINVAL;
    return -1;
}

} // extern "C"
