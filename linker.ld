cat > linker.ld << 'EOF'
ENTRY(_start)

SECTIONS
{
    . = 0x80000000;

    .text : { *(.text*) }
    .rodata : { *(.rodata*) }
    .data : { *(.data*) }
    .bss : { *(.bss*) *(COMMON) }

    . = ALIGN(16);
    __stack_top = . + 0x200000;   /* 2MB stack */
}
EOF
