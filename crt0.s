cat > crt0.s << 'EOF'
.section .text
.global _start
_start:
    la sp, __stack_top
    call main
1:  j 1b
EOF

