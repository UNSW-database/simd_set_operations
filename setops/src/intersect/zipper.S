global zipper_u32_asm
global zipper_i32_asm
global zipper_branch_optimized_u32_asm
global zipper_branchless_u32_asm
global zipper_loop_optimized_u32_asm
global zipper_noindex_u32_asm
global zipper_bad_branch_u32_asm

section .aligned progbits alloc exec nowrite align=64


;                    ;
; === ZIPPER_U32 === ;
;                    ;

; Input registers:
; rdi -> u32 *set0
; rsi -> u32 *set1
; rdx -> u64  set0_len
; rcx -> u64  set1_len
; r8  -> u32 *out
; r9  -> u64  out_len (unused)
;
; Return register:
; rax -> u64 intersection_len
;
; Clobbered registers:
; r10
; r11

align 64
zipper_u32_asm:
        ; if either length is 0 then early exit
        test    rdx, rdx
        setne   r10b
        test    rcx, rcx
        setne   r11b
        and     r10b, r11b
        xor     eax, eax  ; clear rax early to allow for simpler early exit
        cmp     r10b, 1
        jne     .exit
        ; use r10, r11 as set0/set1 indices, and rax as intersection length
        xor     r10d, r10d
        xor     r11d, r11d
        ; padding nops to mitigate Jcc erratum
        nop
        nop
        nop
.loop:
        mov     r9d, dword [rdi + 4*r10]  ; load value from 1st set
        cmp     r9d, dword [rsi + 4*r11]  ; compare with value from 2nd set
        je      .equal
        jb      .second_larger
.first_larger:
        inc     r11                      ; increment 2nd set index
        jmp     .check_second
.equal:
        mov     dword [r8 + 4*rax], r9d  ; write intersected value to out
        inc     rax                      ; increment count
        inc     r11                      ; increment 2nd set index
.second_larger:
        inc     r10                      ; increment 1st set index
        cmp     r10, rdx
        je      .exit
.check_second:
        cmp     r11, rcx
        jne     .loop
.exit:
        ret


;                    ;
; === ZIPPER_I32 === ;
;                    ;

; Input registers:
; rdi -> u32 *set0
; rsi -> u32 *set1
; rdx -> u64  set0_len
; rcx -> u64  set1_len
; r8  -> u32 *out
; r9  -> u64  out_len (unused)
;
; Return register:
; rax -> u64 intersection_len
;
; Clobbered registers:
; r10
; r11

align 64
zipper_i32_asm:
        ; if either length is 0 then early exit
        test    rcx, rcx
        setne   r10b
        test    rdx, rdx
        setne   r11b
        and     r10b, r11b
        xor     eax, eax  ; clear rax early to allow for simpler early exit
        cmp     r10b, 1
        jne     .exit
        ; use r10, r11 as set0/set1 indices, and rax as intersection length
        xor     r10d, r10d
        xor     r11d, r11d
        ; padding nops to mitigate Jcc erratum
        nop
        nop
        nop
.loop:
        mov     r9d, dword [rdi + 4*r10]  ; load value from 1st set
        cmp     r9d, dword [rsi + 4*r11]  ; compare with value from 2nd set
        ; ZF = 1 => values are equal
        ; ZF = 0 =>
        ;     CF = 1 => 2nd value larger
        ;     CF = 0 => 1st value larger
        je      .equal
        jl      .second_larger
.first_larger:
        inc     r11
        jmp     .check_second
.equal:
        mov     dword [r8 + 4*rax], r9d  ; write intersected value to out
        inc     rax                      ; incremente count
        inc     r11                      ; increment 2nd set index
.second_larger:
        inc     r10                      ; increment 1st set index
        cmp     r10, rcx
        je      .exit
.check_second:
        cmp     r11, rdx
        jne     .loop
.exit:
        ret


;                                     ;
; === ZIPPER_BRANCH_OPTIMIZED_U32 === ;
;                                     ;

; Input registers:
; rdi -> u32 *set0
; rsi -> u32 *set1
; rdx -> u64  set0_len
; rcx -> u64  set1_len
; r8  -> u32 *out
; r9  -> u64  out_len (unused)
;
; Return register:
; rax -> u64 intersection_len
;
; Clobbered registers:
; r10
; r11

align 64
zipper_branch_optimized_u32_asm:
        ; if either length is 0 then early exit
        test    rdx, rdx
        setne   r10b
        test    rcx, rcx
        setne   r11b
        and     r10b, r11b
        xor     eax, eax  ; clear rax early to allow for simpler early exit
        cmp     r10b, 1
        jne     .exit
        ; use r10, r11 as set0/set1 indices, and rax as intersection length
        xor     r10d, r10d
        xor     r11d, r11d
        ; use rbx (bl) to store set0 index increment value
        push    rbx
        xor     ebx, ebx
        ; use rbp (bpl) to store set1 index increment value
        push    rbp
        xor     ebp, ebp
.loop:
        mov     r9d, dword [rdi + 4*r10]  ; load value from 1st set
        cmp     r9d, dword [rsi + 4*r11]  ; compare with value from 2nd set
        jne     .index_update
        mov     dword [r8 + 4*rax], r9d   ; write intersected value to out
        inc     rax
        inc     r10
        inc     r11
        jmp     .check_bounds
.index_update:
        setbe   bl        ; set rbx if 2nd value >=
        setae   bpl       ; set rbp if 1st value >=
        add     r10, rbx
        add     r11, rbp
.check_bounds:
        cmp     r10, rdx
        je      .exit
        cmp     r11, rcx
        jne     .loop
.exit:
        pop     rbp
        pop     rbx
        ret


;                               ;
; === ZIPPER_BRANCHLESS_U32 === ;
;                               ;

; Input registers:
; rdi -> u32 *set0
; rsi -> u32 *set1
; rdx -> u64  set0_len
; rcx -> u64  set1_len
; r8  -> u32 *out
; r9  -> u64  out_len (unused)
;
; Return register:
; rax -> u64 intersection_len
;
; Clobbered registers:
; r10
; r11

align 64
zipper_branchless_u32_asm:
        ; if either length is 0 then early exit
        test    rdx, rdx
        setne   r10b
        test    rcx, rcx
        setne   r11b
        and     r10b, r11b
        xor     eax, eax  ; clear rax early to allow for simpler early exit
        cmp     r10b, 1
        jne     .early_exit
        ; use r10, r11 as set0/set1 indices, and rax as intersection length
        xor     r10d, r10d
        xor     r11d, r11d
        ; use rbx (bl) to store set0 index increment value
        push    rbx
        xor     ebx, ebx
        ; use rbp (bpl) to store set1 index increment value
        push    rbp
        xor     ebp, ebp
        ; use r12 to store counter increment
        push    r12
        xor     r12d, r12d
        ; use r13 to store current output value
        push    r13
        xor     r13d, r13d
.loop:
        mov     r9d, dword [rdi + 4*r10]  ; load value from 1st set
        cmp     r9d, dword [rsi + 4*r11]  ; compare with value from 2nd set
        setbe   bl                        ; set rbx if 2nd value >=
        setae   bpl                       ; set rbp if 1st value >=
        sete    r12b                      ; set r12 if values equal
        cmove   r13d, r9d                 ; move new value into writen register if equal
        mov     dword [r8 + 4*rax], r13d  ; unconditionally write r13 to output
        add     rax, r12                  ; conditionally increment counter
        add     r10, rbx                  ; conditionally increment set0 index
        add     r11, rbp                  ; conditionally increment set1 index
        ; loop condition
        cmp     r10, rdx
        je      .exit
        cmp     r11, rcx
        jne     .loop
.exit:
        pop     r13
        pop     r12
        pop     rbp
        pop     rbx
.early_exit:
        ret


;                                   ;
; === ZIPPER_LOOP_OPTIMIZED_U32 === ;
;                                   ;

; NB: Expects set0 to be the set with the smallest last value
;
; Input registers:
; rdi -> u32 *set0
; rsi -> u32 *set1
; rdx -> u64  set0_len
; rcx -> u64  set1_len
; r8  -> u32 *out
; r9  -> u64  out_len (unused)
;
; Return register:
; rax -> u64 intersection_len
;
; Clobbered registers:
; r10
; r11

align 64
zipper_loop_optimized_u32_asm:
        ; if either length is 0 then early exit
        test    rdx, rdx
        setne   r10b
        test    rcx, rcx
        setne   r11b
        and     r10b, r11b
        xor     eax, eax  ; clear rax early to allow for simpler early exit
        cmp     r10b, 1
        jne     .exit
        ; use r10, r11 as set0/set1 indices
        xor     r10d, r10d
        xor     r11d, r11d
        ; padding nops to mitigate Jcc erratum
        nop
        nop
        nop
.loop:
        mov     r9d, dword [rdi + 4*r10]  ; load value from 1st set
        cmp     r9d, dword [rsi + 4*r11]  ; compare with value from 2nd set
        je      .equal
        jb      .second_larger
; first larger
        inc     r11                      ; increment 2nd set index
        jmp     .loop
.equal:
        mov     dword [r8 + 4*rax], r9d  ; write intersected value to out
        inc     rax                      ; increment count
        inc     r11                      ; increment 2nd set index
.second_larger:
        inc     r10                      ; increment 1st set index
        cmp     r10, rdx
        jne     .loop
.exit:
        ret


;                                   ;
; === ZIPPER_NOINDEX_U32 === ;
;                                   ;

; NB: Expects set0 to be the set with the smallest last value
;
; Input registers:
; rdi -> u32 *set0
; rsi -> u32 *set1
; rdx -> u64  set0_len
; rcx -> u64  set1_len
; r8  -> u32 *out
; r9  -> u64  out_len (unused)
;
; Return register:
; rax -> u64 intersection_len
;
; Clobbered registers:
; r10
; r11

align 64
zipper_noindex_u32_asm:
        ; if either length is 0 then early exit
        test    rdx, rdx
        setne   r10b
        test    rcx, rcx
        setne   r11b
        and     r10b, r11b
        xor     eax, eax  ; clear rax early to allow for simpler early exit
        cmp     r10b, 1
        jne     .exit
        ; setup loop end conditions
        lea     rdx, [rdi + 4*rdx]
        ; pad to mitigate jcc erratum
        nop
        nop
        nop
        nop
.loop:
        mov     r9d, dword [rdi]  ; load value from 1st set
        cmp     r9d, dword [rsi]  ; compare with value from 2nd set
        je      .equal
        jb      .second_larger
; first larger
        add     rsi, 4                      ; increment 2nd set index
        jmp     .loop
.equal:
        mov     dword [r8 + 4*rax], r9d  ; write intersected value to out
        inc     rax                      ; increment count
        add     rsi, 4                      ; increment 2nd set index
.second_larger:
        add     rdi, 4                      ; increment 1st set index
        cmp     rdi, rdx
        jne     .loop
.exit:
        ret


;                               ;
; === ZIPPER_BAD_BRANCH_U32 === ;
;                               ;

; Input registers:
; rdi -> u32 *set0
; rsi -> u32 *set1
; rdx -> u64  set0_len
; rcx -> u64  set1_len
; r8  -> u32 *out
; r9  -> u64  out_len (unused)
;
; Return register:
; rax -> u64 intersection_len
;
; Clobbered registers:
; r10
; r11

align 64
zipper_bad_branch_u32_asm:
        ; if either length is 0 then early exit
        test    rdx, rdx
        setne   r10b
        test    rcx, rcx
        setne   r11b
        and     r10b, r11b
        xor     eax, eax  ; clear rax early to allow for simpler early exit
        cmp     r10b, 1
        jne     .exit
        ; use r10, r11 as set0/set1 indices, and rax as intersection length
        xor     r10d, r10d
        xor     r11d, r11d
        nop     ; padding nop to mitigate Jcc erratum
        jmp     .loop
.first_larger:
        inc     r11                      ; increment 2nd set index
.check_second:
        cmp     r11, rcx
        je     .exit
.loop:
        mov     r9d, dword [rdi + 4*r10]  ; load value from 1st set
        cmp     r9d, dword [rsi + 4*r11]  ; compare with value from 2nd set
        jb      .second_larger
        ja      .first_larger
; equal
        mov     dword [r8 + 4*rax], r9d  ; write intersected value to out
        inc     rax                      ; increment count
        inc     r11                      ; increment 2nd set index
.second_larger:
        inc     r10                      ; increment 1st set index
        cmp     r10, rdx
        jne     .check_second
.exit:
        ret


