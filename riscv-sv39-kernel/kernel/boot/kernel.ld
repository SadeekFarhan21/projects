/* Kernel image layout. OpenSBI owns 0x80000000..0x80200000; we load above it.
 * Every boundary that changes page permissions is 4 KiB aligned so the
 * page-table code can give each range its own PTE flags. */
OUTPUT_ARCH(riscv)
ENTRY(_start)

SECTIONS
{
    . = 0x80200000;
    _kernel_start = .;

    .text : {
        KEEP(*(.text.boot))          /* _start must be the first instruction */
        *(.text .text.*)
    }
    . = ALIGN(4096);
    _text_end = .;                   /* text: R-X */

    .rodata : {
        *(.srodata .srodata.*)
        *(.rodata .rodata.*)
    }
    . = ALIGN(4096);
    _rodata_end = .;                 /* rodata: R-- */

    _data_start = .;
    .data : {
        *(.sdata .sdata.*)
        *(.data .data.*)
    }
    .bss (NOLOAD) : {
        . = ALIGN(16);
        _bss_start = .;
        *(.sbss .sbss.*)
        *(.bss .bss.*)
        *(COMMON)
        . = ALIGN(16);
        _bss_end = .;
    }
    . = ALIGN(4096);                 /* data + bss: RW- */

    /* Boot stacks: one slot per hart, each slot = [guard page][stack pages].
     * The guard pages are left unmapped once paging is on. */
    .bootstacks (NOLOAD) : {
        _stacks_start = .;
        *(.bootstacks)
        _stacks_end = .;
    }
    . = ALIGN(4096);
    _kernel_end = .;

    /DISCARD/ : {
        *(.eh_frame .eh_frame_hdr .comment .note .note.*)
    }
}
