#!/usr/bin/env sh

LLVM_MCA=${LLVM_MCA:-llvm-mca}

if [ $# -ne 2 ]; then
	echo "Usage: $0 <cpu> <asm>"
	exit 1
fi

CPU=$1
IN=$2

$LLVM_MCA $IN --all-stats --all-views -march=x86-64 -mcpu=$CPU > $IN.mca
