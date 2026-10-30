#!/usr/bin/env bash
# faer#160 stage 2, probe run on broccoli. Usage: bash run.sh <fork-checkout> <out-dir> [vendor-libs...]
# vendor-libs: "mkl" and/or "openblas" (default: both if found). One thread count per process,
# CPU sets as in stage 1 (whole CCX/CCD domains).
set -euo pipefail
FORK=$(realpath "$1"); OUT=$(realpath -m "$2"); shift 2
VENDORS=("${@:-mkl openblas}"); VENDORS=(${VENDORS[*]})
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
declare -A CPUS=([1]=0 [4]=0-3 [8]=0-7 [16]=0-15 [32]=0-31 [64]=0-63)
guard() {
	local b g
	b=$(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo "?")
	g=$(sort -u /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | tr '\n' ' ')
	echo "# $(date -Is) boost=$b governor=$g load=$(cut -d' ' -f1-3 /proc/loadavg)" | tee -a "$OUT/guard.log"
	[[ "$b" == 0 && "$g" == "performance " ]] || { echo "boost/governor not as required" >&2; exit 1; }
}
( cd "$FORK" && git rev-parse HEAD > "$OUT/fork.sha" \
  && cargo build -j 16 --release -p t4a-faer --features par-trace --example qr_phases_160 --example factor_scaling_160 )
BIN="$FORK/target/release/examples/qr_phases_160"
for T in 1 4 8 16 32 64; do
	reps=3; [[ $T == 8 || $T == 16 ]] && reps=5
	modes=0,5,3,4,2; [[ $T == 1 ]] && modes=0
	guard
	# 1. default block (128 at 4096): the four column-split modes, with the gate trace
	taskset -c "${CPUS[$T]}" "$BIN" --threads $T --n 4096 --modes $modes --reps $reps --trace > "$OUT/fork-4096-T$T.csv"
	guard
	# 2. fixed block widths (the columns of every top-level apply): rule vs no split
	taskset -c "${CPUS[$T]}" "$BIN" --threads $T --n 4096 --blocks 32,64,256 --modes ${modes%%,3*} --reps 3 > "$OUT/fork-4096-blocks-T$T.csv"
	guard
	# 3. the smaller sizes, default block, rule vs no split
	for n in 1024 2048; do
		taskset -c "${CPUS[$T]}" "$BIN" --threads $T --n $n --modes ${modes%%,3*} --reps 5 > "$OUT/fork-$n-T$T.csv"
	done
done
# vendor stages: dgeqrf whole, and dgeqr2 / dlarft / dlarfb composed, nb 32, 64, 128
for v in "${VENDORS[@]}"; do
	case $v in
		mkl) gcc -O2 -o "$OUT/geqrf_stages_mkl" "$HERE/geqrf_stages.c" -Wl,--no-as-needed -lmkl_rt -lpthread -lm -ldl ;;
		openblas) gcc -O2 -o "$OUT/geqrf_stages_openblas" "$HERE/geqrf_stages.c" -lopenblas ;;
	esac
	for T in 1 4 8 16 32 64; do
		guard
		for nb in 32 64 128; do
			MKL_NUM_THREADS=$T MKL_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=$T \
				taskset -c "${CPUS[$T]}" "$OUT/geqrf_stages_$v" 4096 $nb 3 | sed "s/^/$v,$T,/" >> "$OUT/vendor-4096.csv"
		done
	done
done
# acceptance families: the fork (this checkout, the rule and the owner-packed apply) against the
# vendors' LAPACK counterparts, n = 1024/2048/4096 (qr), 1024/2048 (the reductions and Q), and the
# standalone apply of #90
FS="$FORK/target/release/examples/factor_scaling_160"
fam() { # threads op n reps [k cols]
	local T=$1; shift
	for mode in 0 5; do
		FAER_APPLY_MODE=$mode taskset -c "${CPUS[$T]}" "$FS" --threads $T --op "$1" --n "$2" --reps "$3" ${4:+--k $4 --cols $5} | sed "1d;s/^/fork-mode$mode,/" >> "$OUT/families.csv"
	done
	for v in "${VENDORS[@]}"; do
		MKL_NUM_THREADS=$T MKL_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=$T \
			taskset -c "${CPUS[$T]}" "$OUT/lapack_scaling_$v" "$1" "$2" "$3" ${4:+$4 $5} | sed "1d;s/^/$v,$T,/" >> "$OUT/families-vendor.csv"
	done
}
for v in "${VENDORS[@]}"; do
	case $v in
		mkl) gcc -O2 -o "$OUT/lapack_scaling_mkl" "$HERE/lapack_scaling.c" -Wl,--no-as-needed -lmkl_rt -lpthread -lm -ldl ;;
		openblas) gcc -O2 -o "$OUT/lapack_scaling_openblas" "$HERE/lapack_scaling.c" -lopenblas ;;
	esac
done
for T in 1 4 8 16 32 64; do
	guard
	for n in 1024 2048 4096; do fam $T qr $n 3; done
	for n in 1024 2048; do for op in q hess tridiag bidiag; do fam $T $op $n 3; done; done
	for kc in 64:192 96:160 256:256; do fam $T apply 4096 5 ${kc%:*} ${kc#*:}; done
	fam $T apply 16384 3 256 256
done
echo done > "$OUT/DONE"
