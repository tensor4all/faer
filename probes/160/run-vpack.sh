#!/usr/bin/env bash
# faer#160 / #90: the two arms of the block apply, mode 0 (main: each column task re-reads V) and
# mode 5 (V and T packed once per call and shared, per-thread reusable working memory; OpenBLAS's
# level3_thread.c roles), on every family that reaches the block apply.
# Usage: bash run-vpack.sh <fork-checkout> <out-dir> [thread counts, default "1 4 8 16 32 64"]
# CPU sets: broccoli's stage-1 sets; on an 8-core host pass "1 4 8" and set CPUS_8 etc. by env.
set -euo pipefail
FORK=$(realpath "$1"); OUT=$(realpath -m "$2"); shift 2
THREADS=(${@:-1 4 8 16 32 64})
mkdir -p "$OUT"
cpus() { local v="CPUS_$1"; if [[ -n "${!v:-}" ]]; then echo "${!v}"; else case $1 in 1) echo 0;; *) echo "0-$(($1 - 1))";; esac; fi; }
guard() {
	local b g
	b=$(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo "?")
	g=$(sort -u /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | tr '\n' ' ')
	echo "# $(date -Is) boost=$b governor=$g load=$(cut -d' ' -f1-3 /proc/loadavg)" | tee -a "$OUT/guard.log"
	[[ "$b" == 0 && "$g" == "performance " ]] || { echo "boost/governor not as required" >&2; exit 1; }
}
( cd "$FORK" && git rev-parse HEAD > "$OUT/fork.sha" \
  && cargo build -j 8 --release -p t4a-faer --features par-trace --example qr_phases_160 --example factor_scaling_160 )
QR="$FORK/target/release/examples/qr_phases_160"
FS="$FORK/target/release/examples/factor_scaling_160"
fam() { # T op n reps [k cols] -- both arms, adjacent
	local T=$1; shift
	for mode in 0 5 0 5; do
		FAER_APPLY_MODE=$mode taskset -c "$(cpus $T)" "$FS" --threads $T --op "$1" --n "$2" --reps "$3" ${4:+--k $4 --cols $5} \
			2>> "$OUT/calls.log" | sed "1d;s/^/mode$mode,/" >> "$OUT/vpack.csv"
	done
}
for T in "${THREADS[@]}"; do
	guard
	# qr_factor: the apply stage by itself (stage columns), both arms in one process
	for n in 1024 2048 4096; do
		modes=0,5; [[ $T == 1 ]] && modes=0
		taskset -c "$(cpus $T)" "$QR" --threads $T --n $n --modes $modes,0,5 --reps 3 | grep -v '^# trace' > "$OUT/qr-$n-T$T.csv"
	done
	# Q formation (the sequence apply alone), the decompositions whose back-transforms use it, and
	# faer#90's standalone shapes
	for n in 1024 2048 4096; do
		r=2; [[ $n == 4096 && $T -le 4 ]] && r=1
		fam $T q $n $r
		fam $T eigh $n $r
		fam $T svd $n $r
	done
	for kc in 64:192 96:160 256:256; do fam $T apply 4096 5 ${kc%:*} ${kc#*:}; done
	fam $T apply 16384 3 256 256
	# controls: the reductions themselves do not call the block apply (their back-transforms do)
	for n in 1024 2048; do for op in hess tridiag bidiag; do fam $T $op $n 2; done; done
done
echo done > "$OUT/DONE"
