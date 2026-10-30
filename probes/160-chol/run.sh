#!/usr/bin/env bash
# faer#160 cholesky probe on broccoli. Usage: bash run.sh <fork-checkout> <out-dir>
# One thread count per process; boost/governor read back before each block.
set -euo pipefail
FORK=$(realpath "$1"); OUT=$(realpath -m "$2")
mkdir -p "$OUT"
guard() {
	local b g
	b=$(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo "?")
	g=$(sort -u /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | tr '\n' ' ')
	echo "# $(date -Is) boost=$b governor=$g load=$(cut -d' ' -f1-3 /proc/loadavg)" | tee -a "$OUT/guard.log"
	[[ "$b" == 0 && "$g" == "performance " ]] || { echo "boost/governor not as required" >&2; exit 1; }
}
( cd "$FORK" && git rev-parse HEAD > "$OUT/fork.sha" \
  && cargo build -j 16 --release -p t4a-faer --features par-trace --example chol_trace_160 )
BIN="$FORK/target/release/examples/chol_trace_160"
run() { # label cpus threads n reps caps
	guard
	taskset -c "$2" "$BIN" --threads "$3" --n "$4" --reps "$5" --caps "$6" | sed "s/^/$1,/" >> "$OUT/chol.csv"
}
# (1) the 1024/64T collapse: width 64 on all CPUs vs the same team with every gated width capped,
#     vs fewer threads on all or on part of the host (separates "64-wide regions" from "a thread on
#     every CPU", e.g. another process or the observer preempting a spinning member)
for n in 1024 2048; do
	run all64 0-63 64 $n 9 0,63,60,48,32,16
	run all63 0-62 63 $n 9 0
	run hi63 1-63 63 $n 9 0
	run hi60 4-63 60 $n 9 0
	run lo60 0-59 60 $n 9 0
	run all48 0-47 48 $n 9 0
	run all32 0-31 32 $n 9 0
done
# (2) the 4-16T flat efficiency at 2048/4096: the stage split
for n in 2048 4096; do
	for T in 1 4 8 16 32; do
		case $T in 1) c=0;; 4) c=0-3;; 8) c=0-7;; 16) c=0-15;; 32) c=0-31;; esac
		run t$T $c $T $n 5 0
	done
done
echo done > "$OUT/DONE"
