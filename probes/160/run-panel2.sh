#!/usr/bin/env bash
# faer#160 panel probe 2: the panel's breakdown by basis width (pa_w1..pa_w128, panel_factor_ms) and the
# panel run sequentially under a team trailing apply (--panel-seq: P(1) per block, the look-ahead model's
# input), mode 0, blocks default/32/64. Usage: bash run-panel2.sh <fork-checkout> <out-dir>
set -euo pipefail
FORK=$(realpath "$1"); OUT=$(realpath -m "$2")
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
  && cargo build -j 4 --release -p t4a-faer --features par-trace --example qr_phases_160 )
BIN="$FORK/target/release/examples/qr_phases_160"
for T in 1 4 8 16 32 64; do
	for n in 4096 2048 1024; do
		reps=3; [[ $n != 4096 ]] && reps=5
		guard
		taskset -c "${CPUS[$T]}" "$BIN" --threads $T --n $n --modes 0 --blocks 0,32,64 --reps $reps | grep -v '^# trace' > "$OUT/p2-$n-T$T.csv"
		if [[ $T != 1 ]]; then
			taskset -c "${CPUS[$T]}" "$BIN" --threads $T --n $n --modes 0 --blocks 0,32,64 --reps $reps --panel-seq | grep -v '^# trace' > "$OUT/p2-$n-T$T-seq.csv"
		fi
	done
done
echo done > "$OUT/DONE"
