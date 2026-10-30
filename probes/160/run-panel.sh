#!/usr/bin/env bash
# faer#160 stage 2, panel probe (v3). Usage: bash run-panel.sh <fork-checkout> <out-dir>
# The panel's block applies split over the basis rows (mode 6) against the rule (mode 0), at
# outer block widths 32/64/128 (128 is the default at 4096), plus mode 6 with a wider row-split
# threshold. One thread count per process; CPU sets as stage 1; boost/governor read back.
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
  && cargo build -j 16 --release -p t4a-faer --features par-trace --example qr_phases_160 )
BIN="$FORK/target/release/examples/qr_phases_160"
for T in 1 4 8 16 32 64; do
	modes=0,6; [[ $T == 1 ]] && modes=0
	for n in 4096 2048 1024; do
		reps=3; [[ $n != 4096 ]] && reps=5
		guard
		taskset -c "${CPUS[$T]}" "$BIN" --threads $T --n $n --modes $modes --blocks 0,32,64 --reps $reps \
			| grep -v '^# trace' > "$OUT/panel-$n-T$T.csv"
	done
	if [[ $T != 1 ]]; then
		guard
		taskset -c "${CPUS[$T]}" "$BIN" --threads $T --n 4096 --modes 6 --blocks 0,64 --reps 3 --row-split-cols 512 \
			| grep -v '^# trace' > "$OUT/panel-4096-T$T-rs512.csv"
	fi
done
echo done > "$OUT/DONE"
