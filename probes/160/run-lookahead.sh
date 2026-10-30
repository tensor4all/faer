#!/usr/bin/env bash
# faer#160: look-ahead in the blocked QR against the current loop, mode 0, blocks default/32/64.
# Arms: base, la1 (--lookahead, panel on 1 participant), la2, la4 (panel group of 2 / 4).
# Usage: bash run-lookahead.sh <fork-checkout> <out-dir> [thread counts]
set -euo pipefail
FORK=$(realpath "$1"); OUT=$(realpath -m "$2"); shift 2
THREADS=(${@:-4 8 16 32 64})
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
  && cargo build -j 4 --release -p t4a-faer --features par-trace --example qr_phases_160 )
BIN="$FORK/target/release/examples/qr_phases_160"
for T in "${THREADS[@]}"; do
	for n in 4096 2048 1024; do
		reps=3; [[ $n != 4096 ]] && reps=5
		guard
		# arms adjacent, twice, so drift falls on all of them
		for pass in 1 2; do
			for arm in base la1 la2 la4; do
				case $arm in
					base) flags=() ;;
					la1) flags=(--lookahead --lookahead-width 1) ;;
					la2) flags=(--lookahead --lookahead-width 2) ;;
					la4) flags=(--lookahead --lookahead-width 4) ;;
				esac
				taskset -c "$(cpus $T)" "$BIN" --threads $T --n $n --modes 0 --blocks 0,32,64 --reps $reps "${flags[@]}" 2>> "$OUT/steps.log" \
					| grep -v '^# trace' | sed "s/^/$arm,$pass,/" >> "$OUT/la-$n-T$T.csv"
			done
		done
	done
done
echo done > "$OUT/DONE"
