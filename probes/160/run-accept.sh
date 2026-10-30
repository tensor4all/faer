#!/usr/bin/env bash
# faer#160 acceptance (Parallel Scaling): head (the fix branch) against base (main) and the vendors,
# qr_in_place (qr_factor) n = 256/512/1024/2048/4096, 1..64T (two passes, sides adjacent).
# Usage: bash run-accept.sh <probe-checkout> <base-checkout> <head-checkout> <out> [threads...]
# The probe checkout (probe/160-qr-panel) supplies the drivers; base and head are plain checkouts.
set -euo pipefail
PROBE=$(realpath "$1"); BASE=$(realpath "$2"); HEAD=$(realpath "$3"); OUT=$(realpath -m "$4"); shift 4
THREADS=(${@:-1 4 8 16 32 64})
VENDORS=(${VENDORS:-mkl openblas})
mkdir -p "$OUT"
cpus() { local v="CPUS_$1"; if [[ -n "${!v:-}" ]]; then echo "${!v}"; else case $1 in 1) echo 0;; *) echo "0-$(($1 - 1))";; esac; fi; }
guard() {
	local b g
	b=$(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo "?")
	g=$(sort -u /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | tr '\n' ' ')
	echo "# $(date -Is) boost=$b governor=$g load=$(cut -d' ' -f1-3 /proc/loadavg)" | tee -a "$OUT/guard.log"
	[[ "$b" == 0 && "$g" == "performance " ]] || { echo "boost/governor not as required" >&2; exit 1; }
}
for side in BASE HEAD; do
	dir=${!side}
	cp "$PROBE/faer/examples/factor_scaling_160.rs" "$dir/faer/examples/"
	( cd "$dir" && git rev-parse HEAD > "$OUT/$side.sha" && cargo build -j 4 --release -p t4a-faer --example factor_scaling_160 )
done
for v in "${VENDORS[@]}"; do
	case $v in
		mkl) gcc -O2 -o "$OUT/lapack_mkl" "$PROBE/probes/160/lapack_scaling.c" ${MKL_LINK:--Wl,--no-as-needed -lmkl_rt -lpthread -lm -ldl} ;;
		openblas) gcc -O2 -o "$OUT/lapack_openblas" "$PROBE/probes/160/lapack_scaling.c" -lopenblas ;;
	esac
done
for T in "${THREADS[@]}"; do
	guard
	for n in 256 512 1024 2048 4096; do
		reps=5; [[ $n == 4096 ]] && reps=3
		for pass in 1 2; do
			for side in BASE HEAD; do
				taskset -c "$(cpus $T)" "${!side}/target/release/examples/factor_scaling_160" --threads $T --op qr --n $n --reps $reps \
					| sed "1d;s/^/$side,$pass,/" >> "$OUT/accept.csv"
			done
		done
		for v in "${VENDORS[@]}"; do
			MKL_NUM_THREADS=$T MKL_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=$T taskset -c "$(cpus $T)" "$OUT/lapack_$v" qr $n $reps \
				| sed "1d;s/^/$v,$T,/" >> "$OUT/accept-vendor.csv"
		done
	done
done
echo done > "$OUT/DONE"
