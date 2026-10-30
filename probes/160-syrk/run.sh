#!/usr/bin/env bash
# faer#160 Cholesky item (2): the trailing update under the kernel's split (fork) and equal-area
# column ranges (area), the rectangular product of the same shapes (rect), and cholesky_in_place
# under each split, against the vendors' dsyrk / dgemm / dpotrf on the same steps.
# Usage: bash run.sh <fork-checkout at probe/160-syrk> <out-dir> [threads...]
# CPU sets: 1T 0, kT 0..k-1 (override with CPUS_<k>=...); VENDORS="mkl openblas" by default.
set -euo pipefail
FORK=$(realpath "$1"); OUT=$(realpath -m "$2"); shift 2
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
( cd "$FORK" && git rev-parse HEAD > "$OUT/fork.sha" \
  && cargo build -j 16 --release -p t4a-faer --features par-trace --example syrk_probe_160 )
BIN="$FORK/target/release/examples/syrk_probe_160"
for v in "${VENDORS[@]}"; do
	case $v in
		mkl) gcc -O2 -o "$OUT/chol_mkl" "$FORK/probes/160-syrk/chol_lapack.c" ${MKL_LINK:--Wl,--no-as-needed -lmkl_rt -lpthread -lm -ldl} ;;
		openblas) gcc -O2 -o "$OUT/chol_openblas" "$FORK/probes/160-syrk/chol_lapack.c" -lopenblas ;;
	esac
done
for T in "${THREADS[@]}"; do
	guard
	for n in 1024 2048 4096; do
		reps=7; [[ $n == 4096 ]] && reps=5
		taskset -c "$(cpus $T)" "$BIN" --threads $T --n $n --reps $reps | sed 1d >> "$OUT/faer.csv"
		for v in "${VENDORS[@]}"; do
			MKL_NUM_THREADS=$T MKL_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=$T taskset -c "$(cpus $T)" "$OUT/chol_$v" $n $reps \
				| sed "1d;s/^/$v,$T,/" >> "$OUT/vendor.csv"
		done
	done
done
echo done > "$OUT/DONE"
