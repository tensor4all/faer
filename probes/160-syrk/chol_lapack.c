/* Yardstick for tensor4all/faer#160, Cholesky item (2) (not for merging): the vendor's counterparts
 * of syrk_probe_160, through the public API only (threads from MKL_NUM_THREADS / OPENBLAS_NUM_THREADS).
 * usage: chol_lapack N REPS [BLOCK]
 * arms: syrk (dsyrk 'L' over the update steps of a blocked Cholesky of block BLOCK), rect (dgemm of
 * the same shapes), potrf (dpotrf 'L' on an SPD matrix restored outside the clock).
 * output: arm,n,rep,ms (the runner prefixes the vendor and the thread count) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
extern void dsyrk_(const char*, const char*, const int*, const int*, const double*, const double*, const int*, const double*, double*, const int*, size_t, size_t);
extern void dgemm_(const char*, const char*, const int*, const int*, const int*, const double*, const double*, const int*, const double*, const int*, const double*, double*, const int*, size_t, size_t);
extern void dpotrf_(const char*, const int*, double*, const int*, int*, size_t);
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec * 1e-6; }
static void seeded(double* a, size_t len, unsigned long long seed) {
	unsigned long long s = 0x9e3779b97f4a7c15ULL ^ seed;
	for (size_t i = 0; i < len; i++) { s = s * 6364136223846793005ULL + 1442695040888963407ULL; a[i] = (double)(s >> 11) / 9007199254740992.0 - 0.5; }
}
int main(int argc, char** argv) {
	int n = argc > 1 ? atoi(argv[1]) : 2048, reps = argc > 2 ? atoi(argv[2]) : 5, b = argc > 3 ? atoi(argv[3]) : 128, info;
	size_t nn = (size_t)n * n;
	double *panel = malloc((size_t)n * b * sizeof(double)), *c = malloc(nn * sizeof(double));
	double *g = malloc(nn * sizeof(double)), *spd = malloc(nn * sizeof(double)), *a = malloc(nn * sizeof(double));
	seeded(panel, (size_t)n * b, 1); seeded(c, nn, 2); seeded(g, nn, 3);
	double one = 1.0, zero = 0.0, alpha = -1e-3;
	/* spd = g g^T + n I */
	dgemm_("N", "T", &n, &n, &n, &one, g, &n, g, &n, &zero, spd, &n, 1, 1);
	for (int i = 0; i < n; i++) spd[i + (size_t)i * n] += n;
	const char* arms[] = {"syrk", "rect", "potrf"};
	printf("arm,n,rep,ms\n");
	for (int rep = -1; rep < reps; rep++) {
		for (int k = 0; k < 3; k++) {
			if (k == 2) memcpy(a, spd, nn * sizeof(double));
			double t0 = now();
			if (k == 2) {
				dpotrf_("L", &n, a, &n, &info, 1);
			} else {
				for (int j = 0; j + b < n; j += b) {
					int m = n - j - b;
					double *A10 = panel + (j + b), *A11 = c + (j + b) + (size_t)(j + b) * n;
					if (k == 0) dsyrk_("L", "N", &m, &b, &alpha, A10, &n, &one, A11, &n, 1, 1);
					else dgemm_("N", "T", &m, &m, &b, &alpha, A10, &n, A10, &n, &one, A11, &n, 1, 1);
				}
			}
			double ms = now() - t0;
			if (rep >= 0) printf("%s,%d,%d,%.3f\n", arms[k], n, rep, ms);
		}
	}
	return 0;
}
