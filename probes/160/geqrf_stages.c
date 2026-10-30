/* Probe for tensor4all/faer#160: times LAPACK dgeqrf on a square f64 matrix and the same blocked
 * algorithm composed from the public routines (dgeqr2 panel, dlarft factor, dlarfb trailing apply),
 * one stage at a time. Vendor timed through its public API only.
 * usage: geqrf_stages N NB REPS   (threads from MKL_NUM_THREADS / OPENBLAS_NUM_THREADS)
 * output: n,nb,rep,dgeqrf_ms,composed_ms,panel_ms,factor_ms,apply_ms */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
extern void dgeqrf_(const int*, const int*, double*, const int*, double*, double*, const int*, int*);
extern void dgeqr2_(const int*, const int*, double*, const int*, double*, double*, int*);
extern void dlarft_(const char*, const char*, const int*, const int*, const double*, const int*, const double*, double*, const int*, size_t, size_t);
extern void dlarfb_(const char*, const char*, const char*, const char*, const int*, const int*, const int*, const double*, const int*, const double*, const int*, double*, const int*, double*, const int*, size_t, size_t, size_t, size_t);
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec * 1e-6; }
int main(int argc, char** argv) {
	int n = argc > 1 ? atoi(argv[1]) : 4096, nb = argc > 2 ? atoi(argv[2]) : 128, reps = argc > 3 ? atoi(argv[3]) : 3;
	size_t nn = (size_t)n * n;
	double *a0 = malloc(nn * sizeof(double)), *a = malloc(nn * sizeof(double)), *tau = malloc(n * sizeof(double));
	unsigned long long s = 0x9e3779b97f4a7c15ULL ^ (unsigned long long)n;
	for (size_t i = 0; i < nn; i++) { s = s * 6364136223846793005ULL + 1442695040888963407ULL; a0[i] = (double)(s >> 11) / 9007199254740992.0 - 0.5; }
	int lwork = -1, info; double q;
	dgeqrf_(&n, &n, a, &n, tau, &q, &lwork, &info);
	lwork = (int)q; if (lwork < n * nb) lwork = n * nb;
	double *work = malloc((size_t)lwork * sizeof(double)), *T = malloc((size_t)nb * nb * sizeof(double));
	printf("n,nb,rep,dgeqrf_ms,composed_ms,panel_ms,factor_ms,apply_ms\n");
	for (int rep = -1; rep < reps; rep++) { /* rep -1: untimed warm-up */
		memcpy(a, a0, nn * sizeof(double));
		double t0 = now();
		dgeqrf_(&n, &n, a, &n, tau, work, &lwork, &info);
		double whole = now() - t0;
		memcpy(a, a0, nn * sizeof(double));
		double p = 0, f = 0, ap = 0, c0 = now();
		for (int j = 0; j < n; j += nb) {
			int jb = n - j < nb ? n - j : nb, m = n - j, nr = n - j - jb, ld = n;
			double* A = a + (size_t)j * n + j;
			double t = now();
			dgeqr2_(&m, &jb, A, &ld, tau + j, work, &info);
			p += now() - t;
			if (nr > 0) {
				t = now();
				dlarft_("F", "C", &m, &jb, A, &ld, tau + j, T, &nb, 1, 1);
				f += now() - t;
				t = now();
				dlarfb_("L", "T", "F", "C", &m, &nr, &jb, A, &ld, T, &nb, A + (size_t)jb * n, &ld, work, &nr, 1, 1, 1, 1);
				ap += now() - t;
			}
		}
		double composed = now() - c0;
		if (rep >= 0) printf("%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f\n", n, nb, rep, whole, composed, p, f, ap);
	}
	return 0;
}
