/* Acceptance yardstick for tensor4all/faer#160 / #90: the LAPACK counterparts of
 * factor_scaling_160, through the public API only (threads from MKL_NUM_THREADS /
 * OPENBLAS_NUM_THREADS). usage: lapack_scaling OP N REPS [K COLS]
 * OP: qr (dgeqrf) | q (dorgqr after dgeqrf) | hess (dgehrd) | tridiag (dsytrd) | bidiag (dgebrd)
 *     | apply (dlarfb, basis N x K from dgeqrf + dlarft, target N x COLS)
 * output: op,n,k,cols,threads,rep,ms (threads column left empty; the runner prefixes it) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
extern void dgeqrf_(const int*, const int*, double*, const int*, double*, double*, const int*, int*);
extern void dorgqr_(const int*, const int*, const int*, double*, const int*, const double*, double*, const int*, int*);
extern void dgehrd_(const int*, const int*, const int*, double*, const int*, double*, double*, const int*, int*);
extern void dsytrd_(const char*, const int*, double*, const int*, double*, double*, double*, double*, const int*, int*, size_t);
extern void dgebrd_(const int*, const int*, double*, const int*, double*, double*, double*, double*, double*, const int*, int*);
extern void dlarft_(const char*, const char*, const int*, const int*, const double*, const int*, const double*, double*, const int*, size_t, size_t);
extern void dlarfb_(const char*, const char*, const char*, const char*, const int*, const int*, const int*, const double*, const int*, const double*, const int*, double*, const int*, double*, const int*, size_t, size_t, size_t, size_t);
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec * 1e-6; }
static void seeded(double* a, size_t len, unsigned long long seed) {
	unsigned long long s = 0x9e3779b97f4a7c15ULL ^ seed;
	for (size_t i = 0; i < len; i++) { s = s * 6364136223846793005ULL + 1442695040888963407ULL; a[i] = (double)(s >> 11) / 9007199254740992.0 - 0.5; }
}
int main(int argc, char** argv) {
	const char* op = argc > 1 ? argv[1] : "qr";
	int n = argc > 2 ? atoi(argv[2]) : 1024, reps = argc > 3 ? atoi(argv[3]) : 3;
	int k = argc > 4 ? atoi(argv[4]) : 0, cols = argc > 5 ? atoi(argv[5]) : 0, one = 1, info, ilo = 1, ihi = n;
	size_t nn = (size_t)n * n;
	double *a0 = malloc(nn * sizeof(double)), *a = malloc(nn * sizeof(double));
	double *d = malloc(n * sizeof(double)), *e = malloc(n * sizeof(double)), *t1 = malloc(n * sizeof(double)), *t2 = malloc(n * sizeof(double));
	seeded(a0, nn, n);
	if (!strcmp(op, "tridiag")) for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) a[i + (size_t)j * n] = a0[i + (size_t)j * n] + a0[j + (size_t)i * n];
	if (!strcmp(op, "tridiag")) memcpy(a0, a, nn * sizeof(double));
	int lwork = 64 * n + 64 * 64;
	double q;
	/* workspace queries */
	lwork = -1;
	if (!strcmp(op, "qr") || !strcmp(op, "q") || !strcmp(op, "apply")) dgeqrf_(&n, &n, a, &n, t1, &q, &lwork, &info);
	else if (!strcmp(op, "hess")) dgehrd_(&n, &ilo, &ihi, a, &n, t1, &q, &lwork, &info);
	else if (!strcmp(op, "tridiag")) dsytrd_("L", &n, a, &n, d, e, t1, &q, &lwork, &info, 1);
	else dgebrd_(&n, &n, a, &n, d, e, t1, t2, &q, &lwork, &info);
	lwork = (int)q; if (lwork < 64 * n) lwork = 64 * n;
	double* work = malloc((size_t)lwork * sizeof(double));
	/* apply: the basis of a QR of an N x K matrix, its T, and an N x COLS target */
	double *v = NULL, *T = NULL, *c0 = NULL, *c = NULL, *w = NULL;
	if (!strcmp(op, "apply")) {
		v = malloc((size_t)n * k * sizeof(double)); T = malloc((size_t)k * k * sizeof(double));
		c0 = malloc((size_t)n * cols * sizeof(double)); c = malloc((size_t)n * cols * sizeof(double)); w = malloc((size_t)cols * k * sizeof(double));
		seeded(v, (size_t)n * k, 7); seeded(c0, (size_t)n * cols, 11);
		dgeqrf_(&n, &k, v, &n, t1, work, &lwork, &info);
		dlarft_("F", "C", &n, &k, v, &n, t1, T, &k, 1, 1);
	}
	/* q: the factors of one QR, kept outside the clock */
	double* f = NULL;
	if (!strcmp(op, "q")) { f = malloc(nn * sizeof(double)); memcpy(f, a0, nn * sizeof(double)); dgeqrf_(&n, &n, f, &n, t2, work, &lwork, &info); }
	printf("op,n,k,cols,rep,ms\n");
	for (int rep = -1; rep < reps; rep++) {
		if (!strcmp(op, "apply")) memcpy(c, c0, (size_t)n * cols * sizeof(double));
		else if (!strcmp(op, "q")) memcpy(a, f, nn * sizeof(double));
		else memcpy(a, a0, nn * sizeof(double));
		double t0 = now();
		if (!strcmp(op, "qr")) dgeqrf_(&n, &n, a, &n, t1, work, &lwork, &info);
		else if (!strcmp(op, "q")) dorgqr_(&n, &n, &n, a, &n, t2, work, &lwork, &info);
		else if (!strcmp(op, "hess")) dgehrd_(&n, &ilo, &ihi, a, &n, t1, work, &lwork, &info);
		else if (!strcmp(op, "tridiag")) dsytrd_("L", &n, a, &n, d, e, t1, work, &lwork, &info, 1);
		else if (!strcmp(op, "bidiag")) dgebrd_(&n, &n, a, &n, d, e, t1, t2, work, &lwork, &info);
		else dlarfb_("L", "T", "F", "C", &n, &cols, &k, v, &n, T, &k, c, &n, w, &cols, 1, 1, 1, 1);
		double ms = now() - t0;
		if (rep >= 0) printf("%s,%d,%d,%d,%d,%.3f\n", op, n, k, cols, rep, ms);
	}
	(void)one;
	return 0;
}
