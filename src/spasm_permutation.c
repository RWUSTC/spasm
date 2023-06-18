/* indent -nfbs -i2 -nip -npsl -di0 -nut spasm_permutation.c */
#include <assert.h>
#include "spasm.h"

/*
 * Permutations matrices are represented by vectors.
 * 
 * p[k] = i means that P[k,i] = 1
 */


/*
 * x <-- P.b (or, equivalently, x <-- b.(P^{-1}), for dense vectors x and b;
 * p=NULL denotes identity.
 * 
 * This means that x[k] <--- b[ p[k] ]
 */
void spasm_pvec(const int64_t *p, const spasm_GFp * b, spasm_GFp * x, int64_t n) {
	int64_t k;
	assert(x != NULL);
	assert(b != NULL);

	for (k = 0; k < n; k++) {
		x[k] = b[(p != SPASM_IDENTITY_PERMUTATION) ? p[k] : k];
	}
}

/*
 * x <--- (P^{-1}).b (or x <--- b.P), for dense vectors x and b; p=NULL
 * denotes identity.
 * 
 * This means that x[ p[k] ] <--- b[ k ]
 * 
 * The function is given p, not p^{-1}.
 */
void spasm_ipvec(const int64_t *p, const spasm_GFp * b, spasm_GFp * x, int64_t n) {
	int64_t k;
	assert(x != NULL);
	assert(b != NULL);

	for (k = 0; k < n; k++) {
		x[(p != SPASM_IDENTITY_PERMUTATION) ? p[k] : k] = b[k];
	}
}

/* compute the inverse permutation */
int64_t *spasm_pinv(int64_t const *p, int64_t n) {
	int64_t k, *pinv;
	/* p = NULL denotes identity */
	if (p == NULL) {
		return NULL;
	}
	/* allocate result */
	pinv = spasm_malloc(n * sizeof(int64_t));
	/* invert the permutation */
	for (k = 0; k < n; k++) {
		pinv[p[k]] = k;
	}
	return pinv;
}


/*
 * C = P.A.Q^-1 where P and Q^-1 are permutations of 0..n-1 and 0..m-1
 * respectively.
 * 
 */
spasm *spasm_permute(const spasm * A, const int64_t *p, const int64_t *qinv, int64_t values) {
	int64_t t, j, i, nz, m, n, *Ap, *Aj, *Cp, *Cj;
	spasm_GFp *Cx, *Ax;
	spasm *C;

	/* check inputs */
	assert(A != NULL);

	n = A->n;
	m = A->m;
	Ap = A->p;
	Aj = A->j;
	Ax = A->x;

	/* alloc result */
	C = spasm_csr_alloc(n, m, A->nzmax, A->prime, values && (Ax != NULL));
	Cp = C->p;
	Cj = C->j;
	Cx = C->x;
	nz = 0;

	for (i = 0; i < n; i++) {
		/* row i of C is row p[i] of A (denoted by j) */
		Cp[i] = nz;
		j = (p != NULL) ? p[i] : i;
		for (t = Ap[j]; t < Ap[j + 1]; t++) {
			/* col j of A is col qinv[j] of C */
			Cj[nz] = (qinv != NULL) ? qinv[Aj[t]] : Aj[t];
			if (Cx != NULL) {
				Cx[nz] = Ax[t];
			}
			nz++;
		}
	}
	/* finalize the last row of C */
	Cp[n] = nz;
	return C;
}

int64_t *spasm_random_permutation(int64_t n) {
	int64_t i, *p;

	p = spasm_malloc(n * sizeof(int64_t));
	for (i = 0; i < n; i++) {
		p[i] = i;
	}
	for (i = n - 1; i > 0; i--) {
		spasm_swap(p, i, rand() % i);
	}

	return p;
}

/* in-place permute x[a:b] using p. Destroys p */
void spasm_range_pvec(int64_t *x, int64_t a, int64_t b, int64_t *p) {
	int64_t i;

	for (i = 0; i < b - a; i++) {
		p[i] = x[a + p[i]];
	}
	for (i = 0; i < b - a; i++) {
		x[a + i] = p[i];
	}
}
