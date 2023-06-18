/* indent -nfbs -i2 -nip -npsl -di0 -nut spasm_triplet.c */
#include <assert.h>
#include "spasm.h"

/* add an entry to a triplet matrix; enlarge it if necessary */
void spasm_add_entry(spasm_triplet * T, int64_t i, int64_t j, spasm_GFp x) {
	spasm_GFp x_p;

	assert((i >= 0) && (j >= 0));
	int64_t prime = T->prime;

	if (T->nz == T->nzmax)
		spasm_triplet_realloc(T, 1 + 2 * T->nzmax);
	if (T->x != NULL) {
		x_p = ((x % prime) + prime) % prime;
		if (x_p == 0)
			return;
		T->x[T->nz] = x_p;
	}

	// fprintf(stderr, "Adding (%d, %d, %d)\n", i, j, x);

	T->i[T->nz] = i;
	T->j[T->nz] = j;
	T->nz += 1;
	T->n = spasm_max(T->n, i + 1);
	T->m = spasm_max(T->m, j + 1);
}

void spasm_triplet_transpose(spasm_triplet * T) {
	int64_t nz = T->nz;
	int64_t *Ti = T->i;
	int64_t *Tj = T->j;
#pragma omp parallel for schedule(static)
	for (int64_t k = 0; k < nz; k++) {
		int64_t i = Ti[k];
		int64_t j = Tj[k];
		Tj[k] = i;
		Ti[k] = j;
	}
	int64_t tmp = T->m;
	T->m = T->n;
	T->n = tmp;
}


/* in-place */
void spasm_deduplicate(spasm * A) {
	int64_t m = A->m;
	int64_t n = A->n;
	int64_t *Ap = A->p;
	int64_t *Aj = A->j;
	spasm_GFp *Ax = A->x;
	int64_t prime = A->prime;

	int64_t *v = spasm_malloc(m * sizeof(*v));
	for (int64_t j = 0; j < m; j++)
		v[j] = -1;

	int64_t nz = 0;
	for (int64_t i = 0; i < n; i++) {
		int64_t p = nz;
		for (int64_t it = Ap[i]; it < Ap[i + 1]; it++) {
			int64_t j = Aj[it];
			if (v[j] < p) { /* occurs in previous row */
				v[j] = nz;
				Aj[nz] = j;
				if (Ax)
					Ax[nz] = Ax[it];
				nz++;
			} else {
				if (Ax)
					Ax[v[j]] = (Ax[v[j]] + Ax[it]) % prime;
			}
		}
		Ap[i] = p;
	}
	Ap[n] = nz;
	free(v);
	spasm_csr_realloc(A, -1);
}

/* C = compressed-row form of a triplet matrix T */
spasm *spasm_compress(const spasm_triplet * T) {
	int64_t m = T->m;
	int64_t n = T->n;
	int64_t nz = T->nz;
	int64_t *Ti = T->i;
	int64_t *Tj = T->j;
	spasm_GFp *Tx = T->x;
	
	double start = spasm_wtime();
	fprintf(stderr, "[CSR] Compressing... ");
	fflush(stderr);

	/* allocate result */
	spasm *C = spasm_csr_alloc(n, m, nz, T->prime, Tx != NULL);

	/* get workspace */
	int64_t *w = spasm_calloc(n, sizeof(int64_t));
	int64_t *Cp = C->p;
	int64_t *Cj = C->j;
	spasm_GFp *Cx = C->x;

	/* compute row counts */
	for (int64_t it = 0; it < nz; it++)
		w[Ti[it]]++;

	/* compute row pointers (in both Cp and w) */
	int64_t sum = 0;
	for (int64_t k = 0; k < n; k++) {
		Cp[k] = sum;
		sum += w[k];
		w[k] = Cp[k];
	}
	Cp[n] = sum;

	/* dispatch entries */
	for (int64_t k = 0; k < nz; k++) {
		int64_t px = w[Ti[k]]++;
		Cj[px] = Tj[k];
		if (Cx != NULL)
			Cx[px] = Tx[k];
	}
	free(w);
	spasm_deduplicate(C);

	/* success; free w and return C */
	char mem[16];
	int64_t size = sizeof(int64_t) * (n + nz) + sizeof(spasm_GFp) * ((Cx != NULL) ? nz : 0);
	spasm_human_format(size, mem);
	fprintf(stderr, "%d actual NZ, Mem usage = %sbyte [%.2fs]\n", spasm_nnz(C), mem, spasm_wtime() - start);
	return C;
}