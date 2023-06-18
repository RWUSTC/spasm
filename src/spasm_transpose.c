#include "spasm.h"

spasm *spasm_transpose(const spasm * C, int64_t keep_values) {
	int64_t sum, *w;
	spasm *T;

	int64_t m = C->m;
	int64_t n = C->n;
	int64_t *Cp = C->p;
	int64_t *Cj = C->j;
	spasm_GFp *Cx = C->x;

	/* allocate result */
	T = spasm_csr_alloc(m, n, spasm_nnz(C), C->prime, keep_values && (Cx != NULL));
	int64_t *Tp = T->p;
	int64_t *Tj = T->j;
	spasm_GFp *Tx = T->x;

	/* get workspace */
	w = spasm_calloc(m, sizeof(int64_t));

	/* compute column counts */
	for (int64_t i = 0; i < n; i++)
		for (int64_t px = Cp[i]; px < Cp[i + 1]; px++)
			w[Cj[px]]++;

	/* compute column pointers (in both Cp and w) */
	sum = 0;
	for (int64_t j = 0; j < m; j++) {
		Tp[j] = sum;
		sum += w[j];
		w[j] = Tp[j];
	}
	Tp[m] = sum;

	/* dispatch entries */
	for (int64_t i = 0; i < n; i++) {
		for (int64_t px = Cp[i]; px < Cp[i + 1]; px++) {
			int64_t j = Cj[px];
			int64_t py = w[j];
			Tj[py] = i;
			if (Tx != NULL)
				Tx[py] = Cx[px];
			w[j]++;
		}
	}

	free(w);
	return T;
}
