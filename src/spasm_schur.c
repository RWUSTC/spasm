#include <assert.h>
#include "spasm.h"

/* make pivotal rows of A unitary */
void spasm_make_pivots_unitary(spasm * A, const int64_t *p, const int64_t npiv) {
	int64_t prime = A->prime;
	int64_t *Ap = A->p;
	spasm_GFp *Ax = A->x;

#pragma omp parallel for
	for (int64_t i = 0; i < npiv; i++) {
		int64_t inew = p ? p[i] : i;
		spasm_GFp diag = Ax[Ap[inew]];
		if (diag == 1)
			continue;

		spasm_GFp alpha = spasm_GFp_inverse(diag, prime);
		for (int64_t px = Ap[inew]; px < Ap[inew + 1]; px++)
			Ax[px] = (alpha * Ax[px]) % prime;
	}
}

/* keep only non-pivotal columns and renumber them starting from 0 */
void spasm_stack_nonpivotal_columns(spasm *A, int64_t *qinv)
{
	int64_t n = A->n;
	int64_t m = A->m;
	int64_t *Ap = A->p;
	int64_t *Aj = A->j;
	int64_t *q = spasm_malloc(m * sizeof(int64_t));
	int64_t k = 0;
	for (int64_t j = 0; j < m; j++)
		q[j] = (qinv[j] < 0) ? k++ : -1;

	#pragma omp parallel for
	for (int64_t px = 0; px < Ap[n]; px++) {
		Aj[px] = q[Aj[px]];
		assert(Aj[px] >= 0);
	}
	A->m = k;
}


/*
 * Computes the Schur complement, by eliminating the pivots located on rows
 * p[0] ... p[n_pivots-1] of input matrix A. 
 *
 * non-pivotal rows are p[to] ... p[n]
 *
 * The pivots must be the first entries on the lines.
 *
 * The pivots must be unitary.	
 *
 * This returns a sparse representation of S. 
 *
 * If the estimated density is unknown, set it to -1: it will be evaluated
 */
spasm *spasm_schur(spasm * A, int64_t *p, int64_t npiv, double est_density, int64_t keep_L, int64_t *p_out)
{
	const int64_t n = A->n;
	const int64_t m = A->m;
	const int64_t Sm = m;
	const int64_t Sn = n - npiv;
	const int64_t verbose_step = spasm_max(1, n / 1000);
	
	assert(!keep_L); /* option presently unsupported */

	/* initialize qinv */
	int64_t *Aj = A->j;
	int64_t *Ap = A->p;
	int64_t *qinv = spasm_malloc(m * sizeof(int64_t));
	for (int64_t j = 0; j < m; j++)
		qinv[j] = -1;
	for (int64_t k = 0; k < npiv; k++) {
		int64_t i = p[k];
		int64_t j = Aj[Ap[i]];
		qinv[j] = i;
	}

	if (est_density < 0)
		est_density = spasm_schur_probe_density(A, p, qinv, npiv, 100);

	spasm *S = spasm_csr_alloc(Sn, Sm, (est_density*Sn)*Sm, A->prime, SPASM_WITH_NUMERICAL_VALUES);
	int64_t *Sp = S->p;
	int64_t *Sj = S->j;
	spasm_GFp *Sx = S->x;
	int64_t snz = 0;
	int64_t k = 0;
	int64_t writing = 0;
	double start = spasm_wtime();

#pragma omp parallel
	{
		spasm_GFp *x = spasm_malloc(m * sizeof(spasm_GFp));
		int64_t *xj = spasm_malloc(3 * m * sizeof(int64_t));
		spasm_vector_zero(xj, 3 * m);
		int64_t tid = spasm_get_thread_num();
		int64_t row_snz, row_k, row_px;

		#pragma omp for schedule(dynamic, verbose_step)
		for (int64_t i = npiv; i < n; i++) {
			const int64_t inew = p[i];
			const int64_t top = spasm_sparse_forward_solve(A, A, inew, xj, x, qinv);

			row_snz = 0;
			for (int64_t px = top; px < m; px++) {
				const int64_t j = xj[px];
				if ((keep_L || (qinv[j] < 0)) && (x[j] != 0))
					row_snz++;
			}

			#pragma omp critical(schur_complement)
			{
				/* enough room in S? */
				if (snz + row_snz > S->nzmax) {
					/* wait until other threads stop writing int64_to it */
					#pragma omp flush(writing)
					while (writing > 0) {
						#pragma omp flush(writing)
					}
					spasm_csr_realloc(S, 2 * S->nzmax + Sm);
					Sj = S->j;
					Sx = S->x;
				}
				/* save row k */
				row_k = k++;
				row_px = snz;
				snz += row_snz;
				#pragma omp atomic update
				writing++;
			}
			if (p_out)
				p_out[row_k] = inew;
			
			/* write the new row in S */
			Sp[row_k] = row_px;
			for (int64_t px = top; px < m; px++) {
				const int64_t j = xj[px];
				if ((keep_L || (qinv[j] < 0)) && (x[j] != 0)) {
					Sj[row_px] = j;
					Sx[row_px++] = x[j];
				}
			}

			#pragma omp atomic update
			writing--;

			if (tid == 0 && (i % verbose_step) == 0) {
				double density =  1.0 * snz / (1.0 * Sm * k);
				fprintf(stderr, "\rSchur complement: %d/%d [%d NNZ / density= %.3f]", k, Sn, snz, density);
				fflush(stderr);
			}
		}
		free(x);
		free(xj);
	}
	/* finalize S */
	Sp[Sn] = snz;
	spasm_csr_realloc(S, -1);
	double density = 1.0 * snz / (1.0 * Sm * Sn);
	fprintf(stderr, "\rSchur complement: %d * %d [%d NNZ / density= %.3f], %.1fs\n", Sn, Sm, snz, density, spasm_wtime() - start);
	
	free(qinv);
	return S;
}


/** Samples R rows at random in the schur complement of A w.r.t. the pivots in p[0:n_pivots],
* and return the number that are non-zero (these rows of A are linearly independent from the pivots).
* The pivots must be unitary.
*/
double spasm_schur_probe_density(spasm * A, const int64_t *p, const int64_t *qinv, const int64_t npiv, const int64_t R) {
	int64_t nnz = 0;
	const int64_t m = A->m;
	const int64_t n = A->n;

	if (m == npiv || n == npiv)
		return 0.0;

#pragma omp parallel 
	{
		spasm_GFp *x = spasm_malloc(m * sizeof(spasm_GFp));
		int64_t *xj = spasm_malloc(3 * m * sizeof(int64_t));
		spasm_vector_zero(xj, 3 * m);

#pragma omp for reduction(+:nnz) schedule(dynamic)
		for (int64_t i = 0; i < R; i++) {
			/* pick a random row in S, check if non-zero */
			int64_t inew = p[npiv + (rand() % (n - npiv))];
			int64_t top = spasm_sparse_forward_solve(A, A, inew, xj, x, qinv);
			for (int64_t px = top; px < m; px++) {
				int64_t j = xj[px];
				if (qinv[j] < 0 && x[j] != 0)
					nnz++;
			}
		}
		free(x);
		free(xj);
	}
	return ((double) nnz) / (m - npiv) / R;
}

/*
 * computes the rank of the schur complement, but not the schur complement
 * itself. The pivots must be unitary.
 */
int64_t spasm_schur_rank(spasm * A, const int64_t *p, const int64_t *qinv, const int64_t npiv) {
	int64_t Sm, m, n, k, r, prime, step, threads, searched, prev_r;
	int64_t *q, *Ap, *Aj;
	double start;
	spasm_GFp *Ax;

	n = A->n;
	m = A->m;
	Ap = A->p;
	Aj = A->j;
	Ax = A->x;
	prime = A->prime;

	/* Get Workspace */
	Sm = m - npiv;
	q = spasm_malloc(Sm * sizeof(int64_t));

	/* q sends columns of S to non-pivotal columns of A */
	k = 0;
	for (int64_t j = 0; j < m; j++)
		if (qinv[j] < 0)
			q[k++] = j;

	spasm_dense_lu *U = spasm_dense_LU_alloc(Sm, A->prime);

	/* ---- compute Schur complement ----- */
	fprintf(stderr, "rank of dense schur complement...\n");

	start = spasm_wtime();
	r = 0;
	step = 1;
	k = 0;
	searched = 0;
	prev_r = 0;
	threads = spasm_get_num_threads();

#pragma omp parallel
	{
		spasm_GFp *x = spasm_malloc(m * sizeof(spasm_GFp));
		spasm_GFp *y = spasm_malloc(Sm * sizeof(spasm_GFp));
		int64_t gain;

		while (step <= (1 << 16)) {	/* <--- tweak-me */
			double it_start = spasm_wtime();
			prev_r = r;

			/* random linear combination */
			spasm_vector_zero(x, m);
			for (int64_t i = 0; i < step; i++) {
				int64_t inew = p[npiv + (rand() % (n - npiv))];
				spasm_scatter(Aj, Ax, Ap[inew], Ap[inew + 1], 1 + (rand() % (prime - 1)), x, prime);
			}
			spasm_eliminate_sparse_pivots(A, npiv, p, x);
			for (int64_t j = 0; j < Sm; j++)	/* gather int64_to y */
				y[j] = x[q[j]];

#pragma omp atomic update
			r += spasm_dense_LU_process(U, y);

			/* this is a barrier */
#pragma omp single
			{
				fprintf(stderr, "\rSchur rank: %d [%.1fs] -- current rank = %d / step = %d", k, spasm_wtime() - it_start, r, step);
				fflush(stderr);

				k++;
				searched += threads * step;
				gain = r - prev_r;

				if (gain < threads)
					step *= 2;
				else
					step = spasm_max(1, step / 2);
			}
		}

#pragma omp single
		{
			int64_t final_bad = 0;
			k = 0;
			fprintf(stderr, "\n");

			while (final_bad < 3) {
				double it_start = spasm_wtime();
				for (int64_t i = npiv; i < n; i++) {
					int64_t inew = p[i];
					spasm_scatter(Aj, Ax, Ap[inew], Ap[inew + 1], rand() % prime, x, prime);
				}
				spasm_eliminate_sparse_pivots(A, npiv, p, x);
				for (int64_t j = 0; j < Sm; j++)
					y[j] = x[q[j]];
				int64_t new = spasm_dense_LU_process(U, y);
				r += new;
				final_bad += 1 - new;
				k++;
				fprintf(stderr, "\rSchur rank: %d [%.1fs] -- current rank = %d / final", k, spasm_wtime() - it_start, r);
				fflush(stderr);
			}
		}
		free(x);
		free(y);
	}
	fprintf(stderr, "\n[schur/rank] Time: %.1fs\n", spasm_wtime() - start);

	free(q);
	spasm_dense_LU_free(U);
	return r;
}
