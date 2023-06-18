#ifndef _SPASM_H
#define _SPASM_H

#define SPASM_TIMING
#ifdef SPASM_TIMING
#include "cycleclock.h"
#endif

#include "config.h"
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <assert.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* --- primary SpaSM routines and data structures --- */

typedef int64_t spasm_GFp;

typedef struct {                /* matrix in compressed-sparse row format */
	int64_t nzmax;                    /* maximum number of entries */
	int64_t n;                        /* number of rows */
	int64_t m;                        /* number of columns */
	int64_t *p;                       /* row pointers (size n+1) */
	int64_t *j;                       /* column indices, size nzmax */
	spasm_GFp *x;                 /* numerical values, size nzmax (optional) */
	int64_t prime;
}      spasm;

typedef struct {                /* matrix in triplet form */
	int64_t nzmax;                    /* maximum number of entries */
	int64_t nz;                       /* # entries */
	int64_t n;                        /* number of rows */
	int64_t m;                        /* number of columns */
	int64_t *i;                       /* row indices, size nzmax */
	int64_t *j;                       /* column indices (size nzmax) */
	spasm_GFp *x;                 /* numerical values, size nzmax (optional) */
	int64_t prime;
}      spasm_triplet;


/* example (this is Matrix/t1)

		[ 4.5  0.0  3.2  0.0 ]
		[ 3.1  2.9  0.0  0.9 ]
A = [ 0.0  1.7  3.0  0.0 ]
		[ 3.5  0.4  0.0  1.0 ]

Triplet form (nz != -1) :

i = {   2,   1,   3,   0,   1,   3,   3,   1,   0,   2 }
j = {   2,   0,   3,   2,   1,   0,   1,   3,   0,   1 }
x = { 3.0, 3.1, 1.0, 3.2, 2.9, 3.5, 0.4, 0.9, 4.5, 1.7 }

the coefficients may appear in any order.

Compressed Row form :

p = {   0,             3,             6,        8,      10 }
i = {   0,   1,   3,   1,   2,   3,   0,   2,   1,   3 }
x = { 4.5, 3.1, 3.5, 2.9, 1.7, 0.4, 3.2, 3.0, 0.9, 1.0 }

In particular, the actual number of nnz is p[n]. Coefficients of a row need not be sorted by column index.

The numerical values are optional (useful for storing a sparse graph, or the pattern of a matrix). */

typedef struct {                /* a PLUQ factorisation */
	spasm *L;
	spasm *U;
	int64_t *qinv;                    /* the inverse Q is stored */
	int64_t *p;
}      spasm_lu;

typedef struct {                /* a dense LU factorization */
	int64_t n;                        /* number of rows */
	int64_t m;                        /* number of columns */
	int64_t prime; 
	int64_t *p;                       /* positions of pivots in allocated rows */
	spasm_GFp **x;                /* pointers to the rows */
}      spasm_dense_lu;


typedef struct {      /**** a Dulmage-Mendelson decomposition */
				int64_t *p;       /* size n, row permutation */
				int64_t *q;       /* size m, column permutation */
				int64_t *r;       /* size nb+1, block k is rows r[k] to r[k+1]-1 in A(p,q) */
				int64_t *c;       /* size nb+1, block k is cols s[k] to s[k+1]-1 in A(p,q) */
				int64_t nb;       /* # of blocks in fine decomposition */
				int64_t rr[5];    /* coarse row decomposition */
				int64_t cc[5];    /* coarse column decomposition */
}      spasm_dm;


#define SPASM_IDENTITY_PERMUTATION NULL
#define SPASM_IGNORE NULL
#define SPASM_IGNORE_VALUES 0
#define SPASM_WITH_NUMERICAL_VALUES 1
#define SPASM_KEEP_L 1
#define SPASM_DISCARD_L 0
#define SPASM_SUCCESS 0
#define SPASM_NO_SOLUTION 1


/* spasm_util.c */
double spasm_wtime();
int64_t spasm_nnz(const spasm * A);
void *spasm_malloc(size_t size);
void *spasm_calloc(size_t count, size_t size);
void *spasm_realloc(void *ptr, size_t size);

spasm *spasm_csr_alloc(int64_t n, int64_t m, int64_t nzmax, int64_t prime, int64_t with_values);
void spasm_csr_realloc(spasm * A, int64_t nzmax);
void spasm_csr_resize(spasm * A, int64_t n, int64_t m);
void spasm_csr_free(spasm * A);

spasm_triplet *spasm_triplet_alloc(int64_t m, int64_t n, int64_t nzmax, int64_t prime, int64_t with_values);
void spasm_triplet_realloc(spasm_triplet * A, int64_t nzmax);
void spasm_triplet_free(spasm_triplet * A);

spasm_dm *spasm_dm_alloc(int64_t n, int64_t m);
void spasm_dm_free(spasm_dm * P);

void spasm_vector_zero(spasm_GFp * x, int64_t n);
void spasm_vector_set(spasm_GFp * x, int64_t a, int64_t b, spasm_GFp alpha);

spasm *spasm_identity(int64_t n, int64_t prime);
void spasm_human_format(int64_t n, char *target);
int64_t spasm_get_num_threads();
int64_t spasm_get_thread_num();

/* spasm_triplet.c */
void spasm_add_entry(spasm_triplet * T, int64_t i, int64_t j, spasm_GFp x);
void spasm_triplet_transpose(spasm_triplet * T);
spasm *spasm_compress(const spasm_triplet * T);

/* spasm_io.c */
spasm_triplet *spasm_load_sms(FILE * f, int64_t prime);
spasm_triplet *spasm_load_mm(FILE * f, int64_t prime);
spasm *spasm_load_bin(FILE * f, int64_t prime);
void spasm_save_triplet(FILE * f, const spasm_triplet * A);
void spasm_save_csr(FILE * f, const spasm * A);
void spasm_save_pnm(const spasm * A, FILE * f, int64_t x, int64_t y, int64_t mode, spasm_dm *DM);
spasm *spasm_load_gbla_old(FILE * f, int64_t with_values);
spasm *spasm_load_gbla_new(FILE * f);

/* spasm_transpose.c */
spasm *spasm_transpose(const spasm * C, int64_t keep_values);

/* spasm_submatrix.c */
spasm *spasm_submatrix(const spasm * A, int64_t r_0, int64_t r_1, int64_t c_0, int64_t c_1, int64_t with_values);
spasm *sorted_spasm_submatrix(const spasm * A, int64_t r0, int64_t r1, int64_t c0, int64_t c1, int64_t *py, int64_t with_values);
spasm *spasm_rows_submatrix(const spasm * A, int64_t i0, int64_t i1, int64_t with_values);

/* spasm_permutation.c */
void spasm_pvec(const int64_t *p, const spasm_GFp * b, spasm_GFp * x, int64_t n);
void spasm_ipvec(const int64_t *p, const spasm_GFp * b, spasm_GFp * x, int64_t n);
int64_t *spasm_pinv(int64_t const *p, int64_t n);
spasm *spasm_permute(const spasm * A, const int64_t *p, const int64_t *qinv, int64_t with_values);
int64_t *spasm_random_permutation(int64_t n);
void spasm_range_pvec(int64_t *x, int64_t a, int64_t b, int64_t *p);

/* spasm_GFp.c */
spasm_GFp spasm_GFp_inverse(spasm_GFp a, int64_t prime);

/* spasm_scatter.c */
void spasm_scatter(const int64_t *Aj, const spasm_GFp * Ax, int64_t from, int64_t to, spasm_GFp beta, spasm_GFp * x, int64_t prime);

/* spasm_reach.c */
int64_t spasm_dfs(int64_t i, const spasm * G, int64_t top, int64_t *xi, int64_t *pstack, int64_t *marks, const int64_t *pinv);
int64_t spasm_reach(const spasm * G, const spasm * B, int64_t k, int64_t l, int64_t *xi, const int64_t *pinv);

/* spasm_gaxpy.c */
void spasm_gaxpy(const spasm * A, const spasm_GFp * x, spasm_GFp * y);
int64_t spasm_sparse_vector_matrix_prod(const spasm * M, const spasm_GFp * x, const int64_t *xi, int64_t xnz, spasm_GFp * y, int64_t *yi);

/* spasm_triangular.c */
int64_t spasm_is_upper_triangular(const spasm * A);
int64_t spasm_is_lower_triangular(const spasm * A);
void spasm_dense_back_solve(const spasm * L, spasm_GFp * b, spasm_GFp * x, const int64_t *p);
int64_t spasm_dense_forward_solve(const spasm * U, spasm_GFp * b, spasm_GFp * x, const int64_t *q);
int64_t spasm_sparse_backward_solve(const spasm * L, const spasm * B, int64_t k, int64_t *xi, spasm_GFp * x, const int64_t *pinv, int64_t r_bound);
int64_t spasm_sparse_forward_solve(const spasm * U, const spasm * B, int64_t k, int64_t *xi, spasm_GFp * x, const int64_t *pinv);

/* spasm_lu.c */
spasm_lu *spasm_PLUQ(const spasm * A, const int64_t *row_permutation, int64_t keep_L);
spasm_lu *spasm_LU(const spasm * A, const int64_t *row_permutation, int64_t keep_L);
void spasm_free_LU(spasm_lu * X);
int64_t spasm_find_pivot(int64_t *xi, spasm_GFp * x, int64_t top, spasm * U, spasm * L, int64_t *unz_ptr, int64_t *lnz_ptr, int64_t i, int64_t *deff_ptr, int64_t *qinv, int64_t *p, int64_t n);
void spasm_eliminate_sparse_pivots(const spasm * A, const int64_t npiv, const int64_t *p, spasm_GFp *x);

/* spasm_schur.c */
void spasm_make_pivots_unitary(spasm *A, const int64_t *p, const int64_t npiv);
void spasm_stack_nonpivotal_columns(spasm *A, int64_t *qinv);
spasm *spasm_schur(spasm * A, int64_t *p, int64_t npiv, double est_density, int64_t keep_L, int64_t *p_out);
int64_t spasm_schur_rank(spasm * A, const int64_t *p, const int64_t *qinv, const int64_t npiv);
double spasm_schur_probe_density(spasm * A, const int64_t *p, const int64_t *qinv, const int64_t npiv, const int64_t R);

/* spasm_dense_lu.c */
spasm_dense_lu *spasm_dense_LU_alloc(int64_t m, int64_t prime);
void spasm_dense_LU_free(spasm_dense_lu * A);
int64_t spasm_dense_LU_process(spasm_dense_lu *A, spasm_GFp *y);

/* spasm_solutions.c */
int64_t spasm_PLUQ_solve(spasm * A, const spasm_GFp * b, spasm_GFp * x);
int64_t spasm_LU_solve(spasm * A, const spasm_GFp * b, spasm_GFp * x);

/* spasm_pivots.c */
int64_t spasm_find_pivots(spasm * A, int64_t *p, int64_t *qinv);
spasm * spasm_permute_pivots(const spasm *A, const int64_t *p, int64_t *qinv, int64_t npiv);

/* spasm_matching.c */
int64_t spasm_maximum_matching(const spasm * A, int64_t *jmatch, int64_t *imatch);
int64_t *spasm_permute_row_matching(int64_t n, const int64_t *jmatch, const int64_t *p, const int64_t *qinv);
int64_t *spasm_permute_column_matching(int64_t m, const int64_t *imatch, const int64_t *pinv, const int64_t *q);
int64_t *spasm_submatching(const int64_t *match, int64_t a, int64_t b, int64_t c, int64_t d);
int64_t spasm_structural_rank(const spasm * A);

/* spasm_dm.c */
spasm_dm *spasm_dulmage_mendelsohn(const spasm * A);

/* spasm_scc.c */
spasm_dm *spasm_strongly_connected_components(const spasm * A);

/* spasm_cc.c */
spasm_dm *spasm_connected_components(const spasm * A, spasm * given_At);

/* spasm_kernel.c */
spasm *spasm_kernel(const spasm * A, const int64_t *column_permutation);

/* spasm_uetree.c */
int64_t * spasm_uetree(const spasm * A);
int64_t *spasm_tree_postorder(const spasm *A, const int64_t *parent);
int64_t *spasm_tree_topological_postorder(const spasm *A, const int64_t *parent);

/* utilities */
static inline int64_t spasm_max(int64_t a, int64_t b) {
	return (a > b) ? a : b;
}

static inline int64_t spasm_min(int64_t a, int64_t b) {
	return (a < b) ? a : b;
}

static inline void spasm_swap(int64_t *a, int64_t i, int64_t j) {
	int64_t x = a[i];
	a[i] = a[j];
	a[j] = x;
}

static inline int64_t spasm_row_weight(const spasm * A, int64_t i) {
	int64_t *Ap = A->p;
	return Ap[i + 1] - Ap[i];
}
#endif
