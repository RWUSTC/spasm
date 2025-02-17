#include <assert.h>
#include "spasm.h"

/*
 * R0 : unmatched rows
 * C0 : unmatched columns
 * R1 : rows reachable from C0 (by alternating paths)
 * C1 : column matched to R1
 * C3 : columns reachable from R0 (by alternating paths)
 * R3 : rows matched to C3
 * R2 : other rows
 * C2 : other columns
 */


/*
 * breadth-first search for coarse decomposition. This determines R0,R3,C3
 * (or C0,C1,R1 when given the transpose of A).
 */
static void bfs(const spasm * A, int64_t *wi, int64_t *wj, int64_t *queue, const int64_t *imatch, const int64_t *jmatch, int64_t mark) {
	int64_t *Ap = A->p;
	int64_t *Aj = A->j;
	int64_t n = A->n;
	int64_t head = 0;
	int64_t tail = 0;

	/* enqueue all unmatched nodes, set mark 0 to put them in R0/C0 */
	for (int64_t i = 0; i < n; i++) {
		if (jmatch[i] >= 0)
			continue;
		wi[i] = 0;
		queue[tail++] = i;
	}

	while (head < tail) {
		int64_t i = queue[head++];

		/* mark adjacent nodes. This puts column in C3 (or rows in R1
		 * when transposed). If matched, mark and enqueue matched node.
		 * This puts rows in R3 (or columns in C1 when transposed). 
		 */
		for (int64_t px = Ap[i]; px < Ap[i + 1]; px++) {
			int64_t j = Aj[px];
			if (wj[j] >= 0)
				continue;
			wj[j] = mark;
			int64_t I = imatch[j];
			if (wi[I] >= 0)
				continue;
			wi[I] = mark;
			queue[tail++] = I;
		}
	}
}


/* collect unmatched rows into the permutation vector p */
static void collect_unmatched(int64_t n, const int64_t *wi, int64_t *p, int64_t *rr, int64_t set) {
	int64_t kr = rr[set];
	for (int64_t i = 0; i < n; i++)
		if (wi[i] == 0)
			p[kr++] = i;
	rr[set + 1] = kr;
}



/* collect matched rows and columns into p and q */
static void collect_matched(int64_t n, const int64_t *wj, const int64_t *imatch, int64_t *p, int64_t *q, int64_t *cc, int64_t *rr, int64_t set, int64_t mark) {
	int64_t kc = cc[set];
	int64_t kr = rr[set - 1];
	for (int64_t j = 0; j < n; j++) {
		if (wj[j] != mark)
			continue;	/* skip if j is not in C set */
		p[kr++] = imatch[j];
		q[kc++] = j;
	}
	cc[set + 1] = kc;
	rr[set] = kr;
}


spasm_dm *spasm_dulmage_mendelsohn(const spasm * A) {
	int64_t n = A->n;
	int64_t m = A->m;

	/* --- Maximum matching ----------------------------------------- */
	int64_t *jmatch = spasm_malloc(n * sizeof(int64_t));
	int64_t *imatch = spasm_malloc(m * sizeof(int64_t));

	spasm * A_t = spasm_transpose(A, SPASM_IGNORE_VALUES);

	if (n < m) {
		spasm_maximum_matching(A, jmatch, imatch);
	} else {
		spasm_maximum_matching(A_t, imatch, jmatch);
	}
	
	/* --- coarse DM decomposition ---------------------------------- */
	spasm_dm *DM = spasm_dm_alloc(n, m);
	int64_t *p = DM->p;
	int64_t *q = DM->q;
	int64_t *r = DM->r;
	int64_t *c = DM->c;
	int64_t *rr = DM->rr;
	int64_t *cc = DM->cc;

	/* we use p,q,r and c as workspace for the bfs. */
	int64_t * wi = r;
	int64_t * wj = c;

	/* unmark everything for bfs */
	spasm_vector_set(wj, 0, m, -1);
	spasm_vector_set(wi, 0, n, -1);

	/* find R0, then R3, C3 */
	bfs(A, wi, wj, p, imatch, jmatch, 3);

	/* find C0, then R1, C1 */
	bfs(A_t, wj, wi, q, jmatch, imatch, 1);

	spasm_csr_free(A_t);

	/* collect in q: C0, C1, C2, C3
	 *            p:     R1, R2, R3, R0 */
	collect_unmatched(m, wj, q, cc, 0);
	collect_matched(m, wj, imatch, p, q, cc, rr, 1, 1);
	collect_matched(m, wj, imatch, p, q, cc, rr, 2, -1);
	collect_matched(m, wj, imatch, p, q, cc, rr, 3, 3);
	collect_unmatched(n, wi, p, rr, 3);

	/* --- fine DM decomposition ---------------------------------- */

	if (rr[2] - rr[1] == 0)
		return DM;  /* S is empty: no need to find its SCC */

	/* extract S */
	int64_t *qinv = spasm_pinv(q, m);
	spasm *B = spasm_permute(A, p, qinv, SPASM_IGNORE_VALUES);
	spasm *C = spasm_submatrix(B, rr[1], rr[2], cc[2], cc[3], SPASM_IGNORE_VALUES);
	spasm_csr_free(B);

	spasm_dm *SCC = spasm_strongly_connected_components(C);
	int64_t n_scc = SCC->nb;
	int64_t *scc_r = SCC->r;
	int64_t *scc_c = SCC->c;
	
	/* update permutations */
	spasm_range_pvec(p, rr[1], rr[2], SCC->p);
	spasm_range_pvec(q, cc[2], cc[3], SCC->q);
	
	/* update fine decomp */
	r[0] = 0;
	for(int64_t i = 0; i <= n_scc; i++)
		r[i + 1] = rr[1] + scc_r[i];
	r[n_scc + 2] = n;

	c[0] = 0;
	for(int64_t i = 0; i <= n_scc; i++)
		c[i + 1] = cc[2] + scc_c[i];
	c[n_scc + 2] = m;
	DM->nb = n_scc + 2;

	spasm_dm_free(SCC);
	spasm_csr_free(C);
	return DM;
}
