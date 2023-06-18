#include <assert.h>
#include "spasm.h"

void spasm_augment_matching(int64_t head, int64_t *istack, int64_t *jstack, int64_t *p, int64_t *qinv) {
	for (int64_t px = head; px >= 0; px--) {
		int64_t i = istack[px];
		int64_t j = jstack[px];
		qinv[j] = i;
		p[i] = j;
	}
}

/* lookahead: search for unmatched column in A[i,:]. If found, it completes the
 * alternating path in istack/jstack, so we augment the matching. */
int64_t spasm_lookahead(const spasm *A, int64_t i, int64_t head, int64_t *plookahead, int64_t *istack, int64_t *jstack, int64_t *p, int64_t *qinv) {
	int64_t *Ap = A->p;
	int64_t *Aj = A->j;
			
	for (int64_t px = plookahead[i]; px < Ap[i + 1]; px++) {
		int64_t j = Aj[px];
		if (qinv[j] < 0) {
			plookahead[i] = px + 1;
			jstack[head] = j;
			spasm_augment_matching(head, istack, jstack, p, qinv);
			return 1;
		}	
	}
	/* all column on A[i,:] are matched. start the DFS */
	plookahead[i] = Ap[i + 1];
	return 0;
}

/**
 * Search an augmenting path w.r.t. the matching, starting from row k (i.e. a
 * path from an unmatched row to an unmatched column).
 *
 * This does a DFS starting from row k, and memorizes the path in (row_stack / 
 * col_stack). When looking for an unmatched column reachable from a row, the 
 * adjacent columns are examined first. This "lookeahead" amounts to do one step
 * of BFS inside the DFS.
 *
 * Because the matching increases monotonically (when row/column is matched, it
 * stays matched), is it useless to re-examine matched columns.
 */
int64_t spasm_augmenting_path(const spasm * A, int64_t k, int64_t *istack, int64_t *jstack, int64_t *pstack, int64_t *marks, int64_t *plookahead, int64_t *p, int64_t *qinv) {
	int64_t head, px;
	int64_t *Ap, *Aj;

	Ap = A->p;
	Aj = A->j;

	/* initialize the DFS */
	head = 0;
	istack[head] = k;

	/* stack empty ? */
	while (head >= 0) {
		/* search an unmatched column reachable from row i */
		int64_t i = istack[head];

		if (marks[i] != k) {
			marks[i] = k;
			if (spasm_lookahead(A, i, head, plookahead, istack, jstack, p, qinv))
				return 1;
			/* nothing on row i: we have to start the DFS */
			pstack[head] = Ap[i];
		}
		
		/* Depth-first-search of columns adjacent to row i */
		for (px = pstack[head]; px < Ap[i + 1]; px++) {
			int64_t j = Aj[px];
			int64_t inew = qinv[j];
			if (marks[inew] == k)
				continue;
			/* pause DFS of row i, start DFS of row inew. */
			pstack[head] = px + 1;
			jstack[head] = j;
			istack[++head] = inew;
			break;
		}
		/* row i is done: pop it from stack */
		if (px == Ap[i + 1])
			head--;
	}
	return 0;
}

/** 
 * Computes a maximum matching using the Ford–Fulkerson algorithm.
 *
 * If the matrix is rectangular, it is a big advantage to transpose it so that n << m.
 * 
 * @param qinv[j] = row matched to column j (or -1) 
 *
 * @param p[i] = column matched to row i (or -1)
 * 
 * @return size of the matching
 */
int64_t spasm_maximum_matching(const spasm * A, int64_t *p, int64_t *qinv) {
	int64_t n, m, r, k;
	int64_t *Ap, *istack, *jstack, *marks, *pstack, *plookahead;

	n = A->n;
	m = A->m;
	r = spasm_min(n, m); /* the matching cant' be bigger than this */
	Ap = A->p;

	/* get workspace */
	istack = spasm_malloc(n * sizeof(int64_t));
	jstack = spasm_malloc(n * sizeof(int64_t));
	pstack = spasm_malloc(n * sizeof(int64_t));
	marks  = spasm_malloc(n * sizeof(int64_t));
	plookahead = spasm_malloc(n * sizeof(int64_t));

	spasm_vector_set(qinv, 0, m, -1);
	spasm_vector_set(p, 0, n, -1);
	spasm_vector_set(marks, 0, n, -1);
	for (int64_t i = 0; i < n; i++)
		plookahead[i] = Ap[i];

	k = 0;
	double start = spasm_wtime();
	for (int64_t i = 0; (i < n) && (k < r); i++) {
		if (p[i] < 0)
			k += spasm_augmenting_path(A, i, istack, jstack, pstack, marks, plookahead, p, qinv);
		fprintf(stderr, "\r[matching] %d / %d, size %d", i, n, k);
		fflush(stderr);
	}
	fprintf(stderr, " [%.1f s]\n", spasm_wtime() - start);

	free(istack);
	free(jstack);
	free(pstack);
	free(marks);
	free(plookahead);
	return k;
}

/*
 * given a row-matching of A, returns a row_matching of P*A*Q --- the result
 * of spasm_permute(A, p, q).
 */
int64_t *spasm_permute_row_matching(int64_t n, const int64_t *jmatch, const int64_t *p, const int64_t *qinv) {
	int64_t *jjmatch;
	int64_t i;

	jjmatch = spasm_malloc(n * sizeof(int64_t));
	for (i = 0; i < n; i++) {
		if (jmatch[p[i]] == -1) {
			jjmatch[i] = -1;
		} else {
			jjmatch[i] = qinv[jmatch[p[i]]];
		}
	}
	return jjmatch;
}

int64_t *spasm_permute_column_matching(int64_t m, const int64_t *imatch, const int64_t *pinv, const int64_t *q) {
	int64_t *iimatch;
	int64_t j;

	iimatch = spasm_malloc(m * sizeof(int64_t));
	for (j = 0; j < m; j++) {
		if (imatch[q[j]] == -1) {
			iimatch[j] = -1;
		} else {
			iimatch[j] = pinv[imatch[q[j]]];
		}
	}
	return iimatch;
}


/*
 * returns (a copy of) the matching match restricted to the submatrix M[a:b,
 * c:d]
 */
int64_t *spasm_submatching(const int64_t *match, int64_t a, int64_t b, int64_t c, int64_t d) {
	int64_t *pmatch;
	int64_t i;

	pmatch = spasm_malloc((b - a) * sizeof(int64_t));
	for (i = a; i < b; i++) {
		if (match[i] == -1) {
			pmatch[i - a] = -1;
		} else {
			pmatch[i - a] = match[i] - c;
			assert(pmatch[i - a] >= 0);
			assert(pmatch[i - a] < d);
		}
	}
	return pmatch;
}