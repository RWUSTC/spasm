#include <assert.h>
#include "spasm.h"


/*
 * returns the non-trivial strongly connected component of A (which must be
 * square), seen as an directed graph.
 */
/* r must have size n+1 */

/* strategie : quand un noeud est retiré dans une SCC, mettre son index ou son lowlink à n+1 */

spasm_dm *spasm_strongly_connected_components(const spasm * A) {
	int64_t n = A->n;
	int64_t m = A->m;
	int64_t *Ap = A->p;
	int64_t *Aj = A->j;
	assert(n == m);

	spasm_dm *P = spasm_dm_alloc(n, n);
	int64_t p_top = 0;
	int64_t *p = P->p;
	int64_t *rr = P->r;

	int64_t *pstack = spasm_malloc(n * sizeof(int64_t));
	int64_t *marks = spasm_malloc(n * sizeof(int64_t));
	int64_t *prev = spasm_malloc(n * sizeof(int64_t));
	int64_t *stack = spasm_malloc(n * sizeof(int64_t));
	int64_t *lowlink = spasm_malloc(n * sizeof(int64_t));
	
	/* first pass */
	for (int64_t i = 0; i < n; i++) {
		marks[i] = -1;
		prev[i] = -1;
		pstack[i] = Ap[i];
	}

	int64_t n_scc = 0;
	int64_t index = 0;
	rr[n_scc] = 0;
	for (int64_t i = 0; i < n; i++) {
		int64_t head, top;
		if (marks[i] >= 0)
			continue;
		
		/* DFS */
		head = 0;
		top = 0;
		stack[top] = i;
		int64_t j = i;
		while (j >= 0) {
			/* get j from the top of the recursion stack */
			int64_t px, px2;
			if (marks[j] < 0) {
				/* init */
				lowlink[j] = index;
				marks[j] = index++;
			}
			px2 = Ap[j + 1];
			for (px = pstack[j]; px < px2; px++) {
				int64_t k = Aj[px];

				if (marks[k] >= 0) {
					/* update */
					lowlink[j] = spasm_min(lowlink[j], lowlink[k]);
					continue;
				}
				/* push */
				pstack[j] = px + 1;
				stack[++top] = k;
				prev[k] = j;
				j = k;
				break;
			}
			if (px == px2) {
				/* check if we have the root of a SCC */
				if (lowlink[j] == marks[j]) {
					while (stack[top] != j) {
						int64_t k = stack[top--];
						p[p_top++] = k;
						lowlink[k] = n;
					}
					p[p_top++] = j;
					lowlink[j] = n;
					top--;

					rr[++n_scc] = p_top;
				}

				/* pop */
				int64_t k = j;
				j = prev[j];
				if (j >= 0)
					lowlink[j] = spasm_min(lowlink[j], lowlink[k]);
			}
		}
	}
	assert (p_top == n);

	/* at this point, blocks are in reverse order, and inside blocks, nodes are in reverse order */
	int64_t *q = P->q;
	int64_t *cc = P->c;
	for (int64_t i = 0; i < n; i++)
		q[i] = p[n - 1 - i];

	for (int64_t i = 0; i < n; i++)
		p[i] = q[i];

	for (int64_t i = 0; i <= n_scc; i++)
		cc[i] = n - rr[n_scc - i];

	for (int64_t i = 0; i <= n_scc; i++)
		rr[i] = cc[i];


	P->nb = n_scc;

	free(stack);
	free(pstack);
	free(marks);
	free(lowlink);
	return P;
}
