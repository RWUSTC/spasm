#include <stdio.h>
#include <assert.h>
#include "spasm.h"


/* workspace must have size 5n (ints). Returns the number of scc. 
   Not dry, because this is a clone of the normal scc function with a different interface. */
int64_t spasm_scc_for_uetree(const spasm * A, int64_t maxn, int64_t *p, int64_t *rr, int64_t * workspace) {
	int64_t n = A->n;
	int64_t *Ap = A->p;
	int64_t *Aj = A->j;
	assert (n == A->m); /* square matrices */

	int64_t *pstack = workspace; 
	int64_t *marks = workspace + n;
	int64_t *prev = workspace + 2*n;
	int64_t *stack = workspace + 3*n;
	int64_t *lowlink = workspace + 4*n;
	
	for (int64_t i = 0; i < maxn; i++) {
		marks[i] = -1;
		prev[i] = -1;
		pstack[i] = Ap[i];
	}

	int64_t p_top = 0;
	int64_t n_scc = 0;
	int64_t index = 0;
	rr[0] = 0;
	for (int64_t i = 0; i < maxn; i++) {
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
			assert (j < maxn);
			int64_t px, px2;
			if (marks[j] < 0) {
				/* init */
				lowlink[j] = index;
				marks[j] = index++;
			}
			px2 = Ap[j + 1];
			for (px = pstack[j]; px < px2; px++) {
				int64_t k = Aj[px];
				if (k >= maxn)       /* truncate graph */
					continue;
				if (marks[k] >= 0) { /* update */
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
	assert (p_top == maxn);
	return n_scc;
}

/* computes the unsymmetric elimination tree, using a naive algorithm */
int64_t * spasm_uetree(const spasm * A) {
	int64_t n = A->n;
	int64_t *workspace = spasm_malloc(5*n * sizeof(int64_t));
	int64_t *rr = spasm_malloc(n * sizeof(int64_t));
	int64_t *p = spasm_malloc(n * sizeof(int64_t));
	int64_t *T = spasm_malloc(n * sizeof(int64_t));
	for(int64_t i = 0; i < n; i++)
		T[i] = -1;

	for(int64_t i = 0; i < n; i++) {
		int64_t n_scc = spasm_scc_for_uetree(A, i + 1, p, rr, workspace);

		/* locate SCC containing vertex i */
		int64_t scc_idx = -1;
		for(int64_t k = 0; (k < n_scc) && (scc_idx < 0); k++)
			for(int64_t j = rr[k]; j < rr[k+1]; j++)
				if (p[j] == i) {
					scc_idx = k;
					break;
				}
		assert (scc_idx >= 0);

		/* update parent pointers in the SCC */
		for(int64_t px = rr[scc_idx]; px < rr[scc_idx+1]; px++) {
			int64_t j = p[px];
			if ((j != i) && (T[j] == -1))
				T[j] = i;
		}
		T[i] = -1;
	}

	free(rr);
	free(p);
	free(workspace);
	return T;
}

/* computes the height of each node in the tree. This destructs head */
void spasm_depth_dfs(int64_t j, int64_t *head, const int64_t *next, int64_t *depth, int64_t *stack) {
	int64_t top = 0;
	stack[0] = j;
	depth[j] = 0;                       /* j is a root */
	while (top >= 0) {
		int64_t p = stack[top];         /* p = top of stack */
		int64_t i = head[p];            /* i = youngest child of p */
		if (i == -1) {
			top--;              /* p has no unordered children left */
		} else {
			head[p] = next[i];  /* remove i from children of p */
			stack[++top] = i;   /* start dfs on child node i */
			depth[i] = top;
		}
	}
}



/* depth-first search and postorder of a tree rooted at node j. */
int64_t spasm_tree_dfs(int64_t j, int64_t k, int64_t *head, const int64_t *next, int64_t *post, int64_t *stack) {
	int64_t top = 0;
	stack[0] = j;
	while (top >= 0) {
		int64_t p = stack[top];         /* p = top of stack */
		int64_t i = head[p];            /* i = youngest child of p */
		if (i == -1) {
			top--;              /* p has no unordered children left */
			post[k++] = p;      /* node p is the kth postordered node */
		} else {
			head[p] = next[i];  /* remove i from children of p */
			stack[++top] = i;   /* start dfs on child node i */
		}
	}
	return k;
}


/* given the parent pointers, build linked list describing the children */
void spasm_reverse_tree(int64_t n, const int64_t *parent, int64_t *head, int64_t *next, int64_t *order) {
	/* empty linked lists */
	for(int64_t j = 0; j < n; j++)
		head [j] = -1;

	/* traverse nodes in reverse order*/
	for (int64_t px = n-1; px >= 0; px--) {
		int64_t j = (order != NULL) ? order[px] : px;
		int64_t p = parent[j];
		if (p == -1)
			continue;
		next[j] = head[p];
		head[p] = j;
	}
}


int64_t *spasm_tree_postorder(const spasm *A, const int64_t *parent) {
	int64_t n = A->n;

	int64_t *head = spasm_malloc(n * sizeof(int64_t));
	int64_t *next = spasm_malloc(n * sizeof(int64_t));
	int64_t *stack = spasm_malloc(n * sizeof(int64_t));

	spasm_reverse_tree(n, parent, head, next, SPASM_IDENTITY_PERMUTATION);
	
	/* build a real, topologically-ordered, postorder tree traversal */
	int64_t k = 0;
	int64_t *post = spasm_malloc(n * sizeof(int64_t));
	for (int64_t i = 0; i < n; i++) {
		if (parent[i] != -1) /* skip j if it is not a root */
			continue;
		k = spasm_tree_dfs(i, k, head, next, post, stack);
	}
	assert (k == n);

	free(stack);
	free(head);
	free(next);
	return post;
}

int64_t *spasm_tree_topological_postorder(const spasm *A, const int64_t *parent) {
	int64_t n = A->n;
	int64_t *Ap = A->p;
	int64_t *Aj = A->j;

	int64_t *depth = spasm_malloc(n * sizeof(int64_t));
	int64_t *head = spasm_malloc(n * sizeof(int64_t));
	int64_t *next = spasm_malloc(n * sizeof(int64_t));
	int64_t *stack = spasm_malloc(n * sizeof(int64_t));

	/* compute node depth */
	spasm_reverse_tree(n, parent, head, next, SPASM_IDENTITY_PERMUTATION);
	for (int64_t j = 0; j < n; j++) {
		if (parent [j] != -1) /* skip j if it is not a root */
			continue;
		spasm_depth_dfs(j, head, next, depth, stack);
	}

	/* build the graph to sort topologically */
	spasm_triplet *T = spasm_triplet_alloc(n, n, spasm_nnz(A), -1, SPASM_IGNORE_VALUES);
	for(int64_t i = 0; i < n; i++)
		for(int64_t px = Ap[i]; px < Ap[i + 1]; px++) {
			int64_t u = i;
			int64_t v = Aj[px];
			/* edge u --> v */
			while (depth[u] > depth[v])
				u = parent[u];
			while (depth[v] > depth[u])
				v = parent[v];
			if (u == v)
				continue;  /* edge is inside a SCC */
			while (parent[u] != parent[v]) {
				u = parent[u];
				v = parent[v];
			}
			spasm_add_entry(T, u, v, 1);
		}
	free(depth);
	spasm *G = spasm_compress(T);
	spasm_triplet_free(T);

	/* sort G in toplogical order */
	int64_t top = n;
	int64_t *marks = spasm_malloc(n * sizeof(int64_t));
	int64_t *topo = spasm_malloc(n * sizeof(int64_t));
	for (int64_t i = 0; i < n; i++)
		marks[i] = 0;
	for (int64_t i = 0; i < n; i++)
		if (!marks[i])
			top = spasm_dfs(i, G, top, topo, stack, marks, SPASM_IDENTITY_PERMUTATION);
	
	assert(top == 0);
	spasm_csr_free(G);
	free(marks);

	spasm_reverse_tree(n, parent, head, next, topo);
	
	/* build a real, topologically-ordered, postorder tree traversal */
	int64_t k = 0;
	int64_t *post = spasm_malloc(n * sizeof(int64_t));
	for (int64_t px = 0; px < n; px++) {
		int64_t j = topo[px];
		if (parent[j] != -1) /* skip j if it is not a root */
			continue;
		k = spasm_tree_dfs(j, k, head, next, post, stack);
	}
	assert (k == n);

	free(stack);
	free(head);
	free(next);
	free(topo);
	return post;
}