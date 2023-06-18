#include "spasm.h"
/*
 * x = x + beta * A[j], where x is a dense vector and A[j] is sparse
 * 
 * low-level operation for maximum flexibility;
 * 
 * This is where all the heavy lifting should take place.
 */
void spasm_scatter(const int64_t *Aj, const spasm_GFp * Ax, int64_t from, int64_t to, spasm_GFp beta, spasm_GFp * x, int64_t prime) {
	for (int64_t px = from; px < to; px++) {
		int64_t j = Aj[px];
		x[j] = (x[j] + ((beta * Ax[px]))) % prime; /* ultra-naive */ 
	}
}
