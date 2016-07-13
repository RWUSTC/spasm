/* indent -nfbs -i2 -nip -npsl -di0 -nut spasm_sort.c */
#include <assert.h>
#include "spasm.h"

#define INSERT_SORT_THRESHOLD 42/* TODO : tune this value */


/* sort up to index right, excluded */
static void insertion_sort(const spasm * A, int *p, const int left, const int right) {
  int i, j, u, v;

  /* if (left <= 0) { */
  for (i = left + 1; i < right; i++) {
    u = p[i];
    v = spasm_row_weight(A, p[i]);
    j = i - 1;
    while (j >= 0 && spasm_row_weight(A, p[j]) > v) {
      p[j + 1] = p[j];
      j--;
    }
    p[j + 1] = u;
  }
  /*
   * } else { TODO, possible optimization : if ( left>0 ), then we know that
   * on the left of the current subfile, there is an element smaller than all
   * the elements of the subfile (because this element was a pivot).
   * Therefore, we don't have to check explicitly that we attained the left
   * boundary of the array...
   */
}


/* standard median-of-three pivoting strategy. Returns the pivot index */
static int choose_pivot(const spasm * A, int *p, const int left, const int right) {
  int mid = (left + right) / 2;

  if (spasm_row_weight(A, p[mid - 1]) > spasm_row_weight(A, p[mid])) {
    spasm_swap(p, mid - 1, mid);
  }
  if (spasm_row_weight(A, p[mid - 1]) > spasm_row_weight(A, p[mid + 1])) {
    spasm_swap(p, mid - 1, mid + 1);
  }
  if (spasm_row_weight(A, p[mid]) > spasm_row_weight(A, p[mid + 1])) {
    spasm_swap(p, mid, mid + 1);
  }
  return mid;
}


/* returns final position of pivot */
static int pivoting(const spasm * A, int *p, const int initial_left, const int initial_right, const int pivotIndex) {
  int pivotValue, left, right;

  spasm_swap(p, pivotIndex, initial_right - 1);
  pivotValue = spasm_row_weight(A, p[initial_right - 1]);

  right = initial_right - 2;
  left = initial_left;

  while (left < right) {
    while (spasm_row_weight(A, p[left]) < pivotValue) {
      left++;
    }
    while (spasm_row_weight(A, p[right]) > pivotValue) {
      right--;
    }

    if (left < right) {
      spasm_swap(p, left, right);
      left++;
    }
  }

  if (spasm_row_weight(A, p[right]) < pivotValue) {
    right++;
  }
  spasm_swap(p, right, initial_right - 1);
  return right;
}

static void spasm_quicksort(const spasm * A, int *p, const int left, const int right) {
  int pivotIndex, new_pivotIndex;

  if (right - left > INSERT_SORT_THRESHOLD) {

    pivotIndex = choose_pivot(A, p, left, right);
    new_pivotIndex = pivoting(A, p, left, right, pivotIndex);

    spasm_quicksort(A, p, left, new_pivotIndex);
    spasm_quicksort(A, p, new_pivotIndex + 1, right);
  } else {
    insertion_sort(A, p, left, right);
  }
}


int *spasm_row_sort(const spasm * A) {
  int *p;
  int i, n;

  n = A->n;
  p = spasm_malloc(n * sizeof(int));
  for (i = 0; i < n; i++) {
    p[i] = i;
  }
  spasm_quicksort(A, p, 0, n);
  return p;
}


/* return the number of pivots found. 
  @param p : row permutations. Pivotal rows are first.
  @param qinv : inverse column permutation. q[j] is the row on which
                the pivot on column j is, or -1 if there is no pivot
                on column j.
  both p and qinv must be preallocated
*/
int spasm_find_pivots(const spasm * A, int *p, int *qinv) {
  int n, m, i, j, k, I, idx_j, px, n_cheap, pxI, head, tail;
  int *Ap, *Aj, *w, *queue;
  spasm_GFp *Ax;

  n = A->n;
  m = A->m;
  Ap = A->p;
  Aj = A->j;
  Ax = A->x;

  w = spasm_malloc(m * sizeof(int));
#if 1
  queue = spasm_malloc(m * sizeof(int));
#endif

  /* --- Faugère-Lachartre pivot detection ------------------ */
  spasm_vector_set(qinv, 0, m, -1);
  
  for (i = 0; i < n; i++) {
    /* find leftmost entry */
    j = -1;
    for (px = Ap[i]; px < Ap[i + 1]; px++) {
      if (j == -1 || Aj[px] < j) {
        j = Aj[px];
        idx_j = px;
      }
    }
    /* Skip empty rows */
    if (j == -1) {
      continue;
    }
    /* make sure leftmost entry is the first of the row */
    spasm_swap(Aj, Ap[i], idx_j);
    if (Ax != NULL) {
      spasm_swap(Ax, Ap[i], idx_j);
    }
    
    /* check if it is a sparser pivot */
    if (qinv[j] == -1 || spasm_row_weight(A, i) < spasm_row_weight(A, qinv[j])) {
      qinv[j] = i;
    }
  }

  /* count the Faugère-Lachartre pivots, and store their rows in p */
  k = 0;
  for (j = 0; j < m; j++) {
    if (qinv[j] != -1) {
      p[k++] = qinv[j];
    }
  }
  n_cheap = k;
  fprintf(stderr, "[pivots] found %d Faugère-Lachartre pivots\n", k);


/* --- free column pivots ----------------------------------*/
  spasm_vector_set(w, 0, m, 1);
  
  /* scatter previous pivot rows */
  for (i = 0; i < k; i++) {
    I = p[i];
    for (px = Ap[I]; px < Ap[I + 1]; px++) {
      j = Aj[px];
      w[j] = 0;
    }
  }

  /* find new pivots */
  for (i = 0; i < n; i++) {
    j = Aj[Ap[i]];
    if (qinv[j] == i) {  /* this row is already pivotal: skip */
      continue;
    }
   
    /* does it have an entry on a possible column? */
    for (px = Ap[i]; px < Ap[i + 1]; px++) {
      j = Aj[px];
      if (w[j] == 0) {
        continue; /* this column is closed, skip this entry */
      }

      /* check if it is a sparser pivot */
      if (qinv[j] == -1 || spasm_row_weight(A, i) < spasm_row_weight(A, qinv[j])) {
        if (qinv[j] == -1) {
          k++;  
        }
        qinv[j] = i;
        spasm_swap(Aj, Ap[i], px);
        if (Ax != NULL) {
          spasm_swap(Ax, Ap[i], px);
        }
        break;   /* this row is done */
      }
    } 
  }
  fprintf(stderr, "[pivots] %d pivots found on free columns\n", k - n_cheap);
  n_cheap = k;

#if 0
  /* --- transitive reduction ------------------------------------- */
  fprintf(stderr, "starting transitive reduction...\n");
  spasm *T = spasm_csr_alloc(k, m, A->nzmax, -1, SPASM_IGNORE_VALUES);
  int *Tp = T->p;
  int *Tj = T->j;
  int *qinv_red = spasm_malloc(m * sizeof(int));
  int tnz = 0, anz = 0;

  /*
   * workspace initialization. Marking scheme : -1 = unmarked 0 = marked by
   * original row (=kept in transitive reduction) 1 = marked by other,
   * reachable, rows (=removed in transitive reduction)
   */
  for (j = 0; j < m; j++) {
    w[j] = -1;
    qinv_red[j] = -1;
  }

  for (i = 0; i < k; i++) {
    /* perform reduction of row A[I], store result in T[i] */
    I = p[k - 1 - i];

    /* initialize BFS in T with A[I] */
    Tp[i] = tnz;

    head = 0;
    tail = 0;
    for (px = Ap[I]; px < Ap[I + 1]; px++) {
      j = Aj[px];
      queue[tail] = j;
      tail++;
      w[j] = 0;
    }
    anz += spasm_row_weight(A, I);

    /* start BFS */
    while (head < tail) {
      j = queue[head];
      head++;

      int Ired = qinv_red[j];
      if (Ired == -1) {
        continue;
      }
      /*
       * trick: the first entry is the pivot, we know it is already marked,
       * so we skip it
       */
      for (pxI = Tp[Ired] + 1; pxI < Tp[Ired + 1]; pxI++) {
        j = Tj[pxI];
        if (w[j] < 0) {         /* not marked : mark and add to queue */
          queue[tail] = j;
          tail++;
        }
        w[j] = 1;
      }
    }

    /* scan w for surviving entries, add them to T */
    for (px = Ap[I]; px < Ap[I + 1]; px++) {
      j = Aj[px];
      if (w[j] == 0) {          /* entry has not been "superseded", so we add
                                 * it to T */
        Tj[tnz] = j;
        tnz++;
      }
    }

    /* reset w */
    for (px = 0; px < tail; px++) {
      j = queue[px];
      w[j] = -1;
    }

    qinv_red[Aj[Ap[I]]] = i;
    fprintf(stderr, "\r%d / %d | %d vs %d", i, k, tnz, anz);
    fflush(stderr);
  }
  fprintf(stderr, "\r                                             \r");
  /* finalize the last row of T */
  Tp[k] = tnz;
  n_cheap = k;
  fprintf(stderr, "done. |A| = %d, |T| = %d\n", anz, tnz);
#endif

  /* --- find less-cheap pivots ----------------------------------- */
#if 0
  n_cheap = k;
  int n_cheap1 = k;
  int processed = 0;
  /* workspace initialization */
  for (j = 0; j < m; j++) {
    w[j] = 0;
  }

  for (i = 0; i < n; i++) {
    if (q[Aj[Ap[i]]] == i) {    /* this row is already pivotal: skip */
      continue;
    }
    if (i % (n / 100) == 0) {
      fprintf(stderr, "\rcheap : %d / %d --- found %d new", processed, n - n_cheap1, n_cheap - n_cheap1);
      fflush(stderr);
    }
    processed++;
    //printf("------------------------ %d\n", i);

    /* scatters non-pivotal columns of A[i] into w */
    for (px = Ap[i]; px < Ap[i + 1]; px++) {
      j = Aj[px];
      if (q[j] != -1) {         /* column is pivotal: skip */
        continue;
      }
      w[j] = 1;
    }

    head = 0;
    tail = 0;
    for (px = Ap[i]; px < Ap[i + 1]; px++) {
      j = Aj[px];
      if (q[j] == -1) {         /* column is not pivotal: skip */
        continue;
      }
      if (w[j] < 0) {           /* already marked: skip */
        continue;
      }
      queue[tail] = j;
      tail++;
      w[j] = -1;
      //printf("amorçage en %d\n", j);
      /* BFS */
      while (head < tail) {
        j = queue[head];
        //assert(w[j] < 0);
        head++;

        I = q[j];
        if (I == -1) {
          continue;             /* nothing to do */
        }
        for (pxI = Ap[I]; pxI < Ap[I + 1]; pxI++) {
          j = Aj[pxI];
          if (w[j] < 0) {
            continue;           /* already marked */
          }
          queue[tail] = j;
          tail++;
          w[j] = -1;
          /*
           * ne peut-on éviter d'empiler des trucs qui déclenchent le
           * "continue" ci-dessus ?
           */
        }
      }
    }

    /* scan w for surviving entries */
    k = -1;
    for (px = Ap[i]; px < Ap[i + 1]; px++) {
      j = Aj[px];
      if ((k == -1) && (w[j] == 1)) {
        k = j;
        idx_j = px;
      }
      w[j] = 0;                 /* utile ? */
    }

    /* reset w */
    for (px = 0; px < tail; px++) {
      j = queue[px];
      w[j] = 0;
    }

    if (k != -1) {
      q[k] = i;

      /* make sure leftmost entry is the first of the row */
      spasm_swap(Aj, Ap[i], idx_j);
      spasm_swap(Ax, Ap[i], idx_j);
      n_cheap++;
    }
  }
  fprintf(stderr, "\r[pivots] found %d cheap pivots (greedy search)\n", n_cheap - n_cheap1);
#endif

  /* --- build corresponding row permutation ---------------------- */

  int *xj = spasm_malloc(m * sizeof(int));
  int *marks = spasm_malloc(m * sizeof(int));
  int *pstack = spasm_malloc(n * sizeof(int));
  
  /* DFS */
  spasm_vector_set(marks, 0, m, 0);
  int top = m;
  for (j = 0; j < m; j++) {
    if (qinv[j] != -1 && !marks[j]) {
      top = spasm_dfs(j, A, top, xj, pstack, marks, qinv);
    }
  }

  /* reorders the first n_cheap rows of p */
  k = 0;
  for (j = m-1; j >= top; j--) {
    i = qinv[xj[j]];
    if (i != -1) {
      p[k] = i;
      k++;
    }
  }
  assert(k == n_cheap);

  free(xj);
  free(pstack);
  free(marks);

  n_cheap = k;

  /* put other (non-empty) rows afterwards */
  for (i = 0; i < n; i++) {
    if (Ap[i] == Ap[i + 1]) {
      continue;
    }
    j = Aj[Ap[i]];
    if (qinv[j] != i) { /* row is non-pivotal */
      assert(k < n);
      p[k] = i;
      k++;
    }
  }

  /* put empty rows last */
  for (i = 0; i < n; i++) {
    if (Ap[i] == Ap[i + 1]) {
      p[k] = i;
      k++;
    }
  }

  free(w);
  return n_cheap;
}

/* returns a pemruted version of A where pivots are pushed to the top-left
* and form an upper-triangular principal submatrix */
spasm * spasm_permute_pivots(const spasm *A, int *p, int *qinv, int npiv) {
  int i, j, k, n, m, *Ap, *Aj;

  n = A->n;
  m = A->m;
  Ap = A->p;
  Aj = A->j;

  /* pivotal column first, in row-order */
  k = 0;
  for (i = 0; i < npiv; i++) {
    j = Aj[Ap[p[i]]];         /* the pivot is the first entry of each row */
    qinv[j] = k++;
  }

  /* put remaining non-pivotal columns afterwards, in any order */
  for (j = 0; j < m; j++) {
    if (qinv[j] == -1) {
      qinv[j] = k++;
    }
  }

  return spasm_permute(A, p, qinv, SPASM_WITH_NUMERICAL_VALUES);
}