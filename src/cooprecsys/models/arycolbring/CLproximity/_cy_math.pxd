# _cy_math.pxd
# Inline math utilities for arycolbring Cython modules.
# All functions defined here are inlined into every module that cimports them.

from _cy_types cimport CSRMatrix, flt

# ── C standard library declarations ─────────────────────────────────────────

cdef extern from "math.h" nogil:
    double sqrt(double)
    double exp(double)
    double log(double)
    double floor(double)

cdef extern from "stdlib.h" nogil:
    void qsort(void *base, int nmemb, int size,
               int(*compar)(const void *, const void *)) nogil noexcept
    void* bsearch(const void *key, const void *base, int nmemb, int size,
                  int(*compar)(const void *, const void *)) nogil noexcept

# ── Pair struct (used by WARP-kOS) ──────────────────────────────────────────

cdef struct Pair:
    int idx
    flt val

# ── PRNG ────────────────────────────────────────────────────────────────────

cdef inline unsigned int temper(unsigned int x) noexcept nogil:
    cdef unsigned int and_1 = 0x9D2C5680
    cdef unsigned int and_2 = 0xEFC60000
    x = x ^ (x >> 11)
    x = x ^ (x << 7  & and_1)
    x = x ^ (x << 15 & and_2)
    x = x ^ (x >> 18)
    return x


cdef inline int rand_r(unsigned int *seed) noexcept nogil:
    seed[0] = seed[0] * 1103515245 + 12345
    return temper(seed[0]) / 2


cdef inline int sample_range(int min_val, int max_val, unsigned int *seed) noexcept nogil:
    cdef int val_range = max_val - min_val
    return min_val + (rand_r(seed) % val_range)

# ── Integer helpers ──────────────────────────────────────────────────────────

cdef inline int int_min(int x, int y) noexcept nogil:
    if x < y:
        return x
    return y


cdef inline int int_max(int x, int y) noexcept nogil:
    if x < y:
        return y
    return x

# ── Comparators for qsort / bsearch ─────────────────────────────────────────

cdef inline int int_compare(const void *a, const void *b) noexcept nogil:
    cdef int va = (<int*>a)[0]
    cdef int vb = (<int*>b)[0]
    if va > vb:
        return 1
    elif va < vb:
        return -1
    return 0


cdef inline int flt_compare(const void *a, const void *b) noexcept nogil:
    cdef flt va = (<flt*>a)[0]
    cdef flt vb = (<flt*>b)[0]
    if va > vb:
        return 1
    elif va < vb:
        return -1
    return 0


cdef inline int reverse_pair_compare(const void *a, const void *b) noexcept nogil:
    cdef flt diff = (<Pair*>a).val - (<Pair*>b).val
    if diff < 0:
        return 1
    return -1

# ── Sigmoid activation ───────────────────────────────────────────────────────

cdef inline flt sigmoid(flt v) noexcept nogil:
    return <flt>(1.0 / (1.0 + exp(-v)))

# ── Positives lookup (binary search in sorted CSR row) ───────────────────────

cdef inline int in_positives(int item_id,
                              int user_id,
                              CSRMatrix interactions) noexcept nogil:
    """
    Return 1 if item_id is in the sorted indices of user_id's CSR row.
    Uses bsearch for O(log k) lookup where k = nnz per row.
    """
    cdef int start_idx = interactions.get_row_start(user_id)
    cdef int stop_idx  = interactions.get_row_end(user_id)

    if bsearch(&item_id,
               &interactions.indices[start_idx],
               stop_idx - start_idx,
               sizeof(int),
               int_compare) == NULL:
        return 0
    return 1