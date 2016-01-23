# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# coding: utf-8
#
# This file is part of RankPy.
#
# RankPy is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RankPy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with RankPy.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
cimport numpy as np
np.import_array()

from cython cimport view

from libc.stdlib cimport calloc, free, rand, srand, qsort, RAND_MAX

from libc.time cimport time
from libc.time cimport time_t

from libc.math cimport log as ln

from numpy import float64 as DOUBLE
from numpy import int32   as INT

# =============================================================================
# Global variables and structure declarations
# =============================================================================

# The minimum value of 32 bit integer.
cdef INT_t INT32_MIN = np.iinfo('i').min
cdef INT_t INT32_MAX = np.iinfo('i').max

# Auxiliary document structure, used for sort-ing and
# argsort-ing.
cdef struct DOCUMENT_t:
    INT_t position      # The document position in the input list ('document ID').
    INT_t nonce         # Randomly generated number used to break ties in ranking scores.
    DOUBLE_t score      # The document ranking score.

cdef enum:
    RAND_R_MAX = 0x7FFFFFFF # alias 2**32 - 1


# =============================================================================
# C function definitions
# =============================================================================


cdef int __compare(const void *a, const void *b) nogil:
    ''' 
    Compare function used in stdlib's qsort. The parameters are 2 instances
    of DOCUMENT_t, which are being sorted in descending order according to
    their ranking scores, i.e. the higher the ranking score the lower the
    resulting rank of the document.
    '''
    cdef DOUBLE_t diff = ((<DOCUMENT_t*>b)).score - ((<DOCUMENT_t*>a)).score
    if diff < 0:
        if diff > -1e-12: # "Equal to zero from left"
            return ((<DOCUMENT_t*>b)).nonce - ((<DOCUMENT_t*>a)).nonce
        else:
            return -1
    else:
        if diff < 1e-12: # "Equal to zero from right"
            return ((<DOCUMENT_t*>b)).nonce - ((<DOCUMENT_t*>a)).nonce
        else:
            return 1


cdef void __argranksort(DOUBLE_t *ranking_scores,
                        DOCUMENT_t *documents,
                        INT_t document_position_offset,
                        INT_t n_documents,
                        unsigned int *seed) nogil:
    ''' 
    Auxiliary function for ranksort and argranksort functions.
    '''
    cdef INT_t i

    for i in range(n_documents):
        documents[i].position = i + document_position_offset
        documents[i].nonce = our_rand_r(seed)
        documents[i].score = ranking_scores[i]

    qsort(documents, n_documents, sizeof(DOCUMENT_t), __compare)


cdef void __argranksort_queries(INT_t *query_indptr,
                                INT_t n_queries,
                                DOUBLE_t *ranking_scores,
                                DOCUMENT_t *documents,
                                unsigned int *seed) nogil:
    ''' 
    Auxiliary function for ranksort_queries and argranksort_queries functions.
    '''
    cdef INT_t i

    for i in range(n_queries):
        __argranksort(ranking_scores + query_indptr[i], documents + query_indptr[i] - query_indptr[0],
                      query_indptr[i], query_indptr[i + 1] - query_indptr[i], seed)


cdef void argranksort_c(DOUBLE_t *ranking_scores,
                        INT_t *ranks,
                        INT_t n_documents,
                        unsigned int *seed) nogil:
    ''' 
    Return the rank position of the documents associated with the specified ranking_scores,
    i.e. `ranks[i]` is the position of the `ranking_scores[i]` within the sorted array
    (in descending order) of `ranking_scores`.
    '''
    cdef:
        INT_t i
        DOCUMENT_t *documents = <DOCUMENT_t *> calloc(n_documents, sizeof(DOCUMENT_t))

    __argranksort(ranking_scores, documents, 0, n_documents, seed)

    for i in range(n_documents):
        ranks[documents[i].position] = i

    free(documents)
    

cdef void argranksort_queries_c(INT_t *query_indptr,
                                INT_t n_queries,
                                DOUBLE_t *ranking_scores,
                                INT_t *ranks,
                                unsigned int *seed) nogil:
    ''' 
    Return the rank position of the documents within the document list of
    specified queries, which is determined using the specified ranking scores.
    '''
    cdef:
        INT_t i, j, r, n_documents = query_indptr[n_queries] - query_indptr[0]
        DOCUMENT_t *documents = <DOCUMENT_t *> calloc(n_documents, sizeof(DOCUMENT_t))

    __argranksort_queries(query_indptr, n_queries, ranking_scores, documents, seed)

    for i in range(n_queries):
        r = 0
        for j in range(query_indptr[i] - query_indptr[0], query_indptr[i + 1] - query_indptr[0]):
            ranks[documents[j].position - query_indptr[0]] = r
            r += 1

    free(documents)


cdef void ranksort_c(DOUBLE_t *ranking_scores,
                     INT_t *ranking,
                     INT_t n_documents,
                     unsigned int *seed) nogil:
    ''' 
    Return the ranking of the documents associated with the specified ranking scores,
    i.e. `ranking[i]` identifies the ranking score which would be placed at i-th
    position within the sorted array (in descending order) of `ranking_scores`.
    '''
    cdef:
        INT_t i
        DOCUMENT_t *documents = <DOCUMENT_t *> calloc(n_documents, sizeof(DOCUMENT_t))

    __argranksort(ranking_scores, documents, 0, n_documents, seed)

    for i in range(n_documents):
        ranking[i] = documents[i].position

    free(documents)


cdef void ranksort_queries_c(INT_t *query_indptr,
                             INT_t n_queries,
                             DOUBLE_t *ranking_scores,
                             INT_t *ranking,
                             unsigned int *seed) nogil:
    ''' 
    Return the ranking of the documents associated with the specified ranking scores,
    i.e. `ranking_scores[ranking[i]]` will be the ranking score which would be placed
    at i-th position within the sorted array (in descending order) of `ranking_scores`.
    '''
    cdef:
        INT_t i, j, n_documents = query_indptr[n_queries] - query_indptr[0]
        DOCUMENT_t *documents = <DOCUMENT_t *> calloc(n_documents, sizeof(DOCUMENT_t))

    __argranksort_queries(query_indptr, n_queries, ranking_scores, documents, seed)

    for i in range(n_queries):
        for j in range(query_indptr[i] - query_indptr[0], query_indptr[i + 1] - query_indptr[0]):
            ranking[j] = documents[j].position - query_indptr[i]

    free(documents)


cdef void ranksort_relevance_scores_c(DOUBLE_t *ranking_scores,
                                      INT_t *relevance_scores,
                                      INT_t n_documents,
                                      INT_t *out,
                                      unsigned int *seed) nogil:
    ''' 
    Rank the specified relevance scores according to the specified ranking scores.
    '''
    cdef:
        INT_t i
        DOCUMENT_t *documents = <DOCUMENT_t *> calloc(n_documents, sizeof(DOCUMENT_t))

    __argranksort(ranking_scores, documents, 0, n_documents, seed)

    for i in range(n_documents):
        out[i] = relevance_scores[documents[i].position]

    free(documents)


cdef void ranksort_relevance_scores_queries_c(INT_t *query_indptr,
                                              INT_t n_queries,
                                              DOUBLE_t *ranking_scores,
                                              INT_t *relevance_scores,
                                              INT_t *out,
                                              unsigned int *seed) nogil:
    ''' 
    Rank the specified relevance scores according to the specified ranking
    scores with respect to the given queries.
    '''
    cdef:
        INT_t i, j, n_documents = query_indptr[n_queries] - query_indptr[0]
        DOCUMENT_t *documents = <DOCUMENT_t *> calloc(n_documents, sizeof(DOCUMENT_t))

    __argranksort_queries(query_indptr, n_queries, ranking_scores, documents, seed)

    for i in range(n_queries):
        for j in range(query_indptr[i] - query_indptr[0], query_indptr[i + 1] - query_indptr[0]):
            out[j] = relevance_scores[documents[j].position]

    free(documents)


cdef int relevance_argsort_v1_c(INT_t *array, INT_t *indices, INT_t size, INT_t maximum=INT32_MIN) nogil:
    ''' 
    Find indices that sort the given array of non-negative integers in descending order.
    The sorting algorithm is 'counting sort' and it should be used for sorting numbers
    with decent maximum value.

    To sort unbounded integers use `relevance_argsort_v2_c` instead.
    '''
    cdef INT_t i
    cdef INT_t *counts = NULL

    if maximum == INT32_MIN:
        for i in range(size):
            if array[i] > maximum:
                maximum = array[i]

    counts = <INT_t *> calloc(maximum + 2, sizeof(INT_t))

    if counts == NULL:
        return -1

    for i in range(size):
        counts[array[i]] += 1

    for i in range(maximum + 1, 0, -1):
        counts[i - 1] += counts[i]

    for i in range(size):
        indices[counts[array[i] + 1]] = i
        counts[array[i] + 1] += 1

    free(counts)

    return 0


cdef void relevance_argsort_v2_c(INT_t *array, INT_t *indices, INT_t size) nogil:
    ''' 
    An alternative argosrt for unbounded integers. See `relevance_argsort_v1_c`.
    '''
    cdef INT_t i

    for i in range(size):
         indices[i] = i

    introargsort_c(array, indices, size, 2 * <INT_t>log(size))


cdef unsigned int get_seed(object random_state=None):
    if random_state is None:
        random_state = np.random.mtrand._rand
    random_state.randint(1, INT32_MAX)


# =============================================================================
# Python bindings for the C functions defined above.
# =============================================================================


cpdef argranksort(DOUBLE_t[::1] ranking_scores, INT_t[::1] ranks, object random_state=None):
    ''' 
    Return the rank position of the documents associated with the specified ranking_scores,
    i.e. `ranks[i]` is the position of the `ranking_scores[i]` within the sorted array
    (in descending order) of `ranking_scores`.
    '''
    cdef unsigned int seed = get_seed(random_state)
    with nogil:
        argranksort_c(&ranking_scores[0], &ranks[0], ranking_scores.shape[0], &seed)


cpdef argranksort_queries(INT_t[::1] query_indptr, DOUBLE_t[::1] ranking_scores, INT_t[::1] ranks, object random_state=None):
    ''' 
    Return the rank position of the documents within the document list of
    specified queries, which is determined using the specified ranking scores.
    '''
    cdef unsigned int seed = get_seed(random_state)
    with nogil:
        argranksort_queries_c(&query_indptr[0], query_indptr.shape[0] - 1, &ranking_scores[0], &ranks[0], &seed)


cpdef ranksort(DOUBLE_t[::1] ranking_scores, INT_t[::1] ranking, object random_state=None):
    ''' 
    Return the ranking of the documents associated with the specified ranking scores,
    i.e. `ranking[i]` identifies the ranking score which would be placed at i-th
    position within the sorted array (in descending order) of `ranking_scores`.
    '''
    cdef unsigned int seed = get_seed(random_state)
    with nogil:
        ranksort_c(&ranking_scores[0], &ranking[0], ranking_scores.shape[0], &seed)


cpdef ranksort_queries(INT_t[::1] query_indptr, DOUBLE_t[::1] ranking_scores, INT_t[::1] ranking, object random_state=None):
    ''' 
    Return the ranking of the documents associated with the specified ranking scores,
    i.e. `ranking_scores[ranking[i]]` will be the ranking score which would be placed
    at i-th position within the sorted array (in descending order) of `ranking_scores`.
    '''
    cdef unsigned int seed = get_seed(random_state)
    with nogil:
        ranksort_queries_c(&query_indptr[0], query_indptr.shape[0] - 1, &ranking_scores[0], &ranking[0], &seed)


cpdef ranksort_relevance_scores(DOUBLE_t[::1] ranking_scores, INT_t[::1] relevance_scores, INT_t[::1] out, object random_state=None):
    ''' 
    Rank the specified relevance scores according to the specified ranking scores.
    '''
    cdef unsigned int seed = get_seed(random_state)
    with nogil:
        ranksort_relevance_scores_c(&ranking_scores[0], &relevance_scores[0], ranking_scores.shape[0], &out[0], &seed)


cpdef ranksort_relevance_scores_queries(INT_t[::1] query_indptr, DOUBLE_t[::1] ranking_scores, INT_t[::1] relevance_scores, INT_t[::1] out, object random_state=None):
    ''' 
    Rank the specified relevance scores according to the specified ranking
    scores with respect to the given queries.
    '''
    cdef unsigned int seed = get_seed(random_state)
    with nogil:
        ranksort_relevance_scores_queries_c(&query_indptr[0], query_indptr.shape[0] - 1, &ranking_scores[0], &relevance_scores[0], &out[0], &seed)


cpdef noise_relenvace_scores(INT_t[::1] relevance_scores, DOUBLE_t [:, ::1] probabilities, object random_state=None):
    ''' 
    Create a new relevance scores by introducing noise into given relevances.
    The probability of score `i` changing into `j` is `probabilities[i, j]`.
    '''
    cdef:
        INT_t i, j, s
        INT_t n_scores = probabilities.shape[0]
        INT_t n_documents = relevance_scores.shape[0]
        np.ndarray[INT_t, ndim=1] scores = np.empty(n_documents, dtype=INT)
        DOUBLE_t p
        unsigned int seed = get_seed(random_state)

    with nogil:
        # Go through all relevance scores and noise them.
        for i in range(n_documents):
            # The document (ground truth) score.
            s = relevance_scores[i]

            # Just to make sure score will always be set.
            scores[i] = n_scores - 1

            # Sample a random number from [0, 1).
            p = rand_uniform_c(0.0, 1.0, &seed)

            # Change the relevance according to p and probabilities.
            for j in range(n_scores):
                if p <= probabilities[s, j]:
                    scores[i] = j
                    break
                else:
                    p -= probabilities[s, j]                

    return scores


cpdef relevance_argsort_v1(INT_t[::1] array, INT_t[::1] indices, INT_t size, INT_t maximum=INT32_MIN):
    ''' 
    Find indices that sort the given array of non-negative integers in descending order.
    The sorting algorithm is 'counting sort' and it should be used for sorting numbers
    with decent maximum value.

    To sort unbounded integers use `relevance_argsort_v2` instead.
    '''
    with nogil:
        relevance_argsort_v1_c(&array[0], &indices[0], size, maximum)


cpdef relevance_argsort_v2(INT_t[::1] array, INT_t[::1] indices, INT_t size):
    ''' 
    An alternative argosrt for unbounded integers. See `relevance_argsort_v1`.
    '''
    with nogil:
        relevance_argsort_v2_c(&array[0], &indices[0], size)


cpdef rand_uniform(DOUBLE_t low, DOUBLE_t high, object random_state=None):
    cdef unsigned int seed = get_seed(random_state)
    return rand_uniform_c(low, high, &seed)


# ================================================================================
# Edited code from scikit-learn v15.2 - see _tree.pyx for original version.
# Original authors: Gilles Louppe <g.louppe@gmail.com>
#                   Peter Prettenhofer <peter.prettenhofer@gmail.com>
#                   Brian Holt <bdholt1@gmail.com>
#                   Noel Dawe <noel@dawe.me>
#                   Satrajit Gosh <satrajit.ghosh@gmail.com>
#                   Lars Buitinck <L.J.Buitinck@uva.nl>
#                   Arnaud Joly <arnaud.v.joly@gmail.com>
#                   Joel Nothman <joel.nothman@gmail.com>
#                   Fares Hedayati <fares.hedayati@gmail.com>
#
# Under BSD 3 clause licence.
#
# Edited by: Tomas Tunys <tunystom@gmail.com>
# ================================================================================


cdef inline void swap(INT_t *indices, INT_t i, INT_t j) nogil:
    indices[i], indices[j] = indices[j], indices[i]


cdef inline int median3(INT_t *array, INT_t *indices, INT_t size) nogil:
    # Median of three pivot selection, after Bentley and McIlroy (1993).
    # Engineering a sort function. SP&E. Requires 8/3 comparisons on average.
    cdef int a = array[indices[0]]
    cdef int b = array[indices[size / 2]]
    cdef int c = array[indices[size - 1]]

    if a < b:
        if b < c:
            return b
        elif a < c:
            return c
        else:
            return a
    elif b < c:
        if a < c:
            return a
        else:
            return c
    else:
        return b


# Sorting Algorithm: Introsort (Musser, SP&E, 1997).
# Introsort with median of 3 pivot selection and 3-way partition function (robust to repeated elements).
cdef void introargsort_c(INT_t *array, INT_t *indices, INT_t size, INT_t max_depth) nogil:
    cdef INT_t pivot, i, l, r

    while size > 1:
        # For really small array chunks use insertion sort.
        if size < 25:
            insertargsort(array, indices, size)
            return

        # Maximum depth limit exceeded ("gone quadratic").
        if max_depth == 0:
            heapargsort(array, indices, size)
            return

        max_depth -= 1

        pivot = median3(array, indices, size)

        # Three-way partition.
        i = l = 0
        r = size

        while i < r:
            if array[indices[i]] > pivot:
                swap(indices, i, l)
                i += 1
                l += 1
            elif array[indices[i]] < pivot:
                r -= 1
                swap(indices, i, r)
            else:
                i += 1

        introargsort_c(array, indices, l, max_depth)

        indices += r
        size -= r


cdef inline void bubbledown(INT_t *array, INT_t *indices, INT_t i, INT_t size) nogil:
    cdef int j = 2 * i  + 1

    while j < size:
        if j + 1 < size and array[indices[j]] > array[indices[j + 1]]:
            j += 1

        if array[indices[i]] > array[indices[j]]:
            swap(indices, i, j)
        else:
            break

        i = j
        j = 2 * i  + 1


cdef void heapargsort(INT_t *array, INT_t *indices, INT_t size) nogil:
    cdef INT_t i

    i = (size - 2) / 2

    while i >= 0:
        bubbledown(array, indices, i, size)
        i -= 1

    i = size - 1

    while i > 0:
        swap(indices, 0, i)
        bubbledown(array, indices, 0, i)
        i -= 1


cdef void insertargsort(INT_t *array, INT_t *indices, int size) nogil:
    cdef INT_t i, j, k

    # Find the maximum element and put it
    # at the first place to play the role
    # of a boundary sentinel.
    j = 0
    for i in range(1, size):
        if array[indices[i]] > array[indices[j]]:
            j = i

    # Put the sentinel at "array[0]".
    swap(indices, 0, j)

    for i in range(1, size):
        j = i
        k = indices[j]
        # We have a sentinel at "array[0]".
        while array[k] > array[indices[j - 1]]:
            indices[j] = indices[j - 1]
            j -= 1
        indices[j] = k


cdef inline double log(double x) nogil:
    return ln(x) / ln(2.0)


# rand_r replacement using a 32bit XorShift generator
# See http://www.jstatsoft.org/v08/i14/paper for details
cdef inline int our_rand_r(unsigned int *seed) nogil:
    seed[0] ^= <unsigned int>(seed[0] << 13)
    seed[0] ^= <unsigned int>(seed[0] >> 17)
    seed[0] ^= <unsigned int>(seed[0] << 5)

    return <int>(seed[0] & <unsigned int>RAND_R_MAX)


cdef inline int rand_int_c(int low, int high, unsigned int *seed) nogil:
    ''' 
    Generate a random integer in [low, end).
    '''
    return low + our_rand_r(seed) % (high - low)


cdef inline double rand_uniform_c(double low, double high, unsigned int *seed) nogil:
    ''' 
    Generate a random double in [low; high).
    '''
    return ((high - low) * <double> our_rand_r(seed) / <double> RAND_R_MAX) + low


# ================================================================================
# Just some timings...
# ================================================================================
#
# Setup: 2**20 integers sampled at random from [0, 5].
# -------------------------------------------------------------
# numpy.argsort:        1000 loops, best of 3: 26.8 ms per loop
# relevance_argsort_v1: 1000 loops, best of 3: 3.33 ms per loop
# relevance_argsort_v2: 1000 loops, best of 3: 14.9 ms per loop
# -------------------------------------------------------------

