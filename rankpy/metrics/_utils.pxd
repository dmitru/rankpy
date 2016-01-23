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


from cython cimport view

cimport numpy as np

ctypedef np.float64_t DOUBLE_t
ctypedef np.int32_t   INT_t

# =============================================================================
# C function definitions
# =============================================================================

cdef void argranksort_c(DOUBLE_t *ranking_scores, INT_t *ranks, INT_t n_documents, unsigned int *seed) nogil
cdef void argranksort_queries_c(INT_t *query_indptr, INT_t n_queries, DOUBLE_t *ranking_scores, INT_t *ranks, unsigned int *seed) nogil

cdef void ranksort_c(DOUBLE_t *ranking_scores, INT_t *ranking, INT_t n_documents, unsigned int *seed) nogil
cdef void ranksort_queries_c(INT_t *query_indptr, INT_t n_queries, DOUBLE_t *ranking_scores, INT_t *ranking, unsigned int *seed) nogil

cdef void ranksort_relevance_scores_c(DOUBLE_t *ranking_scores, INT_t *relevance_scores, INT_t n_documents, INT_t *out, unsigned int *seed) nogil
cdef void ranksort_relevance_scores_queries_c(INT_t *query_indptr, INT_t n_queries, DOUBLE_t *ranking_scores, INT_t *relevance_scores, INT_t *out, unsigned int *seed) nogil

cdef int relevance_argsort_v1_c(INT_t *array, INT_t *indices, INT_t size, INT_t maximum=*) nogil
cdef void relevance_argsort_v2_c(INT_t *array, INT_t *indices, INT_t size) nogil

cdef unsigned int get_seed(object random_state=*)

# =============================================================================
# Python bindings for the C functions defined above.
# =============================================================================

cpdef argranksort(DOUBLE_t[::1] ranking_scores, INT_t[::1] ranks, object random_state=*)
cpdef argranksort_queries(INT_t[::1] query_indptr, DOUBLE_t[::1] ranking_scores, INT_t[::1] ranks, object random_state=*)

cpdef ranksort(DOUBLE_t[::1] ranking_scores, INT_t[::1] ranking, object random_state=*)
cpdef ranksort_queries(INT_t[::1] query_indptr, DOUBLE_t[::1] ranking_scores, INT_t[::1] ranking, object random_state=*)

cpdef ranksort_relevance_scores(DOUBLE_t[::1] ranking_scores, INT_t[::1] relevance_scores, INT_t[::1] out, object random_state=*)
cpdef ranksort_relevance_scores_queries(INT_t[::1] query_indptr, DOUBLE_t[::1] ranking_scores, INT_t[::1] relevance_scores, INT_t[::1] out, object random_state=*)

cpdef relevance_argsort_v1(INT_t[::1] array, INT_t[::1] indices, INT_t size, INT_t maximum=*)
cpdef relevance_argsort_v2(INT_t[::1] array, INT_t[::1] indices, INT_t size)

cpdef noise_relenvace_scores(INT_t[::1] relevance_scores, DOUBLE_t [:, ::1] probabilities, object random_state=*)
cpdef rand_uniform(DOUBLE_t low, DOUBLE_t high, object random_state=*)
