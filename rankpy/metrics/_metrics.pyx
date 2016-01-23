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
# along with RankPy. If not, see <http://www.gnu.org/licenses/>.


from cython cimport view

from cpython cimport Py_INCREF, PyObject

from libc.stdlib cimport malloc, calloc, realloc, free
from libc.string cimport memset
from libc.math cimport log2

from ._utils cimport ranksort_relevance_scores_queries_c

import numpy as np
cimport numpy as np
np.import_array()


# =============================================================================
# Types, constants, inline and helper functions
# =============================================================================


cdef inline INT_t imin(INT_t a, INT_t b) nogil:
    return b if b < a else a


cdef inline DOUBLE_t fabs(DOUBLE_t a) nogil:
    return -a if a < 0 else a


# =============================================================================
# Metric
# =============================================================================


cdef class Metric:
    ''' 
    The interface for an information retrieval evaluation metric.
    '''
    def __cinit__(self, INT_t cutoff, INT_t maximum_relevance, INT_t maximum_documents,
                  unsigned int seed):
        ''' 
        Initialize the metric with the specified cutoff threshold,
        maximum relevance score a document can have, and the maximum
        number of documents per query.

        All these values should allow the metric object to pre-allocate
        and precompute `something`, which may help it to evaluate
        the metric for queries faster.

        cutoff: integer
            If positive, it denotes the maximum rank of a document
            that will be considered for evaluation. If 0 is given
            ValueError is raised.

        maximum_relevance: integer
            The maximum relevance score a document can have.

        maximum_documents: integer
            The maximum number of documents a query can have.
        '''
        self.cutoff = cutoff
        self.seed = seed

        if cutoff == 0:
            raise ValueError('cutoff has to be positive integer or (-1) but 0 was given')

        if seed == 0:
            raise ValueError('seed cannot be 0')


    cpdef evaluate_ranking(self, INT_t[::1] ranking, INT_t[::1] relevance_scores, DOUBLE_t scale_value, DOUBLE_t query_weight):
        ''' 
        Evaluate the metric on the specified document ranking.

        Parameters:
        -----------
        ranking: array of integers, shape = (n_documents,)
            Specify the list of ranked documents.

        relevance_scores: array of integers, shape = (n_documents,)
            Specify the relevance score for each document.

        scale_value: double
            Optional parameter for implementation of metrics which evaluations
            need to be scaled by the ideal metric values.

        query_weight: double
            The weight of the query for which the metric is evaluated.
        '''
        pass


    cpdef evaluate(self, INT_t[::1] ranked_relevance_scores, DOUBLE_t scale_value, DOUBLE_t query_weight):
        ''' 
        Evaluate the metric on the specified ranked list of document relevance scores.

        Parameters:
        -----------
        ranked_relevance_scores: array of integers, shape = (n_documents,)
            Specify list of relevance scores.

        scale_value: double, shape = (n_documents,)
            Optional parameter for implementation of metrics which evaluations
            need to be scaled by the ideal metric values.

        query_weight: double
            The weight of the query for which the metric is evaluated.
        '''
        pass


    cpdef evaluate_queries(self, INT_t[::1] query_indptr, INT_t[::1] relevance_scores, DOUBLE_t[::1] scores,
                           DOUBLE_t[::1] scale_values, DOUBLE_t[::1] query_weights,  DOUBLE_t[::1] out):
        ''' 
        Evaluate the metric on the specified queries. The relevance scores and
        ranking scores of documents, which belong to query `i`, are in
        `relevance_scores[query_indptr[i]:query_indptr[i + 1]]` and  
        `scores[query_indptr[i]:query_indptr[i + 1]]`, respectively.
        
        Parameters:
        -----------
        query_indptr: array of integers, shape = (n_queries + 1,)
            The query index pointer.

        relevance_scores, array of integers, shape = (n_documents,)
            Specify the relevance score for each document.

        scores: array, shape=(n_documents,)
            Specify the ranking score for each document.

        scale_values: array, shape=(n_queries,), optional
            'Optional' parameter for implementation of metrics which evaluations
            need to be scaled by the ideal metric values. Specific metric
            implementations, such as NDCG, may use this parameter to speed
            up their computation.

        query_weights: array of doubles, shape = (n_queries,), optional (default is None)
            The weight given to each query.

        out: array, shape=(n_documents,), optional
            If not None, it will be filled with the metric value
            for each individual query.
        '''
        pass


    cpdef delta(self, INT_t i, INT_t offset, INT_t[::1] document_ranks, INT_t[::1] relevance_scores,
                DOUBLE_t scale_value, DOUBLE_t[::1] out):
        ''' 
        Compute the change in the metric caused by swapping document `i` with every
        document `offset`, `offset + 1`, ... (in turn).

        The relevance score and document rank of document `i` is `relevance_scores[i]`
        and `document_ranks[i]`, respectively.

        Parameters:
        -----------
        i: integer
            The index of the one document that is being swapped with all
            the others.

        offset: integer
            The start index of the sequence of documents that are
            being swapped.

        document_ranks: array of integers
            Specify the rank for each document.

        relevance_scores: array of integers
            Specify the relevance score for each document.

        scale_value: double, shape = (n_documents,)
            Optional parameter for implementation of metrics which evaluations
            need to be scaled by the ideal metric values.

        out: array of doubles
            The output array. The array size is expected to be at least as big
            as the the number of document pairs being swapped, which should be
            `len(document_ranks) - offset`.
        '''
        pass


    cdef void delta_c(self, INT_t i, INT_t offset, INT_t n_documents, INT_t *document_ranks, INT_t *relevance_scores,
                      DOUBLE_t scale_value, DOUBLE_t *out) nogil:
        ''' 
        See description of self.delta(...) method.
        '''
        pass


    cpdef evaluate_queries_ideal(self, INT_t[::1] query_indptr, INT_t[::1] relevance_scores,
                                 DOUBLE_t[::1] ideal_values):
        ''' 
        Compute the ideal metric value for every one of the specified queries.
        The relevance scores of documents, which belong to query `i`, must be
        stored in `relevance_scores[query_indptr[i]:query_indptr[i + 1]]` in
        descending order.

        Parameters:
        -----------
        query_indptr: array of integers, shape = (n_queries + 1,)
            The query index pointer.

        relevance_scores, array of integers, shape = (n_documents,)
            Specify the relevance score for each document.

        ideal_values: output array of doubles, shape=(n_queries,)
            Output array for the ideal metric value of each query.
        '''
        pass


# =============================================================================
# Discounted Cumulative Gain Metric
# =============================================================================


cdef class DiscountedCumulativeGain(Metric):
    ''' 
    Discounted Cumulative Gain (DCG) metric.
    '''

    cdef DOUBLE_t *gain_cache
    cdef INT_t     maximum_relevance

    cdef DOUBLE_t *discount_cache
    cdef INT_t     maximum_documents


    property gain:
        def __get__(self):
            ''' 
            Exposes the gain array to Python as NumPy array.
            '''
            cdef np.npy_intp shape = self.maximum_relevance + 1
            cdef np.ndarray arr = np.PyArray_SimpleNewFromData(1, &shape, np.NPY_DOUBLE, self.gain_cache)
            Py_INCREF(self)
            arr.base = <PyObject*> self
            return arr


    def __cinit__(self, INT_t cutoff, INT_t maximum_relevance,
                  INT_t maximum_documents, unsigned int seed):
        ''' 
        Initialize the metric with the specified cutoff threshold,
        maximum relevance score a document can have, and the maximum
        number of documents per query.

        These values are used to pre-allocate the gain and discount
        for document relevance scores and ranks, respectively.

        cutoff: integer
            If positive, it denotes the maximum rank of a document
            that will be considered for evaluation. If 0 is given
            ValueError is raised.

        maximum_relevance: integer
            The maximum relevance score a document can have.

        maximum_documents: integer
            The maximum number of documents a query can have.
        '''
        cdef INT_t i
        cdef DOUBLE_t gain

        self.gain_cache = NULL
        self.discount_cache = NULL

        if maximum_relevance <= 0:
            maximum_relevance = 8

        if maximum_documents <= 0:
            maximum_documents = 4096

        self.maximum_relevance = maximum_relevance
        self.maximum_documents = maximum_documents

        self.gain_cache = <DOUBLE_t*> calloc(self.maximum_relevance + 1, sizeof(DOUBLE_t))
        self.discount_cache = <DOUBLE_t*> calloc(self.maximum_documents, sizeof(DOUBLE_t))

        gain = 1.0
        for i in range(maximum_relevance + 1):
            self.gain_cache[i] = gain - 1.0
            gain *= 2

        for i in range(maximum_documents):
            self.discount_cache[i] = log2(2.0 + i)


    def __dealloc__(self):
        ''' 
        Clean up the cached gain and discount values.
        '''
        free(self.gain_cache)
        free(self.discount_cache)


    def __reduce__(self):
        ''' 
        Reduce reimplementation, for pickling.
        '''
        return (DiscountedCumulativeGain, (self.cutoff, self.maximum_relevance, self.maximum_documents))


    cpdef evaluate_ranking(self, INT_t[::1] ranking, INT_t[::1] relevance_scores, DOUBLE_t scale_value, DOUBLE_t query_weight):
        ''' 
        Evaluate the metric on the specified document ranking.

        Parameters:
        -----------
        ranking: array of integers, shape = (n_documents,)
            Specify the list of ranked documents.

        relevance_scores: array of integers, shape = (n_documents,)
            Specify the relevance score for each document.

        scale_value: double
            Should be 1.0.

        query_weight: double
            The weight of the query for which the metric is evaluated.
        '''
        cdef:
            INT_t i, n_documents, cutoff
            DOUBLE_t result

        if scale_value == 0.0:
            return 0.0

        with nogil:
            n_documents = ranking.shape[0]
            cutoff = n_documents if self.cutoff < 0 else self.cutoff
            result = 0.0
            for i in range(imin(cutoff, n_documents)):
                result += self.gain_cache[relevance_scores[ranking[i]]] / self.discount_cache[i]
            result /= scale_value
            result *= query_weight

        return result


    cpdef evaluate(self, INT_t[::1] ranked_relevance_scores, DOUBLE_t scale_value, DOUBLE_t query_weight):
        ''' 
        Evaluate the metric on the specified ranked list of document relevance scores.

        Parameters:
        -----------
        ranked_relevance_scores: array of integers, shape = (n_documents,)
            Specify list of relevance scores.

        scale_value: double
            Should be 1.0.

        query_weight: double
            The weight of the query for which the metric is evaluated.
        '''
        cdef:
            INT_t i, n_documents, cutoff
            DOUBLE_t result

        if scale_value == 0.0:
            return 0.0

        with nogil:
            n_documents = ranked_relevance_scores.shape[0]
            cutoff = n_documents if self.cutoff < 0 else self.cutoff
            result = 0.0
            for i in range(imin(cutoff, n_documents)):
                result += self.gain_cache[ranked_relevance_scores[i]] / self.discount_cache[i]
            result /= scale_value
            result *= query_weight

        return result


    cpdef evaluate_queries(self, INT_t[::1] query_indptr, INT_t[::1] relevance_scores, DOUBLE_t[::1] ranking_scores,
                           DOUBLE_t[::1] scale_values, DOUBLE_t[::1] query_weights, DOUBLE_t[::1] out):
        ''' 
        Evaluate the DCG metric on the specified queries. The relevance scores and
        ranking scores of documents, which belong to query `i`, are in
        `relevance_scores[query_indptr[i]:query_indptr[i + 1]]` and
        `ranking_scores[query_indptr[i]:query_indptr[i + 1]]`, respectively.

        Parameters:
        -----------
        query_indptr: array of integers, shape = (n_queries + 1,)
            The query index pointer.

        relevance_scores, array of integers, shape = (n_documents,)
            Specify the relevance score for each document.

        ranking_scores: array, shape=(n_queries,)
            Specify the ranking score for each document.

        scale_values: array, shape=(n_queries,), optional
            Should be None (defaults to all 1s).

        query_weights: array of doubles, shape = (n_queries,), optional (default is None)
            The weight given to each query.

        out: array, shape=(n_documents,), optional
            If not None, it will be filled with the metric value
            for each individual query.
        '''
        cdef:
            INT_t i, j, n_queries, n_documents
            INT_t *ranked_relevance_scores
            DOUBLE_t result, qresult, query_weights_sum

        with nogil:
            n_queries = query_indptr.shape[0] - 1
            n_documents = relevance_scores.shape[0]
            query_weights_sum = 0.0
            
            ranked_relevance_scores = <INT_t*> calloc(n_documents, sizeof(INT_t))

            ranksort_relevance_scores_queries_c(&query_indptr[0], n_queries, &ranking_scores[0], &relevance_scores[0], ranked_relevance_scores, &self.seed)

            result = 0.0

            for i in range(n_queries):
                n_documents = query_indptr[i + 1] - query_indptr[i]
                cutoff = n_documents if self.cutoff < 0 else imin(self.cutoff, n_documents)

                qresult = 0.0
                for j in range(cutoff):
                    qresult += self.gain_cache[ranked_relevance_scores[query_indptr[i] + j]] / self.discount_cache[j]

                if query_weights is not None:
                    qresult *= query_weights[i]
                    query_weights_sum += query_weights[i]

                if scale_values is not None:
                    if scale_values[i] == 0.0:
                        qresult = 0.0
                    else:
                        qresult /= scale_values[i]

                if out is not None:
                    out[i] = qresult

                result += qresult

            if query_weights is None:
                result /= n_queries
            else:
                result /= query_weights_sum

            free(ranked_relevance_scores)
            
        return result


    cpdef delta(self, INT_t i, INT_t offset, INT_t[::1] document_ranks,
                INT_t[::1] relevance_scores, DOUBLE_t scale_value,
                DOUBLE_t[::1] out):
        ''' 
        Compute the change in the metric caused by swapping document `i` with every
        document `offset`, `offset + 1`, ... (in turn)

        The relevance score and document rank of document `i` is `relevance_scores[i]`
        and `document_ranks[i]`, respectively.

        Parameters:
        -----------
        i: integer
            The index of the one document that is being swapped with all
            the others.

        offset: integer
            The start index of the sequence of documents that are
            being swapped.

        document_ranks: array of integers
            Specify the rank for each document.

        relevance_scores: array of integers
            Specify the relevance score for each document.

        scale_value: double, shape = (n_documents,)
            Should be 1.0.

        out: array of doubles
            The output array. The array size is expected to be at least as big
            as the the number of document pairs being swapped, which should be
            `len(document_ranks) - offset`.
        '''
        with nogil:
            self.delta_c(i, offset, document_ranks.shape[0],
                         &document_ranks[0], &relevance_scores[0],
                         scale_value, &out[0])


    cpdef evaluate_queries_ideal(self, INT_t[::1] query_indptr, INT_t[::1] relevance_scores,
                                 DOUBLE_t[::1] ideal_values):
        ''' 
        Compute the ideal DCG metric value for every one of the specified queries.

        The relevance scores of documents, which belong to query `i`, must be
        stored in `relevance_scores[query_indptr[i]:query_indptr[i + 1]]` in
        descending order.

        Parameters:
        -----------
        query_indptr: array of integers, shape = (n_queries + 1,)
            The query index pointer.

        relevance_scores, array of integers, shape = (n_documents,)
            Specify the relevance score for each document. It is expected
            that these values are sorted in descending order.

        ideal_values: output array of doubles, shape=(n_queries,)
            Output array for the ideal metric value of each query.
        '''
        cdef INT_t i, j, n_documents, cutoff

        with nogil:
            for i in range(query_indptr.shape[0] - 1):
                ideal_values[i] = 0.0
                n_documents = query_indptr[i + 1] - query_indptr[i]
                cutoff = n_documents if self.cutoff < 0 else imin(self.cutoff, n_documents)
                for j in range(cutoff):
                    ideal_values[i] += self.gain_cache[relevance_scores[query_indptr[i] + j]] / self.discount_cache[j]


    cdef void delta_c(self, INT_t i, INT_t offset, INT_t n_documents, INT_t *document_ranks, INT_t *relevance_scores,
                      DOUBLE_t scale_value, DOUBLE_t *out) nogil:
        ''' 
        See description of self.delta(...) method.
        '''
        cdef:
            INT_t j, n_swapped_document_pairs, cutoff
            DOUBLE_t i_relevance_score, i_position_discount
            bint i_above_cutoff, j_above_cutoff

        n_swapped_document_pairs = n_documents - offset
        cutoff  = n_documents if self.cutoff < 0 else self.cutoff

        i_relevance_score = self.gain_cache[relevance_scores[i]]
        i_position_discount = self.discount_cache[document_ranks[i]]
        i_above_cutoff = (document_ranks[i] < cutoff)

        # Does the document 'i' influences the evaluation (at all)?
        if i_above_cutoff:
            for j in range(n_swapped_document_pairs):
                out[j] = -i_relevance_score / i_position_discount
        else:
            for j in range(n_swapped_document_pairs):
                out[j] = 0.0

        for j in range(offset, n_documents):
            j_above_cutoff = (document_ranks[j] < cutoff)

            if j_above_cutoff:
                out[j - offset] += (i_relevance_score - self.gain_cache[relevance_scores[j]]) / self.discount_cache[document_ranks[j]]

            if i_above_cutoff:
                out[j - offset] += self.gain_cache[relevance_scores[j]] / i_position_discount

        if scale_value != 1.0:
            if scale_value == 0.0:
                for j in range(n_swapped_document_pairs):
                    out[j] = 0.0
            else:
                for j in range(n_swapped_document_pairs):
                    out[j] = fabs(out[j] / scale_value)
        else:
            for j in range(n_swapped_document_pairs):
                out[j] = fabs(out[j])


# =============================================================================
# Winner Takes All Metric
# =============================================================================


cdef class WinnerTakesAll(Metric):
    ''' 
    Winner Takes All (WTA) metric.

    It is assumed that the relevance scores of the documents are binary,
    i.e. values of 0 and 1. If you want to use this metric on multi-labeled
    relevances, it is recommended to set query weights to reciprocal of
    the maximum relevance score of a corresponding query documents.
    '''

    def __cinit__(self, INT_t cutoff, INT_t maximum_relevance,
                  INT_t maximum_documents, unsigned int seed):
        ''' 
        Initialize the metric with the specified cutoff threshold.
        maximum relevance score a document can have, and the maximum
        number of documents per query is ignored.

        cutoff: integer
            Ignored, but must not be 0.

        maximum_relevance: integer
            Ignored.

        maximum_documents: integer
            Ignored.
        '''
        pass


    def __reduce__(self):
        ''' 
        Reduce reimplementation, for pickling.
        '''
        return (WinnerTakesAll, (-1, 0, 0))


    cpdef evaluate_ranking(self, INT_t[::1] ranking, INT_t[::1] relevance_scores, DOUBLE_t scale_value, DOUBLE_t query_weight):
        ''' 
        Evaluate the metric on the specified document ranking.

        Parameters:
        -----------
        ranking: array of integers, shape = (n_documents,)
            Specify the list of ranked documents.

        relevance_scores: array of integers, shape = (n_documents,)
            Specify the relevance score for each document.

        scale_value: double
            Ignored.

        query_weight: double
            The weight of the query for which the metric is evaluated.
        '''
        cdef DOUBLE_t result
        with nogil:
            result = query_weight * relevance_scores[ranking[0]]
        return result


    cpdef evaluate(self, INT_t[::1] ranked_relevance_scores, DOUBLE_t scale_value, DOUBLE_t query_weight):
        ''' 
        Evaluate the metric on the specified ranked list of document relevance scores.

        Parameters:
        -----------
        ranked_relevance_scores: array of integers, shape = (n_documents,)
            Specify list of relevance scores.

        scale_value: double
           Ignored.

        query_weight: double
            The weight of the query for which the metric is evaluated.
        '''
        cdef DOUBLE_t result
        with nogil:
            result = query_weight * ranked_relevance_scores[0]
        return result


    cpdef evaluate_queries(self, INT_t[::1] query_indptr, INT_t[::1] relevance_scores, DOUBLE_t[::1] ranking_scores,
                           DOUBLE_t[::1] scale_values, DOUBLE_t[::1] query_weights, DOUBLE_t[::1] out):
        ''' 
        Evaluate the WTA metric on the specified queries. The relevance scores and
        ranking scores of documents, which belong to query `i`, are in
        `relevance_scores[query_indptr[i]:query_indptr[i + 1]]` and
        `ranking_scores[query_indptr[i]:query_indptr[i + 1]]`, respectively.

        Parameters:
        -----------
        query_indptr: array of integers, shape = (n_queries + 1,)
            The query index pointer.

        relevance_scores, array of integers, shape = (n_queries + 1,)
            Specify the relevance score for each document.

        ranking_scores: array, shape=(n_queries,)
            Specify the ranking score for each document.

        scale_values: array, shape=(n_queries,), optional
            Ignored.

        query_weights: array of doubles, shape = (n_queries,), optional (default is None)
            The weight given to each query.

        out: array, shape=(n_documents,), optional
            If not None, it will be filled with the metric value
            for each individual query.
        '''
        cdef:
            INT_t i, n_queries, n_documents
            INT_t *ranked_relevance_scores
            DOUBLE_t result, qresult, query_weights_sum

        with nogil:
            n_queries = query_indptr.shape[0] - 1
            n_documents = relevance_scores.shape[0]
            query_weights_sum = 0.0
            
            ranked_relevance_scores = <INT_t*> calloc(n_documents, sizeof(INT_t))

            ranksort_relevance_scores_queries_c(&query_indptr[0], n_queries, &ranking_scores[0], &relevance_scores[0], ranked_relevance_scores, &self.seed)

            result = 0.0

            for i in range(n_queries):
                if query_weights is None:
                    qresult = 1.0
                else:
                    qresult = query_weights[i]
                    query_weights_sum += query_weights[i]

                qresult *= ranked_relevance_scores[query_indptr[i]]

                if out is not None:
                    out[i] = qresult

                result += qresult

            if query_weights is None:
                result /= n_queries
            else:
                result /= query_weights_sum

            free(ranked_relevance_scores)
            
        return result


    cpdef delta(self, INT_t i, INT_t offset, INT_t[::1] document_ranks, INT_t[::1] relevance_scores,
                DOUBLE_t scale_value, DOUBLE_t[::1] out):
        ''' 
        Compute the change in the metric caused by swapping document `i` with every
        document `offset`, `offset + 1`, ... (in turn)

        The relevance score and document rank of document `i` is `relevance_scores[i]`
        and `document_ranks[i]`, respectively.

        Parameters:
        -----------
        i: integer
            The index of the one document that is being swapped with all
            the others.

        offset: integer
            The start index of the sequence of documents that are
            being swapped.

        document_ranks: array of integers
            Specify the rank for each document.

        relevance_scores: array of integers
            Specify the relevance score for each document.

        scale_value: double, shape = (n_documents,)
            Ignored.

        out: array of doubles
            The output array. The array size is expected to be at least as big
            as the the number of document pairs being swapped, which should be
            `len(document_ranks) - offset`.
        '''
        with nogil:
            self.delta_c(i, offset, document_ranks.shape[0],
                         &document_ranks[0], &relevance_scores[0],
                         scale_value, &out[0])


    cpdef evaluate_queries_ideal(self, INT_t[::1] query_indptr, INT_t[::1] relevance_scores,
                                 DOUBLE_t[::1] ideal_values):
        ''' 
        Compute the ideal WTA metric value for every one of the specified queries.

        The relevance scores of documents, which belong to query `i`, must be
        stored in `relevance_scores[query_indptr[i]:query_indptr[i + 1]]` in
        descending order.

        Parameters:
        -----------
        query_indptr: array of integers, shape = (n_queries + 1,)
            The query index pointer.

        relevance_scores, array of integers, shape = (n_documents,)
            Specify the relevance score for each document. It is expected
            that these values are sorted in descending order.

        ideal_values: output array of doubles, shape=(n_queries,)
            Output array for the ideal metric value of each query.
        '''
        cdef INT_t i

        with nogil:
            for i in range(query_indptr.shape[0] - 1):
                ideal_values[i] = relevance_scores[query_indptr[i]]


    cdef void delta_c(self, INT_t i, INT_t offset, INT_t n_documents, INT_t *document_ranks, INT_t *relevance_scores,
                      DOUBLE_t scale_value, DOUBLE_t *out) nogil:
        ''' 
        See description of self.delta(...) method.
        '''
        cdef INT_t j

        if document_ranks[i] == 0:
            for j in range(offset, n_documents):
                out[j - offset] = fabs(relevance_scores[j] - relevance_scores[i])
        else:
            for j in range(offset, n_documents):
                if document_ranks[j] == 0:
                    out[j - offset] = fabs(relevance_scores[j] - relevance_scores[i])
                else:
                    out[j - offset] = 0.0


# =============================================================================
# Kendall Tau Distance
# =============================================================================

cdef class KendallTau:
    cdef INT_t    *mapping        # Buffer for remapping document IDs to 0, 1, 2, ...
    cdef DOUBLE_t *fenwick        # Fenwick tree for fast computation of weighted inversions.
    cdef DOUBLE_t *weights        # The position weights.
    cdef int       size           # The size of the internal arrays.
    cdef object    weights_func   # The Python function computing the weight of a given position.
    
    def __cinit__(self, weights, capacity=1024):
        ''' 
        Creates Kendall Tau distance metric.
        
        Parameters
        ----------
        weights : function
            A non-decreasing function of one integer 
            parameter `i`, which returns the weight
            of a document at position `i` (0-based).
            It has to return a single float number.
            
            Consider this DCG discount weights,
            for example:
            
                weights(i): -1 / log2(i + 2)
            
            Remember that the parameter i is 0-based!

        capacity: int, optional (default is 1024)
            The initial capacity of the array for precomputed
            weight values.
        '''
        self.mapping = NULL
        self.fenwick = NULL
        self.weights = NULL
        self.size = 0
        self.weights_func = weights
        # Initialize the internal arrays.
        self.inflate_arrays()


    def __dealloc__(self):
        ''' 
        Free the allocated memory for internal arrays.
        '''
        free(self.mapping)
        free(self.fenwick)
        free(self.weights)


    def __reduce__(self):
        return (KendallTau, (self.weights_func,), self.__getstate__())


    def __getstate__(self):
        return {}


    def __setstate__(self, d):
        pass
        

    cdef int inflate_arrays(self, capacity=-1):
        ''' 
        Increase the capacity of the internal arrays to
        the given capacity.
        
        As the name of the function suggests, if `capacity`
        is smaller than the current capacity of the internal
        arrays, nothing happens.
        
        Parameters
        ----------
        capacity : int, optional (default is -1)
            The new capacity of the internal arrays. If -1
            is given the capacity of the internal arrays 
            will be doubled.

        Returns
        -------
        code: int
            -1 on failure, 0 on success.
        '''
        cdef int i
        cdef void * ptr
        
        if capacity <= self.size and self.mapping != NULL:
            return 0
        
        if capacity <= -1:
            if self.size == 0:
                # Initial capacity.
                capacity = 1024
            else:
                # Double the current capacity. 
                capacity = 2 * self.size
        
        # Because documents not appearing in both lists
        # are treated as if they were sitting at the 
        # first position following the end of the lists.
        capacity += 1
                
        # Allocate mapping array.
        #########################
        ptr = realloc(self.mapping, capacity * sizeof(INT_t))
        
        if ptr == NULL:
            return -1
        
        self.mapping = <INT_t *> ptr
        
        # Initialize the new elements to -1.
        memset(<void *>(self.mapping + self.size), -1,
               (capacity - self.size) * sizeof(INT_t))
        
        # Allocate fenwick array.
        #########################
        ptr = realloc(self.fenwick, capacity * sizeof(DOUBLE_t))
        
        if ptr == NULL:
            return -1
        
        self.fenwick = <DOUBLE_t *> ptr
        
        # Initialize the new elements to 0.
        memset(<void *>(self.fenwick + self.size), 0,
               (capacity - self.size) * sizeof(DOUBLE_t))
        
        # Allocate weights array.
        #########################
        ptr = realloc(self.weights, capacity * sizeof(DOUBLE_t))
        
        if ptr == NULL:
            return -1
        
        self.weights = <DOUBLE_t *> ptr
        
        # Initialize the values of new weights using `self.weights_func`.
        for i in range(self.size, capacity):
            self.weights[i] = self.weights_func(i)
        
        self.size = capacity
        return 0
    
    
    def evaluate(self, X, check_input=True):
        ''' 
        Computes the Kendall Tau distance between the given
        list X and its ascendingly sorted version.
        '''
        cdef int size
        cdef DOUBLE_t tau
        
        if check_input:
            if not isinstance(X, np.ndarray):
                X = np.array(X, dtype='int32', order='C')

            if X.ndim != 1:
                raise ValueError('X is not one dimensional.')

            if X.dtype != np.int32 or not X.flags.c_contiguous:
                X = np.ascontiguousarray(X, dtype='int32')

        cdef np.ndarray[INT_t, ndim=1] Y = np.sort(X)

        # +1 in case of 0-based permutations.
        size = max(max(X), max(Y)) + 1
        
        # This may cause trouble for huge document IDs!
        if self.inflate_arrays(size) != 0:
            raise MemoryError('Cannot allocate %d bytes for internal arrays.'
                              % (sizeof(DOUBLE_t) * size))
        
        cdef INT_t *x =  <INT_t *> np.PyArray_DATA(X)
        cdef INT_t *y =  <INT_t *> np.PyArray_DATA(Y)
        
        size = min(X.shape[0], Y.shape[0])
        
        with nogil:
            tau = self.kendall_tau(x, y, size)

        return tau


    def distance(self, X, Y, check_input=True):
        ''' 
        Computes the Kendall Tau distance between the given
        lists X and Y.

        X and Y does not necessarily need to contain the same
        set of numbers. In case the numbers differ it is assumed
        that the lists are prefixes of longer lists, which
        were cutoff. In that matter, the lists does not even
        have to be of the same length, if that is the case,
        the minimum length of the two lists is considered.

        If `check_input` is True, X and Y can be lists/iterables
        of integer numbers, these arrays will be converted to
        numpy arrays with `numpy.int32` dtype.

        If `check_input` is False you need to make sure that
        X and Y are numpy arrays with `numpy.int32` dtype,
        unless you want to suffer severe consequences.
        '''
        cdef int size
        cdef DOUBLE_t tau
        
        if check_input:
            if not isinstance(X, np.ndarray):
                X = np.array(X, dtype='int32', order='C')

            if X.ndim != 1:
                raise ValueError('X is not one dimensional.')

            if X.dtype != np.int32 or not X.flags.c_contiguous:
                X = np.ascontiguousarray(X, dtype='int32')

            if not isinstance(Y, np.ndarray):
                Y = np.array(Y, dtype='int32', order='C')

            if Y.ndim != 1:
                raise ValueError('Y is not one dimensional.')

            if Y.dtype != np.int32 or not Y.flags.c_contiguous:
                Y = np.ascontiguousarray(Y, dtype='int32')
        
        # +1 in case of 0-based permutations.
        size = max(max(X), max(Y)) + 1
        
        # This may cause trouble for huge document IDs!
        if self.inflate_arrays(size) != 0:
            raise MemoryError('Cannot allocate %d bytes for internal arrays.'
                              % (sizeof(DOUBLE_t) * size))
        
        cdef INT_t *x =  <INT_t *> np.PyArray_DATA(X)
        cdef INT_t *y =  <INT_t *> np.PyArray_DATA(Y)
        
        size = min(X.shape[0], Y.shape[0])
        
        with nogil:
            tau = self.kendall_tau(x, y, size)

        return tau


    cdef DOUBLE_t kendall_tau(self, INT_t *X, INT_t *Y, int size) nogil:
        return self.kendall_tau_fenwick(X, Y, size)


    cdef inline DOUBLE_t kendall_tau_array(self, INT_t *X, INT_t *Y, int size) nogil:
        ''' 
        Computes Kendall Tau distance between X and Y using a simple array.
        This variant should be prefarable in case of short lists.
        '''
        cdef int i, j
        cdef double tau = 0.0
        
        for i in range(size):
            self.mapping[X[i]] = i
        
        # Process documents of Y.
        for j in range(size):
            i = self.mapping[Y[j]]
            # The document in Y that is not in X is treated
            # as if it was the first document following the
            # end of list X.
            tau += self._update_array(i if i >= 0 else size, j, size)
            if i >= 0:
                # Offset documents that appear in both lists.
                # This becomes useful for finding documents
                # that appeared only in X (see below).
                self.mapping[Y[j]] += size
    
        # Process documents of X that does not appear in Y.
        for j in range(size):
            i = self.mapping[X[j]]
            # j >= size ==> X[i] is in Y, we need to
            # clear it from the array such that it
            # will not interfere with calculation
            # of inversions for X[i]'s that are not
            # in Y.
            if i >= size:
                self._restore_array(i - size, size)
                # Offset the documents back again
                # for restoring the arrays.
                self.mapping[X[j]] -= size
            else:
                tau += self._get_array(i, size)
    
        # Restore the internal arrays.        
        for j in range(size):
            i = self.mapping[Y[j]]
            # Restore the array for documents appearing
            # only in Y. These documents are put to the
            # same position, hence the restoration can
            # be called only once.
            if i < 0:
                self._restore_array(size, size)
                break
        
        # Finish the restoration of the arrays
        # by clearing the mapping.
        for i in range(size):
            self.mapping[X[i]] = -1

        return tau

        
    cdef inline DOUBLE_t _update_array(self, int i, int sigma, int size) nogil:
        ''' 
        Add a document at position `i` and `sigma` in respective lists
        X and Y into an array and compute the weighted number of
        inversions the document is with all previously added documents.
        
        Parameters
        ----------
        i : int
            The position of the document in X, or -1
            if it is not there.
            
        sigma : int
            The position of the document in Y.
        
        size: int
            The length of the document lists.
        
        Return
        ------
        tau: float
            The weighted number of inversions of the document
            with all the previously processed documents.
        '''
        cdef DOUBLE_t weight, tau = 0.0           
                    
        if i == sigma:
            weight = 1.0 # No displacement.
        else:
            # The weight of "bubbling" document from position
            # i to position sigma.
            weight = self.weights[i] - self.weights[sigma]
            # The average weight (denominator makes the weight
            # always positive).
            weight /= i - sigma
            
        sigma = size - i
        
        # Update the array.
        self.fenwick[sigma] += weight
        
        # Compute the weighted number of inversions of
        # the current document with all the documents
        # inserted before it.
        sigma -= 1
        while sigma >= 0:
            tau += self.fenwick[sigma] * weight
            sigma -= 1

        return tau
    
    
    cdef inline DOUBLE_t _get_array(self, int i, int size) nogil:
        ''' 
        Return the weighted number of invertions for i-th document
        of X, which do not appear in list Y.
        '''
        cdef DOUBLE_t weight, tau = 0.0
    
        # The weight of "bubbling" document
        # from position i to the first position
        # beyond the end of the list.
        weight = self.weights[i] - self.weights[size]
        
        # The average weight (denominator makes
        # the weight always positive).
        weight /= i - size
        
        # Compute the weighted number of inversions of
        # the current document with all the documents
        # inserted before it.
        while size >= 0:
            tau += self.fenwick[size] * weight
            size -= 1

        return tau


    cdef inline void _restore_array(self, int i, int size) nogil:
        ''' 
        Remove the weights at position `size - i` from the array.
        '''
        self.fenwick[size - i] = 0.0


    cdef inline DOUBLE_t kendall_tau_fenwick(self, INT_t *X, INT_t *Y, int size) nogil:
        ''' 
        Computes Kendall Tau distance between X and Y using a simple array.
        This variant should be prefarable in case of short lists.
        '''
        cdef int i, j
        cdef double tau = 0.0
        
        for i in range(size):
            self.mapping[X[i]] = i
        
        # Process documents of Y.
        for j in range(size):
            i = self.mapping[Y[j]]
            # The document in Y that is not in X is treated
            # as if it was the first document following the
            # end of list X.
            tau += self._update_fenwick(i if i >= 0 else size, j, size)
            if i >= 0:
                # Offset documents that appear in both lists.
                # This becomes useful for finding documents
                # that appeared only in X (see below).
                self.mapping[Y[j]] += size
    
        # Process documents of X that does not appear in Y.
        for j in range(size):
            i = self.mapping[X[j]]
            # j >= size ==> X[i] is in Y, we need to
            # clear it from the array such that it
            # will not interfere with calculation
            # of inversions for X[i]'s that are not
            # in Y.
            if i >= size:
                self._restore_fenwick(i - size, size)
                # Offset the documents back again
                # for restoring the arrays.
                self.mapping[X[j]] -= size
            else:
                tau += self._get_fenwick(i, size)
    
        # Restore the internal arrays.        
        for j in range(size):
            i = self.mapping[Y[j]]
            # Restore the array for documents appearing
            # only in Y. These documents are put to the
            # same position, hence the restoration can
            # be called only once.
            if i < 0:
                self._restore_fenwick(size, size)
                break
        
        # Finish the restoration of the arrays
        # by clearing the mapping.
        for i in range(size):
            self.mapping[X[i]] = -1

        return tau


    cdef inline DOUBLE_t _update_fenwick(self, int i, int sigma, int size) nogil:
        ''' 
        Insert the weight of a document with displacement |i - sigma|
        into the Fenwick tree and compute the weighted number of invertions
        the document is in with all previously inserted documents.
        '''
        cdef DOUBLE_t weight, tau = 0.0           
                    
        if i == sigma:
            weight = 1.0 # No displacement.
        else:
            # The weight of "bubbling" document from position
            # i to position sigma.
            weight = self.weights[i] - self.weights[sigma]
            # The average weight (denominator makes the weight
            # always positive).
            weight /= i - sigma
            
        sigma = size - i
        
        if sigma != 0:
            tau += self.fenwick[0] * weight

        # Compute the weighted number of inversions of
        # the current document with all the documents
        # inserted before it.
        while sigma > 0:
            tau += self.fenwick[sigma] * weight
            sigma -= sigma & -sigma

        # Invert the indexing.
        sigma = size - i

        # Update the Fenwick tree.
        if sigma == 0:
            # Document below cutoff.
            self.fenwick[0] += weight
        else:
            # Update the Fenwick tree.
            while sigma <= size:
                self.fenwick[sigma] += weight
                sigma += sigma & -sigma

        return tau
    
    
    cdef inline DOUBLE_t _get_fenwick(self, int i, int size) nogil:
        ''' 
        Return the weighted number of invertions for i-th document
        of X, which do not appear in list Y.
        '''
        cdef DOUBLE_t weight, tau = 0.0
    
        # The weight of "bubbling" document
        # from position i to the first position
        # beyond the end of the list.
        weight = self.weights[i] - self.weights[size]
        
        # The average weight (denominator makes
        # the weight always positive).
        weight /= i - size
        
        # Compute the weighted number of inversions of
        # the current document with all the documents
        # inserted before it.
        while size > 0:
            tau += self.fenwick[size] * weight
            size -= size & -size
            
        tau += self.fenwick[0] * weight

        return tau


    cdef inline void _restore_fenwick(self, int i, int size) nogil:
        ''' 
        Remove the weight at position `size - i` from the Fenwick tree.
        '''
        cdef int j, k
        cdef DOUBLE_t weight
        
        # Invert the indexing.
        k = size - i
        
        # Document below cutoff.
        if k == 0:
            self.fenwick[k] = 0.0
        else:
            # Need to find the weight of the document first.
            weight = self.fenwick[k]
            
            j = k - (k & -k)
            k -= 1

            while k > j:
                weight -= self.fenwick[k]
                k -= k & -k
                
            # Remove the weight from the Fenwick tree.
            i = size - i
            while i <= size:
                self.fenwick[i] -= weight
                i += i & -i
