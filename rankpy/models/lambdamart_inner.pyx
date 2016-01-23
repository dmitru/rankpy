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

from cython cimport view

from libc.stdlib cimport malloc, calloc, free

from libc.string cimport memset

from libc.math cimport exp, log

from ..metrics._utils cimport INT_t
from ..metrics._utils cimport DOUBLE_t
from ..metrics._utils cimport get_seed
from ..metrics._utils cimport argranksort_queries_c
from ..metrics._utils cimport relevance_argsort_v1_c

from ..metrics._metrics cimport Metric

cdef DOUBLE_t EPSILON = np.finfo('d').eps


def parallel_compute_lambdas_and_weights(INT_t qstart,
                                         INT_t qend,
                                         INT_t[::1] query_indptr,
                                         DOUBLE_t[::1] ranking_scores,
                                         INT_t[::1] relevance_scores,
                                         INT_t maximum_relevance,
                                         INT_t[:, ::1] relevance_strides,
                                         Metric metric,
                                         DOUBLE_t[::1] scale_values,
                                         DOUBLE_t[:, ::1] influences,
                                         INT_t[::1] leaves_idx,
                                         DOUBLE_t[::1] query_weights,
                                         DOUBLE_t[::1] document_weights,
                                         DOUBLE_t[::1] output_lambdas,
                                         DOUBLE_t[::1] output_weights,
                                         object random_state=None):
    '''
    Helper function computing pseudo-responses (`lambdas`) and 'optimal'
    gradient steps (`weights`) for the documents belonging to the specified
    queries. This method is suitable for doing some heavy computation
    in a multithreading backend (releases GIL).

    Parameters:
    -----------
    qstart: integer
        Start index of the query for which documents the lambas
        and weights will be computed.

    qend: integer
        End index of the query (eclusive) for which documents
        the lambas and weights will be computed.

    query_indptr: array, shape = (n_queries + 1)
        The query index pointer array.

    ranking_scores: array, shape = (n_documents,)
        The ranking scores of the documents.

    relevance_scores: array, shape = (n_documents,)
        The relevance scores of the documents.

    maximum_relevance: int
        The maximum relevance score.

    relevance_strides: array, shape = (n_query, maximum_relevance + 1)
        The array of index pointers for each query. `relevance_strides[i, s]`
        contains the index of the document (with respect to query `i`)
        which has lower relevance than `s`.

    metric: Metric cython backend
        The evaluation metric, for which the lambdas and weights
        are to be computed.

    scale_values: array, shape=(n_queries,), optional (default is None)
        The precomputed metric scale value for every query.

    influences: array, shape=(maximum_relevance + 1, maximum_relevance + 1)
                or None
        Used to keep track of (proportional) contribution from lambdas
        (force interpretation) of low relevant documents.

    leaves_idx: array, shape=(n_documents, n_leaf_nodes) or None:
        The indices of terminal nodes which the documents fall into.
        This parameter can be used to recompute lambdas and weights
        after regression tree is built.

    output_lambdas: array, shape=(n_documents,)
        Computed lambdas for every document.

    output_weights: array, shape=(n_documents,)
        Computed weights for every document.

    random_state: RandomState instance
        Random number generator used for shuffling of documents
        with the same ranking score.

    Returns
    -------
    loss: float
        The LambdaMART loss of the rankings induced by the specified
        ranking scores.
    '''
    cdef unsigned int seed = get_seed(random_state)

    # with nogil seqfaults here... do not know why :-/.
    # Go straight to C for fancy pointer indexing...
    return parallel_compute_lambdas_and_weights_c(
            qstart, qend, &query_indptr[0], &ranking_scores[0],
            &relevance_scores[0], maximum_relevance,
            NULL if relevance_strides is None else &relevance_strides[0, 0],
            metric,
            NULL if scale_values is None else &scale_values[0],
            NULL if influences is None else &influences[0, 0],
            NULL if leaves_idx is None else &leaves_idx[0],
            NULL if query_weights is None else &query_weights[0],
            NULL if document_weights is None else &document_weights[0],
            &output_lambdas[0], &output_weights[0], seed)


cdef DOUBLE_t parallel_compute_lambdas_and_weights_c(
                INT_t qstart, INT_t qend, INT_t *query_indptr,
                DOUBLE_t *ranking_scores, INT_t *relevance_scores,
                INT_t maximum_relevance, INT_t *relevance_strides,
                Metric metric, DOUBLE_t *scale_values, DOUBLE_t *influences,
                INT_t *leaves_idx, DOUBLE_t *query_weights,
                DOUBLE_t *document_weights, DOUBLE_t *output_lambdas,
                DOUBLE_t *output_weights,
                unsigned int seed) nogil:
    '''
    The guts of `parallel_compute_lambdas_and_weights`.
    '''
    cdef:
        INT_t i, j, k, start, rstart, end, n_documents, j_relevance_score
        INT_t *sort_indices = NULL
        INT_t *document_ranks = NULL
        DOUBLE_t *document_deltas = NULL
        DOUBLE_t *influence_by_relevance = NULL
        DOUBLE_t j_push_down, lambda_, weight, rho, scale
        DOUBLE_t query_weight, j_document_weight, document_pair_weight
        DOUBLE_t loss = 0.0
        bint resort = False

    with nogil:
        # Total number of documents to process.
        n_documents = query_indptr[qend] - query_indptr[qstart]

        # More than enough memory to hold what we want.
        document_ranks = <INT_t *> calloc(n_documents, sizeof(INT_t))
        document_deltas = <DOUBLE_t *> calloc(n_documents, sizeof(DOUBLE_t))

        # maximum_relevance + 1 is the row stride of
        # `influence_by_relevance` and `relevance_strides`.
        maximum_relevance += 1

        if influences != NULL:
            influence_by_relevance = <DOUBLE_t *> calloc(maximum_relevance,
                                                         sizeof(DOUBLE_t))

        # Relevance strides were not given, hence we need to build
        # them from the relevances.
        if relevance_strides == NULL:
            # Indicate that we need to resort the arrays back when we are done.
            resort = True

            # Create relevance_strides.
            relevance_strides = <INT_t *> calloc((qend - qstart) *
                                                 maximum_relevance,
                                                 sizeof(INT_t))

            # Allocate memory for sorting and resorting indices.
            sort_indices = <INT_t *> malloc(2 * n_documents * sizeof(INT_t))

            for i in range(qstart, qend):
                start, end = query_indptr[i], query_indptr[i + 1]

                # Make the indexing easier... maybe.
                sort_indices += start - query_indptr[qstart]
                relevance_scores += start
                relevance_strides += (i - qstart) * maximum_relevance
                ranking_scores += start

                if leaves_idx != NULL:
                    leaves_idx += start

                if document_weights != NULL:
                    document_weights += start

                # Get sorting indices (permutation) of the relevance
                # scores for query 'i'.
                relevance_argsort_v1_c(relevance_scores, sort_indices,
                                       end - start, maximum_relevance)

                # Get inverse sort indices.
                for j in range(end - start):
                    sort_indices[sort_indices[j] + n_documents] = j

                # Sort related arrays according to the query relevance scores.
                sort_in_place(sort_indices, end - start,
                              relevance_scores, ranking_scores,
                              leaves_idx, document_weights)

                # Build relevance_strides for query 'i'.
                for j in range(end - start):
                    relevance_strides[relevance_scores[j]] += 1

                # Offset the relevance_strides properly to make it look
                # like it came from the input parameters.
                k = start
                for j in range(maximum_relevance - 1, -1, -1):
                    if relevance_strides[j] == 0:
                        relevance_strides[j] = -1
                    else:
                        relevance_strides[j] += k
                        k = relevance_strides[j]

                # Revert back the offseting.
                sort_indices -= start - query_indptr[qstart]
                relevance_scores -= start
                relevance_strides -= (i - qstart) * maximum_relevance
                ranking_scores -= start

                if leaves_idx != NULL:
                    leaves_idx -= start

                if document_weights != NULL:
                    document_weights -= start

            # Need to offset the `relevance_strides`
            # to make the indexing work later.
            relevance_strides -= qstart * maximum_relevance

        # Find the rank of each document with respect to
        # the ranking scores over all queries.
        argranksort_queries_c(query_indptr + qstart,
                              qend - qstart,
                              ranking_scores,
                              document_ranks,
                              &seed)

        # Clear output array for lambdas since we will be incrementing.
        memset(output_lambdas + query_indptr[qstart], 0,
               n_documents * sizeof(DOUBLE_t))

        # Clear output array for weights since we will be incrementing.
        memset(output_weights + query_indptr[qstart], 0,
               n_documents * sizeof(DOUBLE_t))

        # Loop through the queries and compute lambdas
        # and weights for every document.
        for i in range(qstart, qend):
            # Get query weight (default is 1.0).
            query_weight = 1.0 if query_weights == NULL else query_weights[i]

            # Skip the queries with weight 0.
            if query_weight == 0.0:
                continue

            start, end = query_indptr[i], query_indptr[i + 1]

            scale = 1.0 if scale_values == NULL else scale_values[i]

            # The number of documents of the current query.
            n_documents = end - start

            # Loop through the documents of the current query.
            for j in range(start, end):
                j_relevance_score = relevance_scores[j]

                # Get the document 'j' weight (default is 1.0)
                if document_weights == NULL:
                    j_document_weight = 1.0
                else:
                    j_document_weight = document_weights[j]

                # Skip the documents with weight 0.0
                if j_document_weight == 0.0:
                    continue

                # The smallest index of a document with a lower
                # relevance score than document 'j'.
                rstart = relevance_strides[i * maximum_relevance +
                                           j_relevance_score]

                # Is there any document less relevant than document 'j'?
                if rstart >= end:
                    break

                # Compute the (absolute) changes in the metric caused
                # by swapping document 'j' with all documents 'k'
                # (k >= rstart), which have lower relevance with respect
                # to the query 'i'.
                metric.delta_c(j - start, rstart - start, n_documents,
                               document_ranks + start - query_indptr[qstart],
                               relevance_scores + start, scale, document_deltas)

                # Clear the influences for the current document.
                if influence_by_relevance != NULL:
                    memset(influence_by_relevance, 0,
                           j_relevance_score * sizeof(DOUBLE_t))

                # Current forces pushing document 'j' down.
                j_push_down = output_lambdas[j]

                for k in range(rstart, end):
                    if document_weights != NULL:
                        document_pair_weight = (query_weight *
                                                j_document_weight *
                                                document_weights[k])
                    else:
                        document_pair_weight = query_weight

                    if document_pair_weight == 0.0:
                        continue

                    rho = ((<DOUBLE_t> 1.0) /
                           ((<DOUBLE_t> 1.0) +
                            (<DOUBLE_t> exp(ranking_scores[j] -
                                            ranking_scores[k]))))

                    # Compute the loss for this pair of documents.
                    loss -= (document_deltas[k - rstart] *
                             document_pair_weight *
                             log(EPSILON if 1 - rho < EPSILON else 1 - rho))

                    # If the documents fall into the same terminal node of
                    # the regression tree, their contribution to the gradients
                    # are none.
                    if leaves_idx != NULL and leaves_idx[j] == leaves_idx[k]:
                        continue

                    lambda_ = (rho * document_pair_weight *
                               document_deltas[k - rstart])

                    weight = (1 - rho) * lambda_

                    output_lambdas[j] += lambda_
                    output_lambdas[k] -= lambda_

                    output_weights[j] += weight
                    output_weights[k] += weight

                    if influence_by_relevance != NULL:
                        influence_by_relevance[relevance_scores[k]] += lambda_

                if influence_by_relevance != NULL:
                    for k in range(j_relevance_score):
                        if influence_by_relevance[k] <= output_lambdas[j]:
                            influences[k * maximum_relevance + j_relevance_score] += influence_by_relevance[k] / output_lambdas[j]
                        influences[j_relevance_score * maximum_relevance + k] += influence_by_relevance[k] / (output_lambdas[j] - 2 * j_push_down)

        # `relevance_strides` array has been constructed here.
        # We need to resort all the arrays back and free the memory.
        if resort:
            # Total number of documents sorted.
            n_documents = query_indptr[qend] - query_indptr[qstart]

            # The inverse sort indices are in the second half of the array.
            sort_indices += n_documents

            for i in range(qstart, qend):
                start, end = query_indptr[i], query_indptr[i + 1]

                # Make the indexing easier... maybe.
                sort_indices += start - query_indptr[qstart]
                relevance_scores += start
                ranking_scores += start

                if leaves_idx != NULL:
                    leaves_idx += start

                if document_weights != NULL:
                    document_weights += start

                output_lambdas += start
                output_weights += start

                # Revert back the earlier sort of related arrays.
                sort_in_place(sort_indices, end - start, relevance_scores,
                              ranking_scores, leaves_idx, document_weights,
                              output_lambdas, output_weights)

                # Revert back the offseting.
                sort_indices -= start - query_indptr[qstart]
                relevance_scores -= start
                ranking_scores -= start

                if leaves_idx != NULL:
                    leaves_idx -= start

                if document_weights != NULL:
                    document_weights -= start

                output_lambdas -= start
                output_weights -= start

            # Offset `sort_indices` and `relevance_strides` back.
            relevance_strides += qstart * maximum_relevance
            sort_indices -= n_documents

            free(relevance_strides)
            free(sort_indices)

        free(document_ranks)
        free(document_deltas)
        free(influence_by_relevance)

    return loss


cdef void sort_in_place(INT_t *indices,
                        INT_t n_documents,
                        INT_t *relevance_scores,
                        DOUBLE_t *ranking_scores,
                        INT_t *leaves_idx,
                        DOUBLE_t *document_weights,
                        DOUBLE_t *lambdas=NULL,
                        DOUBLE_t *weights=NULL) nogil:
    '''
    Sort the given arrays according to `indices` in-place. Once done,
    indices will contain identity permutation.
    '''
    cdef INT_t start, end, tmp_relevance_score, tmp_leave_idx
    cdef DOUBLE_t tmp_ranking_score, tmp_document_weight
    cdef DOUBLE_t tmp_lambda, tmp_weight

    for i in range(n_documents):
        # Skipping fixed points (these elements are in the right place).
        if indices[i] != i:
            start = i

            # Temporarily store the items at the beginning
            # of the permutation cycle.
            tmp_relevance_score = relevance_scores[start]
            tmp_ranking_score = ranking_scores[start]

            if leaves_idx != NULL:
                tmp_leave_idx = leaves_idx[start]

            if document_weights != NULL:
                tmp_document_weight = document_weights[start]

            if lambdas != NULL:
                tmp_lambda = lambdas[start]
                tmp_weight = weights[start]

            # merry go round... ihaaa!
            while indices[start] != i:
                end = indices[start]

                relevance_scores[start] = relevance_scores[end]
                ranking_scores[start] = ranking_scores[end]

                if leaves_idx != NULL:
                    leaves_idx[start] = leaves_idx[end]

                if document_weights != NULL:
                    document_weights[start] = document_weights[end]

                if lambdas != NULL:
                    lambdas[start] = lambdas[end]
                    weights[start] = weights[end]

                indices[start] = start
                start = end

            # Move the items from the beginning of
            # the permutation cycle to the end.
            relevance_scores[end] = tmp_relevance_score
            ranking_scores[end] = tmp_ranking_score

            if leaves_idx != NULL:
                leaves_idx[end] = tmp_leave_idx

            if document_weights != NULL:
                document_weights[end] = tmp_document_weight

            if lambdas != NULL:
                lambdas[start] = tmp_lambda
                weights[start] = tmp_weight

            indices[end] = end
