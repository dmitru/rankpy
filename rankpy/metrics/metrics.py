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

from warnings import warn

from ._metrics import WinnerTakesAll as WTA
from ._metrics import DiscountedCumulativeGain as DCG

from ._utils import relevance_argsort_v1

from sklearn.utils import check_random_state


class WinnerTakesAll(object):
    ''' 
    Winner Takes All metric.

    Arguments:
    ----------
    cutoff: int, optional (default is -1)
        Ignored.

    max_relevance: int, optional (default is 4)
        Ignored.

    max_documents: int, optional (default is 8192):
        Ignored.

    queries: list of rankpy.queries.Queries
        Ignored.
    '''
    def __init__(self, cutoff=-1, max_relevance=None, max_documents=None,
                 queries=None, random_state=None):
        # Create the metric cython backend.
        self.random_state = check_random_state(random_state)
        self.metric_ = WTA(-1, 0, 0, self.random_state.randint(1, np.iinfo('i').max))

    def backend(self, copy=True):
        '''
        Returns the backend metric object.
        '''
        if copy:
            return WTA(-1, 0, 0, self.random_state.randint(1, np.iinfo('i').max))
        else:
            self.metric_

    def evaluate(self, ranking=None, labels=None, ranked_labels=None, scales=None, weight=1.0):
        ''' 
        Evaluate the WTA metric on the specified ranked list of document relevance scores.

        The function input can be either ranked list of relevance labels (`ranked_labels`),
        which is most convenient from the computational point of view, or it can be in
        the form of ranked list of documents (`ranking`) and corresponding relevance scores
        (`labels`), from which the ranked document relevance labels are computed.

        Parameters:
        -----------
        ranking: array, shape = (n_documents,)
            Specify list of ranked documents.

        labels: array: shape = (n_documents,)
            Specify relevance score for each document.

        ranked_labels: array, shape = (n_documents,)
            Relevance scores of the ranked documents. If not given, then
            `ranking` and `labels` must not be None, `ranked_labels` will
            be than inferred from them.

        scales: float, optional (default is None)
            Ignored.

        weight: float, optional (default is 1.0)
            The weight of the query for which the metric is evaluated.           
        '''
        if ranked_labels is not None:
            return self.metric_.evaluate(ranked_labels, 1.0, weight)
        elif ranking is not None and labels is not None:
            if ranking.shape[0] != labels.shape[0]:
                raise ValueError('number of ranked documents != number of relevance labels (%d, %d)' \
                                  % (ranking.shape[0], labels.shape[0]))
            return self.metric_.evaluate_ranking(ranking, labels, 1.0, weight)

    def evaluate_queries(self, queries, scores, scales=None, weights=None, out=None):
        ''' 
        Evaluate the WTA metric on the specified set of queries (`queries`). The documents
        are sorted by corresponding ranking scores (`scores`) and the metric is then
        computed as the average of the metric values evaluated on each query document
        list. The ties in ranking are broken probabilistically.
        
        Parameters:
        -----------
        queries: rankpy.queries.Queries
            The set of queries for which the metric will be computed.

        scores: array, shape=(n_documents,)
            The ranking scores for each document in the queries.

        scales: array, shape=(n_queries,), or None
            Ignored.

        weights: array of doubles, shape=(n_queries,), or None
            The weight of each query for which the metric is evaluated.

        out: array, shape=(n_documents,), or None
            If not None, it will be filled with the metric value
            for each individual query.
        '''
        if queries.document_count() != scores.shape[0]:
            raise ValueError('number of documents != number of scores (%d, %d)' \
                             % (queries.document_count(), scores.shape[0]))

        if weights is not None and queries.query_count() != weights.shape[0]:
            raise ValueError('number of queries != size of weights array (%d, %d)' \
                             % (queries.query_count(), out.shape[0]))

        if out is not None and queries.query_count() != out.shape[0]:
            raise ValueError('number of queries != size of output array (%d, %d)' \
                             % (queries.query_count(), out.shape[0]))

        return self.metric_.evaluate_queries(queries.query_indptr, queries.relevance_scores, scores, None, weights, out)

    def compute_delta(self, i, offset, document_ranks, relevance_scores, scales=None, out=None):
        ''' 
        Compute the change in the WTA metric after swapping document 'i' with
        each document in the document list starting at 'offset'.

        The relevance and rank of the document 'i' is 'relevance_scores[i]' and
        'document_ranks[i]', respectively.

        Similarly, 'relevance_scores[j]' and 'document_ranks[j]' for each j starting
        from 'offset' and ending at the end of the list denote the relevance score
        and rank of the document that will be swapped with document 'i'.

        Parameters:
        -----------
        i: int
            The index (zero-based) of the document that will appear in every pair
             of documents that will be swapped.

        offset: int
            The offset pointer to the start of the documents that will be swapped.

        document_ranks: array
            The ranks of the documents.

        relevance_scores: array
            The relevance scores of the documents.

        scales: float or None, optional (default is None)
            Ignored.

        out: array, optional (default is None)
            The output array. The array size is expected to be at least equal to
            the number of documents shorter by the ofset.
        '''
        n_documents = len(document_ranks)

        if out is None:
            out = np.empty(n_documents - offset, dtype=np.float64)

        if out.shape[0] < n_documents - offset:
            raise ValueError('output array is too small (%d < %d)' \
                             % (out.shape[0], n_documents - offset))

        if document_ranks.shape[0] != relevance_scores.shape[0]:
            raise ValueError('document ranks size != relevance scores (%d != %d)' \
                              % (document_ranks.shape[0], relevance_scores.shape[0]))

        self.metric_.delta(i, offset, document_ranks, relevance_scores, 1.0, out)

        return out

    def compute_scale(self, queries, relevance_scores=None):
        ''' 
        Since WTA is not normalized (or it can be said that it is already normal),
        return None.
        '''
        return None

    def __str__(self):
        ''' 
        Return the textual description of the metric.
        '''
        return 'WTA' if self.metric_.cutoff < 0 else 'WTA@%d' % self.metric_.cutoff


class DiscountedCumulativeGain(object):
    ''' 
    Discounted Cumulative Gain metric.

    Note that it is really recommended to use the optional parameter `queries`
    to pre-allocate the memory for gain and discount components of DCG metric
    for faster computation. Optionally, you can specify `max_relevance` and
    `max_documents`, but they should be inferred from the queries anyway.

    Ignoring the note above can lead only to 2 outcomes:
      1) Everything will be just fine.
      2) You will run into Segmentation Fault.

    Arguments:
    ----------
    cutoff: int, optional (default is -1)
        If positive, it denotes the maximum rank of a document
        that will be considered for evaluation.

    max_relevance: int, optional (default is 4)
        The maximum relevance score a document can have. This must be
        set for caching purposes. If the evaluated document list contain
        a document with higher relevance than the number specified,
        IndexError will be raised.

    max_documents: int, optional (default is 8192):
        The maximum number of documents a query can be associated with. This
        must be set for caching purposes. If the evaluated document list is
        bigger than the specified number, IndexError will be raised.

    queries: list of rankpy.queries.Queries
        The collections of queries that are known to be evaluated by this metric.
        These are used to compute `max_relevance` and `max_documents`.
    '''
    def __init__(self, cutoff=-1, max_relevance=None, max_documents=None,
                 queries=None, random_state=None):
        # Get the maximum relevance score and maximum number of documents
        # per a query from the specified set(s) of queries...
        if queries is not None:
            max_relevance = max([qs.highest_relevance() for qs in queries])
            max_documents = max([qs.longest_document_list() for qs in queries])
        else:
            # or use the parameters given. None values indicate that explicit
            # values were not given which may lead to unexpected results and
            # to runtime errors, hence a user warnings are issued.
            if max_relevance is None:
                max_relevance = 8
                warn('Maximum relevance label was not explicitly specified ' \
                     '(using default value 8). This should be avoided in order ' \
                     'not to encounter runtime error (segfault)!')

            if max_documents is None:
                max_documents = 8192
                warn('Maximum number of documents per query was not explicitly specified ' \
                     '(using default value 8192). This should be avoided in order not to ' \
                     'encounter runtime error (segfault)!')

        self.cutoff = cutoff
        self.max_relevance = max_relevance
        self.max_documents = max_documents
        self.random_state = check_random_state(random_state)

        # Create the metric cython backend.
        self.metric_ = DCG(cutoff, max_relevance, max_documents,
                           self.random_state.randint(1, np.iinfo('i').max))

    def backend(self, copy=True):
        '''
        Returns the backend metric object.
        '''
        if copy:
            return DCG(self.cutoff, self.max_relevance, self.max_documents,
                       self.random_state.randint(1, np.iinfo('i').max))
        else:
            self.metric_

    def evaluate(self, ranking=None, labels=None, ranked_labels=None, scales=None, weight=1.0):
        ''' 
        Evaluate the DCG metric on the specified ranked list of document relevance scores.

        The function input can be either ranked list of relevance labels (`ranked_labels`),
        which is most convenient from the computational point of view, or it can be in
        the form of ranked list of documents (`ranking`) and corresponding relevance scores
        (`labels`), from which the ranked document relevance labels are computed.

        Parameters:
        -----------
        ranking: array, shape = (n_documents,)
            Specify list of ranked documents.

        labels: array: shape = (n_documents,)
            Specify relevance score for each document.

        ranked_labels: array, shape = (n_documents,)
            Relevance scores of the ranked documents. If not given, then
            `ranking` and `labels` must not be None, `ranked_labels` will
            be than inferred from them.

        scales: float, optional (default is None)
            Ignored.

        weight: float, optional (default is 1.0)
            The weight of the query for which the metric is evaluated.
        '''
        if ranked_labels is not None:
            return self.metric_.evaluate(ranked_labels, 1.0, weight)
        elif ranking is not None and labels is not None:
            if ranking.shape[0] != labels.shape[0]:
                raise ValueError('number of ranked documents != number of relevance labels (%d, %d)' \
                                  % (ranking.shape[0], labels.shape[0]))
            return self.metric_.evaluate_ranking(ranking, labels, 1.0, weight)

    def evaluate_queries(self, queries, scores, scales=None, weights=None, out=None):
        ''' 
        Evaluate the DCG metric on the specified set of queries (`queries`). The documents
        are sorted by corresponding ranking scores (`scores`) and the metric is then
        computed as the average of the metric values evaluated on each query document
        list. The ties in ranking are broken probabilistically.
        
        Parameters:
        -----------
        queries: rankpy.queries.Queries
            The set of queries for which the metric will be computed.

        scores: array, shape=(n_documents,)
            The ranking scores for each document in the queries.

        scales: array, shape=(n_queries,), or None
            Ignored.

        weights: array of doubles, shape=(n_queries,), or None
            The weight of each query for which the metric is evaluated.

        out: array, shape=(n_documents,), or None
            If not None, it will be filled with the metric value
            for each individual query.
        '''
        if queries.document_count() != scores.shape[0]:
            raise ValueError('number of documents != number of scores (%d, %d)' \
                             % (queries.document_count(), scores.shape[0]))

        if weights is not None and queries.query_count() != weights.shape[0]:
            raise ValueError('number of queries != size of weights array (%d, %d)' \
                             % (queries.query_count(), out.shape[0]))

        if out is not None and queries.query_count() != out.shape[0]:
            raise ValueError('number of queries != size of output array (%d, %d)' \
                             % (queries.query_count(), out.shape[0]))

        return self.metric_.evaluate_queries(queries.query_indptr, queries.relevance_scores, scores, None, weights, out)

    def compute_delta(self, i, offset, document_ranks, relevance_scores, scales=None, out=None):
        ''' 
        Compute the change in the DCG metric after swapping document 'i' with
        each document in the document list starting at 'offset'.

        The relevance and rank of the document 'i' is 'relevance_scores[i]' and
        'document_ranks[i]', respectively.

        Similarly, 'relevance_scores[j]' and 'document_ranks[j]' for each j starting
        from 'offset' and ending at the end of the list denote the relevance score
        and rank of the document that will be swapped with document 'i'.

        Parameters:
        -----------
        i: int
            The index (zero-based) of the document that will appear in every pair
             of documents that will be swapped.

        offset: int
            The offset pointer to the start of the documents that will be swapped.

        document_ranks: array
            The ranks of the documents.

        relevance_scores: array
            The relevance scores of the documents.

        scales: float or None, optional (default is None)
            Ignored.

        out: array, optional (default is None)
            The output array. The array size is expected to be at least equal to
            the number of documents shorter by the ofset.
        '''
        n_documents = len(document_ranks)

        if out is None:
            out = np.empty(n_documents - offset, dtype=np.float64)

        if out.shape[0] < n_documents - offset:
            raise ValueError('output array is too small (%d < %d)' \
                             % (out.shape[0], n_documents - offset))

        if document_ranks.shape[0] != relevance_scores.shape[0]:
            raise ValueError('document ranks size != relevance scores (%d != %d)' \
                              % (document_ranks.shape[0], relevance_scores.shape[0]))

        self.metric_.delta(i, offset, document_ranks, relevance_scores, 1.0, out)

        return out


    def compute_scale(self, queries, relevance_scores=None):
        ''' 
        Since DCG is not normalized, return None.
        '''
        return None


    def __str__(self):
        ''' 
        Return the textual description of the metric.
        '''
        return 'DCG' if self.metric_.cutoff < 0 else 'DCG@%d' % self.metric_.cutoff


class NormalizedDiscountedCumulativeGain(object):
    ''' 
    Normalized Discounted Cumulative Gain metric.

    Note that it is really recommended to use the optional parameter `queries`
    to pre-allocate the memory for gain and discount components of NDCG metric
    for faster computation.

    Optionally, you can specify `max_relevance` and `max_documents`, but they
    should be inferred from the queries anyway.

    Ignoring the note above can lead only to 2 outcomes:
      1) Everything will be just fine.
      2) You will run into Segmentation Fault.

    Arguments:
    ----------
    cutoff: int, optional (default is -1)
        If positive, it denotes the maximum rank of a document
        that will be considered for evaluation.

    max_relevance: int, optional (default is 4)
        The maximum relevance score a document can have. This must be
        set for caching purposes. If the evaluated document list contain
        a document with higher relevance than the number specified,
        IndexError will be raised.

    max_documents: int, optional (default is 8192):
        The maximum number of documents a query can be associated with. This
        must be set for caching purposes. If the evaluated document list is
        bigger than the specified number, IndexError will be raised.

    queries: list of rankpy.queries.Queries
        The collections of queries that are known to be evaluated by this metric.
        These are used to compute `max_relevance` and `max_documents`.
    '''
    def __init__(self, cutoff=-1, max_relevance=None, max_documents=None,
                 queries=None, random_state=None):
        # Get the maximum relevance score and maximum number of documents
        # per a query from the specified set(s) of queries...
        if queries is not None and len(queries) > 0:
            print('yay!')
            max_relevance = max([qs.highest_relevance() for qs in queries])
            max_documents = max([qs.longest_document_list() for qs in queries])
        else:
            # or use the parameters given. None values indicate that explicit
            # values were not given which may lead to unexpected results and
            # to runtime errors, hence a user warnings are issued.
            if max_relevance is None:
                max_relevance = 8
                warn('Maximum relevance label was not explicitly specified ' \
                     '(using default value 8). This should be avoided in order ' \
                     'not to encounter runtime error (SegFault)!')

            if max_documents is None:
                max_documents = 8192
                warn('Maximum number of documents per query was not explicitly specified ' \
                     '(using default value 8192). This should be avoided in order not to ' \
                     'encounter runtime error (SegFault)!')

        self.cutoff = cutoff
        self.max_relevance = max_relevance
        self.max_documents = max_documents
        self.random_state = check_random_state(random_state)
        
        # Create the metric cython backend.
        self.metric_ = DCG(cutoff, max_relevance, max_documents,
                           self.random_state.randint(1, np.iinfo('i').max))

    def backend(self, copy=True):
        '''
        Returns the backend metric object.
        '''
        if copy:
            return DCG(self.cutoff, self.max_relevance, self.max_documents,
                       self.random_state.randint(1, np.iinfo('i').max))
        else:
            self.metric_

    def evaluate(self, ranking=None, labels=None, ranked_labels=None, scales=None, weight=1.0):
        ''' 
        Evaluate NDCG metric on the specified ranked list of document relevance scores.

        The function input can be either ranked list of relevance labels (`ranked_labels`),
        which is most convenient from the computational point of view, or it can be in
        the form of ranked list of documents (`ranking`) and corresponding relevance scores
        (`labels`), from which the ranked document relevance labels are computed.

        Parameters:
        -----------
        ranking: array, shape = (n_documents,)
            Specify list of ranked documents.

        labels: array: shape = (n_documents,)
            Specify relevance score for each document.

        ranked_labels: array, shape = (n_documents,)
            Relevance scores of the ranked documents. If not given, then
            `ranking` and `labels` must not be None, `ranked_labels` will
            be than inferred from them.

        scales: float, optional (default is None)
            The ideal DCG value on the given documents. If None is given
            it will be computed from the document relevance scores.

        weight: float, optional (default is 1.0)
            The weight of the query for which the metric is evaluated.
        '''
        if ranked_labels is not None:
            return self.metric_.evaluate(ranked_labels, scales or self.metric_.evaluate(np.ascontiguousarray(np.sort(ranked_labels)[::-1]), 1.0, weight), weight)
        elif ranking is not None and labels is not None:
            if ranking.shape[0] != labels.shape[0]:
                raise ValueError('number of ranked documents != number of relevance labels (%d, %d)' \
                                  % (ranking.shape[0], labels.shape[0]))
            return self.metric_.evaluate_ranking(ranking, labels, scales or self.metric_.evaluate(np.ascontiguousarray(np.sort(labels)[::-1]), 1.0, weight), weight)

    def evaluate_queries(self, queries, scores, scales=None, weights=None, out=None):
        ''' 
        Evaluate the NDCG metric on the specified set of queries (`queries`). The documents
        are sorted by corresponding ranking scores (`scores`) and the metric is then
        computed as the average of the metric values evaluated on each query document
        list.
        
        Parameters:
        -----------
        queries: rankpy.queries.Queries
            The set of queries for which the metric will be computed.

        scores: array, shape=(n_documents,)
            The ranking scores for each document in the queries.

        scales: array, shape=(n_queries,) or None, optional (default is None)
            The ideal DCG values for each query. If None is given it will be
            computed from the document relevance scores.

        weights: array of doubles, shape=(n_queries,), or None
            The weight of each query for which the metric is evaluated.

        out: array, shape=(n_documents,), or None
            If not None, it will be filled with the metric value
            for each individual query.
        '''
        if queries.document_count() != scores.shape[0]:
            raise ValueError('number of documents != number of scores (%d, %d)' \
                             % (queries.document_count(), scores.shape[0]))
        if scales is None:
            scales = np.empty(queries.query_count(), dtype=np.float64)
            self.metric_.evaluate_queries_ideal(queries.query_indptr, queries.relevance_scores, scales)

        if queries.query_count() != scales.shape[0]:
            raise ValueError('number of queries != number of scaling factors (%d != %d)' \
                             % (queries.query_count(), scales.shape[0]))

        if weights is not None and queries.query_count() != weights.shape[0]:
            raise ValueError('number of queries != size of weights array (%d, %d)' \
                             % (queries.query_count(), out.shape[0]))

        if out is not None and queries.query_count() != out.shape[0]:
            raise ValueError('number of queries != size of output array (%d, %d)' \
                             % (queries.query_count(), out.shape[0]))

        return self.metric_.evaluate_queries(queries.query_indptr, queries.relevance_scores, scores, scales, weights, out)

    def compute_delta(self, i, offset, document_ranks, relevance_scores, scales=None, out=None):
        ''' 
        Compute the change in the NDCG metric after swapping document 'i' with
        each document in the document list starting at 'offset'.

        The relevance and rank of the document 'i' is 'relevance_scores[i]' and
        'document_ranks[i]', respectively.

        Similarly, 'relevance_scores[j]' and 'document_ranks[j]' for each j starting
        from 'offset' and ending at the end of the list denote the relevance score
        and rank of the document that will be swapped with document 'i'.

        Parameters:
        -----------
        i: int
            The index (zero-based) of the document that will appear in every pair
             of documents that will be swapped.

        offset: int
            The offset pointer to the start of the documents that will be swapped.

        document_ranks: array
            The ranks of the documents.

        relevance_scores: array
            The relevance scores of the documents.

        scales: float or None, optional (default is None)
            The ideal DCG value for the query the documents are associated with.
            If None is given, the scales will be computed from the relevance scores.

        out: array, optional (default is None)
            The output array. The array size is expected to be at least equal to
            the number of documents shorter by the ofset.
        '''
        n_documents = len(document_ranks)

        if out is None:
            out = np.empty(n_documents - offset, dtype=np.float64)

        if out.shape[0] < n_documents - offset:
            raise ValueError('output array is too small (%d < %d)' \
                             % (out.shape[0], n_documents - offset))

        if document_ranks.shape[0] != relevance_scores.shape[0]:
            raise ValueError('document ranks size != relevance scores (%d != %d)' \
                             % (document_ranks.shape[0], relevance_scores.shape[0]))

        if scales is None:
            scales = self.metric_.evaluate(np.ascontiguousarray(np.sort(relevance_scores)[::-1]), 1.0, weight)

        self.metric_.delta(i, offset, document_ranks, relevance_scores, scales, out)

        return out

    def compute_scale(self, queries, relevance_scores=None):
        ''' 
        Return the ideal DCG value for each query. Optionally, external
        relevance assessments can be used instead of the relevances
        present in the queries.

        Parameters
        ----------
        queries: Queries
            The queries for which the ideal DCG should be computed.

        relevance_scores: array of integers, optional, (default is None)
            The relevance scores that should be used instead of the 
            relevance scores inside queries. Note, this argument is
            experimental.
        '''
        ideal_values = np.empty(queries.query_count(), dtype=np.float64)

        if relevance_scores is not None:
            if queries.document_count() != relevance_scores.shape[0]:
                raise ValueError('number of documents and relevance scores do not match')

            # Need to sort the relevance labels first.
            indices = np.empty(relevance_scores.shape[0], dtype=np.intc)
            relevance_argsort_v1(relevance_scores, indices, relevance_scores.shape[0])
            # Creates a copy.
            relevance_scores = relevance_scores[indices]
        else:
            # Assuming these are sorted.
            relevance_scores = queries.relevance_scores

        self.metric_.evaluate_queries_ideal(queries.query_indptr, relevance_scores, ideal_values)

        return ideal_values


    def __str__(self):
        '''
        Return the textual description of the metric.
        '''
        return 'NDCG' if self.metric_.cutoff < 0 else 'NDCG@%d' % self.metric_.cutoff


class ExpectedReciprocalRank(object):
    def __init__(self, cutoff=-1, max_relevance=None, max_documents=None, queries=None):
        # Get the maximum relevance score and maximum number of documents
        # per a query from the specified set(s) of queries...
        self.cutoff = cutoff
        if queries is not None:
            self.max_relevance = max([qs.highest_relevance() for qs in queries])
            self.max_documents = max([qs.longest_document_list() for qs in queries])
        else:
            # or use the parameters given. None values indicate that explicit
            # values were not given which may lead to unexpected results and
            # to runtime errors, hence a user warnings are issued.
            if max_relevance is None:
                self.max_relevance = 8
                warn('Maximum relevance label was not explicitly specified ' \
                     '(using default value 8). This should be avoided in order ' \
                     'not to encounter runtime error (SegFault)!')

            if max_documents is None:
                self.max_documents = 8192
                warn('Maximum number of documents per query was not explicitly specified ' \
                     '(using default value 8192). This should be avoided in order not to ' \
                     'encounter runtime error (SegFault)!')
        self.Ri_array = np.zeros(self.max_relevance + 1, np.double)
        for label in xrange(self.max_relevance + 1):
            self.Ri_array[label] = (2.0 ** label - 1) / (2.0 ** self.max_relevance)



    def Ri(self,label):
        return self.Ri_array[label]

    def get_score_from_labels_list(self,labels_list):
        score = 0.0
        evaluated_size = len(labels_list) if (self.cutoff > len(labels_list) or self.cutoff <= 0) else self.cutoff
        cumulated_product = 1.0
        for r in xrange(1, evaluated_size + 1):
            Rr = self.Ri(labels_list[r - 1])
            score += Rr / r * cumulated_product
            cumulated_product *= (1.0 - Rr)
        return score

    def evaluate(self, ranking=None, labels=None, ranked_labels=None, scales=None):
        '''
        Evaluate NDCG metric on the specified ranked list of document relevance scores.

        The function input can be either ranked list of relevance labels (`ranked_labels`),
        which is most convenient from the computational point of view, or it can be in
        the form of ranked list of documents (`ranking`) and corresponding relevance scores
        (`labels`), from which the ranked document relevance labels are computed.

        Parameters:
        -----------
        ranking: array, shape = (n_documents,)
            Specify list of ranked documents.

        labels: array: shape = (n_documents,)
            Specify relevance score for each document.

        ranked_labels: array, shape = (n_documents,)
            Relevance scores of the ranked documents. If not given, then
            `ranking` and `labels` must not be None, `ranked_labels` will
            be than inferred from them.

        scales: float, optional (default is None)
            The ideal DCG value on the given documents. If None is given
            it will be computed from the document relevance scores.
        '''
        if ranked_labels is not None:
            return self.get_score_from_labels_list(ranked_labels)
        elif ranking is not None and labels is not None:
            if ranking.shape[0] != labels.shape[0]:
                raise ValueError('number of ranked documents != number of relevance labels (%d, %d)' \
                                  % (ranking.shape[0], labels.shape[0]))
            ranked_labels = np.array(sorted(labels, key=dict(zip(labels,ranking)).get, reverse=True), dtype=np.intc)
            return self.get_score_from_labels_list(ranked_labels)

    def evaluate_queries(self, queries, scores, scales=None, out=None):
        total_score = 0.0
        for i in range(queries.query_count()):
            y = queries.relevance_scores[queries.query_indptr[i]:queries.query_indptr[i + 1]]
            ranking = scores[queries.query_indptr[i]:queries.query_indptr[i + 1]]
            total_score += self.evaluate(labels=y, ranking=ranking)
        return total_score / float(queries.query_count())

    def __str__(self):
        '''
        Return the textual description of the metric.
        '''
        return 'ERR' if self.cutoff < 0 else 'ERR@%d' % self.cutoff


class SeznamRank(object):
    def __init__(self, cutoff=20, max_relevance=None, max_documents=None, queries=None, bw_navig=[], bw_info=[], pw_navig=[], pw_info=[]):
        """
        :param cutoff:
        :param max_relevance:
        :param max_documents:
        :param queries:
        :param bw_navig: constants provided by Seznam.cz
        :param bw_info: constants provided by Seznam.cz
        :param pw_navig: constants provided by Seznam.cz
        :param pw_info: constants provided by Seznam.cz
        :return:
        """

        self.cutoff = cutoff
        if queries is not None:
            self.max_relevance = max([qs.highest_relevance() for qs in queries])
            self.max_documents = max([qs.longest_document_list() for qs in queries])
        else:
            # or use the parameters given. None values indicate that explicit
            # values were not given which may lead to unexpected results and
            # to runtime errors, hence a user warnings are issued.
            if max_relevance is None:
                self.max_relevance = 8
                warn('Maximum relevance label was not explicitly specified ' \
                     '(using default value 8). This should be avoided in order ' \
                     'not to encounter runtime error (SegFault)!')

            if max_documents is None:
                self.max_documents = 8192
                warn('Maximum number of documents per query was not explicitly specified ' \
                     '(using default value 8192). This should be avoided in order not to ' \
                     'encounter runtime error (SegFault)!')
        self.BOX_WEIGHTS = (bw_info, bw_navig)
        self.POS_WEIGHTS = (pw_info, pw_navig)



    def get_score_from_labels_list(self, labels_list):
        is_navig = True if self.max_relevance in labels_list else False
        return min(sum(self.BOX_WEIGHTS[is_navig][a - 1] * b for a, b
                       in zip(labels_list, self.POS_WEIGHTS[is_navig])), 100.0) / 100.0

    def evaluate(self, ranking=None, labels=None, ranked_labels=None, scales=None):
        '''
        Evaluate NDCG metric on the specified ranked list of document relevance scores.

        The function input can be either ranked list of relevance labels (`ranked_labels`),
        which is most convenient from the computational point of view, or it can be in
        the form of ranked list of documents (`ranking`) and corresponding relevance scores
        (`labels`), from which the ranked document relevance labels are computed.

        Parameters:
        -----------
        ranking: array, shape = (n_documents,)
            Specify list of ranked documents.

        labels: array: shape = (n_documents,)
            Specify relevance score for each document.

        ranked_labels: array, shape = (n_documents,)
            Relevance scores of the ranked documents. If not given, then
            `ranking` and `labels` must not be None, `ranked_labels` will
            be than inferred from them.

        scales: float, optional (default is None)
            The ideal DCG value on the given documents. If None is given
            it will be computed from the document relevance scores.
        '''
        if ranked_labels is not None:
            return self.get_score_from_labels_list(ranked_labels)
        elif ranking is not None and labels is not None:
            if ranking.shape[0] != labels.shape[0]:
                raise ValueError('number of ranked documents != number of relevance labels (%d, %d)' \
                                  % (ranking.shape[0], labels.shape[0]))
            ranked_labels = np.array(sorted(labels, key=dict(zip(labels,ranking)).get, reverse=True), dtype=np.intc)
            return self.get_score_from_labels_list(ranked_labels)

    def evaluate_queries(self, queries, scores, scales=None, out=None):
        total_score = 0.0
        for i in range(queries.query_count()):
            y = queries.relevance_scores[queries.query_indptr[i]:queries.query_indptr[i + 1]]
            ranking = scores[queries.query_indptr[i]:queries.query_indptr[i + 1]]
            total_score += self.evaluate(labels=y, ranking=ranking)
        return total_score / float(queries.query_count())

    def __str__(self):
        '''
        Return the textual description of the metric.
        '''
        return 'SR' if self.cutoff < 0 else 'SR@%d' % self.cutoff


class MetricFactory(object):
    name2metric = {'WTA': WinnerTakesAll.__class__,
                   'DCG': DiscountedCumulativeGain.__class__,
                   'NDCG': NormalizedDiscountedCumulativeGain }

    @staticmethod
    def __new__(cls, name, queries, random_state=None):
        name, _, cutoff = name.partition('@')
        cutoff = int(cutoff) if len(cutoff) > 0 else -1
        try:
            return cls.name2metric[name](cutoff, queries=queries,
                                         random_state=random_state)
        except KeyError:
            raise ValueError('unknown metric: %s' % name)
