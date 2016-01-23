# -*- coding: utf-8 -*-
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


import logging
import numpy as np
import scipy.sparse as sp

from itertools import chain
from .utils import pickle, unpickle

from warnings import warn


logger = logging.getLogger(__name__)


class QueryDocumentSlices(object):
    '''
    Helper class for getting slices of
    query-document index array.
    '''

    def __init__(self, queries):
        self.queries = queries

    def __getitem__(self, i):
        return slice(self.queries.query_indptr[i],
                     self.queries.query_indptr[i + 1])


class Queries(object):
    '''
    Data structure representing queries used for training learning to
    rank algorithms. It is created from query-document feature vectors,
    corresponding relevance scores, and index mapping from queries to
    associated query-document feature vectors.

    Parameters:
    -----------
    feature_vectors: array, shape = (# of query-document pairs, # of features)
        The feature vectors for query-document pairs.

    relevance_scores: array, shape = (# of query-document pairs,)
        The relevance scores corresponding to the feature vectors.

    query_indptr: array
        The query index pointer into the feature_vectors and relevance_scores
        array, i.e. the document feature vectors,
        ``feature_vectors[query_indptr[i]:query_indptr[i + 1]]``, and the
        corresponding relevance scores,
        ``relevance_scores[query_indptr[i]:query_indptr[i + 1]]``,
        are the feature vectors and relevance scores for the i-th
        query documents.

    max_score: int, optional (default is None)
        The maximum relevance score value. If None, the value is derived
        from the specified relevance scores.

    has_sorted_relevances: bool, optional (default is False)
        If True, it indicates that the relevance scores are sorted
        in decreasing order. Use to surpass sorting when it has been
        already done.

    feature_indices: array, shape = (# of features,), or None
        The set of original feature indices (relevant only for reporting).
    '''
    def __init__(self, feature_vectors, relevance_scores, query_indptr,
                 max_score=None, has_sorted_relevances=False,
                 query_ids=None, feature_indices=None):

        self.feature_vectors = np.asanyarray(feature_vectors,
                                             dtype=np.float32)

        self.relevance_scores = np.asanyarray(relevance_scores,
                                              dtype=np.intc).ravel()

        self.query_indptr = np.asanyarray(query_indptr,
                                          dtype=np.intc).ravel()

        self.n_queries = self.query_indptr.shape[0] - 1
        self.n_feature_vectors = self.feature_vectors.shape[0]

        if query_ids is not None:
            self.query_ids = np.asanyarray(query_ids).ravel()
            assert self.query_ids.shape[0] == self.n_queries, ('the number of '
                    'queries (%d) != the number of query ids (%d)'
                    % (self.n_queries, self.query_ids.shape[0]))

        if feature_indices is not None:
            self.feature_indices = np.asanyarray(feature_indices).ravel()
            assert self.feature_vectors.shape[1] == feature_indices.shape[0],\
                    ('the number of features (%d) != the number of feature '
                     'indices (%d)' % (self.feature_vectors.shape[1],
                                       feature_indices.shape[0]))

        # Make sure shapes are consistent.
        assert self.n_feature_vectors == self.relevance_scores.shape[0],\
                ('the number of documents does not equal the number of '
                 'relevance scores')

        assert self.n_feature_vectors == self.query_indptr[-1], ('the query '
                'index pointer is not correct (number of indexed items is not '
                'the same as the number of documents)')

        if max_score is None:
            max_score = self.relevance_scores.max()\
                            if self.n_queries > 0 else 0

        self.max_score = max_score

        if not has_sorted_relevances:
            # Find the indices to make the documents sorted according to their
            # relevance scores. We create one huge array of indices to tackle
            # the potentially big beast (feature_vectors) at once (since fancy
            # indexing makes a copy it can be more advantageous to do it once
            # -- but it needs 2x as much memory, of course).
            relevance_scores_sorted_indices = \
                np.fromiter(chain(*[self.query_indptr[i] +
                                  np.argsort(self.relevance_scores[self.query_indptr[i]:self.query_indptr[i + 1]])[::-1] for i in range(self.n_queries)]),
                            dtype=np.intc,
                            count=self.n_feature_vectors)

            self.feature_vectors = self.feature_vectors[relevance_scores_sorted_indices, :]
            self.relevance_scores = self.relevance_scores[relevance_scores_sorted_indices]

        # Store for each query where in its document list
        # the relevance scores changes.
        self.query_relevance_strides = np.empty((self.n_queries,
                                                 self.max_score + 1),
                                                dtype=np.intc)
        self.query_relevance_strides.fill(-1)

        # Compute relevance score strides for each query (the scores need to
        # be sorted first), i.e. query_relevance_strides[i,j] denotes
        # the smallest index of a document that is less relevant
        # than the relevance score j in the document list of the i-th query.
        for i in range(self.n_queries):
            query_relevance_scores = self.relevance_scores[self.query_indptr[i]:self.query_indptr[i + 1]]
            query_relevance_starts = np.where(np.diff(query_relevance_scores))[0]
            # Special cases: all documents have the same relevance
            # or there is no irrelevant document.
            if query_relevance_starts.size == 0 or query_relevance_scores[-1] > 0:
                query_relevance_starts = np.append(query_relevance_starts,
                                                   query_relevance_scores.size - 1)

            query_relevance_starts += self.query_indptr[i]

            self.query_relevance_strides[i, self.relevance_scores[query_relevance_starts]] = query_relevance_starts + 1

        # Just to make use of the space (which might be found useful later).
        self.query_relevance_strides[:, 0] = self.query_indptr[1:]

        self.qds = QueryDocumentSlices(self)

    def __str__(self):
        '''
        Return the textual representation of Queries.
        '''
        return ('Queries (%d queries, %d documents, %d features, '
                '%d max. relevance)' % (self.n_queries, self.n_feature_vectors,
                                        self.feature_vectors.shape[1],
                                        self.max_score))

    @staticmethod
    def load_from_text(filepaths, dtype=np.float32, max_score=None,
                       min_feature=None, max_feature=None,
                       has_sorted_relevances=False, purge=False):
        '''
        Load queries in the svmlight format from the specified file(s).

        SVMlight format example (one line):

            5[\s]qid:8[\s]103:1.0[\s]...[\s]981:1.0 982:1.0 # comment[\n]

        Parameters:
        -----------
        filepath: string or list of strings
            The location of the dataset file(s).

        dtype: data-type, optional (default is np.float32)
            The desired data-type for the document feature vectors. Here,
            the default value (np.float32) is chosen for mere optimization
            purposes.

        max_score: int, optional (default is None)
            The maximum relevance score value. If None, the value is derived
            from the relevance scores in the file.

        min_feature: int or None, optional (default is None)
            The minimum feature identifier, which is present in the dataset. If
            None, this value is read from the data. This parameter is important
            because of internal feature remapping: in case of loading different
            parts of a dataset (folds), some features may be present in one
            part and may not be present in another (because all its values are
            0) - this would create inconsistent feature mappings between
            the parts.

        max_feature: int or None, optional (default is None)
            The maximum feature identifier, which is present in the dataset. If
            None, this value is read from the data. This parameter is important
            because of internal feature remapping, see `min_feature` for more.

        has_sorted_relevances: bool, optional (default is False)
            If True, it indicates that the relevance scores of the queries
            in the file are sorted in decreasing order.

        purge: bool, optional (default is False)
            If True, all queries which have documents with the same relevance
            labels are removed. If False, no query is removed.
        '''
        # Arrays used to build CSR matrix of query-document vectors.
        data, indices, indptr = [], [], [0]

        # Relevance score, query ID, query hash, and document hash.
        relevances = []

        query_ids = []
        query_indptr = [0]
        prev_qid = None

        # If only single filepath is given, not a list.
        if isinstance(filepaths, str):
            filepaths = [filepaths]

        n_purged_queries = 0
        n_purged_documents = 0

        def purge_query(qid, data, indices, indptr):
            '''Remove the last query added to the set according to `purge`.'''
            if not purge or qid is None:
                return 0

            r = relevances[query_indptr[-2]]

            i = query_indptr[-2]
            while i < query_indptr[-1] and relevances[i] == r:
                i += 1

            if i == query_indptr[-1]:
                n = query_indptr.pop()

                del query_ids[-1]

                del indices[indptr[query_indptr[-1]]:]
                del data[indptr[query_indptr[-1]]:]

                del relevances[query_indptr[-1]:]
                del indptr[query_indptr[-1] + 1:]

                return n - query_indptr[-1]
            else:
                return 0

        for filepath in filepaths:
            lineno = 0  # Used just to report invalid lines (if any).

            logger.info('Reading queries from %s.' % filepath)

            with open(filepath, 'r') as ifile:
                # Loop through every line containing query-document pair.
                for pair in ifile:
                    lineno += 1
                    try:
                        comment_start = pair.find('#')

                        # Remove the line comment first.
                        if comment_start >= 0:
                            pair = pair[:comment_start]

                        pair = pair.strip()

                        # Skip comments and empty lines.
                        if not pair:
                            continue

                        # Sadly, the distinct items on the line are not
                        # properly separated by a single delimiter. We split
                        # using all whitespaces here.
                        items = pair.split()

                        # Query ID follows the second item on the line,
                        # which is 'qid:'.
                        qid = int(items[1].split(':')[1])

                        if qid != prev_qid:
                            # Make sure query is sanitized before being
                            # added to the set.
                            n_purged = purge_query(prev_qid, data,
                                                   indices, indptr)

                            n_purged_documents += n_purged

                            if n_purged > 0:
                                logger.debug('Ignoring query %d (qid) with %d '
                                             'documents because all had the '
                                             'same relevance label.'
                                             % (prev_qid, n_purged))
                                n_purged_queries += 1

                            query_ids.append(qid)
                            query_indptr.append(query_indptr[-1] + 1)
                            prev_qid = qid

                        else:
                            query_indptr[-1] += 1

                        # Relevance is the first number on the line.
                        relevances.append(int(items[0]))

                        # Load the feature vector into CSR arrays.
                        for fidx, fval in map(lambda s: s.split(':'),
                                              items[2:]):
                            data.append(dtype(fval))
                            indices.append(int(fidx))

                        indptr.append(len(indices))

                        if (query_indptr[-1] + n_purged_documents) % 10000 == 0:
                            logger.info('Read %d queries and %d documents '
                                        'so far.' % (len(query_indptr) + 
                                                     n_purged_queries - 1,
                                                     query_indptr[-1] + 
                                                     n_purged_documents))
                    except:
                        # Ill-formated line (it should not happen).
                        # Print line number
                        print('Ill-formated line: %d' % lineno)
                        raise

                # Need to check the last added query.
                n_purged = purge_query(prev_qid, data, indices, indptr)
                n_purged_documents += n_purged

                if n_purged > 0:
                    logger.debug('Ignoring query %d (qid) with %d documents '
                                 'because all had the same relevance label.'
                                 % (prev_qid, n_purged))
                    n_purged_queries += 1

                logger.info('Read %d queries and %d documents out of which '
                            '%d queries and %d documents were discarded.'
                            % (len(query_indptr) + n_purged_queries - 1,
                               query_indptr[-1] + n_purged_documents,
                               n_purged_queries, n_purged_documents))

        # Empty dataset.
        if len(query_indptr) == 1:
            raise ValueError('the input seems to be empty')

        # Set the minimum feature ID, if not given.
        if min_feature is None:
            min_feature = min(indices)

        if max_feature is None:
            # Remap the features for a proper conversion into dense matrix.
            feature_indices = np.unique(np.r_[min_feature, indices])
            indices = np.searchsorted(feature_indices, indices)
        else:
            assert min(indices) >= min_feature, ('there is a feature with id '
                    'smaller than min_feature: %d < %d' % (min(indices),
                                                           min_feature))

            assert max(indices) <= max_feature, ('there is a feature with id '
                    'greater than max_feature: %d > %d' % (max(indices),
                                                           max_feature))

            feature_indices = np.arange(min_feature,
                                        max_feature + 1,
                                        dtype='int32')

            indices = np.array(indices, dtype='int32') - min_feature

        feature_vectors = sp.csr_matrix((data, indices, indptr), dtype=dtype,
                                        shape=(query_indptr[-1],
                                        len(feature_indices)))

        # Free the copies of the feature_vectors in non-Numpy arrays (if any),
        # this is important in order not to waste memory for the transfer of
        # the feature vectors to dense format (default option).
        del data, indices, indptr

        feature_vectors = feature_vectors.toarray()

        # Create and return a Queries object.
        return Queries(feature_vectors, relevances, query_indptr,
                       max_score=max_score,
                       has_sorted_relevances=has_sorted_relevances,
                       query_ids=query_ids,
                       feature_indices=feature_indices)

    def save_as_text(self, filepath, shuffle=False):
        '''
        Save queries into the specified file in svmlight format.

        Parameters:
        -----------
        filepath: string
            The filepath where this object will be saved.

        shuffle: bool
            Specify to shuffle the query document lists prior
            to writing into the file.
        '''
        # Inflate the query_ids array such that each id covers
        # the corresponding feature vectors.
        query_ids = np.fromiter(
            chain(*[[qid] * cnt for qid, cnt in zip(self.query_ids, np.diff(self.query_indptr))]),
            dtype=int)

        relevance_scores = self.relevance_scores
        feature_vectors = self.feature_vectors

        if shuffle:
            shuffle_indices = np.random.permutation(self.document_count())
            reshuffle_indices = np.argsort(query_ids[shuffle_indices])
            document_shuffle_indices = np.arange(self.document_count(),
                                                 dtype=np.intc)[shuffle_indices[reshuffle_indices]]
            query_ids = query_ids[document_shuffle_indices]
            relevance_scores = relevance_scores[document_shuffle_indices]
            feature_vectors = feature_vectors[document_shuffle_indices]

        with open(filepath, 'w') as ofile:
            for score, qid, feature_vector in zip(relevance_scores,
                                                   query_ids,
                                                   feature_vectors):
                ofile.write('%d' % score)
                ofile.write(' qid:%d' % qid)
                for feature in zip(self.feature_indices, feature_vector):
                    output = ' %d:%.12f' % feature
                    ofile.write(output.rstrip('0').rstrip('.'))
                ofile.write('\n')

    @classmethod
    def load(cls, filepath, mmap=None, order=None):
        '''
        Load the previously saved Queries object from the specified file.

        Parameters:
        -----------
        filepath: string
            The filepath, from which a Queries object will be loaded.

        mmap: {None, ‘r+’, ‘r’, ‘w+’, ‘c’}, optional (default is None)
            If not None, then memory-map the feature vectors, using
            the given mode (see `numpy.memmap` for a details). This
            will work only if the feature matrix has been saved separately.

        order: {'C', 'F'} or None, optional (default is None)
            Specify the order for the feature vectors array. 'C' and 'F'
            stand for C-contiguos and F-contiguos order, respectively.
            If None, the order of the features is the same as in time
            when `self.save(...)` was called. Note that, if mmap is not
            None, this parameter is ignored.
        order:
        '''
        logger.info('Loading queries from %s.' % filepath)
        queries = unpickle(filepath)
        if not hasattr(queries, 'feature_vectors'):
            queries.feature_vectors = np.load(filepath + '.feature_vectors.npy', mmap_mode=mmap)
        # Convert (if needed) feature vectors into wanted order.
        if mmap is None:
            queries.feature_vectors = np.asanyarray(queries.feature_vectors, order=order)
        # Recover the query indices pointer array from the relevance strides array.
        setattr(queries, 'query_indptr', np.empty(queries.query_relevance_strides.shape[0] + 1, dtype=np.intc))
        queries.query_indptr[0] = 0
        queries.query_indptr[1:] = queries.query_relevance_strides[:, 0]
        # Recover the relevance scores from the relevance strides array.
        setattr(queries, 'relevance_scores', np.empty(queries.feature_vectors.shape[0], dtype=np.intc))
        iQD = 0
        for iQ in range(queries.n_queries):
            for score in range(queries.max_score, -1, -1):
                while iQD < queries.query_relevance_strides[iQ, score]:
                    queries.relevance_scores[iQD] = score
                    iQD += 1
        if not hasattr(queries, 'qds'):
            queries.qds = QueryDocumentSlices(queries)
        logger.info('Loaded %d queries with %d documents in total.' % (queries.query_count(), queries.document_count()))
        return queries

    def save(self, filepath, order='F', separate=False):
        '''
        Save this Queries object into the specified file.

        Parameters:
        -----------
        filepath: string
            The filepath where this object will be saved.

        order: {'C', 'F'}, optional (default is 'F')
            Specify the order for the feature vectors array. 'C' and 'F'
            stand for C-contiguous and F-contiguous order, respectively.

        separate: bool
            If set to True, feature matrix is saved in a separate
            file, which can be advantageous later because the matrix
            can be memory mapped on load.
        '''
        # The name of the attributes that will be removed
        # from the object before pickling.
        removed_attribute_names = ['query_indptr', 'relevance_scores']

        # Save feature vectors separately to allow memory mapping.
        if separate:
            np.save(filepath + '.feature_vectors.npy',
                    np.asanyarray(self.feature_vectors, order=order))
            removed_attribute_names.append('feature_vectors')

        removed_attributes = {}

        # Delete the listed attributes.
        for attribute in removed_attribute_names:
            removed_attributes[attribute] = getattr(self, attribute)
            delattr(self, attribute)

        # Pickle...
        pickle(self, filepath)

        # ... and restore the attributes.
        for attribute, value in removed_attributes.items():
            setattr(self, attribute, value)


    def __getitem__(self, index):
        ''' 
        Return new Queries object containing queries in the `index`.
        '''
        # Handle slices.
        if isinstance(index, slice):
            start, stop, step = index.indices(self.query_count())

            # Special treatment for continuous slices.
            if step == 1:
                feature_vectors = self.feature_vectors[self.query_indptr[start]:self.query_indptr[stop]]
                relevance_scores = self.relevance_scores[self.query_indptr[start]:self.query_indptr[stop]]
                query_indptr = (np.array(self.query_indptr[start:stop + 1]) - 
                                         self.query_indptr[start])
                query_ids = self.query_ids[start:stop]

                return Queries(feature_vectors, relevance_scores, query_indptr,
                               query_ids=query_ids,
                               feature_indices=self.feature_indices,
                               has_sorted_relevances=True)
            else:
                index = np.arange(start, stop, step)

        # Handle boolean mask.
        if isinstance(index, np.ndarray) and index.ndim == 1 and index.dtype.kind == 'b':
            index = index.nonzero()[0]

        # Handle a single index.
        try:
            if int(index) == index:
                index = [index]
        except:
            pass

        # Handle Python lists.
        index = np.asanyarray(index, dtype=np.intc).ravel()

        feature_vectors = np.vstack([self.feature_vectors[self.query_indptr[i]:self.query_indptr[i + 1]] for i in index])
        relevance_scores = np.concatenate([self.relevance_scores[self.query_indptr[i]:self.query_indptr[i + 1]] for i in index])
        query_indptr = np.r_[0, np.diff(self.query_indptr)[index]].cumsum()
        query_ids = self.query_ids[index]

        return Queries(feature_vectors, relevance_scores, query_indptr,
                       query_ids=query_ids,
                       feature_indices=self.feature_indices,
                       has_sorted_relevances=True)

    def __len__(self):
        ''' 
        Returns the number of queries.
        '''
        return self.query_count()

    def adjust(self, min_score=None, min_documents=None, purge=False,
               scale=False, return_indices=False, remove_features=None):
        ''' 
        Adjust the document set such that the minimum relevance score is
        changed to the given value (all values are adjusted accordingly)
        and queries, which have all documents with the same relevance labels
        are removed (if purge is True).

        Parameters
        ----------
        min_score: int, optional (default is None)
            The minimum relevance score that is forced on this
            set of queries. Effectively it means that the scores
            are incremented/decremented by an amount given by
            the difference of the given min_score and the current
            minimum relevance score. If None is given, nothing
            happens.

        purge: bool, optional (default is False)
            If True, queries which have all documents of the same
            relevance (label) are removed. If False, no query is
            removed.

        scale: bool, optional (default is False)
            If True, feature values will be scaled into [0, 1]
            range using (x - min{X}) / (max{X} - min{X}).

        return_indices: bool, optional (default is False)
            If True, no query is removed, instead indices of queries that
            would have been removed are returned. Applicable only if
            purge is True.

        remove_features: array of booleans or ints
            Index or indicator (bitmask) array of features to be removed
            from the query-document feature vectores.

        Returns
        -------
        indices: array
            Indices of queries that would have been removed
            (if return_indices is True), or None.
        '''
        def purge_queries(good_query_mask, good_document_mask):
            query_document_counts = np.diff(self.query_indptr)
            good_query_document_counts = query_document_counts[good_query_mask]

            self.feature_vectors = self.feature_vectors[good_document_mask]
            self.relevance_scores = self.relevance_scores[good_document_mask]
            self.query_indptr = np.r_[0, good_query_document_counts].cumsum(dtype=np.intc)
            self.query_ids = self.query_ids[good_query_mask]
            self.n_queries = len(self.query_indptr) - 1
            self.n_feature_vectors = self.query_indptr[-1]
            self.query_relevance_strides = ((self.query_relevance_strides -
                                             query_document_counts.cumsum(dtype=np.intc).reshape(-1, 1))[good_query_mask] +
                                            good_query_document_counts.cumsum(dtype=np.intc).reshape(-1, 1))

            # Not needed, but keeps things nice and clean.
            self.query_relevance_strides[np.where(self.query_relevance_strides < 0)] = -1

        if remove_features is not None:
            # Convert to Numpy array with automatic type conversion.
            remove_features = np.asarray(remove_features)

            # Check dimensionality and correct dtype.
            if remove_features.ndim > 1:
                raise ValueError('remove_features has more '
                                 'than 1 dimension')

            if remove_features.dtype.kind == 'b':
                kept_features = ~remove_features
            elif remove_features.dtype.kind != 'i':
                raise ValueError('remove_features must be integer array')
            else:
                kept_features = np.ones(self.feature_vectors.shape[1],
                                        dtype='bool')
                kept_features[remove_features] = False

            self.feature_vectors = self.feature_vectors[:, kept_features]

            if self.feature_indices is not None:
                self.feature_indices = self.feature_indices[kept_features]

        # Force the given minimum score.
        if min_score is not None:
            current_min_score = self.relevance_scores.min()

            if current_min_score != min_score:
                self.relevance_scores -= current_min_score - min_score
                self.max_score = self.relevance_scores.max()

        if purge:
            bad_query_indices = []

            for i in range(self.n_queries):
                if len(np.unique(self.relevance_scores[self.query_indptr[i]:self.query_indptr[i + 1]])) == 1:
                    bad_query_indices.append(i)

            bad_query_indices = np.array(bad_query_indices, dtype=np.intc)

            if return_indices:
                return bad_query_indices

            # Mask for queries that will persist.
            good_query_mask = np.ones(self.n_queries, dtype=np.bool)
            good_query_mask[bad_query_indices] = False

            # Mask for documents (feature vectors) that will persist.
            good_document_mask = np.ones(self.n_feature_vectors, dtype=np.bool)
            for i in bad_query_indices:
                good_document_mask[self.query_indptr[i]:self.query_indptr[i + 1]] = False

            purge_queries(good_query_mask, good_document_mask)

        if min_documents is not None:
            query_document_counts = np.diff(self.query_indptr)

            bad_query_indices = np.where(query_document_counts < min_documents)[0]

            # Mask for queries that will persist.
            good_query_mask = np.ones(self.n_queries, dtype=np.bool)
            good_query_mask[bad_query_indices] = False

            # Mask for documents (feature vectors) that will persist.
            good_document_mask = np.ones(self.n_feature_vectors, dtype=np.bool)
            for i in bad_query_indices:
                good_document_mask[self.query_indptr[i]:self.query_indptr[i + 1]] = False

            purge_queries(good_query_mask, good_document_mask)

        if scale:
            min_feature_values = self.feature_vectors.min(axis=0, keepdims=True)
            scale_feature_values = self.feature_vectors.max(axis=0, keepdims=True) - min_feature_values

            # To avoid division making NaN's in feature vectors.
            scale_feature_values[scale_feature_values == 0.] = 1.

            self.feature_vectors -= min_feature_values
            self.feature_vectors /= scale_feature_values


    def get_query(self, qid):
        ''' 
        Return the query with the given id.
        '''
        i = np.where(self.query_ids == qid)[0]

        if i is None:
            raise KeyError('no query exist with the specified id: %d' % qid)

        return self[i[0]]

    def document_count(self, i=None):
        ''' 
        Return the number of documents for the query. If i is None
        than the total number of "documents" is returned.
        '''
        if i is None:
            return self.n_feature_vectors
        else:
            return self.query_indptr[i + 1] - self.query_indptr[i]


    def get_feature_vectors(self, i):
        ''' 
        Return the feature vectors of the documents for the specified query.
        '''
        return self.feature_vectors[self.query_indptr[i]:self.query_indptr[i + 1]]


    def query_count(self):
        ''' 
        Return the number of queries in this Queries.
        '''
        return self.n_queries


    def highest_relevance(self):
        ''' 
        Return the maximum relevance score of a document.
        '''
        return self.max_score


    def longest_document_list(self):
        ''' 
        Return the maximum number of documents query can have.
        '''
        return np.diff(self.query_indptr).max()


def concatenate(queries):
    ''' 
    Concatenate the given list of queries into a single Queries object.

    queries: list of Queries
        The list of queries to concatenate.
    '''
    feature_vectors = np.concatenate([q.feature_vectors for q in queries])
    relevance_scores = np.concatenate([q.relevance_scores for q in queries])
    query_indptr = np.concatenate(
                        [np.diff(q.query_indptr) for q in queries]).cumsum()
    query_indptr = np.r_[0, query_indptr]
    max_score = max([q.max_score for q in queries])
    query_ids = np.concatenate([q.query_ids for q in queries])

    assert len(query_ids) == len(np.unique(query_ids)), ('some of the queries '
            'is in more than one collection')

    try:
        feature_indices = (np.concatenate(
            [q.feature_indices.reshape(1, -1) for q in queries]) - 
            queries[0].feature_indices)
    except:
        raise ValueError('feature indices for some queries does not match')

    assert not np.any(feature_indices), 'feature indices for some queries does not match'

    feature_indices = queries[0].feature_indices
    
    return Queries(feature_vectors, relevance_scores, query_indptr,
                   max_score, False, query_ids, feature_indices)


def find_constant_features(queries):
    '''
    Returns indicator array which marks constant features in the query-document
    feature vectors of specified queries.

    Parameters
    ----------
    queries: Queries instance or list of Queries instances
        Queries or collection of queries for which constant
        features are searched for.
    '''
    if isinstance(queries, Queries):
        queries = [queries]

    # The mask for the constant features.
    mask = None

    try:
        for qs in queries:
            if not isinstance(qs, Queries):
                raise ValueError()

            if mask is None:
                mask = np.ones(qs.feature_vectors.shape[1], dtype='bool')
            elif len(mask) != qs.feature_vectors.shape[1]:
                raise ValueError('dimensionality of queries feature vectors '
                                 'in the collection does not match')

            feature_ranges = np.vstack((qs.feature_vectors.min(axis=0), 
                                        qs.feature_vectors.max(axis=0)))

            mask &= (feature_ranges[0] == feature_ranges[1])

        return mask

    except:
        raise ValueError('queries is not an instance of Queries or collection '
                         'of such instances')