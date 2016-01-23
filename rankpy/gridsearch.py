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


import copy
import numpy as np
import logging

from .externals.joblib import Parallel, delayed

from .utils import parallel_helper
from .queries import Queries
from .queries import concatenate

from sklearn.utils import check_random_state
from sklearn.grid_search import ParameterGrid


logger = logging.getLogger(__name__)


def train_test_split(queries, train_size=None, test_size=0.2, documents=False,
                     return_weights=False, random_state=None):
    ''' 
    Split the specified set of queries into training and test sets.

    The portion of queries that ends in the training or test set
    is determined by train_size and test_size parameters, respectively.
    The train_size parameter takes precedence (if specified)

    Parameters:
    -----------
    queries: Queries object
        The set of queries that should be partitioned to a training
        and test set.

    train_size: int or float, optional (default is None)
        If float, denotes the portion of (randomly chosen) queries
        that will become part of the training set. If int, the precise
        number of queries will be put into the training set.
        The complement will make the test set.

    test_size: int or float, optional (default is 0.2)
        If float, denotes the portion of (randomly chosen) queries that
        will become part of the test set. If int, the precise number of
        samples will be put into the test set. The complement will
        make the training set.

    documents: boolean, optional (default is False)
        Instead of splitting the queries into training and test sets,
        the documents will be split. This way, the number of queries
        in the two sets will be the same, but the (relative) number
        of documents will be determined by the ``train_size`` and
        ``test_size`` parameters.

    random_state: RandomState instance, int, or None
        Set this to NumPy random number generator or provide a seed number 
        to get reproducible results.
    '''
    random_state = check_random_state(random_state)

    if documents:
        return __train_test_split_documents(queries, train_size,
                                            test_size, random_state)

    n_queries = queries.query_indptr.size - 1

    if train_size is not None:
        if isinstance(train_size, float):
            if train_size >= 1.0 or train_size <= 0.0:
                raise ValueError('the value of train_size must '
                                 'be in [0.0; 1.0] range')
            train_size = int(train_size * n_queries)
        elif train_size >= n_queries:
            raise ValueError('the specified train_size (%d) must be less than '
                             'the number of queries queries (%d)'
                             % (train_size, n_queries))
        elif train_size < 1:
            raise ValueError('the train_size must be at least 1 (%d was given)'
                             % train_size)
        test_size = n_queries - train_size
    elif test_size is not None:
        if isinstance(test_size, float):
            if test_size >= 1.0 or test_size <= 0.0:
                raise ValueError('the value of test_size must be in '
                                 '[0.0; 1.0] range')
            test_size = int(test_size * n_queries)
        elif test_size >= n_queries:
            raise ValueError('the specified test_size (%d) must be less than '
                             'the number of queries queries (%d)'
                             % (test_size, n_queries))
        elif test_size < 1:
            raise ValueError('the test_size must be at least 1 '
                             '(%d was given)' % test_size)
        train_size = n_queries - test_size
    else:
        raise ValueError('train_size and test_size cannot be both None!')

    if return_weights:
        if documents:
            raise ValueError('return_weights and documents cannot be True '
                             'at once')

        indices = np.arange(n_queries, dtype=np.intc)
        random_state.shuffle(indices)

        train_weights = np.zeros(len(queries), dtype='float64')
        train_weights[indices[test_size:]] = 1.0

        test_weights = np.zeros(len(queries), dtype='float64')
        test_weights[indices[:test_size]] = 1.0

        return train_weights, test_weights

    test_queries_indices = np.arange(n_queries, dtype=np.intc)
    random_state.shuffle(test_queries_indices)

    train_queries_indices = test_queries_indices[test_size:]
    test_queries_indices  = test_queries_indices[:test_size]

    n_query_documents = np.diff(queries.query_indptr)

    test_query_indptr = np.concatenate([[0], n_query_documents[test_queries_indices]])
    train_query_indptr = np.concatenate([[0], n_query_documents[train_queries_indices]])

    np.cumsum(test_query_indptr, out=test_query_indptr)
    np.cumsum(train_query_indptr, out=train_query_indptr)

    assert test_query_indptr[-1] + train_query_indptr[-1] == queries.feature_vectors.shape[0]

    test_feature_vectors = np.empty((test_query_indptr[-1],
                                     queries.feature_vectors.shape[1]),
                                    dtype=queries.feature_vectors.dtype)

    train_feature_vectors = np.empty((train_query_indptr[-1],
                                      queries.feature_vectors.shape[1]),
                                     dtype=queries.feature_vectors.dtype)

    test_relevance_scores = np.empty(test_query_indptr[-1],
                                     dtype=queries.relevance_scores.dtype)
    train_relevance_scores = np.empty(train_query_indptr[-1],
                                      dtype=queries.relevance_scores.dtype)

    for i in range(len(test_query_indptr) - 1):
        test_feature_vectors[test_query_indptr[i]:test_query_indptr[i + 1]] = queries.feature_vectors[queries.query_indptr[test_queries_indices[i]]:queries.query_indptr[test_queries_indices[i] + 1]]
        test_relevance_scores[test_query_indptr[i]:test_query_indptr[i + 1]] = queries.relevance_scores[queries.query_indptr[test_queries_indices[i]]:queries.query_indptr[test_queries_indices[i] + 1]]

    for i in range(len(train_query_indptr) - 1):
        train_feature_vectors[train_query_indptr[i]:train_query_indptr[i + 1]] = queries.feature_vectors[queries.query_indptr[train_queries_indices[i]]:queries.query_indptr[train_queries_indices[i] + 1]]
        train_relevance_scores[train_query_indptr[i]:train_query_indptr[i + 1]] = queries.relevance_scores[queries.query_indptr[train_queries_indices[i]]:queries.query_indptr[train_queries_indices[i] + 1]]

    feature_indices = None
    if queries.feature_indices is not None:
        feature_indices = queries.feature_indices

    train_query_ids = None
    test_query_ids = None

    if queries.query_ids is not None:
        train_query_ids = queries.query_ids[train_queries_indices].copy()
        test_query_ids = queries.query_ids[test_queries_indices].copy()
        
    test_queries = Queries(test_feature_vectors, test_relevance_scores, test_query_indptr, queries.max_score, True, query_ids=test_query_ids, feature_indices=feature_indices)
    train_queries = Queries(train_feature_vectors, train_relevance_scores, train_query_indptr, queries.max_score, True, query_ids=train_query_ids, feature_indices=feature_indices)

    return train_queries, test_queries


def __train_test_split_documents(queries, train_size=None, test_size=0.2,
                                 random_state=None):
    ''' 
    Split the specified set of queries into training and test sets.

    Instead of splitting queries into a training and test sets, the documents
    within each query will be divided. The portion of documents ending
    in each set is determined by train_size and test_size parameters.
    The train_size parameter takes precedence (if specified)

    Parameters:
    -----------
    queries: Queries object
        The set of queries that should be partitioned to a training
        and test set.

    train_size: float, optional (default is None)
        Denotes the portion of (randomly chosen) queries that will
        become part of the training set. The complement will make
        the test set.

    test_size: float, optional (default is 0.2)
        Denotes the portion of (randomly chosen) queries that will
        become part of the test set. The complement will make
        the training set.

    random_state: RandomState instance, int, or None
        Set this to NumPy random number generator or provide a seed number 
        to get reproducible results.
    '''
    random_state = check_random_state(random_state)

    query_document_count = np.diff(queries.query_indptr)

    if train_size is not None:
        if not isinstance(train_size, float) or train_size >= 1.0 or train_size <= 0.0:
            raise ValueError('train_size must be float between 0.0 and 1.0')

        query_train_document_count = (train_size * 
                                      query_document_count).astype(np.intc)

        if np.any(query_train_document_count == 0):
            warn('some queries in training set would not contain any document '\
                 'for train_size=%.2f (qid: %r)' % (train_size, np.where(query_train_document_count == 0)[0]))

            query_train_document_count += 2 * (query_train_document_count == 0).astype(np.intc)

            if np.any(query_document_count - query_train_document_count <= 0):
                raise ValueError('queries with less than 2 documents are not supported')
    elif test_size is not None:
        if not isinstance(test_size, float) or test_size >= 1.0 or test_size <= 0.0:
            raise ValueError('test_size must be float between 0.0 and 1.0')

        query_test_document_count = (test_size * query_document_count).astype(np.intc)

        if np.any(query_test_document_count == 0):
            warn('some queries in test set would not contain any document '\
                 'for test_size=%.2f (qid: %r)' % (test_size, np.where(query_test_document_count == 0)[0]))

            query_test_document_count += 2 * (query_test_document_count == 0).astype(np.intc)

            if np.any(query_document_count - query_test_document_count <= 0):
                raise ValueError('queries with less than 2 documents are not supported')

        query_train_document_count = query_document_count - query_test_document_count
    else:
        raise ValueError('train_size and test_size cannot be both None!')

    # Magic that makes the whole thing work (hopefully in every case!).
    fold_document_indices = [[np.array([], dtype=np.intp)] * 2]
    fold_document_counts = []

    for qid in range(queries.query_count()):
        fold_document_indices.append(np.array_split(queries.query_indptr[qid] \
                                      + random_state.permutation(query_document_count[qid]), [query_train_document_count[qid]]))
        fold_document_counts.append([document_indices.shape[0] for document_indices in fold_document_indices[-1]])

    # Using numpy arrays in Fortran-contiguous order for fancy and efficient column indexing.
    fold_document_indices = np.array(fold_document_indices, dtype=object, order='F')
    fold_document_counts = np.array(fold_document_counts, dtype=np.intc, order='F')

    # This will result in a list of arrays (one per fold) holding indices of documents for each query.
    fold_document_indices = [np.concatenate(fold_document_indices[:, i]) for i in range(2)]

    # This will result in a list of array (one per fold) holding number of documents per query.
    fold_document_counts = [fold_document_counts[:, i] for i in range(2)]

    train_folds_document_indices = fold_document_indices[0]
    train_folds_document_counts = fold_document_counts[0]

    train_feature_vectors = queries.feature_vectors[train_folds_document_indices, :]
    train_relevance_scores = queries.relevance_scores[train_folds_document_indices]
    train_query_indptr = np.r_[0, train_folds_document_counts].cumsum()
        
    train_queries = Queries(train_feature_vectors, train_relevance_scores, train_query_indptr, queries.max_score, False, query_ids=queries.query_ids, feature_indices=queries.feature_indices)

    test_fold_document_indices = fold_document_indices[1]
    test_fold_document_counts = fold_document_counts[1]

    test_feature_vectors = queries.feature_vectors[test_fold_document_indices, :]
    test_relevance_scores = queries.relevance_scores[test_fold_document_indices]
    test_query_indptr = np.r_[0, test_fold_document_counts].cumsum()

    test_queries = Queries(test_feature_vectors, test_relevance_scores, test_query_indptr, queries.max_score, False, query_ids=queries.query_ids, feature_indices=queries.feature_indices)

    return train_queries, test_queries


def shuffle_split_queries(queries, n_folds=10, random_state=None,
                          return_weights=False):
    ''' 
    Split the specified set of queries into `n_folds` for cross-validation.

    Parameters:
    -----------
    queries: Queries object
        The set of queries that should be partitioned to a training and test set.

    n_folds: integer, optional (default is 5)
        The number of folds.

    random_state: RandomState instance, int, or None, optional (default is None)
        Set this to NumPy random number generator or provide a seed number 
        to get reproducible results.

    Returns
    -------
    (train_queries, test_queries): pair of Queries instances
        If return_weights is True, `n_folds` pairs of queries for training
        and evaluation of a ranker.

    or

    (train_weights, test_weights): pair of arrays of floats, shape = (n_queries,)
        If return_weights is True, `n_folds` pair of arrays containing non-zero
        weight only for queries which should be included into training and
        testing, respectively.
    '''
    random_state = check_random_state(random_state)

    if return_weights:
        x = np.arange(len(queries), dtype='int32')
        random_state.shuffle(x)

        for indices in np.array_split(x, n_folds):
            train_weights = np.ones(len(queries), dtype='float64')
            train_weights[indices] = 0.0

            test_weights = np.zeros(len(queries), dtype='float64')
            test_weights[indices] = 1.0

            yield train_weights, test_weights
    else:
        query_document_count  = np.diff(queries.query_indptr)

        if np.any(query_document_count < n_folds):
            raise ValueError('queries contain a document with less documents'\
                             ' than the wanted number of folds')

        # Magic that makes the whole thing work (hopefully in every case!).
        fold_document_indices = [[np.array([], dtype=np.intp)] * n_folds]
        fold_document_counts = []

        for qid, n_documents in enumerate(query_document_count):
            fold_document_indices.append(np.array_split(queries.query_indptr[qid] + random_state.permutation(n_documents), n_folds))
            fold_document_counts.append([document_indices.shape[0] for document_indices in fold_document_indices[-1]])

        # Using numpy arrays in Fortran-contiguous order for fancy and efficient column indexing.
        fold_document_indices = np.array(fold_document_indices, dtype=object, order='F')
        fold_document_counts = np.array(fold_document_counts, dtype=np.intc, order='F')

        # This will result in a list of arrays (one per fold) holding indices of documents for each query.
        fold_document_indices = [np.concatenate(fold_document_indices[:, i]) for i in range(n_folds)]

        # This will result in a list of array (one per fold) holding number of documents per query.
        fold_document_counts = [fold_document_counts[:, i] for i in range(n_folds)]

        for valid_fold in range(n_folds):
            valid_fold_document_indices = fold_document_indices[valid_fold]
            valid_fold_document_counts = fold_document_counts[valid_fold]

            valid_feature_vectors = queries.feature_vectors[valid_fold_document_indices, :]
            valid_relevance_scores = queries.relevance_scores[valid_fold_document_indices]
            valid_query_indptr = np.r_[0, valid_fold_document_counts].cumsum()

            valid_queries = Queries(valid_feature_vectors, valid_relevance_scores, valid_query_indptr, queries.max_score, False, query_ids=queries.query_ids, feature_indices=queries.feature_indices)

            # Make a shallow copy of the lists ...
            train_folds_document_indices = list(fold_document_indices)
            train_folds_document_counts = list(fold_document_counts)
            # ... then remove the valid fold ...
            del train_folds_document_indices[valid_fold]
            del train_folds_document_counts[valid_fold]
            # ... and finally concatenate them together
            train_folds_document_indices = np.concatenate(train_folds_document_indices)
            train_folds_document_counts = sum(train_folds_document_counts)

            train_feature_vectors = queries.feature_vectors[train_folds_document_indices, :]
            train_relevance_scores = queries.relevance_scores[train_folds_document_indices]
            train_query_indptr = np.r_[0, train_folds_document_counts].cumsum()
            
            train_queries = Queries(train_feature_vectors, train_relevance_scores, train_query_indptr, queries.max_score, False, query_ids=queries.query_ids, feature_indices=queries.feature_indices)

            yield train_queries, valid_queries


def shuffle_split_documents(queries, n_folds=10, random_state=None):
    ''' 
    Split the specified set of queries into `n_folds` for cross-validation.

    The documents of every query will be evently divided into `n_folds`
    number of folds.

    Parameters:
    -----------
    queries: Queries object
        The set of queries that should be partitioned to a training and test set.

    n_folds: integer, optional (default is 5)
        The number of folds.

    random_state: RandomState instance, int, or None, optional (default is None)
        Set this to NumPy random number generator or provide a seed number 
        to get reproducible results.
    '''
    random_state = check_random_state(random_state)

    query_document_count  = np.diff(queries.query_indptr)

    if np.any(query_document_count < n_folds):
        raise ValueError('queries contain a document with less documents'\
                         ' than the wanted number of folds')

    # Magic that makes the whole thing work (hopefully in every case!).
    fold_document_indices = [[np.array([], dtype=np.intp)] * n_folds]
    fold_document_counts = []

    for qid, n_documents in enumerate(query_document_count):
        fold_document_indices.append(np.array_split(queries.query_indptr[qid] + random_state.permutation(n_documents), n_folds))
        fold_document_counts.append([document_indices.shape[0] for document_indices in fold_document_indices[-1]])

    # Using numpy arrays in Fortran-contiguous order for fancy and efficient column indexing.
    fold_document_indices = np.array(fold_document_indices, dtype=object, order='F')
    fold_document_counts = np.array(fold_document_counts, dtype=np.intc, order='F')

    # This will result in a list of arrays (one per fold) holding indices of documents for each query.
    fold_document_indices = [np.concatenate(fold_document_indices[:, i]) for i in range(n_folds)]

    # This will result in a list of array (one per fold) holding number of documents per query.
    fold_document_counts = [fold_document_counts[:, i] for i in range(n_folds)]

    for valid_fold in range(n_folds):
        valid_fold_document_indices = fold_document_indices[valid_fold]
        valid_fold_document_counts = fold_document_counts[valid_fold]

        valid_feature_vectors = queries.feature_vectors[valid_fold_document_indices, :]
        valid_relevance_scores = queries.relevance_scores[valid_fold_document_indices]
        valid_query_indptr = np.r_[0, valid_fold_document_counts].cumsum()

        valid_queries = Queries(valid_feature_vectors, valid_relevance_scores, valid_query_indptr, queries.max_score, False, query_ids=queries.query_ids, feature_indices=queries.feature_indices)

        # Make a shallow copy of the lists ...
        train_folds_document_indices = list(fold_document_indices)
        train_folds_document_counts = list(fold_document_counts)
        # ... then remove the valid fold ...
        del train_folds_document_indices[valid_fold]
        del train_folds_document_counts[valid_fold]
        # ... and finally concatenate them together
        train_folds_document_indices = np.concatenate(train_folds_document_indices)
        train_folds_document_counts = sum(train_folds_document_counts)

        train_feature_vectors = queries.feature_vectors[train_folds_document_indices, :]
        train_relevance_scores = queries.relevance_scores[train_folds_document_indices]
        train_query_indptr = np.r_[0, train_folds_document_counts].cumsum()
        
        train_queries = Queries(train_feature_vectors, train_relevance_scores, train_query_indptr, queries.max_score, False, query_ids=queries.query_ids, feature_indices=queries.feature_indices)

        yield train_queries, valid_queries


def expand_related_parameters(d):
    '''
    Expand grouped parameters into individual ones.

    Parameters
    ----------
    d : dict
        The dictionary mapping parameter names to their values.

    Returns
    -------
    d : dict
        The same input object is returned.
    '''
    deep = False

    for ps in d.keys():
        if not isinstance(ps, basestring):
            vs = d.pop(ps)

            if len(ps) != len(vs):
                # Push back the values for error checking.
                d[ps] = vs

                raise ValueError('number of names does not match '
                                 'number of values: %d != %d'
                                 % (len(ps), len(vs)))

            for p, v in zip(ps, vs):
                if p in d:
                    raise ValueError('parameter %s was specified multiple '
                                     'times' % p)

                if not isinstance(p, basestring):
                    deep = True

                d[p] = v

    return (d if not deep else expand_related_parameters(d))


def parallel_fit_ranker(ranker_cls, parameters, training_queries,
                        estopping_queries, validation_queries):
    '''
    Helper method used with Parallel to train and evaluate
    multiple rankers at once.
    '''
    # Expand grouped parameters into individual ones.
    expand_related_parameters(parameters)

    logger.info('Fitting %s using parameters: %r'
                % (ranker_cls.__name__, parameters))

    # Create a ranker, ...
    ranker = ranker_cls(**parameters)

    # ... train it on queries with early stopping...
    ranker.fit(training_queries, validation_queries=estopping_queries)

    # ... and finally evaluate the ranker on holdout queries.
    holdout_performance = ranker.evaluate(validation_queries)

    logger.info('%s performance %11.8f (gridsearch holdout %s)'
                % (ranker_cls.__name__, holdout_performance, ranker.metric))

    return ranker, holdout_performance


def gridsearch(ranker_cls, param_grid, training_queries, 
               estopping_queries=None, validation_queries=None,
               return_scores=False, n_jobs=1, random_state=None):
    '''
    Perform grid search for the given ranker model over discrete
    sets of parameters.

    Parameters
    ----------
    ranker_cls: ranker class object
        The ranker class, which is used to instantiate a ranker with
        different parameters from the grid.

    param_grid: dict or ParameterGrid
        The parameter grid over which the model will be evaluated.

    queries: Queries instance
        The set of queries used for cross-validation.

    estopping_queries: Queries instance
        A set of queries used for early stopping during the training of
        a ranker initialized with a set of parameters from the grid.

    validation_queries: Queries instance
        A set of queries used for early stopping during the re-training of
        a ranker with the best parameters setting found in cross-validation.

    return_scores: bool, default: False
        If True, triplets (parameters, model, validation performance) are
        returned together with the best ranking model.

    n_jobs: int, default: 1
        The number of worker threads computing in parallel.

    random_state: RandomState, or int, or None
        The random number generator used for internal randomness, which 
        is involed in splitting the queries.

    Returns
    -------
    ranker: `ranker_cls` instance
        The ranker trained on `queries` (optionally stopped early using
        `estopping_queries`) with the best parameter setting found in
        `param_grid`.

    or

    (ranker, models): `ranker_cls` instance, list of `ranker_cls` instances
        If `return_scores` is True, the best ranker is additionally accompanied
        with the data about all models trained with different parameters.
    '''
    if not isinstance(param_grid, ParameterGrid):
        if isinstance(param_grid, dict):
            param_grid = ParameterGrid(param_grid)
        else:
            raise ValueError('param_grid needs to be an instance of '
                             'sklearn.grid_search.ParameterGrid (or dict), '
                             'but %s was found' % type(param_grid))

    # Train rankers in parallel.
    models_scores = Parallel(n_jobs=n_jobs, backend='threading')(
                        delayed(parallel_fit_ranker, check_pickle=False)(
                            ranker_cls, parameters, training_queries,
                            estopping_queries, validation_queries)
                        for parameters in param_grid
                    )

    models, scores = zip(*models_scores)

    # Find the best ranker based on the performance
    # on the validation queries.
    best_grid_point = np.argmax(scores)

    if return_scores:
        return models[best_grid_point], zip(param_grid, models, scores)
    else:
        return models[best_grid_point]
        