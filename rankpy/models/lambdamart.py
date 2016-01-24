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


import os
import logging
import sklearn
import warnings

import numpy as np

from ..externals.joblib import Parallel, delayed

from sklearn.ensemble import RandomForestRegressor

from sklearn.utils import check_random_state

from pines.estimators import DecisionTreeRegressor as PinesDecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.tree._tree import TREE_UNDEFINED, TREE_LEAF

from shutil import rmtree
from tempfile import mkdtemp

from ..utils import pickle
from ..utils import unpickle
from ..utils import parallel_helper
from ..utils import _get_partition_indices
from ..utils import _get_n_jobs
from ..utils import aslist

from ..metrics import MetricFactory
from ..metrics._utils import ranksort_queries

from .lambdamart_inner import parallel_compute_lambdas_and_weights


logger = logging.getLogger(__name__)


def compute_lambdas_and_weights(queries, ranking_scores, metric,
                                output_lambdas, output_weights,
                                query_scales=None, relevance_scores=None,
                                query_weights=None, document_weights=None,
                                influences=None, indices=None,
                                random_state=None, n_jobs=1):
    '''
    Compute first and second derivatives (`lambdas` and `weights`) of an
    implicit cost function derived from `metric` and rankings of query
    documents induced by `ranking_scores`.

    Parameters
    ----------
    queries : Queries instance
        The set of queries for which lambdas and weights are computed.

    ranking_scores : array of floats, shape = [n_documents]
        A ranking score for each document in the set of queries.

    metric : Metric instance
        The evaluation metric from which the lambdas and the weights
        are computed.

    output_lambdas : array of floats, shape = [n_documents]
        Computed lambdas for every document.

    output_weights : array of floats, shape = [n_documents]
        Computed weights for every document.

    query_scales : array of floats, shape = [n_queries] or None
        The precomputed metric scale value for every query.

    influences : array of floats, shape = [max_relevance, max_relevance]
                 or None
        Used to keep track of (proportional) contribution in high relevant
        documents' lambdas from low relevant documents.

    indices : array of ints, shape = [n_documents, n_nodes] or None
        The indices of terminal nodes which the documents fall into.
        This parameter can be used to recompute lambdas and weights
        after regression tree is built.

    n_jobs : int, optional (default=1)
        The number of workers which are used to compute lambdas and weights
        in parallel.

    Returns
    -------
    implicit_loss : float
        The (implicit) loss derived from `metric` and the rankings
        induced by `ranking_scores`.
    '''
    # Partition the queries into contiguous parts based on the number of jobs.
    partitions = _get_partition_indices(0, len(queries), n_jobs)

    if relevance_scores is None:
        # Use the relevance scores associated with queries.
        relevance_scores = queries.relevance_scores
        query_relevance_strides = queries.query_relevance_strides
    else:
        # These will be computed in `parallel_compute_lambdas_and_weights`
        # from the given `relevance_scores`.
        query_relevance_strides = None

    return sum(Parallel(n_jobs=n_jobs, backend="threading",
                        pre_dispatch='all', batch_size=1)(
        delayed(parallel_compute_lambdas_and_weights, check_pickle=False)(
            partitions[i], partitions[i + 1], queries.query_indptr,
            ranking_scores, relevance_scores, queries.max_score,
            query_relevance_strides, metric.backend(), query_scales,
            influences, indices, query_weights, document_weights,
            output_lambdas, output_weights, random_state
        )
        for i in range(partitions.shape[0] - 1))
    )


def compute_newton_gradient_steps(estimator, queries, ranking_scores, metric,
                                  lambdas, weights, query_scales=None,
                                  relevance_scores=None, query_weights=None,
                                  document_weights=None, random_state=None,
                                  n_jobs=1, use_pines=False):
    '''
    Compute a single gradient step for each terminal node of the given
    regression tree estimator using Newton's method.

    Parameters:
    -----------
    estimator : DecisionTreeRegressor or RandomForestRegressor instance
        A regression tree or an ensemble of regression trees for which
        the gradient steps are computed.

    queries : Queries instance
        The query documents determine which terminal nodes of the tree
        the corresponding lambdas and weights fall into.

    ranking_scores : array, shape = [n_documents]
        A ranking score for each document in the set of queries.

    metric : Metric instance
        An evaluation metric from which the lambdas and the weights
        are computed.

    lambdas : array of floats, shape = [n_documents]
        A current 1st order derivatives of the (implicit) LambdaMART
        loss function.

    weights : array of floats, shape = [n_documents]
        A current 2nd order derivatives of the (implicit) LambdaMART
        loss function.

    query_scales : array of floats, shape = [n_queries] or None
        A precomputed metric scale value for every query.

    n_jobs : int, optional (default=1)
        The number of workers, which are used to compute the lambdas
        and weights in parallel.
    '''
    if isinstance(estimator, RandomForestRegressor):
        estimators = estimator.estimators_
    else:
        estimators = [estimator]

    for estimator in estimators:
        # Get the number of nodes in the current regression tree.
        if use_pines:
            node_count = estimator._tree.num_of_nodes() + 1
            indices = estimator._tree.apply(
                                    queries.feature_vectors).astype('int32')
        else:
            node_count = estimator.tree_.node_count
            indices = estimator.tree_.apply(
                                    queries.feature_vectors).astype('int32')

        # To get mathematically correct Newton steps in every terminal node
        # of the tree we need to recompute the lambdas and weights with the
        # information about what terminal nodes the documents fall into.
        #
        # Note that this is actually necessary to get correct weights.
        #
        # XXX: After getting consistently (and significantly) better results
        #      without this 'correction step', this step is no longer used.
        # 
        # compute_lambdas_and_weights(queries, ranking_scores, metric,
        #                             lambdas, weights, query_scales,
        #                             relevance_scores, query_weights,
        #                             document_weights, None, indices,
        #                             random_state, n_jobs)

        gradients = np.bincount(indices, lambdas, node_count)

        with np.errstate(invalid='ignore'):
            np.divide(gradients, np.bincount(indices, weights, node_count),
                      out=gradients)

        # Find terminal nodes which got NaN as a resulting Newton's step.
        if use_pines:
            nan_gradients_mask = (np.isnan(gradients) & estimator._tree.leaf_mask())
        else:
            nan_gradients_mask = (np.isnan(gradients) &
                                  (estimator.tree_.children_left == TREE_LEAF))

        if nan_gradients_mask.any():
            # If there is a NaN in gradients it means there is a tree leaf
            # into which only documents with 0 weights and lambdas fell.
            # Since the number of such documents is usually small it is
            # wasteful to waste the tree capacity this way. To remedy this
            # it suffices to increase the `min_samples_leaf` parameter to
            # (at least) the value printed in the warning.
            #
            # XXX: This leaves the internal node values set to NaN! It is
            #      expected these will be never used. 
            gradients[nan_gradients_mask] = 0.
            logger.warn('Regression tree terminal node values (indices: %s) '
                        'were computed from samples (counts: %s) with zero '
                        'weights, it might be a good idea to drop the cutoff '
                        'rank in the metric or increase `min_samples_leaf` '
                        'parameter in order not to waste the tree capacity '
                        'this way.'
                        % (', '.join([str(x) for x \
                                        in nan_gradients_mask.nonzero()[0]]),
                           ', '.join([str(x) for x in np.bincount(
                                          indices,
                                          np.ones(len(indices),
                                                  dtype='int32'),
                                          node_count)[nan_gradients_mask]])))

        # Remarkably, numpy.copyto method side-steps the access protection
        # of `tree_.value` array, which is supposed to be 'read-only'.
        if use_pines:
            estimator._tree._leaf_values[:gradients.size] = gradients
        else:
            np.copyto(estimator.tree_.value, gradients.reshape(-1, 1, 1))


class LambdaMART(object):
    '''
    LambdaMART learning to rank model.

    Parameters
    ----------
    metric : string, optional (default="NDCG")
            Specify evaluation metric which will be used as a utility
            function (i.e. metric of `goodness`) optimized by this model.
            Supported metrics are "NDCG", "WTA", "ERR", with an optional
            suffix "@{N}", where {N} can be any positive integer.

    n_estimators : int, optional (default=1000)
        The maximum number of regression trees that will be trained.

    max_depth : int or None, optional (default=None)
        The maximum depth of the regression trees. This parameter is ignored
        if `max_leaf_nodes` is specified (see description of `max_leaf_nodes`).

    max_leaf_nodes : int or None, optional (default=7)
        The maximum number of leaf nodes. If not None, the `max_depth`
        parameter will be ignored. The tree building strategy also changes
        from depth search first to best search first, which can lead to
        substantial decrease in training time.

    max_features : int, float, or None, optional (default=None)
        The maximum number of features that is considered for splitting when
        regression trees are being built. If float is given it is interpreted
        as a percentage. If None, all feature will be used.

    min_samples_split : int, optional (default=2)
        The minimum number of documents required to split an internal node.

    min_samples_leaf : int, optional (default=1)
        The minimum number of documents required to be at a terminal node.

    shrinkage : float, optional (default=0.1)
        The learning rate (shrinkage factor) that will be used to regularize
        the regression trees in a way which prevents them from making the full
        (Newton's method) gradient step.

    use_newton_method : bool, optional (default=True)
        Estimate the gradient step in each terminal node of regression
        trees using Newton-Raphson method.

    use_random_forest : int, optional (default=0):
        If positive, specify the number of trees in the random forest
        which will be used instead of a single regression tree.

    estopping : int, optional (default=50)
        The number of subsequent iterations after which the training is
        stopped early if no improvement is observed on the training 
        or validation queries (if provided).

    min_n_estimators: int, optional (default=1)
        The minimum number of regression trees that will be trained regardless
        of the `n_estimators` and `estopping` values. Beware that using this
        option may lead to suboptimal performance of the model.

    base_model : Base learning to rank model, optional (default=None)
        The base model that is used to get initial ranking scores.

    n_jobs : int, optional (default=1)
        The number of background working threads that will be spawned to
        compute the desired values faster. If -1 is given then the number
        of available CPUs will be used.

    Attributes
    ----------
    training_performance: array of doubles
        The performance of the model measured after training each
        tree/forest regression estimator on training queries.

    validation_performance: array of doubles
        The performance of the model measured after training each
        tree/forest regression estimator on validation queries.

    XXX: Finish this!!!
    '''
    def __init__(self, metric='NDCG', n_estimators=1000, max_depth=None,
                 max_leaf_nodes=7, max_features=None, min_samples_split=2,
                 min_samples_leaf=1, shrinkage=0.1, use_newton_method=True,
                 use_random_forest=0, random_thresholds=False,
                 use_logit_boost=False, estopping=50, min_n_estimators=1,
                 base_model=None, n_jobs=1, random_state=None, use_pines=False):
        self.estimators = []
        self.n_estimators = n_estimators
        self.min_n_estimators = min_n_estimators
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.estopping = estopping
        self.metric = metric
        self.base_model = base_model
        self.shrinkage = shrinkage
        self.use_logit_boost = use_logit_boost
        self.use_newton_method = use_newton_method
        self.use_random_forest = use_random_forest
        self.random_thresholds = random_thresholds
        self.n_jobs = _get_n_jobs(n_jobs)
        self.training_performance = None
        self.validation_performance = None
        self.best_performance = None
        self.use_pines = use_pines
        self.random_state = check_random_state(random_state)

        # Force the use of newer version of scikit-learn.
        if int(sklearn.__version__.split('.')[1]) < 17:
            raise ValueError('LambdaMART is built on scikit-learn '
                             'implementation of regression trees '
                             'version 17. Please, update your'
                             'scikit-learn package before using RankPy.')

    def fit(self, training_queries, training_query_weights=None,
            validation_queries=None, validation_query_weights=None,
            presort=False, trace=None):
        '''
        Train a LambdaMART model on given training queries. Optionally,
        use validation queries for finding an optimal number of trees
        using early stopping.

        Parameters
        ----------
        training_queries : Queries instance
            The set of queries from which the model will be trained.

        training_query_weights : array of floats, shape = [n_queries],
                                 or None
            The weight given to each training query, which is used to
            measure its importance. Queries with 0.0 weight will never
            be used in training.

        validation_queries : Queries instance or None
            The set of queries used for early stopping.

        validation_query_weights : array of floats, shape = [n_queries]
                                   or None
            The weight given to each validation query, which is used to
            measure its importance. Queries with 0.0 weight will never
            be used in validation.

        presort : bool, optional (default=False)
            Whether to presort the data to speed up the finding of best
            splits in training of regression trees.

        trace : list of strings, optional (default=None)
            Supported values are: `lambdas`, `gradients`, and `influences`.
            Since the number of documents and estimators can be large it is
            not advised to use the values together. When `lambdas` is given,
            then the true and estimated lambdas will be stored, and similarly,
            when `gradients` are given, then the true and estimated Newton
            steps will be stored. Use `influences` if you want to track
            (proportional) contribution of lambdas from lower relevant
            documents on high relevant ones.

        Returns
        -------
        self : object
            Returns self.
        '''
        metric = MetricFactory(self.metric, aslist(training_queries,
                               validation_queries), self.random_state)

        if self.base_model is None:
            training_scores = 1e-6 * self.random_state.rand(
                training_queries.document_count()).astype('float64')
        else:
            training_scores = np.ascontiguousarray(
                                  self.base_model.predict(training_queries,
                                                          n_jobs=self.n_jobs),
                                  dtype='float64')

        if training_query_weights is None:
            training_query_weights = np.ones(len(training_queries),
                                             dtype='float64')

        # Check the weight array shape and dtype.
        if (getattr(training_query_weights, 'dtype', None) != 'float64' or
            not training_query_weights.flags.contiguous):
            training_query_weights = np.ascontiguousarray(
                                training_query_weights, dtype='float64')

        if training_query_weights.shape != (len(training_queries), ):
            raise ValueError('query weights array shape != (%d, )'
                             % len(training_queries))

        if (training_query_weights < 0.0).any():
            raise ValueError('query weights must be non-negative')

        document_weights = np.zeros(training_queries.document_count(),
                                    dtype='float64')

        # Set document weights for documents of queries
        # with non-zero weight to 1.0.
        for i, qw in enumerate(training_query_weights):
            if qw > 0.0:
                document_weights[training_queries.qds[i]] = 1.0

        massless_document_indices = (document_weights == 0.0).nonzero()[0]

        # If the metric used for training is normalized, it is advantageous
        # to precompute the scaling factor for each query in advance.
        training_query_scales = metric.compute_scale(training_queries)

        # Keep the training performance of LambdaMART
        # for every stage of training.
        self.training_performance = np.empty(self.n_estimators,
                                             dtype='float64')
        self.training_performance.fill(np.nan)

        self.training_losses = np.zeros(self.n_estimators,
                                        dtype='float64')
        self.training_losses.fill(np.nan)

        # The pseudo-responses (lambdas) for each document.
        training_lambdas = np.empty(training_queries.document_count(),
                                    dtype='float64')

        # The optimal gradient descent step sizes for each document.
        training_weights = np.empty(training_queries.document_count(),
                                    dtype='float64')

        # The lambdas and predictions may be kept for late analysis.
        if trace is not None:
            # Create temporary directory for traced data.
            TEMP_DIRECTORY_NAME = mkdtemp(prefix='lambdamart.trace.data.tmp',
                                          dir='.')

            logger.info('Created temporary directory (%s) for traced data.'
                        % TEMP_DIRECTORY_NAME)

            self.trace_lambdas = trace.count('lambdas') > 0
            self.trace_gradients = trace.count('gradients') > 0
            self.trace_influences = trace.count('influences') > 0

            # The pseudo-responses (lambdas) for each document:
            # the true and estimated values.
            if self.trace_lambdas:
                # Use memory mapping to store large matrices.
                self.stage_training_lambdas_truth = np.memmap(os.path.join(TEMP_DIRECTORY_NAME, 'training.lambdas.truth.tmp.npy'), dtype='float64', mode='w+', shape=(self.n_estimators, training_queries.document_count()))
                self.stage_training_lambdas_predicted = np.memmap(os.path.join(TEMP_DIRECTORY_NAME, 'training.lambdas.predicted.tmp.npy'), dtype='float64', mode='w+', shape=(self.n_estimators, training_queries.document_count()))

                if validation_queries is not None:
                    self.stage_validation_lambdas_truth = np.memmap(os.path.join(TEMP_DIRECTORY_NAME, 'validation.lambdas.truth.tmp.npy'), dtype='float64', mode='w+', shape=(self.n_estimators, validation_queries.document_count()))
                    self.stage_validation_lambdas_predicted = np.memmap(os.path.join(TEMP_DIRECTORY_NAME, 'validation.lambdas.predicted.tmp.npy'), dtype='float64', mode='w+', shape=(self.n_estimators, validation_queries.document_count()))

            if self.trace_gradients and not self.use_newton_method:
                warnings.warn('tracing gradients is possible only if '
                              'use_newton_method is True -- trace ignored')
                self.trace_gradients = False

            # The (loss) gradient steps for each query-document pair:
            # the true and estimated by the regression trees.
            if self.trace_gradients:
                # Use memory mapping to store large matrices.
                self.stage_training_gradients_truth = np.memmap(os.path.join(TEMP_DIRECTORY_NAME, 'training.gradients.truth.tmp.npy'), dtype='float64', mode='w+', shape=(self.n_estimators, training_queries.document_count()))
                self.stage_training_gradients_predicted = np.memmap(os.path.join(TEMP_DIRECTORY_NAME, 'training.gradients.predicted.tmp.npy'), dtype='float64', mode='w+', shape=(self.n_estimators, training_queries.document_count()))

                if validation_queries is not None:
                    self.stage_validation_gradients_truth = np.memmap(os.path.join(TEMP_DIRECTORY_NAME, 'validation.gradients.truth.tmp.npy'), dtype='float64', mode='w+', shape=(self.n_estimators, validation_queries.document_count()))
                    self.stage_validation_gradients_predicted = np.memmap(os.path.join(TEMP_DIRECTORY_NAME, 'validation.gradients.predicted.tmp.npy'), dtype='float64', mode='w+', shape=(self.n_estimators, validation_queries.document_count()))

            if self.trace_influences:
                self.stage_training_influences = np.zeros((self.n_estimators, training_queries.highest_relevance() + 1, training_queries.highest_relevance() + 1), dtype='float64')

                if validation_queries is not None:
                    self.stage_validation_influences = np.zeros((self.n_estimators, validation_queries.highest_relevance() + 1, validation_queries.highest_relevance() + 1), dtype='float64')

                # Can work only in single threaded mode.
                if self.n_jobs > 1:
                    warnings.warn('cannot use multi-threaded training while '
                                  'tracing influences -- setting n_jobs to 1')
                    self.n_jobs = 1

            # Used when the model is saved to get rid of it.
            self.tmp_directory = TEMP_DIRECTORY_NAME
        else:
            self.trace_lambdas = False
            self.trace_gradients = False
            self.trace_influences = False

        # Initialize the same components for validation
        # queries as the training queries.
        if validation_queries is not None:
            if validation_query_weights is None:
                validation_query_weights = np.ones(len(validation_queries),
                                                   dtype='float64')

            # Check the weight array shape and dtype.
            if (getattr(validation_query_weights, 'dtype', None) != 'float64' or
                not validation_query_weights.flags.contiguous):
                validation_query_weights, = np.ascontiguousarray(
                                validation_query_weights, dtype='float64')

            if validation_query_weights.shape != (len(validation_queries), ):
                raise ValueError('validation query weights array '
                                 'shape != (%d, )' % len(validation_queries))

            if (validation_query_weights < 0.0).any():
                raise ValueError('validation query weights must '
                                 'be non-negative')

            validation_document_weights = np.zeros(validation_queries.document_count(),
                                                   dtype='float64')

            # Set document weights for documents of validation
            # queries with non-zero weight to 1.0.
            for i, qw in enumerate(validation_query_weights):
                if qw > 0.0:
                    validation_document_weights[validation_queries.qds[i]] = 1.0

            massless_validation_document_indices = (validation_document_weights == 0.0).nonzero()[0]

            validation_query_scales = metric.compute_scale(validation_queries)

            # Keep the validation performance of LambdaMART
            # for every stage of training.
            self.validation_performance = np.empty(self.n_estimators,
                                                   dtype='float64')
            self.validation_performance.fill(np.nan)

            self.validation_losses = np.empty(self.n_estimators,
                                              dtype='float64')
            self.validation_losses.fill(np.nan)

            if self.base_model is None:
                validation_scores = 1e-6 * self.random_state.rand(validation_queries.document_count()).astype(np.float64)
            else:
                validation_scores = np.ascontiguousarray(
                                        self.base_model.predict(validation_queries),
                                        dtype='float64')

            if self.trace_lambdas or self.trace_influences:
                # The pseudo-responses (lambdas) for each document
                # in validation queries.
                validation_lambdas = np.empty(validation_queries.document_count(),
                                              dtype='float64')
                # The optimal gradient descent step sizes for each document
                # in validation queries.
                validation_weights = np.empty(validation_queries.document_count(),
                                              dtype='float64')

        # Presort feature values for faster training of the regression trees?
        if presort:
            feature_vectors_idx_sorted = \
                np.asfortranarray(np.argsort(training_queries.feature_vectors,
                                  axis=0), dtype='int32')
        else:
            feature_vectors_idx_sorted = None

        float64_eps = np.finfo('float64').eps

        # The best iteration index and performance value
        # on validation (or training) queries.
        best_performance = -np.inf
        best_performance_k = -1

        # How many iterations the performance has not improved
        # on validation (or training) queries.
        performance_not_improved = 0

        logger.info('Training of LambdaMART model has started.')

        self.n_estimators = max(self.n_estimators, self.min_n_estimators)

        # Iteratively build a sequence of regression trees.
        self.range = range
        for k in self.range(self.n_estimators):    
            training_influences = self.stage_training_influences[k] if self.trace_influences else None

            # Computes the pseudo-responses (lambdas) and gradient step
            # factors (weights) for the current regression tree.
            self.training_losses[k] = compute_lambdas_and_weights(
                                            training_queries, training_scores,
                                            metric, training_lambdas,
                                            training_weights,
                                            training_query_scales, None,
                                            training_query_weights, 
                                            document_weights,
                                            training_influences,
                                            random_state=self.random_state,
                                            n_jobs=self.n_jobs)

            # Build the predictor for the gradients of the loss using either
            # decision tree or random forest.
            if self.use_random_forest > 0:
                if self.use_pines:
                    raise NotImplementedError('Random forest is not implemented in pines library')
                estimator = RandomForestRegressor(
                                n_estimators=self.use_random_forest,
                                max_depth=self.max_depth,
                                max_leaf_nodes=self.max_leaf_nodes,
                                max_features=self.max_features,
                                min_samples_split=self.min_samples_split,
                                min_samples_leaf=self.min_samples_leaf,
                                random_state=self.random_state,
                                n_jobs=self.n_jobs)

                # Train the regression forest.
                estimator.fit(training_queries.feature_vectors, training_lambdas,
                              sample_weight=(document_weights * 
                                             (training_weights > 0)))

            else:
                if self.random_thresholds:
                    if self.use_pines:
                        raise NotImplementedError('Random thresholds are not implemented in pines library')
                    estimator = ExtraTreeRegressor(
                                    max_depth=self.max_depth,
                                    max_leaf_nodes=self.max_leaf_nodes,
                                    max_features=self.max_features,
                                    min_samples_split=self.min_samples_split,
                                    min_samples_leaf=self.min_samples_leaf,
                                    random_state=self.random_state)
                else:
                    if self.use_pines:
                        estimator = PinesDecisionTreeRegressor(
                                        max_depth=self.max_depth,
                                        #max_leaf_nodes=self.max_leaf_nodes,
                                        #max_features=self.max_features,
                                        #min_samples_split=self.min_samples_split,
                                        min_samples_per_leaf=self.min_samples_leaf,
                                        #random_state=self.random_state
                                    )
                    else:
                        estimator = DecisionTreeRegressor(
                                        max_depth=self.max_depth,
                                        max_leaf_nodes=self.max_leaf_nodes,
                                        max_features=self.max_features,
                                        min_samples_split=self.min_samples_split,
                                        min_samples_leaf=self.min_samples_leaf,
                                        random_state=self.random_state)

                if self.use_logit_boost:
                    with np.errstate(invalid='ignore'):
                        target = np.nan_to_num(training_lambdas / 
                                               training_weights)

                    # Clip the target values to fixed range.
                    np.clip(target, a_min=-4.0, a_max=4.0, out=target)

                    sample_weight = (document_weights *
                                     np.clip(training_weights,
                                             a_min=(2 * float64_eps),
                                             a_max=np.inf))
                else:
                    target = training_lambdas
                    sample_weight = document_weights

                # Train the regression tree.
                estimator.fit(training_queries.feature_vectors, target,
                              sample_weight=sample_weight,
                              check_input=False,
                              X_idx_sorted=feature_vectors_idx_sorted)

            # Store the estimated lambdas for later analysis (if wanted).
            if self.trace_lambdas:
                np.copyto(self.stage_training_lambdas_truth[k],
                          training_lambdas)

                np.copyto(self.stage_training_lambdas_predicted[k],
                          estimator.predict(training_queries.feature_vectors))

                # Set training lambdas of documents with 0 weight to
                # NaN indicating that they were never computed.
                self.stage_training_lambdas_truth[k, massless_document_indices] = np.nan

            # Store the true and estimated gradients for later analysis.
            if self.trace_gradients:
                with np.errstate(invalid='ignore'):
                    np.copyto(self.stage_training_gradients_truth[k],
                              training_lambdas)
                    np.divide(self.stage_training_gradients_truth[k],
                              training_weights,
                              out=self.stage_training_gradients_truth[k])
                    self.stage_training_gradients_truth[k, np.isnan(self.stage_training_gradients_truth[k])] = 0.0

            if validation_queries is not None:
                if self.trace_lambdas or self.trace_influences:
                    validation_influences = self.stage_validation_influences[k] if self.trace_influences else None

                    self.validation_losses[k] = \
                        compute_lambdas_and_weights(validation_queries,
                                                    validation_scores, metric,
                                                    validation_lambdas,
                                                    validation_weights,
                                                    validation_query_scales,
                                                    None,
                                                    validation_query_weights,
                                                    validation_document_weights,
                                                    validation_influences,
                                                    random_state=self.random_state,
                                                    n_jobs=self.n_jobs)

                if self.trace_lambdas:
                    np.copyto(self.stage_validation_lambdas_truth[k],
                              validation_lambdas)

                    np.copyto(self.stage_validation_lambdas_predicted[k],
                              estimator.predict(validation_queries.feature_vectors))

                    # Set validation lambdas of documents with 0 weight to
                    # NaN indicating that they were never computed.
                    self.stage_validation_lambdas_truth[k, massless_validation_document_indices] = np.nan

                if self.trace_gradients:
                    with np.errstate(invalid='ignore'):
                        np.copyto(self.stage_validation_gradients_truth[k],
                                  validation_lambdas)
                        np.divide(self.stage_validation_gradients_truth[k],
                                  validation_weights,
                                  out=self.stage_validation_gradients_truth[k])
                        self.stage_validation_gradients_truth[k, np.isnan(self.stage_validation_gradients_truth[k])] = 0.0

            # Estimate the optimal gradient steps using Newton's method.
            if self.use_newton_method:
                compute_newton_gradient_steps(estimator, training_queries,
                                              training_scores, metric,
                                              training_lambdas,
                                              training_weights,
                                              training_query_scales,
                                              None, training_query_weights,
                                              document_weights,
                                              random_state=self.random_state,
                                              n_jobs=self.n_jobs,
                                              use_pines=self.use_pines)

                # Store the true and estimated gradients for later analysis.
                if self.trace_gradients:
                    np.copyto(self.stage_training_gradients_predicted[k],
                              estimator.predict(training_queries.feature_vectors))

                    if validation_queries is not None:
                        np.copyto(self.stage_validation_gradients_predicted[k],
                                  estimator.predict(validation_queries.feature_vectors))
            
            # Update the document scores using the new gradient predictor.
            if self.trace_gradients:
                training_scores += self.shrinkage * self.stage_training_gradients_predicted[k]
            else:
                training_scores += self.shrinkage * estimator.predict(training_queries.feature_vectors)

            # Add the new tree(s) to the company.
            self.estimators.append(estimator)

            self.training_performance[k] = metric.evaluate_queries(
                                               training_queries, training_scores,
                                               scales=training_query_scales,
                                               weights=training_query_weights)

            if validation_queries is None:
                logger.info('#%08d: %s (training): %11.8f (%11.8f)'
                            % (k + 1, metric, self.training_performance[k],
                               self.training_losses[k]))

            # If validation queries have been given, estimate the model
            # performance on them and decide whether the training should not
            # be stopped early due to no significant performanceimprovements.
            if validation_queries is not None:
                if self.trace_gradients:
                    validation_scores += self.shrinkage * self.stage_validation_gradients_predicted[k]
                else:
                    validation_scores += self.shrinkage * self.estimators[-1].predict(validation_queries.feature_vectors)

                self.validation_performance[k] = metric.evaluate_queries(
                                                     validation_queries,
                                                     validation_scores,
                                                     scales=validation_query_scales,
                                                     weights=validation_query_weights)

                if np.isnan(self.validation_losses[k]):
                    logger.info('#%08d: %s (training):   %11.8f (%11.8f)  | '
                                ' (validation):   %11.8f'
                                % (k + 1, metric,
                                   self.training_performance[k],
                                   self.training_losses[k],
                                   self.validation_performance[k]))
                else:
                    logger.info('#%08d: %s (training):   %11.8f (%11.8f)  | '
                                ' (validation):   %11.8f (%11f)'
                                % (k + 1, metric,
                                   self.training_performance[k],
                                   self.training_losses[k],
                                   self.validation_performance[k],
                                   self.validation_losses[k]))

                if self.validation_performance[k] > best_performance:
                    best_performance = self.validation_performance[k]
                    best_performance_k = k
                    performance_not_improved = 0
                else:
                    performance_not_improved += 1

            elif self.training_performance[k] > best_performance:
                    best_performance = self.training_performance[k]
                    best_performance_k = k

            if performance_not_improved >= self.estopping and self.min_n_estimators <= k + 1:
                logger.info('Stopping early since no improvement on '
                            'validation queries has been observed for '
                            '%d iterations (since iteration %d)'
                            % (self.estopping, best_performance_k + 1))
                break

        logger.info('Final model performance (%s) on %s queries: %11.8f'
                    % (metric,
                       'training' if validation_queries is None else 'validation',
                       best_performance))

        # Make sure the model has the wanted size.
        best_performance_k = max(best_performance_k, self.min_n_estimators - 1)

        # Leave the estimators that led to the best performance,
        # either on training or validation set.
        del self.estimators[best_performance_k + 1:]

        # Correct the number of trees.
        if self.n_estimators != len(self.estimators):
            self.n_estimators = len(self.estimators)
            logger.info('Setting the number of trees of the model to %d.'
                        % self.n_estimators)

        # Set these for further inspection.
        self.training_performance = np.resize(self.training_performance, k + 1)
        self.training_losses = np.resize(self.training_losses, k + 1)

        if validation_queries is not None:
            self.validation_performance = \
                np.resize(self.validation_performance, k + 1)

        if validation_queries is None:
            self.best_performance = [(self.training_performance[best_performance_k],
                                     'training')]
        else:
            self.best_performance = [(self.training_performance[best_performance_k],
                                     'training'),
                                     (self.validation_performance[best_performance_k],
                                     'validation')]

        if self.trace_influences:
            self.stage_training_influences = \
                np.resize(self.stage_training_influences,
                          (k + 1, training_queries.highest_relevance() + 1,
                           training_queries.highest_relevance() + 1))

            influences_normalizer = \
                np.bincount(training_queries.relevance_scores,
                            minlength=training_queries.highest_relevance() + 1)

            influences_normalizer = \
                np.triu(np.ones((training_queries.highest_relevance() + 1, 1)) *
                        influences_normalizer, 1)

            influences_normalizer += influences_normalizer.T

            # Normalize training influences appropriately.
            with np.errstate(divide='ignore', invalid='ignore'):
                self.stage_training_influences /= influences_normalizer
                self.stage_training_influences[np.isnan(self.stage_training_influences)] = 0.0

            if validation_queries is not None:
                self.stage_validation_influences = \
                    np.resize(self.stage_validation_influences,
                              (k + 1, validation_queries.highest_relevance() + 1,
                               validation_queries.highest_relevance() + 1))

                influences_normalizer = \
                    np.bincount(validation_queries.relevance_scores,
                                minlength=validation_queries.highest_relevance() + 1)

                influences_normalizer = \
                    np.triu(np.ones((validation_queries.highest_relevance() + 1, 1)) *
                            influences_normalizer, 1)
                influences_normalizer += influences_normalizer.T

                # Normalize training influences appropriately.
                with np.errstate(divide='ignore', invalid='ignore'):
                    self.stage_validation_influences /= influences_normalizer
                    self.stage_validation_influences[np.isnan(self.stage_validation_influences)] = 0.0

        logger.info('Training of LambdaMART model has finished.')

        return self

    @staticmethod
    def __predict(trees, shrinkage, feature_vectors, output):
        for tree in trees:
            output += tree.predict(feature_vectors, check_input=False)
        output *= shrinkage

    def predict(self, queries, n_jobs=1):
        '''
        Predict the ranking score for each individual document
        in the given queries.

        n_jobs: int, optional (default is 1)
            The number of working threads that will be spawned to compute
            the ranking scores. If -1, the current number of CPUs will be used.
        '''
        if len(self.estimators) == 0:
            raise ValueError('the model has not been trained yet')

        n_jobs = _get_n_jobs(n_jobs)

        if self.base_model is not None:
            predictions = np.ascontiguousarray(
                              self.base_model.predict(queries,
                                                      n_jobs=n_jobs),
                              dtype='float64')
        else:
            predictions = np.zeros(queries.document_count(), dtype='float64')

        indices = _get_partition_indices(0, queries.document_count(),
                                         self.n_jobs)

        Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(parallel_helper, check_pickle=False)(
                LambdaMART, '_LambdaMART__predict',
                self.estimators, self.shrinkage,
                queries.feature_vectors[indices[i]:indices[i + 1]],
                predictions[indices[i]:indices[i + 1]]
            )
            for i in range(indices.size - 1)
        )

        return predictions

    def evaluate(self, queries, metric=None, n_jobs=1):
        '''
        Evaluate the performance of the model on the given queries.

        Parameters
        ----------
        queries : Queries instance
            Queries used for evaluation of the model.

        metric : string or None, optional (default=None)
            Specify evaluation metric which will be used as a utility
            function (i.e. metric of `goodness`) optimized by this model.
            Supported metrics are "DCG", NDCG", "WTA", "ERR", with an optional
            suffix "@{N}", where {N} can be any positive integer. If None,
            the model is evaluated with a metric for which it was trained.

        n_jobs: int, optional (default is 1)
            The number of working threads that will be spawned to compute
            the ranking scores. If -1, the current number of CPUs will be used.
        '''
        if metric is None:
            metric = self.metric

        scores = self.predict(queries, n_jobs=_get_n_jobs(n_jobs))

        return MetricFactory(metric, aslist(queries),
                      self.random_state).evaluate_queries(queries, scores)

    def predict_rankings(self, queries, compact=False, n_jobs=1):
        '''
        Predict rankings of the documents for the given queries.

        If `compact` is set to True then the output will be one
        long 1d array containing the rankings for all the queries
        instead of a list of 1d arrays.

        The compact array can be subsequently index using query
        index pointer array, see `queries.query_indptr`.

        query: Query
            The query whose documents should be ranked.

        compact: bool
            Specify to return rankings in compact format.

         n_jobs: int, optional (default is 1)
            The number of working threads that will be spawned to compute
            the ranking scores. If -1, the current number of CPUs will be used.
        '''
        # Predict the ranking scores for the documents.
        predictions = self.predict(queries, n_jobs)

        rankings = np.zeros(queries.document_count(), dtype=np.intc)

        ranksort_queries(queries.query_indptr, predictions, rankings)

        if compact or len(queries) == 1:
            return rankings
        else:
            return np.array_split(rankings, queries.query_indptr[1:-1])

    def feature_importances(self):
        '''
        Return the feature importances.
        '''
        if len(self.estimators) == 0:
            raise ValueError('the model has not been trained yet')

        importances = Parallel(n_jobs=self.n_jobs, backend="threading")(
                          delayed(getattr, check_pickle=False)(
                              tree, 'feature_importances_'
                          )
                          for tree in self.estimators
                      )

        return sum(importances) / self.n_estimators

    @classmethod
    def load(cls, filepath, mmap='r', load_traced=False):
        '''
        Load the previously saved LambdaMART model from the specified file.

        Parameters:
        -----------
        filepath: string
            The filepath, from which a LambdaMART object will be loaded.

        mmap: {None, ‘r+’, ‘r’, ‘w+’, ‘c’}, optional (default is 'r')
            If not None, then memory-map the traced data (if any), using
            the given mode (see `numpy.memmap` for a details).
        '''
        logger.info("Loading %s object from %s" % (cls.__name__, filepath))

        obj = unpickle(filepath)

        if not load_traced:
            return obj

        if obj.trace_lambdas:
            logger.info('Loading traced (true) lambda values from %s.training.lambdas.truth.npy' % filepath)
            setattr(obj, 'stage_training_lambdas_truth', np.load(filepath + '.training.lambdas.truth.npy', mmap_mode=mmap))

            logger.info('Loading traced (predicted) lambda values from %s.training.lambdas.predicted.npy' % filepath)
            setattr(obj, 'stage_training_lambdas_predicted', np.load(filepath + '.training.lambdas.predicted.npy', mmap_mode=mmap))

            if hasattr(obj, 'validation_performance'):
                logger.info('Loading traced (true) lambda values from %s.validation.lambdas.truth.npy' % filepath)
                setattr(obj, 'stage_validation_lambdas_truth', np.load(filepath + '.validation.lambdas.truth.npy', mmap_mode=mmap))

                logger.info('Loading traced (predicted) lambda values from %s.validation.lambdas.predicted.npy' % filepath)
                setattr(obj, 'stage_validation_lambdas_predicted', np.load(filepath + '.validation.lambdas.predicted.npy', mmap_mode=mmap))

        if obj.trace_gradients:
            logger.info('Loading traced (true) gradient values from %s.training.gradients.truth.npy' % filepath)
            setattr(obj, 'stage_training_gradients_truth', np.load(filepath + '.training.gradients.truth.npy', mmap_mode=mmap))

            logger.info('Loading traced (predicted) gradient values from %s.training.gradients.predicted.npy' % filepath)
            setattr(obj, 'stage_training_gradients_predicted', np.load(filepath + '.training.gradients.predicted.npy', mmap_mode=mmap))

            if hasattr(obj, 'validation_performance'):
                logger.info('Loading traced (true) gradient values from %s.validation.gradients.truth.npy' % filepath)
                setattr(obj, 'stage_validation_gradients_truth', np.load(filepath + '.validation.gradients.truth.npy', mmap_mode=mmap))

                logger.info('Loading traced (predicted) gradient values from %s.validation.gradients.predicted.npy' % filepath)
                setattr(obj, 'stage_validation_gradients_predicted', np.load(filepath + '.validation.gradients.predicted.npy', mmap_mode=mmap))

        return obj

    def save(self, filepath):
        '''
        Save te LambdaMART model into the specified file.

        Parameters:
        -----------
        filepath: string
            The filepath where this object will be saved.
        '''
        logger.info("Saving %s object into %s" % (self.__class__.__name__, filepath))

        # Deal with saving the memory-mapped arrays: only the used part of the arrays are saved
        # with the model, separately, i.e., the arrays are standalone *.npy files. The temporary
        # directory is removed after this.
        if self.trace_lambdas:
            logger.info('Saving traced (true) lambda values into %s.training.lambdas.truth.npy' % filepath)
            np.save(filepath + '.training.lambdas.truth.npy', self.stage_training_lambdas_truth[:self.training_performance.shape[0]])
            del self.stage_training_lambdas_truth

            logger.info('Saving traced (predicted) lambda values into %s.training.lambdas.predicted.npy' % filepath)
            np.save(filepath + '.training.lambdas.predicted.npy', self.stage_training_lambdas_predicted[:self.training_performance.shape[0]])
            del self.stage_training_lambdas_predicted

            if hasattr(self, 'validation_performance'):
                logger.info('Saving traced (true) lambda values into %s.validation.lambdas.truth.npy' % filepath)
                np.save(filepath + '.validation.lambdas.truth.npy', self.stage_validation_lambdas_truth[:self.validation_performance.shape[0]])
                del self.stage_validation_lambdas_truth

                logger.info('Saving traced (predicted) lambda values into %s.validation.lambdas.predicted.npy' % filepath)
                np.save(filepath + '.validation.lambdas.predicted.npy', self.stage_validation_lambdas_predicted[:self.validation_performance.shape[0]])
                del self.stage_validation_lambdas_predicted

        if self.trace_gradients:
            logger.info('Saving traced (true) gradient values into %s.training.gradients.truth.npy' % filepath)
            np.save(filepath + '.training.gradients.truth.npy', self.stage_training_gradients_truth[:self.training_performance.shape[0]])
            del self.stage_training_gradients_truth

            logger.info('Saving traced (predicted) gradient values into %s.training.gradients.predicted.npy' % filepath)
            np.save(filepath + '.training.gradients.predicted.npy', self.stage_training_gradients_predicted[:self.training_performance.shape[0]])
            del self.stage_training_gradients_predicted

            if hasattr(self, 'validation_performance'):
                logger.info('Saving traced (true) gradient values into %s.validation.gradients.truth.npy' % filepath)
                np.save(filepath + '.validation.gradients.truth.npy', self.stage_validation_gradients_truth[:self.validation_performance.shape[0]])
                del self.stage_validation_gradients_truth

                logger.info('Saving traced (predicted) gradient values into %s.validation.gradients.predicted.npy' % filepath)
                np.save(filepath + '.validation.gradients.predicted.npy', self.stage_validation_gradients_predicted[:self.validation_performance.shape[0]])
                del self.stage_validation_gradients_predicted

        # Get rid of the temporary directory.
        if hasattr(self, 'tmp_directory'):
            logger.info('Deleting temporary directory (%s) for traced data.' % self.tmp_directory)
            rmtree(self.tmp_directory)
            del self.tmp_directory

        pickle(self, filepath)

        if self.trace_lambdas:
            self.stage_training_lambdas_truth = np.load(filepath + '.training.lambdas.truth.npy', mmap_mode='r')
            self.stage_training_lambdas_predicted = np.load(filepath + '.training.lambdas.predicted.npy', mmap_mode='r')

            if hasattr(self, 'validation_performance'):
                self.stage_validation_lambdas_truth = np.load(filepath + '.validation.lambdas.truth.npy', mmap_mode='r')
                self.stage_validation_lambdas_predicted = np.load(filepath + '.validation.lambdas.predicted.npy', mmap_mode='r')

        if self.trace_gradients:
            self.stage_training_gradients_truth = np.load(filepath + '.training.gradients.truth.npy', mmap_mode='r')
            self.stage_training_gradients_predicted = np.load(filepath + '.training.gradients.predicted.npy', mmap_mode='r')

            if hasattr(self, 'validation_performance'):
                self.stage_validation_gradients_truth = np.load(filepath + '.validation.gradients.truth.npy', mmap_mode='r')
                self.stage_validation_gradients_predicted = np.load(filepath + '.validation.gradients.predicted.npy', mmap_mode='r')

    def save_as_text(self, filepath):
        '''
        Save the model into the file in an XML format.
        '''
        if self.use_random_forest > 0:
            raise ValueError('cannot save model to text when it is based on random forest')

        with open(filepath, 'w') as ofile:
            padding = '\t\t'

            ofile.write('<LambdaMART>\n')
            ofile.write('\t<parameters>\n')
            ofile.write('\t\t<trees> %d </trees>\n' % self.n_estimators)
            if self.max_leaf_nodes is not None:
                ofile.write('\t\t<leaves> %d </leaves>\n' % self.max_leaf_nodes)
            else:
                ofile.write('\t\t<depth> %d </depth>\n' % self.max_depth)
            ofile.write('\t\t<features> %d </features>\n' % -1 if self.max_features is None else self.max_features)
            ofile.write('\t\t<shrinkage> %.2f </shrinkage>\n' % self.shrinkage)
            ofile.write('\t\t<estopping> %d </estopping>\n' % self.estopping)
            ofile.write('\t<ensemble>\n')
            for id, tree in enumerate(self.estimators, start=1):
                # Getting under the tree bark...
                tree = tree.tree_

                ofile.write(padding + '<tree id="%d">\n' % id)

                # Stack of 3-tuples: (depth, parent, node_id).
                stack = [(1, TREE_UNDEFINED, 0)]

                while stack:
                    depth, parent, node_id = stack.pop()

                    # End of split mark.
                    if node_id < 0:
                        ofile.write(padding + (depth * '\t'))
                        ofile.write('</split>\n')
                        continue

                    ofile.write(padding + (depth * '\t'))
                    ofile.write('<split')

                    if parent == TREE_UNDEFINED:
                        ofile.write('>\n')
                    else:
                        pos = 'left' if tree.children_left[parent] == node_id else 'right'
                        ofile.write(' pos="%s">\n' % pos)

                    # If the node is a leaf.
                    if tree.children_left[node_id] == TREE_LEAF:
                        ofile.write(padding + ((depth + 1) * '\t'))
                        ofile.write('<output> %.17f </output>\n' % tree.value[node_id])
                        ofile.write(padding + (depth * '\t'))
                        ofile.write('</split>\n')
                    else:
                        ofile.write(padding + ((depth + 1) * '\t'))
                        # FIXME: Feature indexing should be marked somewhere if it
                        # realy is 0-based or not. Here we are assuming it is NOT!
                        ofile.write('<feature> %d </feature>\n' % (tree.feature[node_id] + 1))
                        ofile.write(padding + ((depth + 1) * '\t'))
                        ofile.write('<threshold> %.9f </threshold>\n' % tree.threshold[node_id])

                        # Push the end of split mark first... then push the right and left child.
                        stack.append((depth, parent, -1))
                        stack.append((depth + 1, node_id, tree.children_right[node_id]))
                        stack.append((depth + 1, node_id, tree.children_left[node_id]))

                ofile.write(padding + '</tree>\n')
            ofile.write('\t</ensemble>\n')
            ofile.write('</LambdaMART>\n')

    def __del__(self):
        '''
        Cleanup the temporary directory for traced lambdas and gradients.
        '''
        # Get rid of the temporary directory and all the memory-mapped arrays.
        if hasattr(self, 'tmp_directory'):
            if self.trace_lambdas:
                del self.stage_training_lambdas_truth
                del self.stage_training_lambdas_predicted

                if hasattr(self, 'validation_performance'):
                    del self.stage_validation_lambdas_truth
                    del self.stage_validation_lambdas_predicted

            if self.trace_gradients:
                del self.stage_training_gradients_truth
                del self.stage_training_gradients_predicted

                if hasattr(self, 'validation_performance'):
                    del self.stage_validation_gradients_truth
                    del self.stage_validation_gradients_predicted

            logger.info('Deleting temporary directory (%s) for traced data.'
                        % self.tmp_directory)
            rmtree(self.tmp_directory)
            del self.tmp_directory

    def __str__(self):
        '''
        Return textual representation of the LambdaMART model.
        '''
        return ('LambdaMART(trees=%d, max_depth=%s, max_leaf_nodes=%s, '
                'shrinkage=%.2f, max_features=%s, use_newton_method=%s)'
                % (self.n_estimators,
                   self.max_depth if self.max_leaf_nodes is None else '?',
                   '?' if self.max_leaf_nodes is None else self.max_leaf_nodes,
                   self.shrinkage,
                   'all' if self.max_features is None else self.max_features,
                   self.use_newton_method))
