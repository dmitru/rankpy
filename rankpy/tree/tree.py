# Adopted version of tree.py from sklearn v15.2.
#
# Original authors: Gilles Louppe <g.louppe@gmail.com>
#                   Peter Prettenhofer <peter.prettenhofer@gmail.com>
#                   Brian Holt <bdholt1@gmail.com>
#                   Joel Nothman <joel.nothman@gmail.com>
#                   Arnaud Joly <arnaud.v.joly@gmail.com>
#
# Under BSD 3 clause licence.
#
# Edited by: Tomas Tunys <tunystom@gmail.com>

import numpy as np

from sklearn.utils import array2d
from sklearn.utils import check_random_state

from ..externals.joblib import Parallel, delayed, cpu_count

from . import _tree
from ._tree import Tree
from ._tree import TreeGrower

DTYPE = _tree.DTYPE
DOUBLE = _tree.DOUBLE
BOOL = _tree.BOOL

MAX_INT = np.iinfo(np.int32).max


__all__ = ['OnlineRandomForestRegressor']


# =============================================================================
# Online Extremely Randomized Regression Tree
# =============================================================================


class OnlineRegressionTree(object):
    ''' 
    Online Regression Tree based on [1].

    Parameters
    ----------
    lambda_: float
        Controls the number of candidate features for a node split.
        It is the mean of the Poisson distribution used to sample
        the number of features.

    n_thresholds: int
        The total number of thresholds per candidate feature that
        will be used for splitting.

    max_depth: int or None
        The maximum depth of a tree. If None is given trees are
        grown indefinitely.

    alpha: function
        Monotonous function of integers. It is supposed to return
        the minimum number of (estimation) samples that both child
        nodes of any candidate split must receive before the split
        can be made.

    beta: function
        Monotonous function of integers. It is supposed to return
        the number of (estimation) samples that a node must receive
        before it starts to ignore the quality of candidate splits
        (in terms of MSE impurity) and forces the best available
        (valid) split.

    tau: float
        The minimum improvement of MSE impurity a candidate split
        must achieve before it is considered for splitting.

    n_fringe: int or None, optional (default is None)
        The size of fringe (per tree): active set of (leaf) nodes,
        which use received (training) samples to estimate the quality
        of their randomly generated candidate splits. Inactive nodes
        use the samples to estimate the upper-bound on the MSE
        for which they are responsible. Whenever a free space
        is allocated on the fringe (for example, a node is split)
        the node with the highest estimate (potentially highest
        decrease of MSE) is activated. Each time a node is split
        its child nodes are inactive and need to 'compete' with
        other nodes to  make it to the fringe. If None is given,
        the fringe is unbounded.
        
    bias: float or None, optional (default is 0)
        The initial prediction value for the root nodes of the trees.

    uniform: bool, optional (default is False)
        If True, the candidate thresholds are going to be picked
        uniformly at random from [0; 1]. This assumes that the
        feature values are normalized or within [0, 1] range.

    batch: bool or None, optional (default is None)
        If True, the nodes are split only after all the samples
        in one call of `self.fit(...)` has been processed.

    random_state : int, RandomState instance or None, optional (default is None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    References
    ----------
    [1] "Consistency of Online Random Forests" by Denil, Misha and Matheson, David
         and de Freitas, Nando. ICML'13.
    '''
    def __init__(self, lambda_, n_thresholds, max_depth, alpha, beta, tau,
                 n_fringe=None, bias=None, uniform=False, batch=False,
                 random_state=None):
        self.lambda_ = lambda_
        self.n_features = None
        self.n_thresholds = n_thresholds
        self.max_depth = 2**31-1 if max_depth is None else max_depth
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.n_fringe = 2**31-1 if n_fringe is None else n_fringe
        self.bias = 0.0 if bias is None else bias
        self.uniform = uniform
        self.batch = batch
        self.random_state = check_random_state(random_state)
        self.tree_ = Tree()
        self.grower = None


    def fit(self, X, y, i, check_input=True):
        ''' 
        Build a decision tree from the training set (X, y).

        Parameters
        ----------
        X : array, shape = [n_samples, n_features]
            The training input samples. Internally, it will be converted to
            `np.float32`, unless check_input is False, in that case if the
            data type is not `np.float32` and exception is raised.

        y : array, shape = (n_samples,)
            The target values. Internally, it will be converted to
            `np.float64` 1-d array, unless check_input is False, in
            that case the array is expected in this format.

        i : array, shape = (n_samples,)
            Binary array (0, 1) indicating from which stream (structure/
            estimation) each sample comes from.

        check_input : boolean, (default=True)
            Allow to bypass several input checking.

        Returns
        -------
        self : object
            Returns self.
        '''
        if check_input:
            X = array2d(X, dtype=DTYPE, copy=False, force_all_finite=False)

            if y.ndim != 1:
                raise ValueError('y must be 1-d array')

            if y.dtype != DOUBLE or not y.flags.c_contiguous:
                y = np.ascontiguousarray(y, dtype=DOUBLE)

            if len(y) != X.shape[0]:
                raise ValueError('Number of labels (%d) does not match '
                                 'number of samples (%d).' % (len(y), X.shape[0]))

            if i.ndim != 1 or np.any(np.unique(i) != (0, 1)):
                raise ValueError('i must be 1-d binary array')

            if i.dtype != DOUBLE or not i.flags.c_contiguous:
                i = np.ascontiguousarray(y, dtype=DOUBLE)

            if len(i) != X.shape[0]:
                raise ValueError('Number of stream indicators (%d) does not match '
                                 'number of samples (%d).' % (len(i), X.shape[0]))

        n_samples, n_features = X.shape

        # First call to fit?
        if self.grower is None:
            self.n_features = n_features
            self.grower = TreeGrower(self.tree_, self.n_features, self.n_thresholds,
                                     self.max_depth, self.alpha, self.beta, self.tau,
                                     self.lambda_, self.uniform, self.bias, 
                                     self.n_fringe, self.batch, self.random_state)

        if self.n_features != n_features:
            raise ValueError('Number of features (%d) does not match number '
                             'of features in previous call to fit (%d).' \
                             % (n_features, self.n_features))

        self.grower.grow(X, y, i)

        return self


    def predict(self, X):
        ''' 
        Predict regression value for X.

        Parameters
        ----------
        X : array, shape = (n_samples, n_features)
            The input samples. Internally, it will be converted to
            `dtype=np.float32`.

        Returns
        -------
        y : array, shape = (n_samples,)
            The predict values.
        '''
        X = array2d(X, dtype=DTYPE, copy=False, force_all_finite=False)

        n_samples, n_features = X.shape

        if self.grower is None:
            return self.bias

        if self.n_features != n_features:
            raise ValueError('Number of features of the model must '
                             ' match the input. Model n_features is %s and '
                             ' input n_features is %s '
                             % (self.n_features_, n_features))

        return self.tree_.predict(X)


# =============================================================================
# Online Extremely Randomized Random Forest
# =============================================================================


def _parallel_build_trees(tree, X, y, tree_idx, n_trees, p=0.5, verbose=0):
    ''' 
    Private function used to fit a single tree in parallel.
    '''
    if verbose > 1:
        print('Building tree %d of %d.' % (tree_idx + 1, n_trees))

    # Assign samples randomly into the structure/estimation streams.
    # The sample comes from structure stream (1) with probability `p`.
    i = tree.random_state.choice(np.array([0, 1], dtype=np.uint8),
                                 size=X.shape[0], p=[1 - p, p])

    # Fit the regression tree.
    tree.fit(X, y, i, check_input=False)


def _parallel_helper(obj, methodname, *args, **kwargs):
    ''' 
    Private helper to workaround Python 2 pickle limitations.
    '''
    return getattr(obj, methodname)(*args, **kwargs)


class OnlineRandomForestRegressor(object):
    ''' 
    Online Random Forest based on [1].

    Parameters
    ----------
    n_estimators: int
        The number of trees in the forest.

    lambda_: float
        Controls the number of candidate features for a node split.
        It is the mean of the Poisson distribution used to sample
        the number of features.

    n_thresholds: int
        The total number of thresholds per candidate feature that
        will be used for splitting.

    max_depth: int or None
        The maximum depth of a tree. If None is given trees are
        grown indefinitely.

    alpha: function
        Monotonous function of integers. It is supposed to return
        the minimum number of (estimation) samples that both child
        nodes of any candidate split must receive before the split
        can be made.

    beta: function
        Monotonous function of integers. It is supposed to return
        the number of (estimation) samples that a node must receive
        before it starts to ignore the quality of candidate splits
        (in terms of MSE impurity) and forces the best available
        (valid) split.

    tau: float
        The minimum improvement of MSE impurity a candidate split
        must achieve before it is considered for splitting.

    n_fringe: int or None, optional (default is None)
        The size of fringe (per tree): active set of (leaf) nodes,
        which use received (training) samples to estimate the quality
        of their randomly generated candidate splits. Inactive nodes
        use the samples to estimate the upper-bound on the MSE
        for which they are responsible. Whenever a free space
        is allocated on the fringe (for example, a node is split)
        the node with the highest estimate (potentially highest
        decrease of MSE) is activated. Each time a node is split
        its child nodes are inactive and need to 'compete' with
        other nodes to  make it to the fringe. If None is given,
        the fringe is unbounded.

    bias: float or None, optional (default is 0)
        The initial prediction value for the root nodes of the trees.

    uniform: bool, optional (default is False)
        If True, the candidate thresholds are going to be picked
        uniformly at random from [0; 1]. This assumes that the
        feature values are normalized or within [0, 1] range.

    batch: bool or None, optional (default is None)
        If True, the nodes are split only after all the samples
        in one call of `self.fit(...)` has been processed.

    prob: float, optional (default is 0.5)
        The probability of sample originating from structure stream.
        This parameter effectively influences the robustness of the
        structure/estimation statistics and can be used to trade off
        quality of one in expence of another.

    n_jobs : int, optional (default is 1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    verbose : int, optional (default is 0)
        Controls the verbosity of the tree building process.

    random_state : int, RandomState instance or None, optional (default is None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    References
    ----------
    [1] "Consistency of Online Random Forests" by Denil, Misha and Matheson, David
         and de Freitas, Nando. ICML'13.
    '''
    def __init__(self, n_estimators, lambda_, n_thresholds, max_depth, alpha, beta,
                 tau, n_fringe=None, bias=None, uniform=False, batch=False, prob=0.5,
                 n_jobs=1, verbose=0, random_state=None):
        self.n_estimators = n_estimators
        self.estimators_ = []
        self.lambda_ = lambda_
        self.n_features = None
        self.n_thresholds = n_thresholds
        self.max_depth = 2**31-1 if max_depth is None else max_depth
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.n_fringe = 2**31-1 if n_fringe is None else n_fringe
        self.bias = 0.0 if bias is None else bias
        self.uniform = uniform
        self.batch = batch
        self.prob = prob

        if n_jobs < 0:
            self.n_jobs = max(cpu_count() + 1 + n_jobs, 1)
        elif n_jobs == 0:
            raise ValueError('Are you serious, n_jobs == 0?')
        else:
            self.n_jobs = n_jobs

        self.verbose = verbose
        self.random_state = check_random_state(random_state)


    def apply(self, X):
        ''' 
        Apply trees in the forest to X, return leaf indices.

        Parameters
        ----------
        X : array, shape = (n_samples, n_features)
            The input samples. Internally, it will be converted to
            `dtype=np.float32`.

        Returns
        -------
        X_leaves : array, shape = (n_samples, n_estimators)
            For each sample x in X and for each tree in the forest,
            return the index of the leaf x ends up in.
        '''
        X = array2d(X, dtype=DTYPE, copy=False, force_all_finite=False)

        results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                           backend="threading")(
            delayed(_parallel_helper)(tree.tree_, 'apply', X)
            for tree in self.estimators_)

        return np.array(results).T


    def _initialize_estimators(self):
        ''' 
        Create and initialize the online regression trees
        making the forest.
        '''
        for _ in range(self.n_estimators):
            random_state = self.random_state.randint(MAX_INT)
            self.estimators_.append(OnlineRegressionTree(self.lambda_, self.n_thresholds,
                                                         self.max_depth, self.alpha, self.beta,
                                                         self.tau, self.n_fringe, self.bias,
                                                         self.uniform, self.batch, random_state))


    def fit(self, X, y, check_input=True):
        ''' 
        Build a forest of trees from the available chunk of training set (X, y).

        Parameters
        ----------
        X : array, shape = (n_samples, n_features)
            The training input samples. Internally, it will be converted to
            `dtype=np.float32`.

        y : array, shape = (n_samples,)
            The target values.

        Returns
        -------
        self : object
            Returns self.
        '''
        if check_input:
            X = array2d(X, dtype=DTYPE, copy=False, force_all_finite=False)

            if y.ndim != 1:
                raise ValueError('y must be 1-d array')

            if len(y) != X.shape[0]:
                raise ValueError('Number of labels (%d) does not match '
                                 'number of samples (%d).' % (len(y), X.shape[0]))

            if y.dtype != DOUBLE or not y.flags.c_contiguous:
                y = np.ascontiguousarray(y, dtype=DOUBLE)

        # First call to fit(...)?
        if not self.estimators_:
            self._initialize_estimators()

        # Parallel loop: we use the threading backend as the Cython code
        # for fitting the trees is internally releasing the Python GIL
        # making threading always more efficient than multiprocessing in
        # that case.
        Parallel(n_jobs=self.n_jobs, verbose=self.verbose, backend="threading")\
                (delayed(_parallel_build_trees)(tree, X, y, tree_idx,
                 len(self.estimators_), self.prob, verbose=self.verbose)
                 for tree_idx, tree in enumerate(self.estimators_))

        return self


    def predict(self, X):
        ''' 
        Predict regression target for X.

        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the trees in the forest.

        Parameters
        ----------
        X : array, shape = (n_samples, n_features)
            The input samples. Internally, it will be converted to
            `dtype=np.float32`.

        Returns
        -------
        y : array, shape = (n_samples, )
            The predicted values.
        '''
        # A call to predict(...) preceding a call to fit(...).
        if not self.estimators_:
            return self.bias

        X = array2d(X, dtype=DTYPE, copy=False, force_all_finite=False)

        all_y_hat = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, backend="threading")\
                            (delayed(_parallel_helper)(tree, 'predict', X)
                             for tree in self.estimators_)

        return sum(all_y_hat) / len(self.estimators_)
