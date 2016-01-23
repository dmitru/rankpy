# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Adopted version of _tree.pyx from sklearn v15.2.
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
cimport numpy as np
np.import_array()

from libc.stdlib cimport malloc, realloc, calloc, free

from libc.string cimport memcpy, memset

from libc.math cimport log as ln

from cpython cimport Py_INCREF, PyObject

cdef extern from "numpy/arrayobject.h":
    object PyArray_NewFromDescr(object subtype, np.dtype descr,
                                int nd, np.npy_intp* dims,
                                np.npy_intp* strides,
                                void* data, int flags, object obj)

ctypedef np.npy_uint8   BOOL_t    # Boolean
ctypedef np.npy_int32   INTC_t    # 32-bit signed integer
ctypedef np.npy_float32 DTYPE_t   # Type of samples
ctypedef np.npy_float64 DOUBLE_t  # Type of targets

ctypedef Splitter* SplitterPtr

from numpy import float32 as DTYPE
from numpy import float64 as DOUBLE
from numpy import uint8   as BOOL

cdef double NaN = np.nan
cdef double INFINITY = np.inf
cdef double ALMOST_INFINITY = np.finfo('d').max / 2 

TREE_LEAF = -1
TREE_UNDEFINED = -2

cdef int _TREE_LEAF = TREE_LEAF
cdef int _TREE_UNDEFINED = TREE_UNDEFINED

# Mitigate precision differences between 32 bit and 64 bit
cdef DTYPE_t FEATURE_THRESHOLD = 1e-7

cdef enum:
    RAND_R_MAX = 0x7FFFFFFF # alias 2**32 - 1


# =============================================================================
# Tree
# =============================================================================


cdef struct Node:        # Regression tree node object.
    int    left_child    # Index of the left child of the node.
    int    right_child   # Index of the right child of the node.
    int    feature       # Index of the feature used for splitting the node.
    double threshold     # Threshold value at the node.
    double impurity      # Impurity of the node.
    int    n_samples     # Number of samples at the node.
    double value         # The node prediction value.
    
    
# Repeat Node definition for NumPy structured array.
NODE_DTYPE = np.dtype({
    'names': ['children_left', 'children_right', 'feature',
              'threshold', 'impurity','n_samples', 'value'],
    'formats': [np.intc, np.intc, np.intc, np.float64,
                np.float64, np.intc, np.float64],
    'offsets': [
        <Py_ssize_t> &(<Node*> NULL).left_child,
        <Py_ssize_t> &(<Node*> NULL).right_child,
        <Py_ssize_t> &(<Node*> NULL).feature,
        <Py_ssize_t> &(<Node*> NULL).threshold,
        <Py_ssize_t> &(<Node*> NULL).impurity,
        <Py_ssize_t> &(<Node*> NULL).n_samples,
        <Py_ssize_t> &(<Node*> NULL).value
    ]
})


cdef class Tree:
    # The Tree object is a binary tree structure constructed by the
    # TreeGrower. The tree structure is used for predictions and
    # feature importances.
    cdef Node*      nodes        # Array of tree nodes.
    cdef public int n_nodes      # Number of tree nodes.
    cdef public int capacity     # Capacity of tree, in terms of nodes.
    cdef public int n_features   # Number of input features.
    cdef public int max_depth    # Maximum depth of the tree.


    property children_left:
        def __get__(self):
            return self._get_node_ndarray()['children_left']


    property children_right:
        def __get__(self):
            return self._get_node_ndarray()['children_right']


    property features:
        def __get__(self):
            return self._get_node_ndarray()['feature']


    property thresholds:
        def __get__(self):
            return self._get_node_ndarray()['threshold']


    property impurities:
        def __get__(self):
            return self._get_node_ndarray()['impurity']


    property n_node_samples:
        def __get__(self):
            return self._get_node_ndarray()['n_samples']


    property values:
        def __get__(self):
            return self._get_node_ndarray()['value']


    def __cinit__(self):
        ''' 
        Constructor.
        '''
        self.nodes = NULL
        self.n_nodes = 0
        self.capacity = 0
        self.max_depth = 0


    def __dealloc__(self):
        ''' 
        Destructor.
        '''
        free(self.nodes)


    def __reduce__(self):
        ''' 
        Reduce re-implementation, for pickling.
        '''
        return (Tree, tuple(), self.__getstate__())


    def __getstate__(self):
        ''' 
        Getstate re-implementation, for pickling.
        '''
        d = {}
        d['nodes'] = self._get_node_ndarray()
        d['n_nodes'] = self.n_nodes
        d['max_depth'] = self.max_depth
        return d


    def __setstate__(self, d):
        ''' 
        Setstate re-implementation, for unpickling.
        '''
        self.max_depth = d['max_depth']
        self.n_nodes = d['n_nodes']
        node_ndarray = d['nodes']

        if (node_ndarray.ndim != 1 or
            node_ndarray.dtype != NODE_DTYPE or
            not node_ndarray.flags.c_contiguous):
            raise ValueError('Did not recognise loaded array layout.')

        if self._resize_c(node_ndarray.shape[0]) != 0:
            raise MemoryError("Resizing tree to %d." % self.capacity)

        nodes = memcpy(self.nodes, (<np.ndarray> node_ndarray).data,
                       self.capacity * sizeof(Node))
                       

    cdef void _resize(self, int capacity=-1) except *:
        ''' 
        Resize the nodes array to `capacity`, if `capacity` == -1,
        then double the size of the array.
        '''            
        if self._resize_c(capacity) != 0:
            raise MemoryError()


    cdef int _resize_c(self, int capacity=-1) nogil:
        ''' 
        Guts of _resize. Returns 0 for success, -1 for error.
        '''
        if capacity == self.capacity and self.nodes != NULL:
            return 0

        if capacity == -1:
            if self.capacity == 0:
                capacity = 3
            else:
                capacity = 2 * self.capacity + 1
        
        cdef void* ptr = realloc(self.nodes, capacity * sizeof(Node))        
        
        if ptr == NULL:
            return -1
        
        self.nodes = <Node*> ptr

        # Shrink the n_nodes accordingly.
        if capacity < self.n_nodes:
            self.n_nodes = capacity

        self.capacity = capacity
        return 0


    cdef int _add_node(self, int parent, bint is_left, bint is_leaf,
                       int feature, double threshold, double impurity,
                       double value, double n_samples) nogil:
        ''' 
        Add a node to the tree. The new node registers itself as the child
        of its parent. Returns -1 on error.
        '''
        cdef int node_idx = self.n_nodes

        if node_idx >= self.capacity and self._resize_c() != 0:
            return -1

        self.nodes[node_idx].impurity = impurity
        self.nodes[node_idx].n_samples = <int> n_samples

        if parent != _TREE_UNDEFINED:
            # Explicitly show this is internal node.
            self.nodes[parent].value = NaN
            if is_left:
                self.nodes[parent].left_child = node_idx
            else:
                self.nodes[parent].right_child = node_idx

        if is_leaf:
            self.nodes[node_idx].left_child = _TREE_LEAF
            self.nodes[node_idx].right_child = _TREE_LEAF
            self.nodes[node_idx].feature = -1
            self.nodes[node_idx].threshold = NaN
            self.nodes[node_idx].value = value
        else:
            self.nodes[node_idx].feature = feature
            self.nodes[node_idx].threshold = threshold

        self.n_nodes += 1

        return node_idx
    
         
    cdef void _split_leaf_node(self, int node_idx, int feature, double threshold,
                               double impurity, double n_samples) nogil:
        '''
        Used to split a leaf node. The call to this method should be followed
        by two call to `self._add_node` with parent node set to `node_idx`.
        '''
        self.nodes[node_idx].value = NaN
        self.nodes[node_idx].feature = feature
        self.nodes[node_idx].threshold = threshold
        self.nodes[node_idx].impurity = impurity
        self.nodes[node_idx].n_samples = <int> n_samples


    cpdef np.ndarray predict(self, DTYPE_t[:, ::1] samples):
        ''' 
        Predict target for the given samples.
        '''
        return self._get_value_ndarray().take(self.apply(samples),
                                              axis=0, mode='clip')


    cdef inline int _apply_c(self, DTYPE_t[::1] sample) nogil:
        '''
        Finds the index of the leaf node for the given sample.
        '''
        cdef Node* node = self.nodes
        
        while node.left_child != _TREE_LEAF:
            if sample[node.feature] <= node.threshold + FEATURE_THRESHOLD:
                node = &self.nodes[node.left_child]
            else:
                node = &self.nodes[node.right_child]
                
        return <int>(node - self.nodes)


    cpdef np.ndarray apply(self, DTYPE_t[:, ::1] samples):
        ''' 
        Finds the index of the leaf node for each sample.
        '''
        cdef int i, n_samples = samples.shape[0]
        cdef np.ndarray[INTC_t] indices = np.zeros((n_samples,), dtype=np.intc)
        cdef int* p_indices = <int*> indices.data

        with nogil:
            for i in range(n_samples):
                p_indices[i] = self._apply_c(samples[i])
                
        return indices


    cdef void _apply_ext_c(self, DTYPE_t[:, ::1] samples, int *sample_indices,
                           int *node_indices, int *n_node_indices, int *n_node_samples) nogil:
        ''' 
        Finds the index of the leaf node for each sample and retrieve them "bucket sorted",
        i.e. `sample_indices[:n_node_samples[0]]` will contain the indices of samples
        falling into the node with index `node_indices[0]`.
             
        Samples for other nodes can be retrieved by offseting the `sample_indices` array
        by the proper amount, which is the cumulative sum of `n_node_samples`.
        
        `n_node_indices` holds the number of leaf nodes receiving at least one sample.
        '''
        cdef int i, j, n_samples = samples.shape[0]
        
        if n_samples == 0:
            return
        
        for i in range(n_samples):
            sample_indices[i] = i
            node_indices[i] = self._apply_c(samples[i])

        if n_samples == 1:
            n_node_samples[0] = 1
            n_node_indices[0] = 1
        else:
            # Sort (jointly) the sample indices according to the node_indices.
            sort(node_indices, sample_indices, n_samples)

            j = 0
            n_node_samples[0] = 0

            # Collapse the node indices and count the number of
            # samples falling to the same leaf node.
            for i in range(n_samples):
                if node_indices[j] != node_indices[i]:
                    j += 1
                    n_node_indices[j] = node_indices[i]
                    n_node_samples[j] = 0
                n_node_samples[j] += 1

            # The number of leaf nodes receiving at least one sample.
            n_node_indices[0] = j + 1


    cdef np.ndarray _get_value_ndarray(self):
        ''' 
        Wraps predicted targets as a 1-d NumPy array.

        The array keeps a reference to this Tree, which manages
        the underlying memory.
        '''
        return self._get_node_ndarray()['value']


    cdef np.ndarray _get_node_ndarray(self):
        ''' 
        Wraps the tree nodes into a NumPy structured array.

        The array keeps a reference to this Tree, which manages the underlying
        memory. Individual fields are publicly accessible as properties of the
        Tree.
        '''
        cdef np.npy_intp shape = self.n_nodes
        cdef np.npy_intp strides = sizeof(Node)
        cdef np.ndarray out

        Py_INCREF(NODE_DTYPE)
        out = PyArray_NewFromDescr(np.ndarray, <np.dtype> NODE_DTYPE, 1, &shape,
                                   &strides, <void*> self.nodes, np.NPY_DEFAULT, None)
        Py_INCREF(self)
        out.base = <PyObject*> self
        return out
    

# =============================================================================
# ImpurityMSE
# =============================================================================


cdef struct ImpurityMSE: 
    # Mean squared error impurity measure.
    #
    # Computes the mean squared error for the prediction of the target values
    # in the node using Welfords algorithm for rolling variance (linear time
    # complexity in the number of samples):
    #
    #    var = sum([(y_i - y_bar)**2 for i in range(n_samples)])
    #        = sum([y_i**2 for i in range(n_samples)]) - n_samples * y_bar**2
    #
    double n_structure_samples
    double n_estimation_samples
    double structure_sum_targets
    double structure_sum_squared_targets
    double estimation_sum_targets

    
# =============================================================================
# Mean Squared Error Impurity Methods
# =============================================================================


cdef inline void ImpurityMSE_reset(ImpurityMSE *self) nogil:
    ''' 
    Reset the impurity. (For object recycling).
    '''
    self.structure_sum_targets = 0.0
    self.structure_sum_squared_targets = 0.0
    self.estimation_sum_targets = 0.0
    self.n_structure_samples = 0.0
    self.n_estimation_samples = 0.0


cdef inline void ImpurityMSE_update(ImpurityMSE * self, DOUBLE_t target, BOOL_t structure) nogil:
    ''' 
    Update the split statistics with the given sample. A sample can be either
    from structure stream (structure is True) or from estimation stream
    (structure is False). 
    '''        
    if structure:
        # This is a structure point.
        self.structure_sum_targets += target
        self.structure_sum_squared_targets += target**2
        self.n_structure_samples += 1
    else:
        # This is an estimation point.
        self.estimation_sum_targets += target
        self.n_estimation_samples += 1


cdef inline double ImpurityMSE_node_impurity(ImpurityMSE *self, double *n_samples=NULL) nogil:
    ''' 
    Evaluate the impurity of the node. `n_samples`
    should be used to get weighted impurity.
    '''
    cdef double result = 0.0
    
    if self.n_structure_samples > 0.0:
        result += self.structure_sum_squared_targets / self.n_structure_samples
        result -= (self.structure_sum_targets / self.n_structure_samples)**2

        if n_samples != NULL:
            result *= self.n_structure_samples / n_samples[0]
    else:
        result = ALMOST_INFINITY

    return result


cdef inline double ImpurityMSE_node_value(ImpurityMSE *self) nogil:
    ''' 
    Compute the node value from the estimation samples into value.
    '''
    if self.n_estimation_samples > 0:
        return self.estimation_sum_targets / self.n_estimation_samples
    else:
        return 0.0


cdef inline double ImpurityMSE_impurity_improvement(ImpurityMSE *self,
                                                    ImpurityMSE *impurity_left,
                                                    ImpurityMSE *impurity_right) nogil:
    ''' 
    Compute impurity improvement:

       impurity - L / N * impurity_left - R / N * impurity_right

    where N is the number of structure samples that arrived in the
    current node, L is the number of such samples in the left child
    and R is the number of such samples in the right child.
    '''
    cdef double impurity = ImpurityMSE_node_impurity(self, NULL)
    cdef double weighted_impurity_left = ImpurityMSE_node_impurity(impurity_left,
                                                                   &self.n_structure_samples)
    cdef double weighted_impurity_right = ImpurityMSE_node_impurity(impurity_right,
                                                                    &self.n_structure_samples)

    return impurity - weighted_impurity_left - weighted_impurity_right


# =============================================================================
# Splitter
# =============================================================================


cdef struct Splitter:
    # The splitter searches in the input space for a feature and a threshold
    # to split the samples.
    #
    # The splitter tracks the quality of each candidate split and when
    # certain conditions are met, the node will be split.
    #
    # The impurity computations are delegated to Impurity object.

    int           node_idx          # The index of the splitter's node.
    int           depth             # The depth of the splitter's node.
    
    int           n_created         # The number of samples the tree saw
                                    # when the splitter was created.

    int           stack_idx         # The index of the splitter on the stack.
    int           heap_idx          # The index of the splitter on the heap.
                                    # of inactive nodes.
    int           split_idx         # The index of the best split.
    
    ImpurityMSE   impurity          # Impurity of the node.
    ImpurityMSE  *impurity_lefts    # Impurity of the left candidate child nodes.
    ImpurityMSE  *impurity_rights   # Impurity of the right candidate child nodes.
    
    int          *features          # Features to split on.
    int           n_features        # Number of candidate features.
    
    double       *thresholds        # What thresholds to split at.
    int          *i_thresholds      # Number of current thresholds for the split.
    int           n_thresholds      # Number of thresholds per feature.
    
    double       *improvements      # Impurity improvements for each threshold.

    int           alpha             # The least number of estimation points that
                                    # each child node of a candidate split need to
                                    # receive before the actual split can happen.
    int           beta              # The number of estimation points that a child
                                    # node needs to receive before a split is forced.
    double        tau               # The minimum impurity improvement before the
                                    # the actual split can happen.
    bint          terminal          # Indicates that the splitter is at terminal node.

        
# =============================================================================
# Splitter Interface Methods
# =============================================================================


cdef inline int Splitter_init(SplitterPtr self, int node_idx, int depth, bint terminal,
                              int n_features, int n_thresholds, int alpha, int beta,
                              double tau, int n_samples) nogil:
    ''' 
    Initializes the splitter on a new node. Returns -1 when out of memory,
    otherwise, returns 0.
    
    Parameters:
    -----------
    node_idx: int
        The index of the node this splitter is at.
        
    depth: int
        The depth of the node this splitter is at.
    
    terminal: bool
        Indicates that this splitter is on a terminal node
        and therefore should never be split.
    
    n_features: int
        The number of candidate features for splitting. This
        number of features will be selected from the total
        number of features (`n_max_features`) uniformly at
        random.
    
    n_thresholds: int
        The number of thresholds for each individual candidate
        feature split.
        
    alpha: int
        The minimum number of estimation points candidate child
        nodes need to receive in order for the split to become
        valid.
        
    beta: int
        The minimum number of estimation points the node needs
        to see before the splitting is being forced.
        
    tau: double
        The minimum improvement in impurity a valid candidate
        split must create before the split is made, unless
        the split is forced.
        
    n_samples: int
        The number of samples the tree has seen up til now.
    '''   
    self.node_idx = node_idx
    self.depth = depth
    self.terminal = terminal
    self.n_created = n_samples    
    self.stack_idx = -1
    self.heap_idx = -1
    self.split_idx = -1
    self.n_features = n_features
    self.n_thresholds = n_thresholds
    self.alpha = alpha
    self.beta = beta
    self.tau = tau
    self.impurity_lefts = NULL
    self.impurity_rights = NULL
    self.features = NULL
    self.i_thresholds = NULL
    self.thresholds = NULL
    self.improvements = NULL

    # Initialize impurity of the node.
    ImpurityMSE_reset(&self.impurity)


cdef int Splitter_activate(SplitterPtr self, int n_max_features,
                           unsigned int *rand_r_state) nogil:
    '''
    Activates the splitter, i.e. splitter generates a random candidate splits
    and starts measuring their statistics.
    
    Parameters:
    -----------
    self: Splitter pointer
        The splitter to activate.

    n_max_features: int
        The total number of candidate features for splitting.
        
    rand_r_state: unsigned int*
        The seed for the module's random number generator. Used to random
        sampling of features.
    '''
    cdef int *features = NULL
    
    # Terminal node is active by its creation because
    # it does not need to create candidate splits but
    # it only needs to accumulate statistics.
    if self.terminal:
        return 0
    
    # Allocate the necessary arrays.
    if Splitter_create(self, self.n_features, self.n_features * self.n_thresholds) != 0:
        return -1
    
    # Resetting impurity.
    ImpurityMSE_reset(&self.impurity)
    
    for i in range(self.n_features * self.n_thresholds):
        ImpurityMSE_reset(&self.impurity_lefts[i])
        ImpurityMSE_reset(&self.impurity_rights[i])
              
    # Create temporary array for feature sampling.    
    features = <int*> malloc(n_max_features * sizeof(int))
    
    if features == NULL:
        return -1
    
    for i in range(n_max_features):
        features[i] = i

    # Fisher-Yates shuffle algorithm to pick `n_features`
    # features uniformly at random from the set of all 
    # `n_max_features` features.
    for i in range(self.n_features):
        j = rand_int(i, n_max_features, rand_r_state)
        features[i], features[j] = features[j], features[i]
        
        # Each of the features is selected with probability
        # `n_features` / `n_max_features`.
        self.features[i] = features[i]
    
    free(features)
    
    # Marks the splitter as active.
    self.heap_idx = -1
    
    return 0
    

cdef inline int Splitter_create(Splitter *self, int n_features, int n_candidate_splits) nogil:
    '''
    Allocates the memory for the splitter. Returns -1 if out of memory,
    otherwise, returns 0.
    '''
    cdef void* ptr = NULL

    self.impurity_lefts = <ImpurityMSE*> malloc(n_candidate_splits * sizeof(ImpurityMSE))
    self.impurity_rights = <ImpurityMSE*> malloc(n_candidate_splits * sizeof(ImpurityMSE))
    self.features = <int*> malloc(n_features * sizeof(int))
    self.i_thresholds = <int*> calloc(n_features, sizeof(int))
    self.thresholds = <double*> malloc(n_candidate_splits * sizeof(double))
    self.improvements = <double*> malloc(n_candidate_splits * sizeof(double))
    
    if (self.impurity_lefts == NULL or
        self.impurity_rights == NULL or
        self.features == NULL or
        self.i_thresholds == NULL or
        self.thresholds == NULL or
        self.improvements == NULL):
        return -1
    
    return 0


cdef inline void Splitter_destroy(Splitter *self) nogil:
    ''' 
    De-allocates all the memory used by the splitter.
    '''
    free(self.impurity_lefts)
    free(self.impurity_rights)
    free(self.features)
    free(self.i_thresholds)
    free(self.thresholds)
    free(self.improvements)


cdef inline bint Splitter_is_active(Splitter *self) nogil:
    '''
    Activates the splitter, i.e. it starts estimating the statistics
    for its impurity and child nodes impurities and values for each
    of its randomly selected splits.
    '''
    # Splitter is active only if it is not on the heap.
    return self.heap_idx == -1


cdef inline bint Splitter_is_split_valid(Splitter *self, int split_idx) nogil:
    ''' 
    Returns True only if the split `split_idx` creates child nodes with at least
    `self.alpha` estimation samples, otherwise, returns False.
    '''
    return (self.impurity_lefts[split_idx].n_estimation_samples >= self.alpha and
            self.impurity_rights[split_idx].n_estimation_samples >= self.alpha)


cdef inline bint Splitter_should_split(Splitter *self) nogil:
    '''
    Returns True if the node should be split, otherwise, returns False.
    
    Note that the method should be called only if the previous call to
    Splitter_update's returned True, otherwise, it has no effect.
    '''
    if self.split_idx != -1 and self.improvements[self.split_idx] > self.tau:
        return True

    return False


cdef inline bint Splitter_must_split(Splitter *self) nogil:
    '''
    Returns True if the node must be split, i.e. it has received more
    than `beta` estimation samples.
    
    Note that the method should be called only if the previous call to
    Splitter_update's returned True, otherwise, it has no effect.
    '''
    if self.split_idx != -1 and self.impurity.n_estimation_samples >= self.beta:
        return True

    return False


cdef inline bint Splitter_update(Splitter *self, DTYPE_t[::1] sample,
                                 DOUBLE_t target, BOOL_t structure,
                                 unsigned int *random_state=NULL) nogil:
    '''
    Update the statistics for the splitter from the given sample.

    Parameters:
    -----------
    self: Splitter pointer
        Spliter which statistics are updated.
    
    sample: array of floats, shape = (n_features,)
        The feature vector of the sample.
    
    target: double
        The target value associated with the sample.
    
    structure: bool
        Indicator varible. If True, the point is coming
        from estimation stream, otherwise, it is coming
        from structure stream.
        
    random_state: unsigned int pointer or NULL
        If not NULL it indicates that thresholds should
        be picke as a number from [0; 1] interval sampled
        uniformly at random.
        
    Returns:
    --------
    can_split: bool
        It is True if the splitter after update contains
        at least one valid split, otherwise, it is False.
    '''
    cdef int i, j, fidx, tidx
    cdef bint can_split = 0

    # We do not care which stream the sample came from
    # in case of terminal nodes. We use every point
    # for estimating the target variable here.
    # FIXME: How it is actually done in the paper?
    if self.terminal:
        ImpurityMSE_update(&self.impurity, target, False)
        return False
    
    # We do not care which stream the sample came from
    # in case of inactive nodes. We use every point
    # for estimating the upper-bound on prediction
    # error here.
    # FIXME: How it is actually done in the paper?
    if not Splitter_is_active(self):
        ImpurityMSE_update(&self.impurity, target, False)
        ImpurityMSE_update(&self.impurity, target, True)
        return False
    
    # First, update the impurity and statistics of the current node.      
    ImpurityMSE_update(&self.impurity, target, structure)
    
    # Second, update the impurity and statistics of candidate splits.
    for i in range(self.n_features):
        fidx = self.features[i]

        # If the candidate split on the feature (fidx) has not gathered
        # enought estimation points for thresholds yet.
        if self.i_thresholds[i] < self.n_thresholds and structure:
            if random_state != NULL:
                # Sample a random threshold from [0, 1] interval.
                self.thresholds[i * self.n_thresholds + self.i_thresholds[i]] = rand_uniform(0, 1, random_state)
            else:
                # FIXME: Make sure there are no duplicate thresholds???
                self.thresholds[i * self.n_thresholds + self.i_thresholds[i]] = sample[fidx]
            self.i_thresholds[i] += 1

        for j in range(self.i_thresholds[i]):
            tidx = i * self.n_thresholds + j

            # Update the candidate split (fidx, tidx) with the sample.
            if sample[fidx] <= self.thresholds[tidx] + FEATURE_THRESHOLD:
                ImpurityMSE_update(&self.impurity_lefts[tidx], target, structure)
            else:
                ImpurityMSE_update(&self.impurity_rights[tidx], target, structure)

            # Update impurity of the candidate child nodes, the parent node has been already updated.
            self.improvements[tidx] = ImpurityMSE_impurity_improvement(&self.impurity,
                                                                       &self.impurity_lefts[tidx],
                                                                       &self.impurity_rights[tidx])
            
            # See whether the candate split is valid.
            if Splitter_is_split_valid(self, tidx):
                if self.split_idx == -1 or self.improvements[tidx] > self.improvements[self.split_idx]:
                    self.split_idx = tidx
                can_split = True

    # Returns True if at least one of the candidate splits' become valid.
    return can_split


cdef inline void Splitter_update_node_value(SplitterPtr self, Node* nodes) nogil:
    '''
    Update the node target prediction.
    '''
    nodes[self.node_idx].value = ImpurityMSE_node_value(&self.impurity)


cdef inline double Splitter_heap_impurity(Splitter *splitter, double n_tree_samples) nogil:
    '''
    Returns the impurity of the splitter, which in case of ImpurityMSE
    is the upper-bound on the improvement of the mean squared regression
    error (MSE).
    
    Note that the value is weighted by the estimated probability of 
    a sample falling into the splitter's node.
    
    This way the node retrived from the heap should have the highest
    value in terms of potential decrease of the MSE.
    
    Parameters:
    -----------
    splitter: Splitter pointer
        The splitter which impurity is wanted.
    
    n_tree_samples: double
        The number of samples the tree has seen
        during its training.
    
    Returns:
    --------
    impurity: double
        The impurity of the splitter's node weighted
        by the estimated probability that a random
        sample falls to the node.
    '''
    cdef double node_rprobability
    
    # If there is no way to estimate the probability
    # retrive negative infinity - we do not want to
    # pick the sample in any way, when we have not
    # enough information.
    if splitter.impurity.n_structure_samples == 0:
        return -INFINITY
    
    # Compute the reciprocal of the probability of a sample
    # ending in the splitter's node.

    # The number of samples having chance falling into the node...
    node_rprobability = n_tree_samples - splitter.n_created
    
    # ... divided by the actual number of samples falling into it.
    node_rprobability /= splitter.impurity.n_structure_samples
    
    return ImpurityMSE_node_impurity(&splitter.impurity, &node_rprobability)               


cdef inline void Splitter_heap_up(SplitterPtr *heap, int index,
                                  double n_tree_samples) nogil:
    '''
    Move the splitter at the given position up in the heap.
    '''
    cdef int father, son
    
    son = index
    father = (son - 1) / 2

    # Until we reach the top of the heap.
    while son > 0:
        if Splitter_heap_impurity(heap[son], n_tree_samples) > Splitter_heap_impurity(heap[father], n_tree_samples):
            heap[father], heap[son] = heap[son], heap[father]
            # Keep the splitters' positions in the heap consistent.
            heap[son].heap_idx = son
        else:
            break
            
        # Pass the fight to the next generation.
        son = father
        father = (son - 1) / 2
            
    # Keep the splitters' positions in the heap consistent.
    heap[son].heap_idx = son    


cdef inline void Splitter_heap_down(SplitterPtr *heap, int index, int heap_size,
                                    double n_tree_samples) nogil:
    '''
    Move the splitter at the given position down in the heap.
    '''
    cdef int father, son

    father = index
    son = 2 * father + 1
    
    # Until we reach the bottom of the heap.
    while son < heap_size:
        # Fight between brothers?
        if (son + 1 < heap_size and 
            Splitter_heap_impurity(heap[son], n_tree_samples) < Splitter_heap_impurity(heap[son + 1], n_tree_samples)):
            son += 1

        # A son is better than father?
        if Splitter_heap_impurity(heap[father], n_tree_samples) < Splitter_heap_impurity(heap[son], n_tree_samples):
            heap[father], heap[son] = heap[son], heap[father]
            # Keep the splitters' positions in the heap consistent.
            heap[father].heap_idx = father
        else:
            break
            
        # Pass the fight to the next generation.
        father = son
        son = 2 * father + 1
        
    # Keep the splitters' positions in the heap consistent.
    heap[father].heap_idx = father    


cdef inline bint Splitter_heap_contains(SplitterPtr splitter) nogil:
    '''
    Returns True if the splitter is on the heap, otherwise,
    returns False.
    '''
    return splitter.heap_idx != -1


cdef inline void Splitter_heap_restore(SplitterPtr splitter, SplitterPtr *heap,
                                       int heap_size, double n_tree_samples) nogil:
    '''
    Restore the heap order after messing with the impurity value of
    the splitter. The most impure splitter is pushed to the top of
    the heap.
    
    Parameters:
    -----------
    splitter: Splitter pointer
        The splitter which impurity has changed, hence the heap
        order needs to be restored.
        
    heap: array of Splitter pointers
        The heap of splitters to restore.
    
    heap_size: int
        The number of splitters in the heap.
    
    n_tree_samples: double
        The number of samples the tree has seen
        during its training.
    '''
    cdef int index = splitter.heap_idx
    
    # Since we have no idea where to move the given splitter
    # (whether is impurity rose or fell), we try both ways.
    if index > 0:
        Splitter_heap_up(heap, index, n_tree_samples)
       
    # If the splitter did not move (or was already at the top).
    if splitter.heap_idx == index:
        Splitter_heap_down(heap, index, heap_size, n_tree_samples)


cdef inline int Splitter_heap_push(SplitterPtr self, SplitterPtr **heap, int *heap_size,
                                   int *heap_capacity, double n_tree_samples) nogil:
    '''
    Push the splitter in the heap. Inflates the heap if necessary,
    by doubling its current capacity.
    
    Parameters:
    -----------
    self: Splitter pointer
        Splitter to be pushed.
        
    heap: Splitter array pointer
        Reference to heap.
    
    heap_size: int pointer
        Reference to the current heap size.
    
    heap_capacity: int pointer
        Reference to the heap capacity.
    
    n_tree_samples: double
        The number of samples the tree has seen
        during its training.
        
    Returns:
    --------
    code: int
        -1 on out of memory, 0 otherwise.
    ''' 
    # Inflate the heap if required.
    if heap_size[0] == heap_capacity[0]:
        if heap_capacity[0] == 0:
            heap_capacity[0] = 3
        else:
            heap_capacity[0] = 2 * heap_capacity[0] + 1

        ptr = realloc(heap[0], heap_capacity[0] * sizeof(SplitterPtr))

        if ptr == NULL:
            return -1

        heap[0] = <SplitterPtr*> ptr
    
    # Save the current position of the splitter
    # for its subsequent restoration.
    self.heap_idx = heap_size[0]

    # Add the splitter into the inactive splitter queue.
    heap[0][heap_size[0]] = self
    heap_size[0] += 1
    
    # Restore the heap.
    if heap_size[0] > 1:
        Splitter_heap_up(heap[0], heap_size[0] - 1, n_tree_samples)
    
    return 0

        
cdef inline SplitterPtr Splitter_heap_pop(SplitterPtr *heap, int *heap_size,
                                          double n_tree_samples) nogil:
    '''
    Return the best splitter in the heap. The best is determined by the level
    of impurity -- the higher, the better.
    
    Parameters:
    -----------
    heap: array of Splitter pointers
        The heap of splitters to pop from.
    
    heap_size: int pointer
        The reference to the number of splitters in the heap.
    
    Returns:
    --------
    splitter: Splitter pointer
        The splitter with the largest impurity, or NULL
        if the heap is empty.
        
    '''
    cdef SplitterPtr splitter = NULL
    
    if heap_size[0] == 0:
        return NULL
    
    # The best is at the top.
    splitter = heap[0]
    heap_size[0] -= 1

    # Move the last splitter (if any) to the top.
    # and restore the heap.
    if heap_size[0] > 0:
        heap[0] = heap[heap_size[0]]
        heap[0].heap_idx = 0
        Splitter_heap_down(heap, 0, heap_size[0], n_tree_samples)

    return splitter


cdef inline int Splitter_stack_push(SplitterPtr splitter, SplitterPtr **stack,
                                   int *stack_size, int *stack_capacity) nogil:
    '''
    Push the splitter on the stack. Inflates the stack if necessary,
    by doubling its current capacity.
    
    Parameters:
    -----------
    splitter: Splitter pointer
        Splitter to be pushed.
        
    heap: Splitter array pointer
        Reference to heap.
    
    heap_size: int pointer
        Reference to the current heap size.
    
    heap_capacity: int pointer
        Reference to the heap capacity.
        
    Returns:
    --------
    code: int
        -1 on out of memory, 0 otherwise.
    ''' 
    # Inflate the heap if required.
    if stack_size[0] == stack_capacity[0]:
        if stack_capacity[0] == 0:
            stack_capacity[0] = 8
        else:
            stack_capacity[0] = 2 * stack_capacity[0]

        ptr = realloc(stack[0], stack_capacity[0] * sizeof(SplitterPtr))

        if ptr == NULL:
            return -1

        stack[0] = <SplitterPtr*> ptr
    
    # Push the splitter on the top of the stack.
    splitter.stack_idx = stack_size[0]
    stack[0][stack_size[0]] = splitter
    stack_size[0] += 1
    
    return 0

        
cdef inline SplitterPtr Splitter_stack_pop(SplitterPtr *stack, int *stack_size) nogil:
    '''
    Return the splitter on the top of the stack.
    
    Parameters:
    -----------
    stack: array of Splitter pointers
        The stack of splitters to pop from.
    
    stack_size: int pointer
        The reference to the number of splitters in the stack.
    
    Returns:
    --------
    splitter: Splitter pointer
        The splitter on the top of the stack, or NULL
        if the stack is empty.
        
    '''
    if stack_size[0] == 0:
        return NULL    
    
    stack_size[0] -= 1
    stack[stack_size[0]].stack_idx = -1
    
    return stack[stack_size[0]]


cdef inline bint Splitter_stack_contains(SplitterPtr splitter) nogil:
    '''
    Returns True if the given splitter is on the stack,
    otherwise, returns False.
    '''
    return splitter.stack_idx != -1


# =============================================================================
# Tree Grower
# =============================================================================

cdef class TreeGrower:
    cdef Tree         tree           # The tree being grown.
    
    cdef object       alphas         # The functional dependence between the tree depth
                                     # and the least number of estimation points per leaf.
    cdef object       betas          # The functional dependence between the tree depth
                                     # and the number of estimation points that force
                                     # a split.
    cdef double       tau            # The minimum impurity improvement a candidate
                                     # split must have before being executed (in regular
                                     # conditions)
    cdef double       lambda_        # The mean of Poisson distribution used to sample
                                     # the maximum number of candidate features on which
                                     # a node can be split.
            
    cdef int          max_depth      # The maximum depth of the tree.
    
    cdef int          n_samples      # The number of samples the tree was trained on, so far.
    cdef int          n_features     # The number of features.
    cdef int          n_thresholds   # The number of candidate thresholds per feature.
    
    cdef int          n_fringe       # The number of available places in growing fringe
                                     # for inactive splitters.
    
    cdef object       random_state   # Random state
    cdef unsigned int rand_r_state   # our_rand_r random number state
    cdef unsigned int *uniform       # Points to rand_r_state if uniform sampling is used,
                                     # otherwise, it is NULL.

    cdef SplitterPtr *splitters      # The pool of tree node splitters.
    cdef int          capacity       # The size of the splitters pool.
    
    cdef SplitterPtr *queue          # The priority queue of inactive splitters.
    cdef int          n_queue        # The number of inactive splitters.
    cdef int          n_queue_cap    # The queue capacity.
    
    cdef SplitterPtr *stack          # The stack of splitters ready to split.
    cdef int          n_stack        # The number of splitters on the stack.
    cdef int          n_stack_cap    # The stack capacity.
    
    cdef bint         batch_update   # Indicates that node prediction values should
                                     # be updated after call to `self.grow(...)`.
    ''' 
    Creates an regression tree growing on the fly with each new sample added to it.    
    '''
    def __cinit__(self, Tree tree, int n_features, int n_thresholds, int max_depth,
                  object alphas, object betas, double tau, double lambda_, bint uniform,
                  double value, int n_fringe, bint batch_update, object random_state):
        ''' 
        Plant a decision tree and wait for samples from which it will grow.
        Both `alpha` and `beta` should be Python functions with a single 
        integer parameter.
        '''
        cdef int i, node_idx, n_candidates

        self.tree = tree
        
        self.alphas = alphas
        self.betas = betas
        self.tau = tau
        self.lambda_ = lambda_
        self.n_fringe = max(8, n_fringe)
        self.max_depth = max_depth
        self.n_samples = 0
        self.n_features = n_features
        self.n_thresholds = n_thresholds

        self.random_state = random_state
        self.rand_r_state = random_state.randint(0, RAND_R_MAX)
        
        self.uniform = &self.rand_r_state if uniform else NULL
        
        self.splitters = NULL
        self.capacity = 0   
        
        self.queue = NULL
        self.n_queue = 0
        self.n_queue_cap = 0
        
        self.stack = NULL
        self.n_stack = 0
        self.n_stack_cap = 0
        
        self.batch_update = batch_update
        
        # Sample the number of candidate features from Poisson(lambda).
        n_candidates = min(1 + random_state.poisson(lambda_), n_features)
    
        # Add root node into the tree.
        node_idx = tree._add_node(_TREE_UNDEFINED, False, True, 0, 0.0, INFINITY, value, 0)
        
        # Put a splitter on the root node.
        if self._add_splitter(node_idx, 0, alphas(0), betas(0), n_candidates) != 0:
            raise MemoryError()


    def __dealloc__(self):
        ''' 
        Destructor.
        '''
        cdef int i

        for i in range(self.capacity):
            if self.splitters[i] != NULL:
                Splitter_destroy(self.splitters[i])
                free(self.splitters[i])
        free(self.splitters)
        
        # We do not need to call free on the inactive
        # splitters because they are freed above.
        free(self.stack)
        free(self.queue)


    cdef int _add_splitter(self, int node_idx, int depth, int alpha,
                           int beta, int n_candidate_features) nogil:
        '''
        Adds a splitter on a particular node of the tree. The splitter is
        initially inactive unless there is room in the fringe of the tree.
        
        Parameters:
        -----------
        node_idx: int
            The index of the node this splitter is placed at.
        
        depth: int
            The depth of the node this splitter is placed at.

        alpha: int
            The minimum number of estimation points candidate child
            nodes need to receive in order for the split to become
            valid.

        beta: int
            The minimum number of estimation points the node needs
            to see before the splitting is being forced.

        n_candidate_features: int
            The number of candidate features for splitting. This
            number of features will be selected from the total
            number of features (`self.n_features`) uniformly at
            random.

        Returns:
        --------
        code: int
            Either 0 on success or -1 on error (out of memory).
        '''
        cdef void * ptr
        cdef bint is_terminal
        cdef SplitterPtr splitter
        
        # Inflate the splitters' array (keep pace with growing of the tree).
        if self.tree.capacity > self.capacity:
            ptr = realloc(self.splitters, self.tree.capacity * sizeof(SplitterPtr*))
            
            if ptr == NULL:
                return -1
            
            self.splitters = <SplitterPtr*> ptr
            
            # To make __dealloc__ work properly.
            memset(<void*>(self.splitters + self.capacity), 0,
                   (self.tree.capacity - self.capacity) * sizeof(SplitterPtr))
            
            self.capacity = self.tree.capacity
            
        splitter = <SplitterPtr> malloc(sizeof(Splitter))
        
        if splitter == NULL:
            return -1
        
        # Put the splitter on the node.
        self.splitters[node_idx] = splitter
        
        is_terminal = (depth == self.max_depth)
        
        # Initialize the splitter.
        Splitter_init(splitter, node_idx, depth, is_terminal, n_candidate_features,
                      self.n_thresholds, alpha, beta, self.tau, self.n_samples)
        
        # Terminal nodes are not pushed on the heap
        # because they are automatically activated.
        if is_terminal:
            return 0
        
        # Put the new splitter in an inactive splitters priority queue.
        if Splitter_heap_push(splitter, &self.queue, &self.n_queue,
                              &self.n_queue_cap, self.n_samples) != 0:
            return -1

        # Pop the best inactive splitter if there is room in the fringe
        # and there is an inactive splitter.
        if self.n_fringe > 0 and self.n_queue > 0:            
            splitter = Splitter_heap_pop(self.queue, &self.n_queue, self.n_samples)

            if Splitter_activate(splitter, self.n_features, &self.rand_r_state) != 0:
                return -1
            
            self.n_fringe -= 1

        return 0
        

    cdef int _split_node(self, SplitterPtr splitter):
        '''
        Splits the node with the given splitter and destroys it.
        Two new splitters (initially inactive) are created for
        each new child node.
        
        Parameters:
        -----------
        splitter: Splitter pointer
            The splitter used to split its node.
        
        Returns:
        --------
        code: int
            Either 0 on success or -1 on error (out of memory).
        '''
        cdef int left_idx, right_idx, idx = splitter.split_idx

        cdef int alpha = self.alphas(splitter.depth + 1)
        cdef int beta = self.betas(splitter.depth + 1)

        cdef int n_candidates_left = min(1 + self.random_state.poisson(self.lambda_), self.n_features)
        cdef int n_candidates_right = min(1 + self.random_state.poisson(self.lambda_), self.n_features)

        with nogil:
            # Just one vacant seat for a new leaf on the fringe.
            self.n_fringe += 1

            # Split the former leaf node.
            self.tree._split_leaf_node(splitter.node_idx,
                                       splitter.features[idx / self.n_thresholds],
                                       splitter.thresholds[idx],
                                       ImpurityMSE_node_impurity(&splitter.impurity),
                                       splitter.impurity.n_structure_samples)

            # Add left child node.
            left_idx = self.tree._add_node(splitter.node_idx, True, True, 0, 0.0,
                                           ImpurityMSE_node_impurity(&splitter.impurity_lefts[idx]),
                                           ImpurityMSE_node_value(&splitter.impurity_lefts[idx]),
                                           splitter.impurity_lefts[idx].n_structure_samples)
            if left_idx == -1:
                return -1

            # Put a splitter on the left child node.
            if self._add_splitter(left_idx, splitter.depth + 1, alpha, beta, n_candidates_left) != 0:
                return -1

            # Add right child node.
            right_idx = self.tree._add_node(splitter.node_idx, False, True, 0, 0.0,
                                            ImpurityMSE_node_impurity(&splitter.impurity_rights[idx]),
                                            ImpurityMSE_node_value(&splitter.impurity_rights[idx]),
                                            splitter.impurity_rights[idx].n_structure_samples)
            if right_idx == -1:
                return -1

            # Put a splitter on the right child node.
            if self._add_splitter(right_idx, splitter.depth + 1, alpha, beta, n_candidates_right) != 0:
                return -1

            # Remove the splitter.
            self.splitters[splitter.node_idx] = NULL # Because of __dealloc__.
            Splitter_destroy(splitter)
            
        return 0


    cpdef grow(self, DTYPE_t[:, ::1] samples, DOUBLE_t[::1] targets, BOOL_t[::1] structure):
        ''' 
        Update the tree with the given samples and targets.
        '''
        cdef SplitterPtr splitter
        cdef int sample_idx, rc = 0, q = 0
        cdef int n_samples = samples.shape[0]

        with nogil:
            for sample_idx in range(n_samples):
                # Tree sees another sample.
                self.n_samples += 1

                # Find the splitter on the leaf for the sample.
                splitter = self.splitters[self.tree._apply_c(samples[sample_idx])]

                # If there is at least one valid candidate split...
                if Splitter_update(splitter, samples[sample_idx], targets[sample_idx],
                                   structure[sample_idx], self.uniform):
                    # ... try whether it is good enough or it has to be forced.
                    if Splitter_should_split(splitter) or Splitter_must_split(splitter):
                        if self.batch_update:
                            if not Splitter_stack_contains(splitter):
                                Splitter_stack_push(splitter, &self.stack, &self.n_stack,
                                                    &self.n_stack_cap)
                        else:
                            # GIL is needed for sampling out of Poisson her.
                            with gil:
                                if self._split_node(splitter) != 0:
                                    rc = -1
                                    break

                        # In batch mode we update the splitter's node
                        # value in the end. The splitter also becomes
                        # invalid after splitting, so we need to avoid
                        # using it for updating its node's value
                        # (see below).
                        continue

                # Update the node's prediction value?
                # Splitter_update_node_value(splitter, self.tree.nodes)

                if Splitter_heap_contains(splitter):
                    Splitter_heap_restore(splitter, self.queue, self.n_queue, self.n_samples)
            
            # Split the nodes waiting to be split in the batch mode.
            if self.batch_update:
                splitter = Splitter_stack_pop(self.stack, &self.n_stack)
                while splitter != NULL:
                    with gil:
                        if self._split_node(splitter) != 0:
                            rc = -1
                            break
                    splitter = Splitter_stack_pop(self.stack, &self.n_stack)

        if rc == -1:
            raise MemoryError()


# =============================================================================
# Utils
# =============================================================================


# rand_r replacement using a 32bit XorShift generator
# See http://www.jstatsoft.org/v08/i14/paper for details
cdef inline int our_rand_r(unsigned int *seed) nogil:
    seed[0] ^= <unsigned int>(seed[0] << 13)
    seed[0] ^= <unsigned int>(seed[0] >> 17)
    seed[0] ^= <unsigned int>(seed[0] << 5)

    return <int>(seed[0] & <unsigned int>RAND_R_MAX)


cdef inline int rand_int(int low, int high, unsigned int *random_state) nogil:
    ''' 
    Generate a random integer in [low, end).
    '''
    return low + our_rand_r(random_state) % (high - low)


cdef inline double rand_uniform(double low, double high, unsigned int *random_state) nogil:
    ''' 
    Generate a random double in [low; high).
    '''
    return ((high - low) * <double> our_rand_r(random_state) / <double> RAND_R_MAX) + low


# =============================================================================
# Utils not used but might become handy in the future.
# =============================================================================

# Sorting Algorithm: Introsort (Musser, SP&E, 1997).
cdef inline void sort(int *node_indices, int *sample_indices, int n_samples) nogil:
    cdef int max_depth = 2 * <int>log(n_samples)
    introsort(node_indices, sample_indices, n_samples, max_depth)


cdef inline void swap(int *node_indices, int *sample_indices, int i, int j) nogil:
    node_indices[i], node_indices[j] = node_indices[j], node_indices[i]
    sample_indices[i], sample_indices[j] = sample_indices[j], sample_indices[i]


cdef inline int median3(int *node_indices, int n_samples) nogil:
    # Median of three pivot selection, after Bentley and McIlroy (1993).
    # Engineering a sort function. SP&E. Requires 8/3 comparisons on average.
    cdef int a = node_indices[0]
    cdef int b = node_indices[n_samples / 2]
    cdef int c = node_indices[n_samples - 1]

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


# Introsort with median of 3 pivot selection and 3-way partition function
# (robust to repeated elements, e.g. lots of samples in the same nodes).
cdef void introsort(int *node_indices, int *sample_indices, int n_samples, int max_depth) nogil:
    cdef int pivot, i, l, r

    while n_samples > 1:
        if max_depth <= 0: # max depth limit exceeded ("gone quadratic")
            heapsort(node_indices, sample_indices, n_samples)
            return

        max_depth -= 1

        pivot = median3(node_indices, n_samples)

        # Three-way partition.
        i = l = 0
        r = n_samples
        while i < r:
            if node_indices[i] < pivot:
                swap(node_indices, sample_indices, i, l)
                i += 1
                l += 1
            elif node_indices[i] > pivot:
                r -= 1
                swap(node_indices, sample_indices, i, r)
            else:
                i += 1

        introsort(node_indices, sample_indices, l, max_depth)

        node_indices += r
        sample_indices += r
        n_samples -= r


cdef inline void sift_down(int *node_indices, int* sample_indices, int start, int end) nogil:
    # Restore heap order in node_indices[start:end] by moving the max element to start.
    cdef int child, maxind, root

    root = start

    while True:
        child = root * 2 + 1

        # find max of root, left child, right child
        maxind = root
        if child < end and node_indices[maxind] < node_indices[child]:
            maxind = child
        if child + 1 < end and node_indices[maxind] < node_indices[child + 1]:
            maxind = child + 1

        if maxind == root:
            break
        else:
            swap(node_indices, sample_indices, root, maxind)
            root = maxind


cdef void heapsort(int * node_indices, int *sample_indices, int n_samples) nogil:
    cdef int start = (n_samples - 2) / 2
    cdef int end = n_samples

    while True:
        sift_down(node_indices, sample_indices, start, end)
        if start == 0:
            break
        start -= 1

    end = n_samples - 1

    while end > 0:
        swap(node_indices, sample_indices, 0, end)
        sift_down(node_indices, sample_indices, 0, end)
        end = end - 1


cdef inline double log(double x) nogil:
    return ln(x) / ln(2.0)
