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
import numpy as np
import scipy.sparse

try:
    import cPickle as _pickle
except ImportError:
    import pickle as _pickle

from .externals.joblib import cpu_count


def pickle(obj, filepath, protocol=-1):
    '''
    Pickle the object into the specified file.

    Parameters:
    -----------
    obj: object
        The object that should be serialized.

    filepath:
        The location of the resulting pickle file.
    '''
    with open(filepath, 'wb') as fout:
        _pickle.dump(obj, fout, protocol=protocol)


def unpickle(filepath):
    '''
    Unpicle the object serialized in the specified file.

    Parameters:
    -----------
    filepath:
        The location of the file to unpickle.
    '''
    with open(filepath, 'rb') as fin:
        return _pickle.load(fin)


def save_spmatrix(filename, X, compress=False, tempdir=None):
    '''
    Serializes X into file using numpy .npz format
    with an optional compression.
    '''
    is_csc = False
    if scipy.sparse.isspmatrix_csc(X):
        is_csc = True
    elif not scipy.sparse.isspmatrix_csr(X):
        raise TypeError('X is not a sparse matrix.')

    is_csc = np.asarray(is_csc)

    npz = dict(data=X.data, indices=X.indices, indptr=X.indptr,
               shape=X.shape, is_csc=is_csc)
    save_fcn = np.savez_compressed if compress else np.savez

    old_tmpdir = os.environ.get('TMPDIR')
    os.environ['TMPDIR'] = os.getcwd() if tempdir is None else tempdir

    try:
        save_fcn(filename, **npz)
    finally:
        if old_tmpdir is None:
            del os.environ['TMPDIR']
        else:
            os.environ['TMPDIR'] = old_tmpdir


def load_spmatrix(filename):
    '''
    Load a sparse matrix X from the specified .npz file.
    '''
    npz = np.load(filename)

    if npz['is_csc']:
        return scipy.sparse.csc_matrix((npz['data'], npz['indices'],
                                        npz['indptr']), shape=npz['shape'])
    else:
        return scipy.sparse.csr_matrix((npz['data'], npz['indices'],
                                        npz['indptr']), shape=npz['shape'])


def parallel_helper(obj, methodname, *args, **kwargs):
    '''
    Helper function to avoid pickling problems when using Parallel loops.
    '''
    return getattr(obj, methodname)(*args, **kwargs)


def _get_n_jobs(n_jobs):
    '''
    Get number of jobs for the computation.

    This function reimplements the logic of joblib to determine the actual
    number of jobs depending on the cpu count:
        - If ``n_jobs`` is -1 all CPUs are used.
        - If ``n_jobs`` is greater than 0, ``n_jobs`` jobs is used.
        - If ``n_jobs`` is less than 0, ``n_cpus + n_jobs + 1`` jobs is used.
        - If ``n_jobs`` is 0, ``ValueError`` exception is raised.

    In all but last case, the returned number of jobs is at least 1 and
    at most the number of CPUs.

    Parameters
    ----------
    n_jobs : int
        The wanted number of jobs stated in joblib convention.

    Returns
    -------
    n_jobs : int
        The actual number of jobs as positive integer smaller than ``maximum``
        (if given).
    '''
    if n_jobs < 0:
        return max(cpu_count() + 1 + n_jobs, 1)
    elif n_jobs == 0:
        raise ValueError('n_jobs == 0 has no meaning')
    else:
        return min(n_jobs, cpu_count())


def _get_partition_indices(start, end, n_jobs):
    '''
    Get boundary indices for ``n_jobs`` number of sub-arrays dividing
    a (contiguous) array of indices starting with ``start`` (inclusive)
    and ending with ``end`` (exclusive) into equal parts.
    '''
    if (end - start) >= n_jobs:
        return np.linspace(start, end, n_jobs + 1).astype(np.intc)
    else:
        return np.arange(end - start + 1, dtype=np.intc)

def aslist(*args):
    '''
    Helper method which wraps the parameters into a list and removes
    any None element from it.
    '''
    return list(filter(None, args))