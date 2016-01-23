# -*- coding: utf-8 -*-
#
# Author: Tomas Tunys
# Source: Original Perl script by Jun Xu can be found here: http://research.microsoft.com/en-us/projects/mslr/eval-ttest-mslr.pl.txt

''' 
Significance test for learning to rank algorithms.

The input files are expected to be in the following
format:

qid1	measure_value1
qid2	measure_value2
...
qidN	measure_valueN

where each line specifies both the query ID and 
the IR evaluation measure of a L2R algorithm on
that query.

The input files are expected to contain the same
number of queries with the same query IDs not
necessarily in the same order.
'''

import argparse
import numpy as np

from scipy.stats import ttest_rel


def load_evaluation_file(filename):
    with open(filename, 'r') as ifile:
        qids, values = zip(*map(str.split, ifile.readlines()))
        return np.array(qids, dtype=np.intp), np.array(values, dtype=np.float64)


def ttest(filename1, filename2):
    qids1, values1 = load_evaluation_file(arguments.filename1)
    qids2, values2 = load_evaluation_file(arguments.filename2)

    if qids1.shape[0] != qids2.shape[0]:
        raise ValueError('number of queries in files do not match (%d != %d)'\
                         % (qids1.shape[0], qids2.shape[0]))

    qids1_sort_idxs = np.argsort(qids1)
    qids2_sort_idxs = np.argsort(qids2)

    qids1 = qids1[qids1_sort_idxs]
    qids2 = qids2[qids2_sort_idxs]

    if np.any(qids1 != qids2):
        raise ValueError('files do not contain the same queries')

    values1 = values1[qids1_sort_idxs]
    values2 = values2[qids2_sort_idxs]

    mean1 = np.mean(values1)
    mean2 = np.mean(values2)

    t_statistic, p_value = ttest_rel(values1, values2)    

    return values1.shape[0], mean1, mean2, t_statistic, p_value

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    parser.add_argument('filename1', help='specify evaluation file from algorithm 1')
    parser.add_argument('filename2', help='specify evaluation file from algorithm 2')

    arguments = parser.parse_args()

    n_queries, mean1, mean2, t_statistic, p_value = ttest(arguments.filename1, arguments.filename2)

    print 'Number of queries:    %d' % n_queries
    print 'Mean perf. measure 1: %.8f' % mean1
    print 'Mean perf. measure 2: %.8f' % mean2
    print 't-statistic:          %.8f' % t_statistic
    print 'p-value:              %.8f' % p_value
