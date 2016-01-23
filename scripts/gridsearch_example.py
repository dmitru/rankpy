# -*- coding: utf-8 -*-

import numpy as np

import logging

from rankpy.queries import Queries
from rankpy.queries import find_constant_features

from rankpy.models import LambdaMART

from rankpy.gridsearch import gridsearch
from rankpy.gridsearch import train_test_split

# Turn on logging.
logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : '
                    '%(message)s', level=logging.INFO)

# Load the query datasets.
training_queries = Queries.load_from_text('data/MQ2007/Fold1/train.txt')
validation_queries = Queries.load_from_text('data/MQ2007/Fold1/vali.txt')
test_queries = Queries.load_from_text('data/MQ2007/Fold1/test.txt')

logging.info('=' * 80)

# Save them to binary format ...
training_queries.save('data/MQ2007/Fold1/training')
validation_queries.save('data/MQ2007/Fold1/validation')
test_queries.save('data/MQ2007/Fold1/test')

# ... because loading them will be then faster.
training_queries = Queries.load('data/MQ2007/Fold1/training')
validation_queries = Queries.load('data/MQ2007/Fold1/validation')
test_queries = Queries.load('data/MQ2007/Fold1/test')

logging.info('=' * 80)

# Set this to True in order to remove queries containing all documents
# of the same relevance score -- these are useless for LambdaMART.
remove_useless_queries = False

# Find constant query-document features.
cfs = find_constant_features([training_queries,
                              validation_queries,
                              test_queries])

# Get rid of constant features and (possibly) remove useless queries.
training_queries.adjust(remove_features=cfs, purge=remove_useless_queries)
validation_queries.adjust(remove_features=cfs, purge=remove_useless_queries)
test_queries.adjust(remove_features=cfs)

# Print basic info about query datasets.
logging.info('Train queries: %s' % training_queries)
logging.info('Valid queries: %s' % validation_queries)
logging.info('Test queries: %s' % test_queries)

logging.info('=' * 80)

param_grid = {'metric':              ['NDCG@10'],
              'max_features':        [0.5, None],
              'max_leaf_nodes':      [4, 8],
              ('min_samples_split',
               'min_samples_leaf') : [(200, 100)],
              'shrinkage':           [0.1, 0.5],
              'estopping':           [50],
              'random_state':        [42]}

# Split validation queries (50/50) for early stopping. We cannot just use 
# validation queries for both early stopping and model selection because
# of "data snooping" bias.
estop_queries, validation_queries = train_test_split(validation_queries,
                                                test_size=0.5)

model, scores = gridsearch(LambdaMART, param_grid, training_queries,
                           estopping_queries=estop_queries,
                           validation_queries=validation_queries,
                           return_scores=True, n_jobs=-1,
                           random_state=23)

logging.info('=' * 80)
logging.info('Gridsearch results')
logging.info('=' * 80)

for gparams, gmodel, gscore in scores:
    logging.info('Parameters: %r' % gparams)
    logging.info('Model: %s' % gmodel)
    logging.info('Performance: %11.8f (holdout %s)' % (gscore, gmodel.metric))
    logging.info('-' * 80)

logging.info('Best model: %s' % model)
logging.info('Performance: %11.8f (test %s)' % (model.evaluate(
                                                    validation_queries),
                                                model.metric))
