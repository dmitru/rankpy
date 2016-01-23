# -*- coding: utf-8 -*-

import numpy as np

import logging

from rankpy.queries import Queries
from rankpy.queries import find_constant_features

from rankpy.models import LambdaMART


# Turn on logging.
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)

# Load the query datasets.
training_queries = Queries.load_from_text('data/MQ2007/Fold1/train.txt')
validation_queries = Queries.load_from_text('data/MQ2007/Fold1/vali.txt')
test_queries = Queries.load_from_text('data/MQ2007/Fold1/test.txt')

logging.info('================================================================================')

# Save them to binary format ...
training_queries.save('data/MQ2007/Fold1/training')
validation_queries.save('data/MQ2007/Fold1/validation')
test_queries.save('data/MQ2007/Fold1/test')

# ... because loading them will be then faster.
training_queries = Queries.load('data/MQ2007/Fold1/training')
validation_queries = Queries.load('data/MQ2007/Fold1/validation')
test_queries = Queries.load('data/MQ2007/Fold1/test')

logging.info('================================================================================')

# Print basic info about query datasets.
logging.info('Train queries: %s' % training_queries)
logging.info('Valid queries: %s' % validation_queries)
logging.info('Test queries: %s' %test_queries)

logging.info('================================================================================')

# Set this to True in order to remove queries containing all documents
# of the same relevance score -- these are useless for LambdaMART.
remove_useless_queries = False

# Find constant query-document features.
cfs = find_constant_features([training_queries, validation_queries, test_queries])

# Get rid of constant features and (possibly) remove useless queries.
training_queries.adjust(remove_features=cfs, purge=remove_useless_queries)
validation_queries.adjust(remove_features=cfs, purge=remove_useless_queries)
test_queries.adjust(remove_features=cfs)

# Print basic info about query datasets.
logging.info('Train queries: %s' % training_queries)
logging.info('Valid queries: %s' % validation_queries)
logging.info('Test queries: %s' % test_queries)

logging.info('================================================================================')

model = LambdaMART(metric='NDCG@10', max_leaf_nodes=7, shrinkage=0.1,
                   estopping=50, n_jobs=-1, min_samples_leaf=50,
                   random_state=42)

model.fit(training_queries, validation_queries=validation_queries)

logging.info('================================================================================')

logging.info('%s on the test queries: %.8f'
             % (model.metric, model.evaluate(test_queries, n_jobs=-1)))

model.save('LambdaMART_L7_S0.1_E50_' + model.metric)