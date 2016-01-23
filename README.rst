RankPy: Learning to Rank with Python
====================================

Currently, the RankPy project is under intensive development. The goal is to provide reliable, efficient, and stable implementations of the state of the art
learning to rank algorithms. Why not start with one of the most famous algorithms, which is available right now: LambdaMART.


Prerequisities
--------------
- Python (2.7)
- Cython (0.22)
- NumPy  (1.9.2)
- SciPy  (0.15.1)
- Scikit-learn (0.16.1)

Installation
------------
Install the prerequisites and RankPY as follows::

    $ git clone https://bitbucket.org/tunystom/rankpy.git
    $ cd rankpy
    $ pip install -r requirements.txt
    $ python setup.py install


Using RankPy: Simple Example
----------------------------
1) Prepare data in svmlight format, e.g., download the *MQ2007* (see next section on `Data`_) ::

        $ mkdir data
        $ wget http://research.microsoft.com/en-us/um/beijing/projects/letor/LETOR4.0/Data/MQ2007.rar -O data/MQ2007.rar
        $ unrar x data/MQ2007.rar data/


2) Create a python script file with the following content::

        $ cat << EOF > run_lambdamart.py
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
        EOF

3) Run the script::

        $ python run_lambdamart.py

4) Enjoy the results::

        2016-01-23 11:10:11,967 : Reading queries from data/MQ2007/Fold1/train.txt.
        2016-01-23 11:10:13,452 : Read 244 queries and 10000 documents so far.
        2016-01-23 11:10:14,951 : Read 479 queries and 20000 documents so far.
        2016-01-23 11:10:16,424 : Read 720 queries and 30000 documents so far.
        2016-01-23 11:10:17,901 : Read 963 queries and 40000 documents so far.
        2016-01-23 11:10:18,219 : Read 1017 queries and 42158 documents out of which 0 queries and 0 documents were discarded.
        2016-01-23 11:10:19,247 : Reading queries from data/MQ2007/Fold1/vali.txt.
        2016-01-23 11:10:20,835 : Read 246 queries and 10000 documents so far.
        2016-01-23 11:10:21,402 : Read 339 queries and 13813 documents out of which 0 queries and 0 documents were discarded.
        2016-01-23 11:10:21,727 : Reading queries from data/MQ2007/Fold1/test.txt.
        2016-01-23 11:10:23,232 : Read 245 queries and 10000 documents so far.
        2016-01-23 11:10:23,771 : Read 336 queries and 13652 documents out of which 0 queries and 0 documents were discarded.
        2016-01-23 11:10:24,086 : ================================================================================
        2016-01-23 11:10:24,139 : Loading queries from data/MQ2007/Fold1/training.
        2016-01-23 11:10:24,356 : Loaded 1017 queries with 42158 documents in total.
        2016-01-23 11:10:24,356 : Loading queries from data/MQ2007/Fold1/validation.
        2016-01-23 11:10:24,426 : Loaded 339 queries with 13813 documents in total.
        2016-01-23 11:10:24,426 : Loading queries from data/MQ2007/Fold1/test.
        2016-01-23 11:10:24,502 : Loaded 336 queries with 13652 documents in total.
        2016-01-23 11:10:24,502 : ================================================================================
        2016-01-23 11:10:24,502 : Train queries: Queries (1017 queries, 42158 documents, 46 features, 2 max. relevance)
        2016-01-23 11:10:24,502 : Valid queries: Queries (339 queries, 13813 documents, 46 features, 2 max. relevance)
        2016-01-23 11:10:24,502 : Test queries: Queries (336 queries, 13652 documents, 46 features, 2 max. relevance)
        2016-01-23 11:10:24,502 : ================================================================================
        2016-01-23 11:10:24,561 : Train queries: Queries (1017 queries, 42158 documents, 41 features, 2 max. relevance)
        2016-01-23 11:10:24,561 : Valid queries: Queries (339 queries, 13813 documents, 41 features, 2 max. relevance)
        2016-01-23 11:10:24,561 : Test queries: Queries (336 queries, 13652 documents, 41 features, 2 max. relevance)
        2016-01-23 11:10:24,561 : ================================================================================
        2016-01-23 11:10:24,568 : Training of LambdaMART model has started.
        2016-01-23 11:10:25,402 : #00000001: NDCG@10 (training):    0.41002253 (7311.70920937)  |  (validation):    0.41130524
        2016-01-23 11:10:26,168 : #00000002: NDCG@10 (training):    0.41005427 (8956.40257430)  |  (validation):    0.41114890
        2016-01-23 11:10:26,888 : #00000003: NDCG@10 (training):    0.41298782 (8537.53497895)  |  (validation):    0.41217137
        2016-01-23 11:10:27,659 : #00000004: NDCG@10 (training):    0.41282811 (8262.79183708)  |  (validation):    0.41191528
        2016-01-23 11:10:28,371 : #00000005: NDCG@10 (training):    0.41365948 (8011.38396906)  |  (validation):    0.41236446
        2016-01-23 11:10:29,086 : #00000006: NDCG@10 (training):    0.41383714 (7833.12660515)  |  (validation):    0.41327476
        2016-01-23 11:10:29,867 : #00000007: NDCG@10 (training):    0.41765956 (7679.36439399)  |  (validation):    0.41597709
        2016-01-23 11:10:30,595 : #00000008: NDCG@10 (training):    0.41720731 (7581.25200952)  |  (validation):    0.41592357
        2016-01-23 11:10:31,335 : #00000009: NDCG@10 (training):    0.41759721 (7458.44560063)  |  (validation):    0.41753739
        2016-01-23 11:10:31,907 : #00000010: NDCG@10 (training):    0.41756696 (7372.92064216)  |  (validation):    0.41772050
        (... 79 more lines ...)
        2016-01-23 11:11:29,927 : #00000090: NDCG@10 (training):    0.48020542 (6151.61869473)  |  (validation):    0.46076222
        (... 40 more lines ...)
        2016-01-23 11:11:55,680 : #00000131: NDCG@10 (training):    0.49675316 (5898.02266883)  |  (validation):    0.45780273
        2016-01-23 11:11:56,236 : #00000132: NDCG@10 (training):    0.49732270 (5889.21435296)  |  (validation):    0.45812762
        2016-01-23 11:11:56,803 : #00000133: NDCG@10 (training):    0.49777376 (5888.12108406)  |  (validation):    0.45807680
        2016-01-23 11:11:57,541 : #00000134: NDCG@10 (training):    0.49752163 (5884.32866622)  |  (validation):    0.45778792
        2016-01-23 11:11:58,333 : #00000135: NDCG@10 (training):    0.49776780 (5875.30263816)  |  (validation):    0.45773001
        2016-01-23 11:11:59,129 : #00000136: NDCG@10 (training):    0.49769031 (5870.65797954)  |  (validation):    0.45881187
        2016-01-23 11:11:59,692 : #00000137: NDCG@10 (training):    0.49747754 (5859.51664451)  |  (validation):    0.45891329
        2016-01-23 11:12:00,396 : #00000138: NDCG@10 (training):    0.49785502 (5854.70879573)  |  (validation):    0.45773058
        2016-01-23 11:12:01,040 : #00000139: NDCG@10 (training):    0.49872081 (5850.96921858)  |  (validation):    0.45870160
        2016-01-23 11:12:01,742 : #00000140: NDCG@10 (training):    0.49931365 (5844.96868533)  |  (validation):    0.45921750
        2016-01-23 11:12:01,743 : Stopping early since no improvement on validation queries has been observed for 50 iterations (since iteration 90)
        2016-01-23 11:12:01,743 : Final model performance (NDCG@10) on validation queries:  0.46076222
        2016-01-23 11:12:01,743 : Setting the number of trees of the model to 90.
        2016-01-23 11:12:01,743 : Training of LambdaMART model has finished.
        2016-01-23 11:12:01,743 : ================================================================================
        2016-01-23 11:12:01,886 : NDCG@10 on the test queries: 0.48676943
        2016-01-23 11:12:01,887 : Saving LambdaMART object into LambdaMART_L7_S0.1_E50_NDCG@10

Data
----
RankPy acceptes data formatted in the SVMlight (see http://svmlight.joachims.org/) format.
You can download learning to rank data sets here:

- **GOV**: http://research.microsoft.com/en-us/um/beijing/projects/letor/LETOR3.0/Gov.rar (you'll need files in QueryLevelNorm)
- **OHSUMED**: http://research.microsoft.com/en-us/um/beijing/projects/letor/LETOR3.0/OHSUMED.zip
- **MQ2007**: http://research.microsoft.com/en-us/um/beijing/projects/letor/LETOR4.0/Data/MQ2007.rar (files for supervised learning)
- **MQ2008**: http://research.microsoft.com/en-us/um/beijing/projects/letor/LETOR4.0/Data/MQ2008.rar (files for supervised learning)
- **Yahoo!**: http://webscope.sandbox.yahoo.com/catalog.php?datatype=c
- **MSLR-WEB10K**: http://research.microsoft.com/en-us/um/beijing/projects/mslr/data/MSLR-WEB10K.zip
- **MSLR-WEB30K**: http://research.microsoft.com/en-us/um/beijing/projects/mslr/data/MSLR-WEB30K.zip
- **Yandex Internet Mathematics 2009**: http://imat2009.yandex.ru/academic/mathematic/2009/en/datasets (query identifier need to be parsed out of comment into qid feature)

All credit for making this list goes to Anne Schuth -- check out [Lerot: an Online Learning to Rank Framework](https://bitbucket.org/ilps/lerot).

Acknowledgements
----------------
Parts of this project were created during my visit at the ILPS research group at the University of Amsterdam, which was funded by ESF (European Science Foundation) and CTU Media Lab Foundation.

License
-------
This program is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with this program. If not, see http://www.gnu.org/licenses/.
