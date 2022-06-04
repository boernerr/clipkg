'''This file will be the "glue" file that facilitates __main__.py  interaction with k-means and creating data.'''

import logging

from os import path

from clipkg.src import mk_data_from_pipe
from clipkg.src.models import kmeans_document_cluster, srsp_text_classification, srsp_predict

# LOGGER = logging.getLogger(path.basename(__name__))
LOGGER = logging.getLogger(__name__)

def configure_logging():
    '''Generic logging func.'''
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)-8s %(name)s - %(message)s',
        datefmt='%d %b %Y %H:%M:%S'
        # ,filename='test_log_file.txt' # file storing logs
    )

def train_model():
    # section 1: for running mk_data_from_pipe.py
    created_file, text_col = mk_data_from_pipe.main()
    LOGGER.debug(f'CREATED transformed data file: {created_file}')
    LOGGER.debug(f'text col supplied: {text_col}')

    # section 2: Feed outputs of section 1 -> kmeans_document_cluster.py
    kmeans_file = kmeans_document_cluster.main(created_file, text_col)
    LOGGER.debug(f'CREATED k-means file: {kmeans_file}')

    # section 3: Feed outputs from section 2 -> srsp_text_classification.py
    pickled_model_file = srsp_text_classification.main(kmeans_file)
    if pickled_model_file:
        LOGGER.debug(f'CREATED {pickled_model_file}')
    else:
        LOGGER.debug('CREATED .pickle file')

    # TODO ***** It may be best to have predicted capability housed in a separate function altogether *****
    # TODO section 4: Accept new data to predict on model trained in section 3 -> srsp_predict.py
    # srsp_predict.main()

def mk_data():
    '''Stand alone function call to only normalize data.'''
    mk_data_from_pipe.main()
    LOGGER.debug('Ran Transform option')

def predict():
    '''Stand-alone function to use a pre-trained model on a raw .csv file.

    This should remain a stand-alone function because train_model() '''
    created_file, text_col = mk_data_from_pipe.main()
    # Right now, we have the modified raw .csv and the text column
    srsp_predict.main(created_file, text_col)