'''
Logging and testing.

Author: Dudko Anastasiia
Date: 03082022
'''
import os
import logging

from churn_library import (import_data,
                           perform_eda,
                           encoder_helper,
                           perform_feature_engineering,
                           train_models)

from constants import LOG_FILE_PATH


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    try:
        assert os.path.isdir('./images/eda')
        assert len(os.listdir('./images/eda')) == 6
        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError as error:
        logging.info("Testing perform_eda:  Dir with eda plot does not exist")
        raise error


def test_encoder_helper(encoder_helper, df):
    '''
    test encoder helper
    '''
    category_lst = ['Gender']
    encoded_df = encoder_helper(df, category_lst)

    try:
        assert encoded_df.shape[0] > 0
        assert encoded_df.shape[1] > 0
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as error:
        logging.info(
            "Testing encoder_helper: Encoded dataframe contains 0 rows or columns ")
        raise error
    return encoded_df


def test_perform_feature_engineering(perform_feature_engineering, df):
    '''
        test perform_feature_engineering
        '''
    x_train, x_test, y_train, y_test = perform_feature_engineering(df)
    try:
        assert len(x_train) > 0
        assert len(x_test) > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        logging.info(
            "Testing perform_feature_engineering: SUCCESS - \
                    train and test dataframes are not empty")
    except AssertionError as error:
        logging.info(
            "Testing perform_feature_engineering: train and test dataframes should have length > 0")
        raise error

    try:
        assert len(x_train) == len(y_train)
        assert len(x_test) == len(y_test)
        logging.info(
            "Testing perform_feature_engineering: SUCCESS - x and y have the same length")
    except AssertionError as error:
        logging.info(
            "Testing perform_feature_engineering: x and y should have the same length")
        raise error
    return x_train, x_test, y_train, y_test


def test_train_models(train_models):
    '''
    test train_models
    '''
    try:
        assert os.path.isdir('./images/results')
        assert len(os.listdir('./images/results')) == 3
        logging.info(
            "Testing train_models: SUCCESS - dir with model results exists")
    except AssertionError as error:
        logging.info(
            "Testing perform_eda: Dir with models results does not exist")
        raise error

    try:
        assert os.path.isdir('./models')
        assert len(os.listdir('./models')) == 2
        logging.info(
            "Testing train_models: SUCCESS - dir with saved model exists")
    except AssertionError as error:
        logging.info(
            "Testing perform_eda: Dir with saved models does not exist")
        raise error


if __name__ == "__main__":
    logging.basicConfig(
        filename=LOG_FILE_PATH,
        level=logging.INFO,
        filemode='w',
        format='%(name)s - %(levelname)s - %(message)s')

    df = import_data('./data/bank_data.csv')
    test_import(import_data)
    test_eda(perform_eda)
    encoded_df = test_encoder_helper(encoder_helper, df)
    test_perform_feature_engineering(
        perform_feature_engineering, encoded_df)
    test_train_models(train_models)
    logging.info('All tests successufully passed.')
    print('All tests successufully passed.')
