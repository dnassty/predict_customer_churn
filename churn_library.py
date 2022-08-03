'''
This module provide code to identify credit card customers that are most likely to churn.

Author: Dudko Anastasiia
Date: 03082022
'''


from constants import EDA_PLOTS_PATH, RESULTS_PATH, MODEL_PATH
import os
import logging
import joblib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, plot_roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split

import seaborn as sns
sns.set()


os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
        pth: a path to the csv
    output:
        df: pandas dataframe
    '''
    try:
        df = pd.read_csv(pth)
        df['Churn'] = df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
    except FileNotFoundError:
        print('File with {} not found'.format(pth))
    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
        df: pandas dataframe
    output:
        None
    '''
    cat_cols = df.dtypes[df.dtypes == "object"].index.drop('Attrition_Flag')

    for col in cat_cols:
        plt.figure(figsize=(20, 10))
        df[col].hist()
        plt.savefig(os.path.join(EDA_PLOTS_PATH, col + '_distribution.png'))

    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(os.path.join(EDA_PLOTS_PATH, 'heatmap.png'))
    logging.info('SUCESS: EDA performed and images saved.')


def encoder_helper(df, category_lst):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features

    output:
            df: pandas dataframe with new columns for
    '''
    for category in category_lst:
        category_proprotion_lst = []
        category_groups = df.groupby(category).mean()['Churn']

        for val in df[category]:
            category_proprotion_lst.append(category_groups.loc[val])

        df[category + '_Churn'] = category_proprotion_lst
    return df


def perform_feature_engineering(df):
    '''
    input:
              df: pandas dataframe

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    cat_columns = df.dtypes[df.dtypes == "object"].index.drop('Attrition_Flag')
    df = encoder_helper(df, cat_columns)

    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    y = df['Churn']
    X = pd.DataFrame()

    X[keep_cols] = df[keep_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(os.path.join(RESULTS_PATH, '/Random_Forest_Train_Test.png'))

    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(
        os.path.join(
            RESULTS_PATH,
            'Logistic_Regresion_Train_Test.png'))


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth)


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    rfc = RandomForestClassifier(random_state=42)

    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    classification_report(y_test, y_test_preds_rf)
    classification_report(y_train, y_train_preds_rf)
    feature_importance_plot(
        cv_rfc.best_estimator_,
        X_train,
        os.path.join(
            RESULTS_PATH,
            'Random_Forest_feature_importance.jpg'))

    rfc_disp = plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test)
    rfc_disp.plot().figure_.savefig(
        os.path.join(
            RESULTS_PATH,
            'Random_Forest_roc_curve.png'))

    classification_report(y_test, y_test_preds_lr)
    classification_report(y_train, y_train_preds_lr)
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    lrc_plot.plot().figure_.savefig(
        os.path.join(
            RESULTS_PATH,
            'Logistic_Regresion_roc_curve.png'))

    joblib.dump(
        cv_rfc.best_estimator_,
        os.path.join(
            MODEL_PATH,
            'rfc_model.pkl'))
    joblib.dump(lrc, os.path.join(MODEL_PATH, 'logistic_model.pkl'))


if __name__ == '__main__':
    data = import_data('./data/bank_data.csv')
    perform_eda(data)
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(data)
    train_models(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)
