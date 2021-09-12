
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import churn_library_solution as cls

CSV_DATA = 'Customer Churn/data/bank_data.csv'

def plot_report(y_true, y_pred, plot_name):
    plt.rc('figure', figsize=(7, 7))
    plt.text(0.01, 1.25, str(plot_name),{'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_true, y_pred)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('/Users/jelaleddin/MLOps-Udacity-Projects/Customer Churn/images/{}.png'.format(plot_name))
    plt.close()

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
    '''plt.rc('figure', figsize=(5, 5))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('/Users/jelaleddin/MLOps-Udacity-Projects/Customer Churn/images/last_result.png')
    plt.close()
    
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
    plt.savefig('/Users/jelaleddin/MLOps-Udacity-Projects/Customer Churn/images/last_result2.png')
    plt.close()'''

    plot_report(y_test, y_test_preds_rf, 'RandomForestTraining')
    plot_report(y_train, y_train_preds_rf, 'RandomForestTest')

    plot_report(y_train, y_train_preds_lr, 'LogRegTrain')
    plot_report(y_test, y_test_preds_lr, 'LogRegTest')


def func1(x_train, x_test, y_train):

    cv_randomforest = joblib.load('/Users/jelaleddin/MLOps-Udacity-Projects/Customer Churn/models/rfc_model.pkl')
    log_reg = joblib.load('/Users/jelaleddin/MLOps-Udacity-Projects/Customer Churn/models/logistic_model.pkl')

    y_train_preds_rf = cv_randomforest.predict(x_train)
    print(y_train_preds_rf)
    y_test_preds_rf = cv_randomforest.predict(x_test)
    y_train_preds_lr = log_reg.predict(x_train)
    y_test_preds_lr = log_reg.predict(x_test)

    return y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf 


def main():
    '''
    ML Pipeline for Customer Churn prediction project

    '''
    category_lst = [
        'Gender',
        'Income_Category',
        'Marital_Status',
        'Education_Level',
        'Card_Category']
    
    dt_frame = cls.import_data(CSV_DATA)
    cls.perform_eda(dt_frame)
    updated_df = cls.encoder_helper(dt_frame, category_lst)
    x_train, x_test, y_train, y_test = cls.perform_feature_engineering(
        updated_df)


    y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf = func1(
        x_train, x_test, y_train)
    #cls.plot_training_results(x_test, y_test)
    #cls.feature_importance_plot(x_test)
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)


if __name__ == '__main__':
    main()
