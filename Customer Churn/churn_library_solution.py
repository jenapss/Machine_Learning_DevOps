# library doc string

"""
The Library Docstring

"""

# import libraries
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
sns.set()
IMG_PATH = 'Customer Churn/images/eda/'
CSV_DATA = 'Customer Churn/data/bank_data.csv'


def import_data(pth):
    """
    returns dataframe for the csv found at pth

        input:
                pth: a path to the csv
        output:
                df: pandas dataframe

    """
    return pd.read_csv(pth)


def perform_eda(dt_frame):
    """perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    """

    dt_frame['Churn'] = dt_frame['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    fig1 = dt_frame['Churn'].hist().get_figure()
    fig1.savefig(IMG_PATH + 'churn_dist.jpg')
    plt.close()

    fig2 = dt_frame['Customer_Age'].hist().get_figure()
    fig2.savefig(IMG_PATH + 'customer_age.jpg')
    plt.close()
    fig3 = dt_frame.Marital_Status.value_counts(
        'normalize').plot(kind='bar').get_figure()
    fig3.savefig(IMG_PATH + 'martial_status_dist.jpg')
    plt.close()
    fig4 = sns.distplot(dt_frame['Total_Trans_Ct']).get_figure()
    fig4.savefig(IMG_PATH + 'total_transaction_dist.jpg')
    plt.close()
    fig5 = sns.heatmap(
        dt_frame.corr(),
        annot=False,
        cmap='Dark2_r',
        linewidths=2).get_figure()
    fig5.savefig(IMG_PATH + 'heatmap.jpg')
    plt.close()

def encoder_helper(dt_frame, category_lst):  # deleted one argument
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name
            [optional argument that could be used for naming variables or index y_data column]

    output:
            df: pandas dataframe with new columns for
    '''
    dt_frame['Churn'] = dt_frame['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    for category in category_lst:
        lst = []
        groups = dt_frame.groupby(category).mean()['Churn']
        for val in dt_frame[category]:
            lst.append(groups.loc[val])
        dt_frame['{}_Churn'.format(category)] = lst
    return dt_frame


def perform_feature_engineering(dt_frame):
    '''
    input:
            df: pandas dataframe
            response: string of response name
            [optional argument that could be used for naming variables or index y_data column]

    output:
              x_train: x_data training data
              x_test: x_data testing data
              y_train: y_data training data
              y_test: y_data testing data
    '''
    # drop unneeded columns in simpler way than manually listing all needed
    # columns.
    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
             'Income_Category_Churn', 'Card_Category_Churn']

    
    x_data = pd.DataFrame()
    x_data[keep_cols] = dt_frame[keep_cols]
    y_data = dt_frame['Churn']

    # train test split
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.3, random_state=42)

    return x_train, x_test, y_train, y_test


def model():
    """
    Model Initialization

    input: None

    output: cv_rfc, lrc

                cv_rfc - Cross validation of Random Forest Classifier model
                lrc - Logistic Regression model

    """

    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=1000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    return cv_rfc, lrc


def train_models(x_train, x_test, y_train):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: x_data training data
              x_test: x_data testing data
              y_train: y_data training data
              y_test: y_data testing data
    output:
              None
    '''
    # grid search
    cv_randomforest, log_reg = model()
    print('TRAINING STARTED...')
    cv_randomforest.fit(x_train, y_train)
    # logreg
    log_reg.fit(x_train, y_train)

    y_train_preds_rf = cv_randomforest.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_randomforest.best_estimator_.predict(x_test)
    y_train_preds_lr = log_reg.predict(x_train)
    y_test_preds_lr = log_reg.predict(x_test)
    print('TRAINING ENDED...')
    # save the best model
    joblib.dump(cv_randomforest.best_estimator_, 'Customer Churn/models/rfc_model.pkl')
    joblib.dump(log_reg, 'Customer Churn/models/logistic_model.pkl')

    return y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf


def plot_training_results(x_test, y_test):
    '''
    Plott training results

    '''
    log_reg = joblib.load('Customer Churn/models/logistic_model.pkl')
    cv_randomforest = joblib.load('Customer Churn/models/rfc_model.pkl')
    # ----- PLOTS ----- PLOTS ----
    # logistics regression results
    plt.figure(1,figsize=(20, 12))
    axis = plt.gca()
    
    plot_roc_curve(cv_randomforest,
                   x_test, y_test, ax=axis, alpha=0.8)
    plot_roc_curve(log_reg, x_test, y_test, ax=axis, alpha=0.8)
    plt.savefig('Customer Churn/images/results2.png')
    plt.close()

def feature_importance_plot(x_test):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_test: pandas dataframe of x_data values
            output_pth: path to store the figure

    output:
             None

    '''

    # LOAD THE MODEL
    cv_randomforest = joblib.load('Customer Churn/models/rfc_model.pkl')

    # END LOADING THE MODEL
    explainer = shap.TreeExplainer(cv_randomforest)
    shap_values = explainer.shap_values(x_test)
    shap.summary_plot(shap_values, x_test, plot_type='bar',show=False)

    # save plot
    plt.savefig('Customer Churn/images/results/shap.png')
    plt.close()

    # Calculate feature importances
    importances = cv_randomforest.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    # Rearrange feature names so they match the sorted feature names
    names = [x_test.columns[i] for i in indices]

    # Create a plot and save it
    plt.figure(figsize=(20, 5))
    plt.title('Feature Importances')
    plt.ylabel('Importance')
    # Add bars
    plt.bar(range(x_test.shape[1]), importances[indices])
    # Add feature names as x-axis labels
    plt.xticks(range(x_test.shape[1]), names, rotation=90)
    plt.savefig('Customer Churn/images/results/feat_importance.png')
    plt.close()


def plot_report(y_true, y_pred, plot_name):
    plt.rc('figure', figsize=(10, 10))
    plt.text(0.01, 1.25, str(plot_name),{'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_true, y_pred)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('Customer Churn/images/results/{}.png'.format(plot_name))
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
    plot_report(y_test, y_test_preds_rf, 'RandomForestTest')
    plot_report(y_train, y_train_preds_rf, 'RandomForestTrain')

    plot_report(y_train, y_train_preds_lr, 'LogRegTrain')
    plot_report(y_test, y_test_preds_lr, 'LogRegTest')



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
    
    dt_frame = import_data(CSV_DATA)
    perform_eda(dt_frame)
    updated_df = encoder_helper(dt_frame, category_lst)
    x_train, x_test, y_train, y_test = perform_feature_engineering(
        updated_df)


    y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf = train_models(
        x_train, x_test, y_train)
    plot_training_results(x_test, y_test)
    feature_importance_plot(x_test)
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)


if __name__ == '__main__':
    main()
