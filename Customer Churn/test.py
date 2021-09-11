
import os
import logging
import churn_library_solution as cls
import pandas as pd

IMG_PATH = 'Customer Churn/images'
MODEL_PATH = 'Customer Churn/models'

logging.basicConfig(
    filename='Customer Churn/logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


category_lst = [
    'Gender',
    'Income_Category',
    'Marital_Status',
    'Education_Level',
    'Card_Category']

'''df = pd.read_csv('/Users/jelaleddin/MLOps-Udacity-Projects/Customer Churn/data/bank_data.csv')
df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
gender_lst = []
gender_groups = df.groupby('Gender').mean()['Churn']

for val in df['Gender']:
    gender_lst.append(gender_groups.loc[val])

df['Gender_Churn'] = gender_lst    

print(df)'''




def test():
    try:
        df = cls.import_data('Customer Churn/data/bank_data.csv')
        category_lst = [
            'Gender',
            'Income_Category',
            'Marital_Status',
            'Education_Level',
            'Card_Category']
        updated_df = cls.encoder_helper(df, category_lst)
        print(type(updated_df))
        assert (category_lst[0] + '_Churn' in updated_df) == True
        assert (category_lst[1] + '_Churn' in updated_df) == True
        assert (category_lst[2] + '_Churn' in updated_df) == True
        assert (category_lst[3] + '_Churn' in updated_df) == True
        assert (category_lst[4] + '_Churn' in updated_df) == True
        logging.info('SUCCESS = Testing encoder_helper: Columns are created')
        print('hello')
    except AssertionError as err:
        logging.error("ERROR = Testing encoder_helper :   Columns are not created!")
        raise err

test()