import os
import pandas as pd

df = pd.read_csv('Customer Churn/data/bank_data.csv')
ls = ['Income']

#category_lst = ['Gender', 'Income_Category', 'Marital_Status', 'Education_Level', 'Card_Category']

print((ls[0] + '_Category' in df) == True)