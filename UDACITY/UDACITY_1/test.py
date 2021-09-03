import pandas as pd
import matplotlib.pyplot as plt
from churn_library2 import import_data, perform_eda


df = import_data('/Users/jelaleddin/UDACITY/UDACITY_1/data/bank_data.csv')

print(df.head())

perform_eda(df)

