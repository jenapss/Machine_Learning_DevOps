# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

# Project Description

### This is the first out of four capstone projects for Machine Learning DevOps Engineer degree at Udacity.

The Customer Churn Prediction is about identifying credit card customers that are most likely to churn based on customer data with several features.

Raw view for customer dataset is below:


![Screenshot](https://github.com/jenapss/Machine_Learning_DevOps/blob/main/Customer%20Churn/readme_files/sample_dataset.png?raw=True)

## Requirements

In order to be able to run the project files, first install all the required libraries and dependencies by following command: ```pip install -r requirements.txt```

Description some of important packages: 

| Packages       |  Package's primary function    |
|----------------|--------------------------------|
|Pandas          | Tabular Data Manupilation      |
|sklearn         | Machine learning algorithms    |
|numpy           |Array & Vector data Manupilation|
|matplotlib      | Plotting                       |


## Running Files


### Model Training

 * For Explatory Data Analysis and Model training run ```churn_library_solution.py``` by this command: ``` python churn_library_solution.py```. This single python script does all EDA, data preprocessing and model training and all at once. Corresponding plots of EDA, data preprocessing and model training and testing metrics will be saved in ```/Customer Chunr/images/ ``` directory.

### Testing Driven Development

 * As it is important to follow software engineering best practices, testing script has been written to ensure that all functionality of ``` churn_library_solution.py``` works properly. To test previously mentioned script, separete script was created - ``` churn_libary_logging_and_test.py``` or ```pytest  churn_libary_logging_and_test.py``` (2nd should be run from terminal). The result of tests and logs is stored at ```/Customer Churn/logs/``` folder.
