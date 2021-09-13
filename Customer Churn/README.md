# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

# Project Description

### This is the first out of four capstone projects for Machine Learning DevOps Engineer degree at Udacity.

The Customer Churn Prediction is about identifying credit card customers that are most likely to churn based on customer data with several features.

Raw view for customer dataset is below:


![Screenshot](https://github.com/jenapss/Machine_Learning_DevOps/blob/main/Customer%20Churn/readme_files/sample_dataset.png?raw=True)



## Running Files


### Model Training

1) For Explatory Data Analysis and Model training you should run ```churn_library_solution.py``` from your terminal by this command: ``` python churn_library_solution.py``` and corresponding training notifications will displayed on terminal. This single python script does all EDA, data preprocessing and model training and all at once. Corresponding plots of EDA, data preprocessing and model training and testing metrics will be save in ```/Customer Chunr/images/ ``` directory.

### Testing Driven Development

As it is important to follow software engineering best practices, I made testing script to ensure that all functionality of ``` churn_library_solution.py``` works properly. To test previously mentioned script I made separete script - ``` churn_libary_logging_and_test.py```. This file tests the behavior of all functions from ```churn_library_solution.py```. The result of tests is stored at ```/Customer Churn/logs/``` folder.
