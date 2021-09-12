import os
import logging
import churn_library_solution as cls

IMG_PATH = '/Users/jelaleddin/MLOps-Udacity-Projects/Customer Churn/images/'
MODEL_PATH = '/Users/jelaleddin/MLOps-Udacity-Projects/Customer Churn/models'

logging.basicConfig(
    filename='Customer Churn/logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import():
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''

	try:
		df = cls.import_data("/Users/jelaleddin/MLOps-Udacity-Projects/Customer Churn/data/bank_data.csv")
		logging.info("SUCCESS - Testing import_data")
	except FileNotFoundError as err:
		logging.error("Testing import_eda: The file wasn't found")
		raise err

	try:
		assert df.shape[0] > 0
	except AssertionError as err:
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err


def test_eda():
	'''
	test perform eda function
	'''
	df = cls.import_data("/Users/jelaleddin/MLOps-Udacity-Projects/Customer Churn/data/bank_data.csv")
	cls.perform_eda(df)
	assert os.path.exists(IMG_PATH + '1.jpg') == True
	assert os.path.exists(IMG_PATH + '2.jpg') == True
	assert os.path.exists(IMG_PATH + '3.jpg') == True
	assert os.path.exists(IMG_PATH + '4.jpg') == True
	assert os.path.exists(IMG_PATH + '5.jpg') == True
	
	


def test_encoder_helper():
	'''
	test encoder helper
	'''
	try:
		df = cls.import_data('/Users/jelaleddin/MLOps-Udacity-Projects/Customer Churn/data/bank_data.csv')
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
		logging.info('SUCCESS - Testing encoder_helper: Columns are created!')
	except AssertionError as err:
		logging.error("ERROR - Testing encoder_helper: Columns are not created!")
		raise err

def test_perform_feature_engineering():
	'''
	test perform_feature_engineering
	'''
	
	df = cls.import_data('/Users/jelaleddin/MLOps-Udacity-Projects/Customer Churn/data/bank_data.csv')
	category_lst = [
		'Gender',
		'Income_Category',
		'Marital_Status',
		'Education_Level',
		'Card_Category']
	updated_df = cls.encoder_helper(df, category_lst)
	X_train, X_test, y_train, y_test = cls.perform_feature_engineering(updated_df)
	try:
		assert X_train.shape[0] > 0
		assert X_train.shape[1] == 19 
		assert X_test.shape[0] > 0
		assert X_test.shape[1] == 19 
		assert y_train.shape[0] > 0
		assert y_test.shape[0] > 0
		logging.info("SUCCESS - Feature Engineering")
	except AssertionError as err:
		logging.error("ERROR - Feature Engineering")
		raise err
		



def test_train_models():
	'''
	test train_models
	'''
	try:
		assert os.path.exists(MODEL_PATH + '/rfc_model.pkl') == True
		assert os.path.exists(MODEL_PATH + '/logistic_model.pkl') == True
		logging.info('SUCCESS - MODELS SAVED')
	except AssertionError as err:
		logging.error("ERROR - MODELS NOT SAVED")
		raise err


 


if __name__ == "__main__":
	test_import()
	test_eda()
	test_encoder_helper()
	test_perform_feature_engineering()
	test_train_models()








