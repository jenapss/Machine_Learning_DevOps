import os
import logging
import churn_library_solution as cls

IMG_PATH = './images/'


logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

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
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err


def test_eda(perform_eda,IMG_PATH):
	'''
	test perform eda function
	'''
	df = cls.import_data("./data/bank_data.csv")
	try:
		perform_eda(df) 
		assert os.path.exists(IMG_PATH + '/images/1.jpg') == True
		assert os.path.exists(IMG_PATH + '/images/2.jpg') == True
		assert os.path.exists(IMG_PATH + '/images/3.jpg') == True
		assert os.path.exists(IMG_PATH + '/images/4.jpg') == True
		assert os.path.exists(IMG_PATH + '/images/5.jpg') == True
		logging.info('Testing perform_data: SUCCESS')
	except AssertionError as err:
		logging.error("Testing perform_data: The plot have not been saved")
		raise err


def test_encoder_helper(encoder_helper):
	'''
	test encoder helper
	'''
	df = cls.import_data('./data/bank_csv.csv')
	category_lst = ['Gender', 'Income_Category', 'Marital_Status', 'Education_Level', 'Card_Category']

	try:
		encoder_helper(df, category_lst)
		assert (category_lst[0] + '_Churn' in df) == True
		assert (category_lst[1] + '_Churn' in df) == True
		assert (category_lst[2] + '_Churn' in df) == True
		assert (category_lst[3] + '_Churn' in df) == True
		assert (category_lst[4] + '_Churn' in df) == True
		logging.info('SUCCESS = Testing encoder_helper: Columns are created')
	except AssertionError: as err:
		logging.error("ERROR = Testing encoder_helper :   Columns are not created!")
		raise err


def test_perform_feature_engineering(perform_feature_engineering):
	'''
	test perform_feature_engineering
	'''


def test_train_models(train_models):
	'''
	test train_models
	'''



def main():
    '''
	main call function
	'''
    pass

if __name__ == "__main__":
	pass








