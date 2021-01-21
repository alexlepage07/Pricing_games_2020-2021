"""In this module, we ask you to define your pricing model, in Python."""

import pickle
import numpy as np
import pandas as pd
import sklearn
import itertools
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate

from preprocessing import *

# TODO: import your modules here.
# Don't forget to add them to requirements.txt before submitting.

# Feel free to create any number of other functions, constants and classes to use
# in your model (e.g., a preprocessing function).


def fit_model(X_raw, y_raw):
	"""
	Model training function: given training data (X_raw, y_raw), train this pricing model.

	Parameters
	----------
	X_raw : Pandas dataframe, with the columns described in the data dictionary.
		Each row is a different contract. This data has not been processed.
	y_raw : a Numpy array, with the value of the claims, in the same order as contracts in X_raw.
		A one dimensional array, with values either 0 (most entries) or >0.

	Returns
	-------
	self: this instance of the fitted model. This can be anything, as long as it is compatible
		with your prediction methods.
	"""
	xgb_params = {
		# Definition of the model to train
		"objective": ["reg:tweedie"],
		"tweedie_variance_power": [1.13],
		"booster": ['gbtree'],
		# Evaluation metric
		"eval_metric": ["rmse"],
		# Parameters for gbtree booster
		'colsample_bytree': [0.85],
		'gamma': [0.2],
		'learning_rate': [0.18],
		'max_depth': [8],
		'min_child_weight': [2],
		'subsample': [0.6],
		'lambda': [1],
		'alpha': [0],
		# 'tree_method': ['gpu_hist'],
		# Additionnal parameters for the training function
		"early_stopping_rounds": [25],
		"num_boost_round": [4000]
	}

	# Preprocessing
	preprocessing = Preprocess_X_data(
		n_occurences_vh_make_model=50,
		drop_id=True
	)
	x = preprocessing.fit_transform(X_raw)
	x = x.drop(columns='id_policy', errors='ignore')

	# Split the data in train and validation dataset according to the year
	x_train, x_valid, y_train, y_valid = train_test_split(
		x, y_raw,
		test_size=0.10,
		shuffle=True,
		random_state=2020
	)

	# Convert de features dataframes into DMatrix so it can be use to train an
	# XGBoost
	dmatrix_train = xgb.DMatrix(x_train.values)
	dmatrix_train.set_label(y_train)
	dmatrix_valid = xgb.DMatrix(x_valid.values)
	dmatrix_valid.set_label(y_valid)

	# Transform xgb_params as a list of every combinations of parameters that
	# needs to be tried during the gridsearch.
	keys, values = zip(*xgb_params.items())
	param_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
	print(f"Number of combinations of parameters to try: {len(param_list)}")

	# Train the XGBoost model
	results_list = list()
	for i, params_dict in enumerate(param_list):
		results_dict = {}
		model = xgb.train(
			params_dict,
			dtrain=dmatrix_train,
			num_boost_round=params_dict["num_boost_round"],
			early_stopping_rounds=params_dict["early_stopping_rounds"],
			evals=[(dmatrix_train, "train"), (dmatrix_valid, "eval")],
			evals_result=results_dict
		)
		results_list.append({
			"eval": float(list(results_dict["eval"].values())[0][-1]),
			"train": float(list(results_dict["train"].values())[0][-1]),
			"params": params_dict
		})
		print(f"Trained model #{i + 1} out of {len(param_list)}")

	print(results_list)

	return model, preprocessing


def predict_expected_claim(model, X_raw, preprocessing):
	"""
	Model prediction function: predicts the expected claim based on the pricing model.

	This functions estimates the expected claim made by a contract (typically, as the product
	of the probability of having a claim multiplied by the expected cost of a claim if it occurs),
	for each contract in the dataset X_raw.

	This is the function used in the RMSE leaderboard, and hence the output should be as close
	as possible to the expected cost of a contract.

	Parameters
	----------
	model: a Python object that describes your model. This can be anything, as long
	as it is consistent with what `fit` outpurs.
	X_raw : Pandas dataframe, with the columns described in the data dictionary.
	Each row is a different contract. This data has not been processed.

	Returns
	-------
	avg_claims: a one-dimensional Numpy array of the same length as X_raw, with one
	expected claim per contract (in same order). These expected claims must be POSITIVE (>0).
	"""
	# Preprocessing
	x_clean = preprocessing.transform(X_raw)
	x_clean = xgb.DMatrix(x_clean.values)

	return model.predict(x_clean)


def predict_premium(model, X_raw, preprocessing):
	"""
	Model prediction function: predicts premiums based on the pricing model.

	This function outputs the prices that will be offered to the contracts in X_raw.
	premium will typically depend on the average claim predicted in
	predict_average_claim, and will add some pricing strategy on top.

	This is the function used in the average profit leaderboard. Prices output here will
	be used in competition with other models, so feel free to use a pricing strategy.

	Parameters
	----------
	model: a Python object that describes your model. This can be anything, as long
		as it is consistent with what `fit` outputs.

	X_raw : Pandas dataframe, with the columns described in the data dictionary.
		Each row is a different contract. This data has not been processed.

	preprocessing : A fitted preprocessing object return from the fit_model function.

	Returns
	-------
	prices: a one-dimensional Numpy array of the same length as X_raw, with one
		price per contract (in same order). These prices must be POSITIVE (>0).
	"""
	expected_claims = predict_expected_claim(model, X_raw, preprocessing)

	return expected_claims * 1.2


def save_model(model, preprocessing):
	"""
	Saves this trained model to a file.

	This is used to save the model after training, so that it can be used for prediction later.

	Do not touch this unless necessary (if you need specific features). If you do, do not
	forget to update the load_model method to be compatible.

	Parameters
	----------
	model: a Python object that describes your model. This can be anything, as long
		as it is consistent with what `fit` outpurs.

	preprocessing: The fitted preprocessing object return from the fit_model function.
	"""
	# Saving the trained model
	model.save_model('trained_model.json')

	# Saving the preprocessing settings
	with open('preprocessed.pickle', 'wb') as target_file:
		pickle.dump(preprocessing, target_file)


def load_model():
	"""
	Load a saved trained model from the file.

	This is called by the server to evaluate your submission on hidden data.
	Only modify this *if* you modified save_model.
	"""
	# Loading the trained model
	trained_model = xgb.Booster()  # init model
	trained_model.load_model('trained_model.json')  # load data

	# Loading the preprocessing settings
	with open('preprocessed.pickle', 'rb') as target:
		preprocessing = pickle.load(target)

	return trained_model, preprocessing
