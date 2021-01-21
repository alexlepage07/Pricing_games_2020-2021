import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import xgboost as xgb
import itertools
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from xgboost.sklearn import XGBClassifier
import matplotlib.pylab as plt

from preprocessing import *


def model_score(X_train, X_test, y_train, y_test, xgb_params):
    """Model training function: given training data (X_raw, y_raw), train this pricing model.

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
    # Convert de features dataframes into DMatrix so it can be use to train an 
    # XGBoost
    dmatrix_train = xgb.DMatrix(X_train)
    dmatrix_train.set_label(y_train)
    dmatrix_valid = xgb.DMatrix(X_test)
    dmatrix_valid.set_label(y_test)

    # Transform xgb_params as a list of every combinations of parameters that 
    # needs to be tried during the gridsearch.

    # Train the XGBoost model

    results_dict = {}
    model = xgb.train(
        xgb_params,
        dtrain=dmatrix_train,
        num_boost_round=4000,
        early_stopping_rounds=50,
        evals=[(dmatrix_train, "train"), (dmatrix_valid, "eval")],
        evals_result=results_dict
    )

    return float(list(results_dict["eval"].values())[0][-1])
