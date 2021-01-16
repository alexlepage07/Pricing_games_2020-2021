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

def fit_model(X_raw, y_raw):
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
    TRAINING_YEARS = [1,2,3]

    xgb_params = {
        "learning_rate" : [0.05],
        "n_estimators" : [500],
        # Definition of the model to train
        "objective": ["reg:tweedie"],
        "tweedie_variance_power" : [1.1],
        "booster" : ['gbtree'],
        "scale_pos_weight": [10],
        # Evaluation metric
        "eval_metric": ["rmse"],
        # Parameters for gbtree booster
        'gamma' : [0.1],
        'lambda':[100],
        "alpha": [0],
        "min_child_weight": [10],
        "max_depth": [8],
        "colsample_bytree": [0.8],
        "subsample": [0.8],
        'tree_method':['gpu_hist'],
        # Additionnal parameters for the training function
        "early_stopping_rounds": [50],
        "num_boost_round": [4000],
        "subsample" : [0.8],
        "colsample_bytree" : [0.8],
        "nthread" : [4]
    }

    models_counts = X_raw.vh_make_model.value_counts()
    df = pd.DataFrame(list(zip(models_counts.index, models_counts)), 
               columns =['vh_make_model', 'model_make_count'])
    
    df['vh_make_model']=df['vh_make_model'].astype(str)
    preprocessing = Preprocess_X_data(n_occurences_vh_make_model=50,
                                      drop_id=True, model_c =df)
    
    # Split the data in train and validation dataset according to the year
    x = X_raw.copy()


    # Preprocessing
    preprocessing.fit(x)
    x = preprocessing.transform(x)


    # No more use of the column year
    #x = x.drop(columns='year')
    x = np.array(x)
    
    x_train, x_valid, y_train, y_valid = train_test_split(x, y_raw, 
                                                          test_size=0.33,
                                                          shuffle = True,
                                                          random_state = 4000
                                                        )


    # Convert de features dataframes into DMatrix so it can be use to train an 
    # XGBoost
    dmatrix_train = xgb.DMatrix(x_train)
    dmatrix_train.set_label(y_train)
    dmatrix_valid = xgb.DMatrix(x_valid)
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
            dtrain = dmatrix_train,
            num_boost_round = params_dict["num_boost_round"],
            early_stopping_rounds = params_dict["early_stopping_rounds"],
            evals = [(dmatrix_train, "train"), (dmatrix_valid, "eval")],
            evals_result = results_dict
        )
        results_list.append({
            "eval": float(list(results_dict["eval"].values())[0][-1]),
            "train": float(list(results_dict["train"].values())[0][-1]),
            "params": params_dict
        })
        print(f"Trained model #{i + 1} out of {len(param_list)}")

    print(results_list)

    return model,results_list
