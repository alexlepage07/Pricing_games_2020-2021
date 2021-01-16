import numpy as np
import pandas as pd
import pickle
import json
from tqdm import tqdm
import xgboost as xgb
import itertools
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from xgboost.sklearn import XGBClassifier
import matplotlib.pylab as plt

from preprocessing import *
from xgb_hyperopt import *
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

def f(para):
    rmse = model_score(X_train,X_test, y_train,y_test,para)
    return {'loss': rmse, 'status': STATUS_OK}

if __name__ == "__main__":
    df = pd.read_csv('training.csv')
    X_raw = df.drop(columns=['claim_amount'])
    y_raw = df['claim_amount']
    
    models_counts = X_raw.vh_make_model.value_counts()
    df = pd.DataFrame(list(zip(models_counts.index, models_counts)), 
               columns =['vh_make_model', 'model_make_count'])
    
    df['vh_make_model']=df['vh_make_model'].astype(str)
    preprocessing = Preprocess_X_data(n_occurences_vh_make_model=50,
                                      drop_id=True, model_c =df)

    # Preprocessing
    preprocessing.fit(X_raw)
    x = preprocessing.transform(X_raw)


    # No more use of the column year
    #x = x.drop(columns='year')
    x = np.array(x)
    
    X_train, X_test, y_train, y_test = train_test_split(x, y_raw, 
                                                          test_size=0.33,
                                                          shuffle = True,
                                                          random_state = 4000
                                                        )

    xgb_params_space = {
        "learning_rate" : hp.uniform("learning_rate",0.001,0.2),
        "n_estimators" : hp.choice("n_estimators",range(1,1000)),
        # Definition of the model to train
        "objective": "reg:tweedie",
        "tweedie_variance_power" : hp.uniform("tweedie_variance_power",1,2),
        "booster" : 'gbtree',
        "scale_pos_weight": hp.choice("scale_pos_weight",range(1,1000)),
        # Evaluation metric
        "eval_metric": "rmse",
        # Parameters for gbtree booster
        'gamma' : hp.uniform("gamma",0,0.4),
        'lambda': hp.choice("lambda",range(1,1000)),
        "alpha": hp.choice("alpha",range(1,1000)),
        "min_child_weight": hp.choice("min_child_weight",range(1,12)),
        "max_depth": hp.choice("max_depth",range(1,10)),
        'tree_method': 'gpu_hist',
        # Additionnal parameters for the training function
        "colsample_bytree": hp.uniform("colsample_bytree",0.5,0.9),
        "subsample": hp.uniform("subsample",0.5,0.9),
    }
    
    trials = Trials()
    print('Start')
    best = fmin(f, xgb_params_space, algo=tpe.suggest, max_evals=100, trials=trials)
    print('best:')
    print(best)
    with open('xgb_best_params.json', 'w') as outfile:
        json.dump(best, outfile)

