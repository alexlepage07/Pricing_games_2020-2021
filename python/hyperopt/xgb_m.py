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
from xgb_hyperopt import *
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

def f(para):
    X_train, X_test, y_train, y_test = train_test_split(x, y_raw, 
                                                          test_size=0.33,
                                                          shuffle = True,
                                                          random_state = para['rnd_st']
                                                        )
    rmse = model_score(X_train,X_test, y_train,y_test,para)
    return {'loss': rmse, 'status': STATUS_OK}

if __name__ == "__main__":
    df = pd.read_csv('../training.csv')
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
    


    xgb_params_space = {
        "learning_rate" : hp.uniform("learning_rate",0.01,0.2),
        "n_estimators" : 1000 ,#hp.choice("n_estimators",range(1,1000)),
        # Definition of the model to train
        "objective": "reg:tweedie",
        "tweedie_variance_power" : hp.uniform("tweedie_variance_power",1.01,1.99),
        "booster" : 'gbtree',
        "scale_pos_weight": 1 ,#hp.choice("scale_pos_weight",range(1,1000)),
        # Evaluation metric
        "eval_metric": "rmse",
        # Parameters for gbtree booster
        'gamma' : 0.1,#hp.uniform("gamma",0,0.4),
        'lambda': 0,#hp.choice("lambda",range(1,1000)),
        "alpha": 0,#hp.choice("alpha",range(1,1000)),
        "min_child_weight": hp.choice("min_child_weight",range(1,12)),
        "max_depth": 10,#hp.choice("max_depth",range(1,10)),
        'tree_method': 'gpu_hist',
        # Additionnal parameters for the training function
        #"colsample_bytree": 0.8,#hp.uniform("colsample_bytree",0.5,0.9),
        #"subsample": 0.8,#hp.uniform("subsample",0.5,0.9),
        'rnd_st':hp.choice('rnd_st',range(1,20000))
    }
    
    trials = Trials()
    print('Start')
    best = fmin(f, xgb_params_space, algo=tpe.suggest, max_evals=100, trials=trials)
    print('best:')
    print(best)

    with open('xgb_best_params_3.pickle', 'wb') as handle:
        pickle.dump(best, handle, protocol=pickle.HIGHEST_PROTOCOL)


