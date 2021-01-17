import lightgbm as lgb

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
import scipy

from preprocessing import *

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

def model_score(lgb_params):
    loss = lgb.cv(lgb_params, train_data, 10, metrics='rmse', nfold=6, stratified=False)
    return np.mean(loss['rmse-mean'])

def f(para):
    rmse = model_score(para)
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
    x = preprocessing.transform(X_raw, one_hot_c = False)


    #msk = np.random.rand(len(x.year)) < 0.8
    #x.insert(0,'train_set', msk)

    for c in x.columns:
        col_type = x[c].dtype
        if col_type == 'object' or col_type.name == 'category':
            x[c] = x[c].astype('category')
    # No more use of the column year
    #x = x.drop(columns='year')
    train_data = lgb.Dataset(x, label=y_raw,free_raw_data=False, feature_name='auto', categorical_feature=['pol_coverage','pol_pay_freq',
                                                                                        'pol_payd','pol_usage','drv_sex1',
                                                                                        'drv_drv2','vh_make_model','vh_fuel',
                                                                                        'vh_type','vh_age_NA','vh_value_NA'])
    
    #test_data = train_data.create_valid('test.svm')

    lgb_params_space = {
        "learning_rate" : hp.uniform("learning_rate",0.001,0.2),
        'num_iterations' : hp.choice('num_iterations',range(20,100)),
        # Definition of the model to train
        "objective": "tweedie",
        "tweedie_variance_power" : hp.uniform("tweedie_variance_power",1,2),
        "boosting" : 'gbdt',
        # Evaluation metric
        "metric": "rmse",
        # Parameters for gbtree booster
        "num_leaf": hp.choice("num_leaf",range(10,500)),
        "max_depth": hp.choice("max_depth",range(1,20)),
        'tree_learner': hp.choice('tree_learner',['serial','feature','data','voting']),
        # Additionnal parameters for the training function
        'scale_pos_weight': hp.choice("scale_pos_weight",range(1,150)),
        'num_threads' : 6,
        #'device_type': 'gpu'
        'bagging_fraction': hp.uniform('bagging_fraction',0,1),
        'bagging_freq': hp.choice('bagging_freq',range(1,10))
    }


    trials = Trials()
    print('Start')
    best = fmin(f, lgb_params_space, algo=tpe.suggest, max_evals=100, trials=trials)
    print('best:')
    print(best)
    with open('lgb_best_params.pickle', 'w') as outfile:
        pickle.dump(best, outfile)

