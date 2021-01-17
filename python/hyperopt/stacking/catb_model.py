import numpy as np
import pandas as pd
import pickle

from tqdm import tqdm
import xgboost as xgb
import itertools
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from catboost import Pool, CatBoostRegressor
import matplotlib.pylab as plt

from preprocessing import *

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

def Catb_train(X_train,X_test, y_train,y_test,para):
    # initialize Pool
    train_pool = Pool(X_train, 
                    y_train, 
                    feature_names=list(X_train.keys()), 
                    cat_features=['pol_coverage','pol_pay_freq',
                    'pol_payd','pol_usage','drv_sex1',
                    'drv_drv2','drv_sex2','vh_make_model','vh_fuel',
                    'vh_type','vh_age_NA','vh_value_NA'])
    test_pool = Pool(X_test, 
                    y_test, 
                    feature_names=list(X_test.keys()), 
                    cat_features=['pol_coverage','pol_pay_freq',
                    'pol_payd','pol_usage','drv_sex1',
                    'drv_drv2','drv_sex2','vh_make_model','vh_fuel',
                    'vh_type','vh_age_NA','vh_value_NA'])

    # specify the training parameters 
    model = CatBoostRegressor(para)
    #train the model
    model.fit(train_pool,eval_set=test_pool)
    # make the prediction using the resulting model
    return model



def predict_catb(X, model):
    X_pool = Pool(X, 
                    feature_names=list(X.keys()), 
                    cat_features=['pol_coverage','pol_pay_freq',
                    'pol_payd','pol_usage','drv_sex1',
                    'drv_drv2','drv_sex2','vh_make_model','vh_fuel',
                    'vh_type','vh_age_NA','vh_value_NA'])
    return model.predict(X_pool)



