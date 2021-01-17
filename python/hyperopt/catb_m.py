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

def Catb_score(X_train,X_test, y_train,y_test,para):
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
    model = CatBoostRegressor(iterations=para['iterations'], 
                            depth=para['depth'], 
                            learning_rate=para['learning_rate'],
                            eval_metric=para['eval_metric'],
                            od_type=para['od_type'],
                            l2_leaf_reg=para['l2_leaf_reg'],
                            #random_strength=para['random_strength'],
                            bagging_temperature=para['bagging_temperature'],
                            grow_policy=para['grow_policy'],
                            loss_function='Tweedie:variance_power='+str(para['variance_power']))
    #train the model
    model.fit(train_pool,eval_set=test_pool)
    # make the prediction using the resulting model
    rmse = model.best_score_
    return np.mean(rmse['validation']['RMSE'])



def f(para):
    X_train, X_test, y_train, y_test = train_test_split(x, y_raw, 
                                                          test_size=0.33,
                                                          shuffle = True,
                                                          random_state = 2000
                                                        )
    rmse = Catb_score(X_train,X_test, y_train,y_test,para)
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

    # No more use of the column year
    #x = x.drop(columns='year')
    
    for c in x.columns:
        col_type = x[c].dtype
        if col_type == 'object' or col_type.name == 'category':
            x[c] = x[c].astype('category')

    xgb_params_space = {
                        'iterations' : hp.choice('iterations',range(10,1000)),
                        'variance_power': hp.uniform('variance_power',1,2), 
                        'depth': hp.choice('depth',range(2,12)), 
                        'learning_rate': hp.uniform('learning_rate',0.001,1),
                        'eval_metric':'RMSE',
                        'od_type': hp.choice('od_type',[None,'IncToDec','Iter']),
                        'l2_leaf_reg': hp.choice('l2_leaf_reg',range(1,1000)),
                        'bagging_temperature':hp.uniform('bagging_temperature',0,1),
                        'grow_policy':hp.choice('grow_policy',[None,'SymmetricTree','Depthwise','Lossguide']),
                        'rnd_st':hp.choice('rnd_st',range(1,20000))
    }
    
    trials = Trials()
    print('Start')
    best = fmin(f, xgb_params_space, algo=tpe.suggest, max_evals=100, trials=trials)
    print('best:')
    print(best)

    with open('catb_best_params.pickle', 'wb') as handle:
        pickle.dump(best, handle, protocol=pickle.HIGHEST_PROTOCOL)


