import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import xgboost as xgb
import itertools
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from xgboost.sklearn import XGBClassifier
import matplotlib.pylab as plt

from xgb_model import *
from catb_model import *
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor

xgb_params = {
    "learning_rate" : 0.2,
    "n_estimators" : 500,
    # Definition of the model to train
    "objective": "reg:tweedie",
    "tweedie_variance_power" : 1.1,
    "booster" : 'gbtree',
    "scale_pos_weight": 1,
    # Evaluation metric
    "eval_metric": "rmse",
    # Parameters for gbtree booster
    'gamma' : 0.1,
    'lambda':100,
    "alpha":0,
    "min_child_weight": 1,
    "max_depth": 12,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    'tree_method':'gpu_hist',
    # Additionnal parameters for the training function
    "early_stopping_rounds": 50,
    "num_boost_round": 4000
 }

catb_param = {
    'bagging_temperature': 0.98, 
    'depth': 12, 
    #'grow_policy': 2, 
    'iterations': 500, 
    'l2_leaf_reg': 271,
    'learning_rate': 0.2, 
    'loss_function': 'Tweedie:variance_power=1.2',
    'eval_metric': 'RMSE'
 }


lgb_param = {
        "learning_rate" : 0.1,
        'num_iterations' : 200,
        # Definition of the model to train
        "objective": "tweedie",
        "tweedie_variance_power" : 1.8,
        "boosting" : 'gbdt',
        # Evaluation metric
        "metric": "rmse",
        # Parameters for gbtree booster
        "num_leaf": 100,
        "max_depth": 12,
        'tree_learner': 'serial', #['serial','feature','data','voting']),
        # Additionnal parameters for the training function
        'scale_pos_weight': 10,
        'num_threads' : 6,
        #'device_type': 'gpu'
        'bagging_fraction': 0.98,
        'bagging_freq': 1
    }

rnd_f_param = {
    'n_estimators': 100,
    'max_depth': 8,
    'min_samples_split': 2,
    'min_samples_leaf': 2,
    'min_weight_fraction_leaf': 0,
    'max_features': 'auto',#['auto', 'sqrt', 'log2']),
    'ccp_alpha' : 1
    }

extra_tree_param = {
    'n_estimators': 10,
    'max_depth': 8,
    'min_samples_split': 2,
    'min_samples_leaf': 2,
    'min_weight_fraction_leaf': 0,
    'max_features': 'auto',#['auto', 'sqrt', 'log2']),
    'ccp_alpha' : 1
    }

xgb_final = {
    "learning_rate" : 0.2,
    "n_estimators" : 969,
    # Definition of the model to train
    "objective": "reg:tweedie",
    "tweedie_variance_power" : 1.0115179435833022,
    "booster" : 'gbtree',
    "scale_pos_weight": 120,
    # Evaluation metric
    "eval_metric": "rmse",
    # Parameters for gbtree booster
    'gamma' : 0.024066338722433883,
    'lambda':844,
    "alpha":625,
    "min_child_weight": 3,
    "max_depth": 7,
    'tree_method':'gpu_hist',
    # Additionnal parameters for the training function
    "early_stopping_rounds": 50,
    "num_boost_round": 4000,
    "subsample" : 0.8,
    'colsample_bytree': 0.6570706296458656
 }


if __name__ == "__main__":
    df = pd.read_csv('../../training.csv')
    X_raw = df.drop(columns=['claim_amount'])
    y = df['claim_amount']

    models_counts = X_raw.vh_make_model.value_counts()
    df = pd.DataFrame(list(zip(models_counts.index, models_counts)), 
               columns =['vh_make_model', 'model_make_count'])
    
    df['vh_make_model']=df['vh_make_model'].astype(str)
    preprocessing = Preprocess_X_data(n_occurences_vh_make_model=50,
                                      drop_id=True, model_c =df)

    # Preprocessing
    preprocessing.fit(X_raw)
    x = preprocessing.transform(X_raw)
    x_cat = preprocessing.transform(X_raw,one_hot_c = False)

    #change typpes for light_gbm
    for c in x_cat.columns:
        col_type = x_cat[c].dtype
        if col_type == 'object' or col_type.name == 'category':
            x_cat[c] = x_cat[c].astype('category')

    holdout_idx = x.sample( frac = 0.1,random_state=0).index

    kf = KFold(n_splits=9, random_state=None, shuffle=True)

    train_data = x.iloc[~x.index.isin(holdout_idx)]

    x_new = x.copy()
    x_new.insert(
            loc=len(x.columns),
            column='xgb_'+str(1)+'_preds',
            value = 0
            )
    x_new.insert(
            loc=len(x.columns),
            column='light_gbm_'+str(1)+'_preds',
            value = 0
            )
    x_new.insert(
            loc=len(x.columns),
            column='catb_'+str(1)+'_preds',
            value = 0
            )
    x_new.insert(
            loc=len(x.columns),
            column='randomForest_'+str(1)+'_preds',
            value = 0
            )
    x_new.insert(
            loc=len(x.columns),
            column='extra_tree_'+str(1)+'_preds',
            value = 0
            )

    i = 1
    split_idx_lst = list()
    
    for train_index, test_index in kf.split(train_data):
        split_idx_lst.append({
            'split nr': i,
            "train_idx": train_index,
            "test_idx": test_index
        })

        X_train, X_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        X_cat_train, X_cat_test = x_cat.iloc[train_index], x_cat.iloc[test_index]

        #train xgb 1
        xgb_m, res = train_xgb(X_train, X_test,y_train, y_test,xgb_params)
        x_new.loc[test_index,['xgb_'+str(1)+'_preds']] = predict_xgb(X_test,xgb_m)
        #train light gbm 1
        d_train = lgb.Dataset(X_cat_train, label=y_train,free_raw_data=False, feature_name='auto', categorical_feature=['pol_coverage','pol_pay_freq',
                                                                                        'pol_payd','pol_usage','drv_sex1',
                                                                                        'drv_drv2','vh_make_model','vh_fuel',
                                                                                        'vh_type','vh_age_NA','vh_value_NA'])
        lgb_m = lgb.train(lgb_param, d_train, 100)                                                                       
        x_new.loc[test_index,['light_gbm_'+str(1)+'_preds']] = lgb_m.predict(X_cat_test)
        #train catboost
        catb_m = Catb_train(X_cat_train, X_cat_test,y_train, y_test,catb_param)
        x_new.loc[test_index,['catb_'+str(1)+'_preds']] = predict_catb(X_cat_test,catb_m)
        #train rnd tree
        rnd_f_m = RandomForestRegressor(**rnd_f_param).fit(X_train,y_train)
        x_new.loc[test_index,['randomForest_'+str(1)+'_preds']] = rnd_f_m.predict(X_test)
        #train extra tree
        extra_tree_m = ExtraTreesRegressor(**extra_tree_param).fit(X_train,y_train)
        x_new.loc[test_index,['extra_tree_'+str(1)+'_preds']] = extra_tree_m.predict(X_test)


        lst_eval_rmse.append(float(list(res["eval"].values())[0][-1]))

        i=1+i
    


    final_train_set = x.iloc[~x.index.isin(holdout_idx)]
    final_cat_train_set = x_cat.iloc[~x.index.isin(holdout_idx)]

    final_test_set = x.iloc[holdout_idx]
    final_cat_test_set = x.iloc[holdout_idx]

    y_train = y.iloc[~x.index.isin(holdout_idx)]
    y_test = y.iloc[holdout_idx]
    #Level 1 models holdout prediction
    #train xgb 1
    xgb_m, res = train_xgb(final_train_set, final_test_set,y_train, y_test,xgb_params)
    x_new.loc[holdout_idx,['xgb_'+str(1)+'_preds']] = predict_xgb(final_test_set,xgb_m)
    #train light gbm 1
    d_train = lgb.Dataset(final_cat_train_set , label=y_train,free_raw_data=False, feature_name='auto', categorical_feature=['pol_coverage','pol_pay_freq',
                                                                                    'pol_payd','pol_usage','drv_sex1',
                                                                                    'drv_drv2','vh_make_model','vh_fuel',
                                                                                    'vh_type','vh_age_NA','vh_value_NA'])
                                                                            
    lgb_m = lgb.train(lgb_param, d_train, 100)                                                                       
    x_new.loc[holdout_idx,['light_gbm_'+str(1)+'_preds']] = lgb_m.predict(final_cat_test_set)
    #train catboost
    catb_m = Catb_train(final_cat_train_set,final_cat_test_set,y_train, y_test,catb_param)
    x_new.loc[holdout_idx,['catb_'+str(1)+'_preds']] = predict_catb(final_cat_test_set,catb_m)
    #train rnd tree
    rnd_f_m = RandomForestRegressor(**rnd_f_param).fit(final_train_set,y_train)
    x_new.loc[holdout_idx,['randomForest_'+str(1)+'_preds']] = rnd_f_m.predict(final_test_set)
    #train extra tree
    extra_tree_m = ExtraTreesRegressor(**extra_tree_param).fit(final_train_set,y_train)
    x_new.loc[holdout_idx,['extra_tree_'+str(1)+'_preds']] = extra_tree_m.predict(final_test_set)

    with open('stacking_new_df.pickle', 'wb') as handle:
        pickle.dump(x_new, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #Level 2 model
    l2_train = x_new.iloc[~x.index.isin(holdout_idx)]
    l2_valid = x_new.iloc[holdout_idx]
    xgb_m, res = train_xgb(l2_train, l2_valid,y_train, y_test,xgb_final)
    print(res)
    


