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

xgb_params = {
    "learning_rate" : 0.05,
    "n_estimators" : 500,
    # Definition of the model to train
    "objective": "reg:tweedie",
    "tweedie_variance_power" : 1.1,
    "booster" : 'gbtree',
    "scale_pos_weight": 10,
    # Evaluation metric
    "eval_metric": "rmse",
    # Parameters for gbtree booster
    'gamma' : 0.1,
    'lambda':100,
    "alpha":0,
    "min_child_weight": 10,
    "max_depth": 8,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    'tree_method':'gpu_hist',
    # Additionnal parameters for the training function
    "early_stopping_rounds": 50,
    "num_boost_round": 4000,
    "subsample" : 0.8,
    "colsample_bytree" : 0.8
 }

if __name__ == "__main__":
    df = pd.read_csv('training.csv')
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
    for i in range(10):
        kf = KFold(n_splits=10, random_state=None, shuffle=True)
        x_new = x.copy()
        x_new.insert(
                loc=len(x.columns),
                column='xgb_'+str(i+1)+'_preds',
                value = 0
                )
        lst_eval_rmse = list()
        for train_index, test_index in kf.split(x):
            X_train, X_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            xgb_m, res = train_xgb(X_train, X_test,y_train, y_test,xgb_params)
            x_new.loc[test_index,['xgb_'+str(i+1)+'_preds']] = predict_xgb(x.iloc[test_index],xgb_m)
            lst_eval_rmse.append(float(list(res["eval"].values())[0][-1]))
        
        x = x_new
        print(x)
        print(np.mean(lst_eval_rmse))


    


