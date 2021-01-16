import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import itertools
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import matplotlib.pylab as plt
from sklearn.ensemble import ExtraTreesRegressor
import json
from preprocessing import *
from xgb_hyperopt import *
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


def hyperopt_train_test(params):
    clf = ExtraTreesRegressor(**params)
    return -cross_val_score(clf, x, y_raw, scoring='neg_mean_squared_error').mean()


def f(para):
    mse = hyperopt_train_test(para)
    return {'loss': mse, 'status': STATUS_OK}


if __name__ == "__main__":
    X_raw = pd.read_csv('../training.csv')
    y_raw = X_raw.pop('claim_amount')

    models_counts = X_raw.vh_make_model.value_counts()
    df = pd.DataFrame(list(zip(models_counts.index, models_counts)), 
                      columns=['vh_make_model', 'model_make_count'])
    
    df['vh_make_model'] = df['vh_make_model'].astype(str)
    preprocessing = Preprocess_X_data(n_occurences_vh_make_model=50,
                                      drop_id=True, model_c=df)

    # Preprocessing
    preprocessing.fit(X_raw)
    x = preprocessing.transform(X_raw)

    # No more use of the column year
    #x = x.drop(columns='year')
    x = np.array(x)
    
    space4extree = {
        'n_estimators': hp.choice('n_estimators', range(20, 1000)),
        'max_depth': hp.choice('max_depth', range(2, 20)),
        'min_samples_split': hp.choice('min_samples_split', range(2, 10)),
        'min_samples_leaf': hp.choice('min_samples_leaf', range(1, 10)),
        'min_weight_fraction_leaf': hp.uniform('min_weight_fraction_leaf', 0, 0.5),
        'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2']),
        'ccp_alpha': hp.uniform('ccp_alpha', 1, 1000)
    }
    
    trials = Trials()
    best = fmin(f, space4extree, algo=tpe.suggest, max_evals=1000, trials=trials)
    print('best:')
    print(best)

    with open('extra_trees_best_params.json', 'w') as outfile:
        json.dump(best, outfile)
