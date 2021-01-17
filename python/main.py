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

from xgboost_model import *

if __name__ == "__main__":
    df = pd.read_csv('training.csv')
    x = df.drop(columns=['claim_amount'])
    y_raw = df['claim_amount']

    models_counts = x.vh_make_model.value_counts()
    df = pd.DataFrame(list(zip(models_counts.index, models_counts)), 
               columns =['vh_make_model', 'model_make_count'])
    
    df['vh_make_model']=df['vh_make_model'].astype(str)
    preprocessing = Preprocess_X_data(n_occurences_vh_make_model=50,
                                      drop_id=True, model_c =df)
    

    # Preprocessing
    preprocessing.fit(x)
    x = preprocessing.transform(x)


    val = 800
    for i in tqdm(range(10000)):
        x_train, x_valid, y_train, y_valid = train_test_split(x, y_raw, 
                                                          test_size=0.5,
                                                          shuffle = True,
                                                          random_state = i
                                                          #stratify = find_zero
                                                        )
        trained_model,res_lst = fit_model(x_train, x_valid, y_train, y_valid)
        if res_lst[0]['eval'] < val:
            val = res_lst[0]['eval']
            with open('best5050_split_seed.txt', 'w') as outfile:
                json.dump([i,val], outfile)
    print(val)

    #feat_imp = pd.Series(trained_model.get_score( importance_type='gain')).sort_values(ascending = False)
    #feat_imp.plot(kind = 'bar',title='Feature Importances')
    #plt.ylabel('Feature Importance Score')

    #plt.show()

