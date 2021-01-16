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

from xgboost_model import *

if __name__ == "__main__":
    df = pd.read_csv('training.csv')
    X_train = df.drop(columns=['claim_amount'])
    y_train = df['claim_amount']
    
    trained_model,res_lst = fit_model(X_train, y_train)

    feat_imp = pd.Series(trained_model.get_score( importance_type='gain')).sort_values(ascending = False)
    feat_imp.plot(kind = 'bar',title='Feature Importances')
    plt.ylabel('Feature Importance Score')

    plt.show()

