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
    X_train = pd.read_csv('training.csv')
    y_train = X_train.pop('claim_amount')

    trained_model, res_lst = fit_model(X_train, y_train)

    feature_importance = pd.Series(trained_model.get_score(importance_type='gain')).sort_values(ascending=False)
    feature_importance.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')

    plt.show()
