from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelBinarizer
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

class NormalizeData:
    '''
    Class used to normalize a dataset according to a standard normal 
    distribution.

    Methods
    -------
    fit : Use the training dataset to calculate the mean and standard deviation
        used for the normalisation of new data.

    transform : Use the parameters calculated in the fit method to normalize 
        new data.
    '''
    def __init__(self):
        self.x_means = 0
        self.x_std = 0

    def fit(self, x_train):
        x_float = x_train.select_dtypes(include=['float', 'int']).drop(
            columns=[ 'pol_no_claims_discount'])
        self.x_means = x_float.mean()
        self.x_std = x_float.std()
        return self

    def transform(self, x_raw):
        for idx in x_raw:
            if idx in self.x_means.index:
                x_raw[idx] = (x_raw[idx] - self.x_means[idx]) / self.x_std[idx]
        return x_raw

class Compress_vh_make_model:
    '''
    Class used to group the labels with low frequency from the feature 
    vh_make_model.

    Methods
    -------
    fit : Use the training dataset to calculate the mean and standard deviation
        used for the normalisation of new data.

    transform : Use the parameters calculated in the fit method to normalize 
        new data.
    '''
    def __init__(self, n_occurences=30,models_c = None):
        self.n_occ = n_occurences
        self.models_c = models_c

    def fit(self, x_train):
        models_counts = x_train.vh_make_model.value_counts()
        self.models_to_group = models_counts[models_counts < self.n_occ].keys()
        return self
    
    def transform(self, x_raw):
        mask_model_to_group = x_raw.vh_make_model.isin(self.models_to_group)
        x_raw.loc[mask_model_to_group, 'vh_make_model'] = 'other_models'
        x_raw = x_raw.merge(self.models_c,on = 'vh_make_model', how ='left')
        x_raw['vh_make_model'] = x_raw['vh_make_model'].fillna(value=1)
        return x_raw


class Preprocess_X_data:
    """
    Class to preprocess the features of the dataset

    Methods
    -------
    add_new_features : Method to include new features

    impute_missing_values : Method to deal with missing values

    fit : Use the training data set to specify the parameters of the 
        prepocessing.

    transform : Use the parameters from the fit method to preprocess new data.

    """
    def __init__(self, n_occurences_vh_make_model=30, drop_id=False, model_c = None):
        self.normalizer = NormalizeData()
        self.compress_models = Compress_vh_make_model(
            n_occurences=n_occurences_vh_make_model,models_c = model_c)
        self.cols_to_binarize = ['pol_payd', 'drv_sex1', 'drv_drv2']
        self.cols_to_one_hot_encode = [
            'pol_coverage','pol_pay_freq', 'pol_usage', 'drv_sex2', 
            'vh_make_model', 'vh_fuel', 'vh_type'
            ]
        self.drop_id = drop_id

    def add_new_features(self, x_raw):
        x = x_raw.copy()
        # Adding new features
        x.insert(
            loc=len(x.columns),
            column='pop_density',
            value = x.population / x.town_surface_area
            )
        x.insert(
            loc=len(x.columns),
            column='vh_speed_drv_age_ratio',
            value = x.vh_speed / x.drv_age1
            )
        x.insert(
            loc=len(x.columns),
            column='potential_force_impact',
            value = x.vh_speed * x.vh_weight
            )

        # Droping not necessay variables
        #x = x.drop(columns='pol_pay_freq')
        if self.drop_id:
            x = x.drop(columns='id_policy')
        return x

    def impute_missing_values(self, x_raw):
        x = x_raw.copy()
        # Adding missing indicators
        x['vh_age_NA'] = x['vh_age'].isnull()
        x['vh_value_NA'] = x['vh_value'].isnull()

        # Impute missing values
        x = x.fillna(0)
        return x

    def fit(self, x_train):
        # Adding new features
        x_train = self.add_new_features(x_train)

        # Compressing the vh_make_model column
        self.compress_models.fit(x_train)
        x_train = self.compress_models.transform(x_train)
        # Normalization
        self.normalizer.fit(x_train)

        return self

    def transform(self, x_raw):
        # Adding new features
        x_prep = self.add_new_features(x_raw)

        # Compressing the vh_make_model column
        x_prep = self.compress_models.transform(x_prep)

        # Normalization
        colnames = x_prep.columns
        x_prep = self.normalizer.transform(x_prep)
        x_prep = pd.DataFrame(x_prep, columns=colnames)

        # Impute missing values
        x_prep = self.impute_missing_values(x_prep)

        # Binarize columns with only two categories
        lb = LabelBinarizer()
        for col in self.cols_to_binarize:
            x_prep[col] = lb.fit_transform(x_prep[col])

        # One-Hot-Encode the other categorical columns
        x_prep = pd.get_dummies(
            data=x_prep,
            prefix = self.cols_to_one_hot_encode,
            columns = self.cols_to_one_hot_encode,
            drop_first=True,
            dtype='int8'
            )

        return x_prep 