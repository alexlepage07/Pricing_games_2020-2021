from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LinearRegression
import pandas as pd


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
            columns=['year', 'pol_no_claims_discount', 'vh_value'])
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

    def __init__(self, n_occurences=30):
        self.n_occ = n_occurences

    def fit(self, x_train):
        self.models_counts = x_train.vh_make_model.value_counts()
        self.models_to_group = self.models_counts[
            self.models_counts < self.n_occ].keys()
        return self

    def transform(self, x_raw):
        # Add a new feature according to the model count
        df_counts = pd.DataFrame(
            list(zip(self.models_counts.index, self.models_counts)),
            columns=['vh_make_model', 'vh_model_counts']
        )
        x_raw = x_raw.merge(
            right=df_counts,
            on='vh_make_model',
            how='left'
        )
        x_raw['vh_model_counts'] = x_raw['vh_model_counts'].fillna(value=1)

        # Compressing the column vh_make_model by grouping low frequency's models.
        mask_model_to_group = x_raw.vh_make_model.isin(self.models_to_group)
        x_raw.loc[mask_model_to_group, 'vh_make_model'] = 'other_models'

        return x_raw


class ImputeVhInformations:
    '''
    Imputation class for missing values
    '''
    def __init__(self):
        self.speed_mean = 0
        self.weight_mean = 0
        self.age_mean = 0
        self.vh_value_imputer = LinearRegression()

    def prefit(self, x_train):
        self.speed_mean = x_train['vh_speed'].mean()
        self.weight_mean = x_train['vh_weight'].mean()
        self.age_mean = x_train['vh_age'].mean()
        return self

    def pretransform(self, x_raw):
        x = x_raw.copy()

        # Adding missing indicators
        x['vh_age_NA'] = x['vh_age'].isnull()
        x['vh_value_NA'] = x['vh_value'].isnull()

        # Impute some of the missings
        x.loc[x.vh_speed.isnull(), 'vh_speed'] = self.speed_mean
        x.loc[x.vh_weight.isnull(), 'vh_weight'] = self.weight_mean
        x.loc[x.vh_age.isnull(), 'vh_age'] = self.age_mean

        return x

    def fit(self, x_train):
        # Starting to impute speed and weight according to the model mean
        self.prefit(x_train)
        x = self.pretransform(x_train)

        # Training a mixed linear regression to impute the vh_value
        variables = ('vh_fuel', 'vh_age', 'vh_type')
        vh_columns = x.columns.str.startswith(variables)
        rows_to_predict = x.vh_value.notnull()

        x_imputation = x.loc[rows_to_predict, vh_columns]
        x_imputation = x_imputation.drop(columns='vh_age_NA')

        self.vh_value_imputer.fit(x_imputation, x.vh_value[rows_to_predict])

        return self

    def transform(self, x_raw):
        # Imputing speed and weight according to the model mean
        x = self.pretransform(x_raw)

        # Predict the missing vh_values
        variables = ('vh_fuel', 'vh_age', 'vh_type')
        vh_columns = x.columns.str.startswith(variables)
        rows_to_predict = x.vh_value.isnull()
        x_to_predict = x.loc[rows_to_predict, vh_columns].drop(columns='vh_age_NA')
        x.loc[rows_to_predict, 'vh_value'] = self.vh_value_imputer.predict(x_to_predict)

        # For the other variables, we simply put zeros.
        x = x.fillna(0)

        return x


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

    def __init__(self, n_occurences_vh_make_model=30, drop_id=False):
        self.normalizer = NormalizeData()
        self.imputer = ImputeVhInformations()
        self.compress_models = Compress_vh_make_model(
            n_occurences=n_occurences_vh_make_model
        )
        self.cols_to_binarize = ['pol_payd', 'drv_sex1', 'drv_drv2']
        self.cols_to_one_hot_encode = [
            'pol_coverage', 'pol_usage', 'drv_sex2',
            'vh_make_model', 'vh_fuel', 'vh_type'
        ]
        self.drop_id = drop_id
        self.vh_value_mean = 0

    def add_new_features(self, x_raw):
        x = x_raw.copy()
        # Adding new features
        x.insert(
            loc=len(x.columns),
            column='pop_density',
            value=x.population / x.town_surface_area
        )
        x.insert(
            loc=len(x.columns),
            column='vh_speed_drv_age_ratio',
            value=x.vh_speed / x.drv_age1
        )
        x.insert(
            loc=len(x.columns),
            column='potential_force_impact',
            value=x.vh_speed * x.vh_weight
        )

        # Droping not necessay variables
        x = x.drop(columns='pol_pay_freq')
        if self.drop_id:
            x = x.drop(columns='id_policy')
        return x

    def fit(self, x_train):
        # Adding new features
        x_train = self.add_new_features(x_train)

        # Compressing the vh_make_model column
        self.compress_models.fit(x_train)
        x_train = self.compress_models.transform(x_train)

        # Normalization
        self.normalizer.fit(x_train)
        colnames = x_train.columns
        x_train = self.normalizer.transform(x_train)
        x_train = pd.DataFrame(x_train, columns=colnames)

        # One-Hot-Encode the categorical columns necessary to imput vh_value
        x_train = pd.get_dummies(
            data=x_train,
            prefix=['vh_type', 'vh_fuel'],
            columns=['vh_type', 'vh_fuel'],
            drop_first=True,
            dtype='int8'
        )

        # Impute missing vh_values
        self.imputer.fit(x_train)

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

        # Binarize columns with only two categories
        lb = LabelBinarizer()
        for col in self.cols_to_binarize:
            x_prep[col] = lb.fit_transform(x_prep[col])

        # One-Hot-Encode the other categorical columns
        x_prep = pd.get_dummies(
            data=x_prep,
            prefix=self.cols_to_one_hot_encode,
            columns=self.cols_to_one_hot_encode,
            drop_first=True,
            dtype='int8'
        )

        # Impute missing values
        x_prep = self.imputer.transform(x_prep)

        return x_prep

    def fit_transform(self, x_raw):
        return self.fit(x_raw).transform(x_raw)
