# import libraries

# Import necessary libraries for data
import os
import tarfile
import urllib
from cgi import test
from urllib.parse import urlparse
from zlib import crc32

import joblib

# To plot pretty figures
#%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedShuffleSplit,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

mpl.rc("axes", labelsize=14)
mpl.rc("xtick", labelsize=12)
mpl.rc("ytick", labelsize=12)


#
# Data link
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/data/data.tgz"

# data fetch from the link
def fetch_data_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# data from local directory
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# custom transformers
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, train_data, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
        self.train_data = train_data

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        col_names = "total_rooms", "total_bedrooms", "population", "households"
        rooms_ix, bedrooms_ix, population_ix, households_ix = [
            self.train_data.columns.get_loc(c) for c in col_names
        ]
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[
                X, rooms_per_household, population_per_household, bedrooms_per_room
            ]

        else:
            return np.c_[X, rooms_per_household, population_per_household]


# top features
def indices_of_top_k(arr, k):
    return np.sort(np.argpartition(np.array(arr), -k)[-k:])


# Feature Engineering: top feature selection
class TopFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importances, k):
        self.feature_importances = feature_importances
        self.k = k

    def fit(self, X, y=None):
        self.feature_indices_ = indices_of_top_k(self.feature_importances, self.k)
        return self

    def transform(self, X):
        return X[:, self.feature_indices_]


# data_cleaning
def clean_housing_data(inc_col="median_income"):
    data = load_housing_data()

    # split data based on income category
    data["income_cat"] = pd.cut(
        data[inc_col], bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf], labels=[1, 2, 3, 4, 5]
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(data, data["income_cat"]):
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index]

    # drop the income_Category
    return strat_train_set, strat_test_set


# preprocessing
def preprocess_data():
    train_data, test_data = clean_housing_data()
    for set_ in (train_data, test_data):
        set_.drop("income_cat", axis=1, inplace=True)

    # train_data_copy = train_data.copy()

    # dependent feature
    x_features = train_data.drop("median_house_value", axis=1)
    y_feature = train_data["median_house_value"].copy()

    return x_features, y_feature


# differentiating dtypes of columns of data
def data_category_list():
    ind_feat = preprocess_data()[0]
    num_attribs = list(ind_feat.select_dtypes(include=["float64"]).columns)
    cat_attribs = list(ind_feat.select_dtypes(include=["object"]).columns)
    return num_attribs, cat_attribs


# Preprocessing Pipeline with numerical columns
num_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        (
            "attribs_adder",
            CombinedAttributesAdder(preprocess_data()[0], add_bedrooms_per_room=True),
        ),
        ("std_scaler", StandardScaler()),
    ]
)

# full pipeline to Preprocess the data
full_pipeline = ColumnTransformer(
    [
        ("num", num_pipeline, data_category_list()[0]),
        ("cat", OneHotEncoder(), data_category_list()[1]),
    ]
)

# feature engineering
def feature_enginnering():
    # Get the feature_importances scores Using Grid Search
    x_features, y_feature = preprocess_data()
    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
    ]

    forest_reg = RandomForestRegressor(random_state=42)
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    grid_search = GridSearchCV(
        forest_reg,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    grid_search.fit(full_pipeline.fit_transform(x_features), y_feature)
    feature_importances = grid_search.best_estimator_.feature_importances_
    return grid_search, feature_importances


# feature selection
def input_top_features():
    k = input("Enter the top features for selection:")
    return int(k)


# preprocessing,feature selection, training pipeline
prepare_select_and_predict_pipeline = Pipeline(
    [
        ("preparation", full_pipeline),
        ("feature_selection", TopFeatureSelector(feature_enginnering()[1], 5),),
        ("random_forest_reg", RandomForestRegressor(),),
    ]
)


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


# train_data:
def train_model():
    x_train, y_train = preprocess_data()
    param_grid = [
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
    ]

    prepare_select_and_predict_pipeline.fit(x_train, y_train)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


# test results
def test_results():
    # test some data
    test_set = clean_housing_data()[1]
    x_test_features = test_set.drop("median_house_value", axis=1)
    y_test_feature = test_set["median_house_value"].copy()

    some_data = x_test_features.iloc[:4]
    some_labels = y_test_feature.iloc[:4]

    predicted_labels = prepare_select_and_predict_pipeline.predict(some_data)
    print("Predictions:\t", prepare_select_and_predict_pipeline.predict(some_data))
    print("Labels:\t\t", list(some_labels))

    mse, mae, r2 = eval_metrics(some_labels, predicted_labels)

    rmse = np.sqrt(mse)

    return rmse, mae, r2


# Create nested runs
with mlflow.start_run(run_name="PARENT_RUN") as parent_run:
    mlflow.log_param("parent", "yes")

    # nested child 1
    with mlflow.start_run(run_name="CHILD_RUN1", nested=True) as child_run1:
        x_feat, y_feat = preprocess_data()
        mlflow.log_param("Data Processing", "yes")

    # nested child 2
    with mlflow.start_run(run_name="CHILD_RUN2", nested=True) as child_run2:
        grid_search = feature_enginnering()[0]
        # print the best features:
        cvres = grid_search.cv_results_
        for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
            print(np.sqrt(-mean_score), params)
        mlflow.log_param("Feature Engineering", "yes")

    # nested child 3
    with mlflow.start_run(run_name="CHILD_RUN3", nested=True) as child_run3:
        # train the model

        train_model()

        rmse, mae, r2 = test_results()
        # Log parameter, metrics, and model to MLflow
        mlflow.log_param("Data Modelling", "yes")
        mlflow.log_param("best_parameters", grid_search.best_params_)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        csv_path = os.path.join(HOUSING_PATH, "housing.csv")
        mlflow.log_artifact(csv_path)

        print("Save to: {}".format(mlflow.get_artifact_uri()))
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        mlflow.sklearn.log_model(prepare_select_and_predict_pipeline, "model")

print("parent run_id: {}".format(parent_run.info.run_id))
print("child run_id : {}".format(child_run1.info.run_id))
print("child run_id : {}".format(child_run2.info.run_id))
print("child run_id : {}".format(child_run3.info.run_id))
print("--")

# Search all child runs with a parent id
query = "tags.mlflow.parentRunId = '{}'".format(parent_run.info.run_id)
results = mlflow.search_runs(filter_string=query)
print(results)

