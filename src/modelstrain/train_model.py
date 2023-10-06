import os
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.datacollection.collect_dataset import DataPathConfig
from src.exception import CustomException
from src.logger import logging
from src.utils import model_trainer


class ModelConfig:
    def __init__(self, train_data_path: str):
        self.train_models_results = DataPathConfig().train_model_results
        self.train_data_path = train_data_path

    def initiat_model_config(self):
        try:
            logging.info("Preparing data for training.")

            train_df = pd.read_csv(self.train_data_path)

            X = np.array(train_df.iloc[:, :-1])
            y = np.array(train_df.iloc[:, -1])

            models = {
                "LinearRigression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "SVR": SVR(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "RandomForestRegressor": RandomForestRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoostRegressor": CatBoostRegressor(verbose=False),
            }

            # params = {
            #     "LinearRigression": {},
            #     "Ridge": {
            #         "solver": [
            #             "auto",
            #             "svd",
            #             "cholesky",
            #             "lsqr",
            #             "sparse_cg",
            #             "sag",
            #             "saga",
            #             "lbfgs",
            #         ],
            #     },
            #     "Lasso": {"selection": ["cyclic", "random"]},
            #     "SVR": {"kernel": ["linear", "poly", "rbf", "sigmoid"]},
            #     "DecisionTreeRegressor": {
            #         "criterion": [
            #             "squared_error",
            #             "friedman_mse",
            #             "absolute_error",
            #             "poisson",
            #         ],
            #         "max_depth": range(10, 101, 10),
            #     },
            #     "RandomForestRegressor": {
            #         "n_estimators": range(10, 101, 10),
            #         "criterion": [
            #             "squared_error",
            #             "friedman_mse",
            #             "absolute_error",
            #             "poisson",
            #         ],
            #         "max_depth": range(10, 101, 10),
            #     },
            #     "AdaBoostRegressor": {"loss": ["linear", "square", "exponential"]},
            #     "XGBRegressor": {},
            #     "CatBoostRegressor": {},
            # }

            trained_models, train_results = model_trainer(X, y, models)
            logging.info("Models are trained.")

            logging.info("Storing train model results")
            train_results.to_csv(self.train_models_results, index=False, header=True)

            return trained_models

        except Exception as e:
            raise CustomException(e, sys)
