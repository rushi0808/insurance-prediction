import os
import pickle
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, object):
    logging.info("stroring model")
    with open(file_path, "wb") as f:
        pickle.dump(object, f)


def load_object(file_path):
    logging.info("Loading model")
    with open(file_path, "rb") as f:
        object = pickle.load(f)
        return object


def model_trainer(X, y, models: dict, params: dict = None):
    try:
        start = datetime.now()
        logging.info("Model training started")
        model_name = []
        model_r2 = []
        model_mse = []
        model_rmse = []
        model_mape = []
        model_acc = []

        trained_models = {}

        for each in models.keys():
            model = models[each]
            if params != None:
                param = params[each]
                grid = GridSearchCV(estimator=model, param_grid=param, cv=5, n_jobs=-1)
                grid.fit(X, y)
                model.set_params(**grid.best_params_)

            model.fit(X, y)

            pred_tr = model.predict(X)

            r2 = r2_score(y, pred_tr)
            mse = mean_squared_error(y, pred_tr)
            rmse = mean_squared_error(y, pred_tr, squared=False)
            mape = mean_absolute_percentage_error(y, pred_tr)
            acc = round(100 - (mape * 100), 2)

            trained_models[each] = model

            model_name.append(each)
            model_r2.append(r2)
            model_mse.append(mse)
            model_rmse.append(rmse)
            model_mape.append(mape)
            model_acc.append(acc)

        train_results = pd.DataFrame()
        train_results["Model_Name"] = model_name
        train_results["Model_accuray"] = model_acc
        train_results["Model_r2_score"] = model_r2
        train_results["Model_mse"] = model_mse
        train_results["Model_rmse"] = model_rmse
        train_results["Model_mape"] = model_mape

        logging.info("Model training finished returning results.")
        end = datetime.now()
        logging.info(f"Time taken for model training {end-start}")

        return (trained_models, train_results)

    except Exception as e:
        raise CustomException(e, sys)


def model_testing(X, y, trained_models: dict):
    try:
        logging.info("model testing started")
        test_model_name = []
        test_r2 = []
        test_acc = []
        test_mse = []
        test_rmse = []
        test_mape = []

        for each in trained_models.keys():
            model = trained_models[each]
            pred_ts = model.predict(X)
            r2 = r2_score(y, pred_ts)
            mse = mean_squared_error(y, pred_ts)
            rmse = mean_squared_error(y, pred_ts, squared=False)
            mape = mean_absolute_percentage_error(y, pred_ts)
            acc = round(100 - (mape * 100), 2)

            test_model_name.append(each)
            test_r2.append(r2)
            test_acc.append(acc)
            test_mse.append(mse)
            test_rmse.append(rmse)
            test_mape.append(mape)

        test_results = pd.DataFrame()
        test_results["Model_name"] = test_model_name
        test_results["r2_score"] = test_r2
        test_results["accuracy"] = test_acc
        test_results["mse"] = test_mse
        test_results["rmse"] = test_rmse
        test_results["mape"] = test_mape

        best_model_name = list(
            test_results.sort_values("accuracy", ascending=False)["Model_name"]
        )[0]

        best_model = trained_models[best_model_name]
        logging.info("Models testing finished.")

        return (best_model, test_results)

    except Exception as e:
        raise CustomException(e, sys)


def make_prediction(customer_input: dict, preprocessor_path, model_path):
    customer_input_df = pd.DataFrame(customer_input, index=[0])
    logging.info("Converted customer data to dataframe")

    logging.info("Loading preprocessor and model")
    preprocessor = load_object(preprocessor_path)
    model = load_object(model_path)

    logging.info("transforming data with preprocessor")
    customer_input_processed = preprocessor.transform(customer_input_df)
    customer_prediction = model.predict(customer_input_processed)
    logging.info("Customer prediction completed")

    return np.exp(customer_prediction)
