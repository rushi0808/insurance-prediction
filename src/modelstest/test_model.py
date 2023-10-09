import os
import sys
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from src.datacollection.collect_dataset import DataPathConfig
from src.exception import CustomException
from src.logger import logging
from src.utils import model_testing, save_object


class TestModelConfig:
    def __init__(self, trained_models: dict, test_data_path):
        self.test_data_path = test_data_path
        self.test_models_results = DataPathConfig().test_model_results
        self.best_model_path = DataPathConfig().best_model_path
        self.trained_models = trained_models

    def initiate_testing(self):
        try:
            start = datetime.now()
            logging.info("Reading test data.")
            test_df = pd.read_csv(self.test_data_path)

            logging.info("preparing X and y")
            X = np.array(test_df.iloc[:, :-1])
            y = np.array(test_df.iloc[:, -1])

            logging.info("model testing initiated.")
            best_model, test_results = model_testing(X, y, self.trained_models)
            logging.info("model testing finished found the best model.")

            logging.info("storing test results")
            test_results.to_csv(self.test_models_results, index=False, header=True)

            logging.info("storing best model")
            save_object(self.best_model_path, best_model)
            end = datetime.now()

            logging.info(f"time taken for model testing {end-start}")

            return (test_results, best_model)

        except Exception as e:
            raise CustomException(e, sys)
