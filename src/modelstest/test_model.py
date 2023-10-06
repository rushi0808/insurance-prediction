import os
import sys
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import model_testing, save_object


@dataclass
class TestResultConfig:
    test_data_path: str = os.path.join("data", "test.csv")
    test_result_path: str = os.path.join("data", "test_results.csv")
    model_path: str = os.path.join("models", "model.pkl")


class TestModelConfig:
    def __init__(self, trained_model: dict):
        self.test_results_path = TestResultConfig()
        self.trained_model = trained_model

    def initiate_testing(self):
        try:
            start = datetime.now()
            logging.info("Reading test data.")
            test_df = pd.read_csv(self.test_results_path.test_data_path)

            logging.info("preparing X and y")
            X = np.array(test_df.iloc[:, :-1])
            y = np.array(test_df.iloc[:, -1])

            logging.info("model testing initiated.")
            best_model, test_df = model_testing(X, y, self.trained_model)
            logging.info("model testing finished found the best model.")

            logging.info("storing test results")
            test_df.to_csv(
                self.test_results_path.test_result_path, index=False, header=True
            )

            logging.info("storing best model")
            save_object(self.test_results_path.model_path, best_model)
            end = datetime.now()

            logging.info(f"time taken for model testing {end-start}")

            return (test_df, best_model)

        except Exception as e:
            raise CustomException(e, sys)
