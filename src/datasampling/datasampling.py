import os
import sys
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split

from src.datacollection.collect_dataset import DataPathConfig
from src.exception import CustomException
from src.logger import logging


class DataSamplingConfig:
    def __init__(self, processed_data_path):
        self.train_data_path = DataPathConfig().train_data_path
        self.test_data_path = DataPathConfig().test_data_path
        self.processed_data_path = processed_data_path

    def initiat_data_config(self):
        try:
            start_time = datetime.now()
            logging.info("Data smapling initiated.")

            logging.info("Reading preprocessed data.")
            processed_data = pd.read_csv(self.processed_data_path)

            logging.info("Spliting data into training and testing.")
            train_data, test_data = train_test_split(processed_data, test_size=0.2)

            logging.info("storing train data.")
            train_data.to_csv(self.train_data_path, index=False, header=True)

            logging.info("storing test data")
            test_data.to_csv(self.test_data_path, index=False, header=True)

            end_time = datetime.now()
            logging.info(f"Time taken for data sampling {end_time-start_time}.")

            return (self.train_data_path, self.test_data_path)

        except Exception as e:
            raise CustomException(e, sys)
