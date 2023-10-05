import os
import sys
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging


@dataclass
class DatasetPathConfig:
    train_data_path: str = os.path.join("data", "train.csv")
    test_data_path: str = os.path.join("data", "test.csv")


class DatasetConfig:
    def __init__(self):
        self.datapath = DatasetPathConfig()
        self.train_df: pd.DataFrame()
        self.test_df: pd.DataFrame()

    def Initiat_data_config(self):
        try:
            start_time = datetime.now()

            logging.info("Reading Data from source")
            df = pd.read_csv("data\insurance.csv")

            logging.info("Data Sampling for train and test")
            self.train_df, self.test_df = train_test_split(df, test_size=0.2)

            logging.info("Storing train data")
            self.train_df.to_csv(
                self.datapath.train_data_path, index=False, header=True
            )

            logging.info("Storing test data")
            self.test_df.to_csv(self.datapath.test_data_path, index=False, header=True)

            logging.info(f"Data Configration Complete!")
            end_time = datetime.now()

            logging.info(f"Time taken for data configration {end_time-start_time}")

            return (self.datapath.train_data_path, self.datapath.test_data_path)

        except Exception as e:
            raise CustomException(e, sys)


# For testing perpose
# if __name__ == "__main__":
#     data_path_obj = DatasetConfig()
#     train_path, test_path = data_path_obj.Initiat_data_config()
