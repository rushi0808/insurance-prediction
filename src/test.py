import sys

from src.data.make_dataset import DatasetConfig
from src.exception import CustomException
from src.logger import logging

if __name__ == "__main__":
    data_path_obj = DatasetConfig()
    train_path, test_path = data_path_obj.initiat_data_config()
    print(train_path, test_path)
