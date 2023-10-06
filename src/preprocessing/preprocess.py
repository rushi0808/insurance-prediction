import os
import sys
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from src.datacollection.collect_dataset import DataPathConfig
from src.exception import CustomException
from src.logger import logging
from src.preprocessing.preprocessobject import PreprocessorObject
from src.utils import load_object


class DataPreprocessing:
    def __init__(self, datapath, target_col_name: str):
        self.preprocessed_data_path = DataPathConfig().preprocessed_data_path
        self.datapath = datapath
        self.target_col_name = target_col_name

    def initiate_preprocesor(self):
        try:
            start = datetime.now()
            logging.info("Proprocessing intiated.")

            df = pd.read_csv(self.datapath)
            logging.info("Data loading completed for preprocessing")

            X = df.drop(columns=self.target_col_name)
            y = df[self.target_col_name]
            logging.info("seperated independent and target variables.")

            num_features = X.select_dtypes(exclude="object").columns
            cate_features = X.select_dtypes(include="object").columns
            logging.info("Seperated numeric and categorical columns.")

            logging.info("Creating preprocessor.")
            preprocess_path = PreprocessorObject(
                num_features, cate_features
            ).buildpreprocessor()

            preprocess_obj = load_object(preprocess_path)

            logging.info("Fitting data to preprocessor.")
            X = preprocess_obj.fit_transform(X)

            X = pd.DataFrame(X)
            y = pd.DataFrame(np.log(y))
            processed_data = pd.concat([X, y], axis=1)
            logging.info("Storing preprocessed data.")
            processed_data.to_csv(
                self.preprocessed_data_path,
                index=False,
                header=True,
            )

            end = datetime.now()
            logging.info(f"Time taken for preprocessing: {end-start}")

            return self.preprocessed_data_path

        except Exception as e:
            raise CustomException(e, sys)
