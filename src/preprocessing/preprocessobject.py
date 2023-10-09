import os
import pickle
import sys
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.datacollection.collect_dataset import DataPathConfig
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


class PreprocessorObject:
    def __init__(self, num_col, cate_col):
        self.preprocessor_obj_path = DataPathConfig().preprocessor_obj_path
        self.num_col = num_col
        self.cate_col = cate_col

    def buildpreprocessor(self):
        try:
            logging.info("Initiated process of building preprocessor.")
            num_pipeline = Pipeline(
                [
                    ("SimpleImputer", SimpleImputer(strategy="mean")),
                    # ("StandardScaler", StandardScaler()),
                ]
            )
            logging.info("Created Numeric Pipeline")

            cate_pipeline = Pipeline(
                [
                    ("SimpleImputer", SimpleImputer(strategy="most_frequent")),
                    ("OneHotEncoder", OneHotEncoder()),
                ]
            )
            logging.info("Created categorical pipeline.")

            preprocessor = ColumnTransformer(
                [
                    ("Num_Pipeline", num_pipeline, self.num_col),
                    ("Cate_Pipeline", cate_pipeline, self.cate_col),
                ]
            )
            logging.info("Preprocessor Build complete.")

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
