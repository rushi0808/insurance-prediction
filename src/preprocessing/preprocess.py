import os
import sys
from dataclasses import dataclass
from datetime import datetime

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.make_dataset import DatasetPathConfig


class DataPreprocess:
    def __init__(self) -> None:
        self.datapath = DatasetPathConfig()

    def initiate_data_preprocess(self):
        pass
