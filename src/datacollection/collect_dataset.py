import os
from dataclasses import dataclass


@dataclass
class DataPathConfig:
    rawdatapath: str = os.path.join("data", "insurance.csv")
    preprocessed_data_path: str = os.path.join("data", "preprocessed_data.csv")
    preprocessor_obj_path: str = os.path.join("preprocessor", "preprocessor.pkl")
    train_data_path: str = os.path.join("data", "train.csv")
    test_data_path: str = os.path.join("data", "test.csv")
    train_model_results: str = os.path.join("data", "train_models_results.csv")
    test_model_results: str = os.path.join("data", "test_models_results.csv")
    best_model_path: str = os.path.join("models", "model.pkl")
