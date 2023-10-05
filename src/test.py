import sys

from src.datacollection.collect_dataset import DataCollectionPath
from src.datasampling.datasampling import DataSamplingConfig
from src.preprocessing.preprocess import DataPreprocessing

if __name__ == "__main__":
    data_path = DataCollectionPath()
    preprocess_obj = DataPreprocessing(data_path.datapath, "charges")
    preprocessed_data_path = preprocess_obj.initiate_preprocesor()
    sampling_obj = DataSamplingConfig(preprocessed_data_path)
    train_path, test_path = sampling_obj.initiat_data_config()
    print(train_path, test_path)
