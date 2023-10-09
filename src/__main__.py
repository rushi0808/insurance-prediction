import sys

from src.datacollection.collect_dataset import DataPathConfig
from src.datasampling.datasampling import DataSamplingConfig
from src.modelstest.test_model import TestModelConfig
from src.modelstrain.train_model import ModelConfig
from src.preprocessing.preprocess import DataPreprocessing


def main():
    rawdatapath = DataPathConfig().rawdatapath
    preprocess_obj = DataPreprocessing(rawdatapath, "charges")
    preprocessed_data_path = preprocess_obj.initiate_preprocesor()
    sampling_obj = DataSamplingConfig(preprocessed_data_path)
    train_path, test_path = sampling_obj.initiat_data_config()
    modelconfig_obj = ModelConfig(train_path)
    trained_models = modelconfig_obj.initiat_model_config()
    test_config_obj = TestModelConfig(trained_models, test_path)
    test_results, best_model = test_config_obj.initiate_testing()
    print(test_results, best_model)


if __name__ == "__main__":
    main()
