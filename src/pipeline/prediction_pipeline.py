import os
import sys

from src.datacollection.collect_dataset import DataPathConfig
from src.exception import CustomException
from src.logger import logging
from src.utils import make_prediction


class PredictionPipeline:
    def __init__(self, customer_input: dict):
        self.model_path = DataPathConfig.best_model_path
        self.preprocessor_path = DataPathConfig.preprocessor_obj_path
        self.customer_input = customer_input

    def initiate_prediction(self):
        try:
            logging.info("Prediction initiated")
            customer_prediction = make_prediction(
                self.customer_input, self.preprocessor_path, self.model_path
            )

            return customer_prediction
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    customer_input = {
        "age": 24,
        "sex": "male",
        "bmi": 20,
        "children": 0,
        "smoker": "no",
        "region": "southwest",
    }
    prediction_obj = PredictionPipeline(customer_input)
    prediction = prediction_obj.initiate_prediction()
    print(prediction)
