import logging
import os
import sys
from datetime import datetime

from src.exception import CustomException

# constant
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


# For testing perpose
# if __name__ == "__main__":
#     try:
#         logging.info("Testing logging")
#         for each in range(1, 10000):
#             print(each)
#         logging.info("Testing done all good!")
#     except Exception as e:
#         raise CustomException(e, sys)
