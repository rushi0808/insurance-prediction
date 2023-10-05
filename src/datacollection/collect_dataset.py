import os
from dataclasses import dataclass


@dataclass
class DataCollectionPath:
    datapath: str = os.path.join("data", "insurance.csv")
