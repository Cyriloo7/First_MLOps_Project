import pandas as pd
import numpy as np
from src.logger.logging import logging
from src.exceptions import customexception


import os
import Sys
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import path

from src.utils.utils import save_object, evaluate_model


@dataclass
class ModelTrainerConfig:
    pass

class MoelTrainer:
    def __init__(self):
        pass
    def initiate_model_trainer(self):
        try:
            pass
        except Exception as e:
            logging.info()
            raise customexception(e, sys)