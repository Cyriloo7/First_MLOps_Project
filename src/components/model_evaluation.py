import pandas as pd
import numpy as np
from src.logger.logging import logging
from src.exception.exception import customexception
from pathlib import Path


import os
import sys
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.DimondPricePrediction.utils.utils import load_object

@dataclass
class ModelEvaluationConfig:
    pass

class ModelEvaluation:
    def __init__(self):
        pass
    def initiate_model_evaluation(self):
        try:
            pass
        except Exception as e:
            logging.info()
            raise customexception(e, sys)