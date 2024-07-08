import pandas as pd
import numpy as np
from src.logger.logging import logging
from src.exception.exception import customexception
from pathlib import Path


import os
import sys
import pickle
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import mlflow
import mlflow.sklearn
from src.utils.utils import load_object
from urllib.parse import urlparse
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

@dataclass
class ModelEvaluationConfig:
    pass

class ModelEvaluation:
    def __init__(self):
        logging.info("Evaluation started...")

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        logging.info(f"RMSE: {rmse}, MAE: {mae}, R2: {r2} is calculated")
        return rmse, mae, r2

    def initiate_model_evaluation(self, train_array, test_array):
        try:
            X_test, y_test = (test_array[:,:-1], test_array[:-1])

            model_path = os.path.join("artifacts","model.pkl")
            model = load_object(model_path)

            #mlflow.set_registry_uri("")
            logging.info("model has been registered")
            tracking_uri_type_store = urlparse(mlflow.get_tracking_uri())
            print(tracking_uri_type_store)
            logging.info("model evaluation started...")

            with mlflow.start_run() as run:
                run.log_param("model_type", "RandomForestRegressor")
                run.log_metric("rmse", self.eval_metrics(y_test, model.predict(X_test))[0])
                run.log_metric("mae", self.eval_metrics(y_test, model.predict(X_test))[1])
                run.log_metric("r2", self.eval_metrics(y_test, model.predict(X_test))[2])
                run.log_artifact(model_path)
            logging.info("Evaluation completed")

            if tracking_uri_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.sklearn.log_model(model, "model", registered_model_name="ml_model")
            else:
                mlflow.sklearn.log_model(model, "model")


        except Exception as e:
            logging.info()
            raise customexception(e, sys)