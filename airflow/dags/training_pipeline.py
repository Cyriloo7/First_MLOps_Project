from __future__ import annotations
import json
from textwrap import dedent
import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator
from src.pipeline.training_pipeline import TrainingPipeline
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


TrainingPipeline = TrainingPipeline()

with DAG(
    "gemstone_training_pipeline",
    default_args={"retries": 2},
    description="this is my training pipeline",
    schedule = "@weekly",
    start_date=pendulum.tz.datetime(2024, 7, 9, tz="UTC"),
    catchup=False,
    tags=["training", "training_pipeline","classification"],
) as dag:
    dag.doc_md = __doc__
    def data_ingestion(**kwargs):
        ti = kwargs['ti']
        train_data_path, test_data_path = DataIngestion.initiate_data_ingestion()
        ti.xcom_push("data_ingestion_artifact", {"train_data_path":train_data_path})

    def data_transformation(**kwargs):
        ti = kwargs['ti']
        data_ingestion_artifact = ti.xcom_pull(key='data_ingestion_artifact', task_ids='data_ingestion')
        train_arr, test_arr = DataTransformation.initiate_data_transformation(data_ingestion_artifact["train_data_path"])
        train_arr = train_arr.tolist()
        test_arr = test_arr.tolist()
        ti.xcom_push("data_transformation_artifact", {"train_arr": train_arr, "test_arr": test_arr})

    def model_training(**kwargs):
        import numpy as np
        ti = kwargs['ti']
        data_transformation_artifact = ti.xcom_pull(key='data_transformation_artifact', task_ids='data_transformation')
        train_arr = np.array(data_transformation_artifact["train_arr"])
        test_arr = np.array(data_transformation_artifact["test_arr"])
        model_trainer = ModelTrainer.initiate_model_training(train_arr, test_arr)

    data_ingestion_task = PythonOperator(
        task_id="data_ingestion",
        python_callable=data_ingestion,
        provide_context=True,
    )
    data_ingestion_task.doc_md = dedent()

    data_transformation_task = PythonOperator(
        task_id="data_transformation",
        python_callable=data_transformation,
        provide_context=True,
    )
    data_transformation_task.doc_md = dedent()

    model_training_task = PythonOperator(
        task_id="model_training",
        python_callable=model_training,
        provide_context=True,
    )
    model_training_task.doc_md = dedent()
    
    data_ingestion_task >> data_transformation_task >> model_training_task