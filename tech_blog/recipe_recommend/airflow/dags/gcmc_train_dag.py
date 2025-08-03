from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

from gcmc_trainer import GCMCDataset, GCMCTrainer
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

def run_training():
    conn_params = {
        "dbname": "mydb",
        "user": "user",
        "password": "pass",
        "host": "my_postgres",
        "port": "5432"
    }

    dataset = GCMCDataset(conn_params)

    model_config = {
        "input_dim": 128,
        "hidden_dims": [72, 32],
        "num_classes": 6,
        "num_basis_functions": 3,
        "accum": "stack",
        "self_connections": False,
        "dropout": 0.5,
        "model_ty": "classify"
    }

    train_config = {
        "epochs": 100,
        "batch_size": 512,
        "patience": 10,
        "num_classes": 6
    }

    trainer = GCMCTrainer(dataset, model_config, train_config)
    trainer.train()
    # 모델 저장은 trainer 내부에서 수행됨

with DAG(
    'gcmc_training_dag',
    default_args=default_args,
    description='Train GCMC model on schedule',
    schedule = '0 3 * * *',  # 매일 오전 3시
    start_date=datetime(2025, 6, 22),
    catchup=False,
    tags=['gcmc', 'ml', 'airflow'],
) as dag:

    train_task = PythonOperator(
        task_id='train_gcmc_model',
        python_callable=run_training
    )
