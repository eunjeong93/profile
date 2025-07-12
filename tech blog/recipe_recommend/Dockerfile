FROM python:3.10-slim

USER root

RUN apt-get update && apt-get install -y gcc

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# ✅ copy total files in airflow/dags/ 
COPY airflow/dags/ /opt/airflow/dags/

# if needed, copy the model code
COPY gcmc_trainer.py /opt/airflow/dags/
COPY GCMC.py /opt/airflow/dags/

ENV PYTHONPATH=/app