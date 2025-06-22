FROM apache/airflow:2.9.0-python3.10

USER root

RUN apt-get update && apt-get install -y gcc

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# ✅ copy total files in airflow/dags/ 
COPY airflow/dags/ /opt/airflow/dags/

# if needed, copy the model code
COPY gcmc_trainer.py /opt/airflow/dags/
COPY GCMC.py /opt/airflow/dags/

USER airflow
