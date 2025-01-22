from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

from functions import extract_clean, transform, load_to_db
from fintech_dashboard import create_dashboard

default_args = {
    "owner": "omar_abdelaty",
    "depends_on_past": False,
    'start_date': days_ago(2),
    "retries": 1,
}

dag = DAG(
    'fintech_pipeline',
    default_args=default_args,
    description='fintech pipeline',
)

with DAG(
    dag_id = 'fintech_pipeline',
    schedule_interval = '@once',
    default_args = default_args,
    tags = ['fintech-pipeline'],
)as dag:
    extract_clean_task = PythonOperator(
        task_id = 'extract_clean',
        python_callable = extract_clean,
        op_kwargs = {
            'fileName': '/opt/airflow/data/fintech_data_18_52_11870.csv'
        }
    )

    transform_task = PythonOperator(
        task_id = 'transform',
        python_callable = transform,
        op_kwargs = {
            'fileName': '/opt/airflow/data/fintech_clean.csv'
        }
    )

    load_to_db_task = PythonOperator(
        task_id = 'load_to_db',
        python_callable = load_to_db,
        op_kwargs = {
            'fileName': '/opt/airflow/data/fintech_transformed.csv'
        }
    )

    run_dashboard_task = PythonOperator(
        task_id = 'create_dashboard',
        python_callable = create_dashboard,
        op_kwargs= {
            'fileName': '/opt/airflow/data/fintech_transformed.csv'
        }
    )

    extract_clean_task >> transform_task >> load_to_db_task >> run_dashboard_task