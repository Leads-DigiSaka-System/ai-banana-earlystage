"""
Minimal example DAG so Airflow UI shows at least one pipeline.
Edit or add more DAGs in dags/ for training, data prep, etc.
Runs in Docker (airflow not required in uv for execution).
Style follows Airflow 3.x: https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/dags.html
"""
import datetime

from airflow.providers.standard.operators.bash import BashOperator  # type: ignore[reportMissingImports]
from airflow.providers.standard.operators.python import PythonOperator  # type: ignore[reportMissingImports]
from airflow.sdk import DAG  # type: ignore[reportMissingImports]

with DAG(
    dag_id="example_hello",
    schedule=None,
    start_date=datetime.datetime(2025, 1, 1),
    catchup=False,
    tags=["example"],
) as dag:
    hello_bash = BashOperator(
        task_id="say_hello",
        bash_command="echo 'Hello from Airflow'",
    )
    hello_python = PythonOperator(
        task_id="hello_python",
        python_callable=lambda: print("Hello from Python task"),
    )
    hello_bash >> hello_python
