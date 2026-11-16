"""Airflow가 이 저장소의 dags/ 폴더를 읽는지만 확인하는 테스트 DAG.

적재 → 정제 → 집계 → 피처 구간의 실제 DAG가 아니다. 수동 실행만 한다(schedule=None).
"""
from datetime import datetime

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

with DAG(
    dag_id="hello_airflow",
    start_date=datetime(2026, 11, 1),
    schedule=None,
    catchup=False,
) as dag:
    BashOperator(task_id="hello", bash_command="echo hello airflow")
