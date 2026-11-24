"""[파이프라인 구간: 정제] raw.served·raw.actions → clean.served·clean.actions 매시 DAG.

같은 시간의 `load_raw` DAG 실행이 끝난 뒤 `transform/clean_events.py`를 작업 이미지 Pod로 실행한다.
두 DAG는 start_date·schedule이 같아서 logical date가 맞물리고, ExternalTaskSensor가 그것을 기다린다.
쓰기는 모두 MERGE라 같은 시간을 다시 돌려도 행이 늘지 않는다. 겹쳐 돌면 같은 행을 두 번 넣을 수
있으므로 max_active_runs=1이다.

Airflow Variables:
  feast_work_image (작업 이미지 태그), gcp_project_id,
  raw_bq_dataset(선택, 기본 raw), clean_bq_dataset(선택, 기본 clean),
  raw_load_lookback_hours(선택, 기본 3. load_raw와 같은 값을 쓴다)
"""
import os
from datetime import datetime, timedelta

from airflow.exceptions import AirflowException
from airflow.models import Variable
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.sensors.external_task import ExternalTaskSensor
from airflow.sdk import DAG
from kubernetes.client import models as k8s

NAMESPACE = os.environ.get("FEAST_WORK_NAMESPACE", "airflow")
SERVICE_ACCOUNT = os.environ.get("FEAST_WORK_SERVICE_ACCOUNT", "airflow-task")
REQUIRED_VARS = ["feast_work_image", "gcp_project_id"]


def check_config(**context) -> None:
    """필수 Variable이 없으면 Pod를 띄우기 전에 실패한다."""
    missing = [key for key in REQUIRED_VARS if not Variable.get(key, default_var=None)]
    if missing:
        raise AirflowException(f"Airflow Variable이 없다: {', '.join(missing)}")


with DAG(
    dag_id="clean_events",
    start_date=datetime(2026, 11, 1),   # load_raw와 같아야 logical date가 맞는다
    schedule=timedelta(hours=1),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["clean"],
) as dag:
    wait_raw = ExternalTaskSensor(
        task_id="wait_load_raw",
        external_dag_id="load_raw",
        external_task_id="load_raw",
        mode="reschedule",
        poke_interval=120,
        timeout=60 * 60,
    )
    check = PythonOperator(task_id="check_config", python_callable=check_config)

    clean = KubernetesPodOperator(
        task_id="clean_events",
        name="clean-events",
        image="{{ var.value.feast_work_image }}",
        cmds=["python", "/app/transform/clean_events.py"],
        namespace=NAMESPACE,
        service_account_name=SERVICE_ACCOUNT,
        image_pull_policy="IfNotPresent",
        container_resources=k8s.V1ResourceRequirements(
            requests={"cpu": "100m", "memory": "256Mi"},
            limits={"cpu": "500m", "memory": "512Mi"},
        ),
        env_vars={
            "LOAD_HOUR_START": "{{ data_interval_start.isoformat() }}",
            "LOAD_LOOKBACK_HOURS": "{{ var.value.get('raw_load_lookback_hours', '3') }}",
            "GCP_PROJECT_ID": "{{ var.value.gcp_project_id }}",
            "RAW_BQ_DATASET": "{{ var.value.get('raw_bq_dataset', 'raw') }}",
            "CLEAN_BQ_DATASET": "{{ var.value.get('clean_bq_dataset', 'clean') }}",
        },
        get_logs=True,
        on_finish_action="delete_pod",
    )

    wait_raw >> check >> clean
