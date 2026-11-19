"""[파이프라인 구간: 피처 → 서빙 내보내기] Feast online store(Valkey) materialize DAG.

Feast는 Airflow 이미지에 없으므로 Python 3.12 작업 이미지를 KubernetesPodOperator로 띄운다.
작업 이미지는 `pyproject.toml`의 `airflow-dag` 묶음으로 만든다.

materialize는 Valkey뿐 아니라 registry에도 적재 구간을 쓴다. 그래서 Airflow 서비스 계정에는
registry 읽기·쓰기 권한이 필요하고, main `feast apply`와 겹치지 않게 apply 전에 DAG를 pause한다.

Airflow Variables:
  feast_work_image, feast_registry_path, valkey_host,
  gcp_project_id, feast_bq_dataset, feast_bq_temp_dataset,
  feast_full_materialize, feast_materialize_start
"""
import os
from datetime import datetime, timedelta

from airflow.exceptions import AirflowException
from airflow.models import Variable
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.providers.standard.operators.python import PythonOperator
from airflow.sdk import DAG
from kubernetes.client import models as k8s

NAMESPACE = os.environ.get("FEAST_WORK_NAMESPACE", "airflow")
SERVICE_ACCOUNT = os.environ.get("FEAST_WORK_SERVICE_ACCOUNT", "airflow-task")
REQUIRED_VARS = [
    "feast_work_image",
    "feast_registry_path",
    "valkey_host",
    "gcp_project_id",
    "feast_bq_dataset",
    "feast_bq_temp_dataset",
]


def check_config(**context) -> None:
    """필수 Variable이 없으면 Pod를 띄우기 전에 실패한다."""
    missing = [key for key in REQUIRED_VARS if not Variable.get(key, default_var=None)]
    if missing:
        raise AirflowException(f"Airflow Variable이 없다: {', '.join(missing)}")
    full = str(Variable.get("feast_full_materialize", default_var="false")).lower() == "true"
    if full and not Variable.get("feast_materialize_start", default_var=None):
        raise AirflowException("feast_full_materialize=true인데 feast_materialize_start가 없다")


with DAG(
    dag_id="feast_materialize",
    start_date=datetime(2026, 11, 1),
    schedule=timedelta(hours=1),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["feast", "feature"],
) as dag:
    check = PythonOperator(task_id="check_config", python_callable=check_config)

    materialize = KubernetesPodOperator(
        task_id="materialize",
        name="feast-materialize",
        image="{{ var.value.feast_work_image }}",
        cmds=["python", "/app/features/feast/materialize.py"],
        namespace=NAMESPACE,
        service_account_name=SERVICE_ACCOUNT,
        image_pull_policy="IfNotPresent",
        container_resources=k8s.V1ResourceRequirements(
            requests={"cpu": "500m", "memory": "2Gi"},
            limits={"cpu": "2", "memory": "4Gi"},
        ),
        env_vars={
            "MATERIALIZE_START": "{{ data_interval_start.isoformat() }}",
            "MATERIALIZE_END": "{{ data_interval_end.isoformat() }}",
            "FEAST_FULL_MATERIALIZE": "{{ var.value.get('feast_full_materialize', 'false') }}",
            "FEAST_MATERIALIZE_START": "{{ var.value.get('feast_materialize_start', '') }}",
            "FEAST_MATERIALIZE_LOOKBACK_HOURS": "{{ var.value.get('feast_materialize_lookback_hours', '3') }}",
            "FEAST_REGISTRY_PATH": "{{ var.value.feast_registry_path }}",
            "VALKEY_HOST": "{{ var.value.valkey_host }}",
            "GCP_PROJECT_ID": "{{ var.value.gcp_project_id }}",
            "FEAST_BQ_DATASET": "{{ var.value.feast_bq_dataset }}",
            "FEAST_BQ_TEMP_DATASET": "{{ var.value.feast_bq_temp_dataset }}",
        },
        get_logs=True,
        on_finish_action="delete_pod",
    )

    check >> materialize
