"""[파이프라인 구간: 적재] GCS 행동 로그 → BigQuery raw.served·raw.actions 매시 적재 DAG.

`export_behavior_logs`가 GCS에 쓴 gzip JSONL을 `ingest/load_raw.py`가 읽어 raw 표에 넣는다.
google-cloud-bigquery는 Airflow 이미지에 있는지 확인되지 않았으므로, materialize와 같은
Python 3.12 작업 이미지(`pyproject.toml`의 `airflow-dag` 묶음)를 KubernetesPodOperator로 띄운다.

같은 시간을 다시 돌려도 이미 넣은 파일은 건너뛴다. 두 실행이 겹치면 같은 파일을 함께
넣을 수 있으므로 max_active_runs=1로 한 번에 하나만 돈다.

Airflow Variables:
  feast_work_image (작업 이미지 태그, materialize와 공유), gcp_project_id,
  raw_gcs_bucket, raw_gcs_prefix(선택), raw_bq_dataset(선택, 기본 raw),
  raw_load_lookback_hours(선택, 기본 3)
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
REQUIRED_VARS = ["feast_work_image", "gcp_project_id", "raw_gcs_bucket"]


def check_config(**context) -> None:
    """필수 Variable이 없으면 Pod를 띄우기 전에 실패한다."""
    missing = [key for key in REQUIRED_VARS if not Variable.get(key, default_var=None)]
    if missing:
        raise AirflowException(f"Airflow Variable이 없다: {', '.join(missing)}")


with DAG(
    dag_id="load_raw",
    start_date=datetime(2026, 11, 1),
    schedule=timedelta(hours=1),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["ingest", "raw"],
) as dag:
    check = PythonOperator(task_id="check_config", python_callable=check_config)

    load = KubernetesPodOperator(
        task_id="load_raw",
        name="load-raw",
        image="{{ var.value.feast_work_image }}",
        cmds=["python", "/app/ingest/load_raw.py"],
        namespace=NAMESPACE,
        service_account_name=SERVICE_ACCOUNT,
        image_pull_policy="IfNotPresent",
        container_resources=k8s.V1ResourceRequirements(
            requests={"cpu": "250m", "memory": "512Mi"},
            limits={"cpu": "1", "memory": "2Gi"},
        ),
        env_vars={
            "LOAD_HOUR_START": "{{ data_interval_start.isoformat() }}",
            "LOAD_LOOKBACK_HOURS": "{{ var.value.get('raw_load_lookback_hours', '3') }}",
            "GCP_PROJECT_ID": "{{ var.value.gcp_project_id }}",
            "RAW_GCS_BUCKET": "{{ var.value.raw_gcs_bucket }}",
            "RAW_GCS_PREFIX": "{{ var.value.get('raw_gcs_prefix', '') }}",
            "RAW_BQ_DATASET": "{{ var.value.get('raw_bq_dataset', 'raw') }}",
        },
        get_logs=True,
        on_finish_action="delete_pod",
    )

    check >> load
