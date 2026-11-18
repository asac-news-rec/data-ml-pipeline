"""[파이프라인 구간: 피처 → 서빙 내보내기] Feast online store(Valkey) materialize DAG.

Feast는 Airflow 이미지에 없으므로 Python 3.12 작업 이미지를 KubernetesPodOperator로 띄운다.
작업 이미지는 `pyproject.toml`의 `airflow-dag` 묶음으로 만든다.

환경변수:
  FEAST_WORK_IMAGE              작업 이미지 (필수)
  FEAST_WORK_NAMESPACE          Pod namespace (기본 airflow)
  FEAST_WORK_SERVICE_ACCOUNT    Pod service account (기본 default)

Airflow Variables:
  feast_registry_path, valkey_host, gcp_project_id,
  feast_bq_dataset, feast_bq_temp_dataset,
  feast_full_materialize, feast_materialize_start
"""
import os
from datetime import datetime, timedelta

from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.sdk import DAG

WORK_IMAGE = os.environ.get("FEAST_WORK_IMAGE", "")
NAMESPACE = os.environ.get("FEAST_WORK_NAMESPACE", "airflow")
SERVICE_ACCOUNT = os.environ.get("FEAST_WORK_SERVICE_ACCOUNT", "default")

with DAG(
    dag_id="feast_materialize",
    start_date=datetime(2026, 11, 1),
    schedule=timedelta(hours=1),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["feast", "feature"],
) as dag:
    KubernetesPodOperator(
        task_id="materialize",
        name="feast-materialize",
        image=WORK_IMAGE or "REPLACE_ME",
        cmds=["python", "/app/features/feast/materialize.py"],
        namespace=NAMESPACE,
        service_account_name=SERVICE_ACCOUNT,
        image_pull_policy="IfNotPresent",
        env_vars={
            "MATERIALIZE_START": "{{ data_interval_start.isoformat() }}",
            "MATERIALIZE_END": "{{ data_interval_end.isoformat() }}",
            "FEAST_FULL_MATERIALIZE": "{{ var.value.get('feast_full_materialize', 'false') }}",
            "FEAST_MATERIALIZE_START": "{{ var.value.get('feast_materialize_start', '') }}",
            "FEAST_REGISTRY_PATH": "{{ var.value.get('feast_registry_path', '') }}",
            "VALKEY_HOST": "{{ var.value.get('valkey_host', 'valkey.valkey.svc.cluster.local') }}",
            "GCP_PROJECT_ID": "{{ var.value.get('gcp_project_id', 'final-project-513508') }}",
            "FEAST_BQ_DATASET": "{{ var.value.get('feast_bq_dataset', 'final-project-513508.dev_yoonmo') }}",
            "FEAST_BQ_TEMP_DATASET": "{{ var.value.get('feast_bq_temp_dataset', 'dev_yoonmo') }}",
        },
        get_logs=True,
        on_finish_action="delete_pod",
    )
