"""[파이프라인 구간: 수집] 가상 독자 행동 로그를 매시 GCS에 쓰는 임시 DAG (한 바퀴 시험용).

`ingest/simulate_readers.py`가 지난 한 시간의 served·actions 로그를 로깅 서버 내보내기와
같은 경로·형식으로 행동 로그 버킷에 쓴다. 다음 `load_raw` 실행이 앞 3시간을 함께 보므로
(raw_load_lookback_hours) 이 DAG가 조금 늦게 끝나도 다음 시간에 적재된다.

실제 독자 로그가 생기면 Variable `simulate_readers_enabled`를 false로 두고, 이후 DAG를 지운다.

Airflow Variables:
  feast_work_image (작업 이미지 태그, materialize와 공유), gcp_project_id, raw_gcs_bucket,
  raw_gcs_prefix(선택), feat_bq_dataset(선택, 기본 feat),
  simulate_readers_enabled(선택, 기본 true), sim_readers(선택, 기본 200)
"""
import os
from datetime import datetime, timedelta

from airflow.exceptions import AirflowException, AirflowSkipException
from airflow.models import Variable
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.providers.standard.operators.python import PythonOperator
from airflow.sdk import DAG
from kubernetes.client import models as k8s

NAMESPACE = os.environ.get("FEAST_WORK_NAMESPACE", "airflow")
SERVICE_ACCOUNT = os.environ.get("FEAST_WORK_SERVICE_ACCOUNT", "airflow-task")
REQUIRED_VARS = ["feast_work_image", "gcp_project_id", "raw_gcs_bucket"]


def check_config(**context) -> None:
    """꺼져 있으면 건너뛰고, 필수 Variable이 없으면 Pod를 띄우기 전에 실패한다."""
    if Variable.get("simulate_readers_enabled", default_var="true").lower() != "true":
        raise AirflowSkipException("simulate_readers_enabled가 true가 아니라 건너뛴다")
    missing = [key for key in REQUIRED_VARS if not Variable.get(key, default_var=None)]
    if missing:
        raise AirflowException(f"Airflow Variable이 없다: {', '.join(missing)}")


with DAG(
    dag_id="simulate_readers",
    start_date=datetime(2026, 11, 1),
    schedule=timedelta(hours=1),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["ingest", "simulator"],
) as dag:
    check = PythonOperator(task_id="check_config", python_callable=check_config)

    simulate = KubernetesPodOperator(
        task_id="simulate_readers",
        name="simulate-readers",
        image="{{ var.value.feast_work_image }}",
        cmds=["python", "/app/ingest/simulate_readers.py"],
        namespace=NAMESPACE,
        service_account_name=SERVICE_ACCOUNT,
        image_pull_policy="IfNotPresent",
        container_resources=k8s.V1ResourceRequirements(
            requests={"cpu": "100m", "memory": "256Mi"},
            limits={"cpu": "500m", "memory": "512Mi"},
        ),
        env_vars={
            "SIM_HOUR_START": "{{ data_interval_start.isoformat() }}",
            "GCP_PROJECT_ID": "{{ var.value.gcp_project_id }}",
            "RAW_GCS_BUCKET": "{{ var.value.raw_gcs_bucket }}",
            "RAW_GCS_PREFIX": "{{ var.value.get('raw_gcs_prefix', '') }}",
            "FEAT_BQ_DATASET": "{{ var.value.get('feat_bq_dataset', 'feat') }}",
            "SIM_READERS": "{{ var.value.get('sim_readers', '200') }}",
        },
        get_logs=True,
        on_finish_action="delete_pod",
    )

    check >> simulate
