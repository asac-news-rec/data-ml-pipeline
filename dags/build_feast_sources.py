"""[파이프라인 구간: 집계 → 피처] clean + feat 기사 표 → Feast 원천 표 4개 매시 DAG.

같은 시간의 `clean_events`가 끝난 뒤 `transform/build_feast_sources.py`를 작업 이미지 Pod로 실행한다.
결과 표(article_profile, reader_profile, reader_activity, reader_article)는 `feast_materialize`가
Valkey로 옮긴다. 기사 임베딩(`embed_articles`)은 따로 돌고, 아직 임베딩이 없는 기사는 다음 시간에 들어간다.

Airflow Variables:
  feast_work_image, gcp_project_id (필수)
  clean_bq_dataset(기본 clean), feat_bq_dataset(기본 feat),
  feast_bq_dataset(결과 표 위치. feast_materialize와 같은 Variable, '프로젝트.데이터셋' 또는 데이터셋 이름, 기본 feat)
  data_valid_from(선택. 이 시각 이후 행동만 피처에 쓴다. 예: 2026-11-26T14:00:00+09:00. 비어 있으면 전부)
    부하 테스트·인프라 변경으로 생긴 시험 로그를 지우지 않고 뺄 때 값만 새 시각으로 옮긴다.
"""
import os
from datetime import datetime, timedelta

from airflow.exceptions import AirflowException
from airflow.models import Variable
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.sensors.external_task import ExternalTaskSensor
from airflow.sdk import DAG
from airflow.timetables.interval import CronDataIntervalTimetable
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
    dag_id="build_feast_sources",
    start_date=datetime(2026, 11, 1),   # clean_events와 같아야 logical date가 맞는다
    # Airflow 3의 timedelta는 켠 시각부터 세서 DAG마다 logical_date가 어긋난다. 정시 한 시간 구간으로 고정한다.
    schedule=CronDataIntervalTimetable("0 * * * *", timezone="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["feast", "feature"],
) as dag:
    wait_clean = ExternalTaskSensor(
        task_id="wait_clean_events",
        external_dag_id="clean_events",
        external_task_id="clean_events",
        mode="reschedule",
        poke_interval=120,
        timeout=60 * 60,
    )
    check = PythonOperator(task_id="check_config", python_callable=check_config)

    build = KubernetesPodOperator(
        task_id="build_feast_sources",
        name="build-feast-sources",
        image="{{ var.value.feast_work_image }}",
        cmds=["python", "/app/transform/build_feast_sources.py"],
        namespace=NAMESPACE,
        service_account_name=SERVICE_ACCOUNT,
        image_pull_policy="IfNotPresent",
        container_resources=k8s.V1ResourceRequirements(
            requests={"cpu": "100m", "memory": "256Mi"},
            limits={"cpu": "500m", "memory": "512Mi"},
        ),
        env_vars={
            "GCP_PROJECT_ID": "{{ var.value.gcp_project_id }}",
            "CLEAN_BQ_DATASET": "{{ var.value.get('clean_bq_dataset', 'clean') }}",
            "FEAT_BQ_DATASET": "{{ var.value.get('feat_bq_dataset', 'feat') }}",
            "FEAST_SOURCE_DATASET": "{{ var.value.get('feast_bq_dataset', 'feat') }}",
            "DATA_VALID_FROM": "{{ var.value.get('data_valid_from', '') }}",
        },
        get_logs=True,
        on_finish_action="delete_pod",
    )

    wait_clean >> check >> build
