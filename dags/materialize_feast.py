"""[파이프라인 구간: 피처 → 서빙 내보내기] Feast online store(Valkey) materialize DAG.

Feast는 Airflow 이미지에 없으므로 Python 3.12 작업 이미지를 KubernetesPodOperator로 띄운다.
작업 이미지는 `pyproject.toml`의 `airflow-dag` 묶음으로 만든다.

materialize는 Valkey뿐 아니라 registry에도 적재 구간을 쓴다. 그래서 Airflow 서비스 계정에는
registry 읽기·쓰기 권한이 필요하고, main `feast apply`와 겹치지 않게 apply 전에 DAG를 pause한다.

같은 시간의 `build_feast_sources`(Feast 원천 표 생성)가 끝난 뒤에 적재한다. 두 DAG는 start_date·schedule이
같아서 logical date가 맞물린다. 원천 표를 아직 로컬 스크립트로 만드는 동안에는 Variable
`feast_wait_for_sources=false`로 기다리지 않게 한다.

Airflow Variables:
  feast_work_image, feast_registry_path, valkey_host,
  gcp_project_id, feast_bq_dataset, feast_bq_temp_dataset,
  feast_full_materialize, feast_materialize_start,
  feast_wait_for_sources(선택, 기본 true)

feast_bq_temp_dataset은 `프로젝트.데이터셋`과 `데이터셋` 둘 다 받는다. Feast BigQuery offline store가
앞에 프로젝트를 붙이므로 작업 Pod에는 데이터셋 이름만 넘긴다. 프로젝트가 gcp_project_id와 다르면 실패한다.
"""
import os
import re
from datetime import datetime, timedelta

from airflow.exceptions import AirflowException
from airflow.models import Variable
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.providers.standard.operators.python import BranchPythonOperator, PythonOperator
from airflow.providers.standard.sensors.external_task import ExternalTaskSensor
from airflow.sdk import DAG
from airflow.timetables.interval import CronDataIntervalTimetable
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


def wait_or_skip(**context) -> str:
    """원천 표 DAG를 기다릴지 정한다. 기다리지 않으면 바로 check_config로 간다."""
    wait = str(Variable.get("feast_wait_for_sources", default_var="true")).lower() == "true"
    return "wait_feast_sources" if wait else "check_config"


def dataset_name(value: str, project: str) -> str:
    """`데이터셋`이나 `프로젝트.데이터셋`에서 데이터셋 이름을 얻는다. 다른 프로젝트·잘못된 형식이면 실패한다."""
    prefix, dot, dataset = value.strip().rpartition(".")
    if not re.fullmatch(r"[A-Za-z0-9_]+", dataset) or (dot and prefix != project):
        raise AirflowException(f"feast_bq_temp_dataset 형식이 잘못됐다: {value!r}")
    return dataset


def check_config(**context) -> None:
    """필수 Variable이 없으면 Pod를 띄우기 전에 실패한다."""
    missing = [key for key in REQUIRED_VARS if not Variable.get(key, default_var=None)]
    if missing:
        raise AirflowException(f"Airflow Variable이 없다: {', '.join(missing)}")
    dataset_name(Variable.get("feast_bq_temp_dataset"), Variable.get("gcp_project_id"))
    full = str(Variable.get("feast_full_materialize", default_var="false")).lower() == "true"
    if full and not Variable.get("feast_materialize_start", default_var=None):
        raise AirflowException("feast_full_materialize=true인데 feast_materialize_start가 없다")


with DAG(
    dag_id="feast_materialize",
    start_date=datetime(2026, 11, 1),
    # Airflow 3의 timedelta는 켠 시각부터 세서 DAG마다 logical_date가 어긋난다. 정시 한 시간 구간으로 고정한다.
    schedule=CronDataIntervalTimetable("0 * * * *", timezone="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["feast", "feature"],
    user_defined_macros={"dataset_name": dataset_name},
) as dag:
    decide = BranchPythonOperator(task_id="decide_wait", python_callable=wait_or_skip)
    wait_sources = ExternalTaskSensor(
        task_id="wait_feast_sources",
        external_dag_id="build_feast_sources",
        external_task_id="build_feast_sources",
        mode="reschedule",
        poke_interval=120,
        timeout=60 * 60,
    )
    check = PythonOperator(task_id="check_config", python_callable=check_config,
                           trigger_rule="none_failed_min_one_success")

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
            "FEAST_BQ_TEMP_DATASET": "{{ dataset_name(var.value.feast_bq_temp_dataset, var.value.gcp_project_id) }}",
        },
        get_logs=True,
        on_finish_action="delete_pod",
    )

    decide >> [wait_sources, check]
    wait_sources >> check >> materialize
