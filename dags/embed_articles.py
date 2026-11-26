"""[파이프라인 구간: 피처] Cloud SQL 새 기사 → feat.article_embedding·feat.article_meta 매시 DAG.

`ingest/embed_articles.py`를 임베딩 작업 이미지(`pyproject.toml`의 `embed` 묶음, CPU torch)
Pod로 실행한다. 모델(multilingual-e5-base)은 이미지에 미리 넣어 두고, Pod가 매번 내려받지 않는다.
Cloud SQL은 generate_news와 같은 cloud-sql-proxy로 접속하되, 읽기 전용 계정을 쓴다
(11/19 멘토링: 목적별 계정·최소 권한).

Airflow Variables:
  embed_work_image, gcp_project_id, embed_db_user, embed_db_password_secret (필수)
  embed_db_name(선택, 기본 newsdb), feat_bq_dataset(선택, 기본 feat),
  embed_model_revision(선택, 기본 main), embed_limit(선택, 기본 500)
"""
import os
from datetime import datetime, timedelta

from airflow.exceptions import AirflowException
from airflow.models import Variable
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.providers.standard.operators.python import PythonOperator
from airflow.sdk import DAG
from airflow.timetables.interval import CronDataIntervalTimetable
from kubernetes.client import models as k8s

NAMESPACE = os.environ.get("FEAST_WORK_NAMESPACE", "airflow")
SERVICE_ACCOUNT = os.environ.get("FEAST_WORK_SERVICE_ACCOUNT", "airflow-task")
REQUIRED_VARS = ["embed_work_image", "gcp_project_id", "embed_db_user", "embed_db_password_secret"]


def check_config(**context) -> None:
    """필수 Variable이 없으면 Pod를 띄우기 전에 실패한다."""
    missing = [key for key in REQUIRED_VARS if not Variable.get(key, default_var=None)]
    if missing:
        raise AirflowException(f"Airflow Variable이 없다: {', '.join(missing)}")


with DAG(
    dag_id="embed_articles",
    start_date=datetime(2026, 11, 1),
    # Airflow 3의 timedelta는 켠 시각부터 세서 DAG마다 logical_date가 어긋난다. 정시 한 시간 구간으로 고정한다.
    schedule=CronDataIntervalTimetable("0 * * * *", timezone="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["feature", "article"],
) as dag:
    check = PythonOperator(task_id="check_config", python_callable=check_config)

    embed = KubernetesPodOperator(
        task_id="embed_articles",
        name="embed-articles",
        image="{{ var.value.embed_work_image }}",
        cmds=["python", "/app/ingest/embed_articles.py"],
        namespace=NAMESPACE,
        service_account_name=SERVICE_ACCOUNT,
        image_pull_policy="IfNotPresent",
        container_resources=k8s.V1ResourceRequirements(
            requests={"cpu": "500m", "memory": "2Gi"},
            limits={"cpu": "2", "memory": "4Gi"},
        ),
        env_vars={
            "GCP_PROJECT_ID": "{{ var.value.gcp_project_id }}",
            "FEAT_BQ_DATASET": "{{ var.value.get('feat_bq_dataset', 'feat') }}",
            "PGHOST": "cloud-sql-proxy",
            "PGPORT": "5432",
            "PGDATABASE": "{{ var.value.get('embed_db_name', 'newsdb') }}",
            "PGUSER": "{{ var.value.embed_db_user }}",
            "PGPASSWORD_SECRET": "{{ var.value.embed_db_password_secret }}",
            "EMBED_MODEL_REVISION": "{{ var.value.get('embed_model_revision', 'main') }}",
            "EMBED_LIMIT": "{{ var.value.get('embed_limit', '500') }}",
            "HF_HUB_OFFLINE": "1",   # 이미지에 넣은 모델만 쓴다
        },
        get_logs=True,
        on_finish_action="delete_pod",
    )

    check >> embed
