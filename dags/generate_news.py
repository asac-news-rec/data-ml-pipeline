"""가상 뉴스 생성 → 기사 검증 → Cloud SQL 적재 배치를 GKE Pod로 실행합니다."""

from datetime import timedelta
from pathlib import Path

import pendulum
from airflow.sdk import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator


# 생성기 이미지의 커밋 SHA는 업로드한 버전으로 교체합니다.
GENERATOR_IMAGE = "asia-northeast3-docker.pkg.dev/final-project-513508/news-rec/news-generator:54317e07df0dc8076c2949f8d66d0368bbdd4af6"
POD_TEMPLATE_FILE = str(Path(__file__).resolve().parent / "pod_templates" / "news-generator.yaml")
TEXT_MODEL = "google/gemini-2.5-flash"
IMAGE_MODEL = "black-forest-labs/flux.2-klein-4b"
IMAGE_BASE_URL = "https://storage.googleapis.com/final-project-513508-news-thumbnails/images"
# Pod 템플릿에서도 같은 경로에 결과 저장소를 마운트해야 합니다.
OUTPUT_ROOT = "/mnt/news-output"

with DAG(
    dag_id="generate_news",
    description="OpenRouter 기사·썸네일 생성 및 Cloud SQL 적재",
    start_date=pendulum.datetime(2026, 11, 18, tz="Asia/Seoul"),
    schedule="*/5 * * * *",
    catchup=False,
    max_active_runs=1,
    is_paused_upon_creation=True,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=5)},
    tags=["news", "generation"],
) as dag:
    KubernetesPodOperator(
        task_id="generate_and_store",
        name="news-generator",
        namespace="airflow",
        image=GENERATOR_IMAGE,
        # 인프라가 준비한 템플릿에서 실행 계정·볼륨·Secret·DB 연결을 받습니다.
        pod_template_file=POD_TEMPLATE_FILE,
        base_container_name="base",
        cmds=["python", "-m", "generate.generate"],
        arguments=[
            "--text-model", TEXT_MODEL,
            "--image-model", IMAGE_MODEL,
            "--count", "1",
            "--output-root", OUTPUT_ROOT,
            "--date", "{{ data_interval_end.in_timezone('Asia/Seoul').strftime('%Y-%m-%d') }}",
            "--run-id", "{{ dag.dag_id }}:{{ run_id }}",
            "--image-base-url", IMAGE_BASE_URL,
            "--write-db",
        ],
        in_cluster=True,
        get_logs=True,
        do_xcom_push=False,
        startup_timeout_seconds=600,
        execution_timeout=timedelta(hours=1),
        on_finish_action="delete_pod",
        log_pod_spec_on_failure=False,
    )
