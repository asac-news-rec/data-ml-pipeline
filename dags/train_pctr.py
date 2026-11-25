"""[파이프라인 구간: 학습 → 서빙] 매일 PCTR 모델을 학습하고, champion이 바뀌면 추론 서버를 롤링 재시작하는 DAG.

train_pctr Pod 하나가 training set을 만들고 바로 학습한다(파일을 같은 Pod 안에서만 주고받는다).
종료 코드 98(champion 유지)·99(클릭 부족)는 skipped가 되고 restart_inference_server도 건너뛴다.
추론 서버는 시작할 때만 models:/pctr@champion을 읽으므로 재시작해야 새 모델이 나간다.
새 Pod가 준비되지 않으면 쿠버네티스가 이전 Pod를 남겨 두고, 이 DAG는 실패한다.

Airflow Variables:
  train_work_image, gcp_project_id, feast_registry_path, feast_bq_temp_dataset, mlflow_tracking_uri (필수)
  train_days(기본 14), clean_bq_dataset(기본 clean), valkey_host, inference_namespace(기본 inference),
  inference_deployment(기본 inference-server)
"""
import os
import time
from datetime import datetime, timedelta, timezone

from airflow.exceptions import AirflowException
from airflow.models import Variable
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.providers.standard.operators.python import PythonOperator
from airflow.sdk import DAG
from airflow.timetables.interval import CronDataIntervalTimetable
from kubernetes.client import models as k8s

NAMESPACE = os.environ.get("FEAST_WORK_NAMESPACE", "airflow")
SERVICE_ACCOUNT = os.environ.get("FEAST_WORK_SERVICE_ACCOUNT", "airflow-task")
REQUIRED_VARS = ["train_work_image", "gcp_project_id", "feast_registry_path",
                 "feast_bq_temp_dataset", "mlflow_tracking_uri"]
ROLLOUT_TIMEOUT_S = 600


def check_config(**context) -> None:
    """필수 Variable이 없으면 Pod를 띄우기 전에 실패한다."""
    missing = [key for key in REQUIRED_VARS if not Variable.get(key, default_var=None)]
    if missing:
        raise AirflowException(f"Airflow Variable이 없다: {', '.join(missing)}")


def rollout_done(deployment) -> bool:
    """kubectl rollout status와 같은 기준: 새 버전 Pod만 남고 모두 사용 가능하다."""
    want, status = deployment.spec.replicas or 1, deployment.status
    return ((status.observed_generation or 0) >= deployment.metadata.generation
            and (status.updated_replicas or 0) == want
            and (status.replicas or 0) == want
            and (status.available_replicas or 0) == want)


def restart_inference_server(**context) -> None:
    """kubectl rollout restart와 같은 annotation을 바꾸고, 새 Pod가 준비될 때까지 기다린다."""
    from kubernetes import client, config

    config.load_incluster_config()
    apps = client.AppsV1Api()
    namespace = Variable.get("inference_namespace", default_var="inference")
    name = Variable.get("inference_deployment", default_var="inference-server")
    now = datetime.now(timezone.utc).isoformat()
    apps.patch_namespaced_deployment(name, namespace, {"spec": {"template": {"metadata": {
        "annotations": {"kubectl.kubernetes.io/restartedAt": now}}}}})
    deadline = time.monotonic() + ROLLOUT_TIMEOUT_S
    while time.monotonic() < deadline:
        if rollout_done(apps.read_namespaced_deployment(name, namespace)):
            print(f"{namespace}/{name} 재시작 끝")
            return
        time.sleep(10)
    raise AirflowException(f"{namespace}/{name}의 새 Pod가 {ROLLOUT_TIMEOUT_S}초 안에 준비되지 않았다")


with DAG(
    dag_id="train_pctr",
    start_date=datetime(2026, 11, 1),
    # 매일 02:30 KST. 정각의 적재 DAG들과 겹치지 않게 30분에 둔다.
    schedule=CronDataIntervalTimetable("30 17 * * *", timezone="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=10)},
    tags=["train", "pctr"],
) as dag:
    check = PythonOperator(task_id="check_config", python_callable=check_config)

    train = KubernetesPodOperator(
        task_id="train_pctr",
        name="train-pctr",
        image="{{ var.value.train_work_image }}",
        cmds=["sh", "-c"],
        arguments=["python /app/training/build_training_set.py && "
                   "python /app/training/train_lgbm.py --training-set /tmp/training_set.parquet "
                   "--model-out /tmp/lgbm.txt"],
        skip_on_exit_code=[98, 99],
        namespace=NAMESPACE,
        service_account_name=SERVICE_ACCOUNT,
        image_pull_policy="IfNotPresent",
        # 가짜 로그 103,244행 학습의 최대 메모리가 2.7GB였다(로컬).
        container_resources=k8s.V1ResourceRequirements(
            requests={"cpu": "500m", "memory": "3Gi"},
            limits={"cpu": "2", "memory": "6Gi"},
        ),
        env_vars={
            "GCP_PROJECT_ID": "{{ var.value.gcp_project_id }}",
            "TRAIN_END": "{{ data_interval_end.isoformat() }}",
            "TRAIN_DAYS": "{{ var.value.get('train_days', '14') }}",
            "TRAINING_SET": "/tmp/training_set.parquet",
            "CLEAN_BQ_DATASET": "{{ var.value.get('clean_bq_dataset', 'clean') }}",
            "FEAST_FS_YAML": "feature_store.gcp.yaml",
            "FEAST_REGISTRY_PATH": "{{ var.value.feast_registry_path }}",
            "FEAST_BQ_TEMP_DATASET": "{{ var.value.feast_bq_temp_dataset }}",
            "VALKEY_HOST": "{{ var.value.get('valkey_host', 'valkey.valkey.svc.cluster.local') }}",
            "MLFLOW_TRACKING_URI": "{{ var.value.mlflow_tracking_uri }}",
        },
        get_logs=True,
        on_finish_action="delete_pod",
    )

    restart = PythonOperator(task_id="restart_inference_server", python_callable=restart_inference_server)

    check >> train >> restart
