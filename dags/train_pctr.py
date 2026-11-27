"""[파이프라인 구간: 학습 → 서빙] 매일 PCTR 모델을 학습하고, champion이 바뀌면 추론 서버를 롤링 재시작하는 DAG.

train_pctr Pod 하나가 training set을 만들고 바로 학습한다(파일을 같은 Pod 안에서만 주고받는다).
종료 코드 98(champion 유지)·99(클릭 부족)는 skipped가 된다.
restart_inference_server는 학습 결과와 상관없이(trigger_rule=all_done) 매번 돈다.
MLflow pctr@champion 버전과 추론 서버 Deployment의 annotation(news-rec/champion-version)이 다를 때만 재시작한다.
그래서 alias를 옮긴 직후 Pod가 죽거나 재시작이 실패해도, 다음 실행(재시도·다음 날)이 다시 맞춘다.
추론 서버는 시작할 때만 models:/pctr@champion을 읽으므로 재시작해야 새 모델이 나간다.
새 Pod가 준비되지 않으면 쿠버네티스가 이전 Pod를 남겨 두고, 이 DAG는 실패한다.

Airflow Variables:
  train_work_image, gcp_project_id, feast_registry_path, feast_bq_temp_dataset, mlflow_tracking_uri (필수)
  train_include_simulator(기본 false. 가상 독자 로그로도 학습할 때만 true),
  data_valid_from(선택. 이 시각 이후 로그만 학습·피처에 쓴다. build_feast_sources와 같은 값), train_days(기본 14), clean_bq_dataset(기본 clean), valkey_host, inference_namespace(기본 inference),
  inference_deployment(기본 inference-server)
  inference_namespace·inference_deployment를 바꾸면 RBAC(airflow-restart-inference)의 대상도 같이 바꿔야 한다.
"""
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

from airflow.exceptions import AirflowException, AirflowSkipException
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
VERSION_ANNOTATION = "news-rec/champion-version"


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


def champion_version(tracking_uri: str, model_name: str = "pctr") -> str | None:
    """MLflow REST로 champion 버전을 읽는다. alias가 없으면 None. Airflow 이미지에 mlflow 패키지가 없어도 된다."""
    query = urllib.parse.urlencode({"name": model_name, "alias": "champion"})
    url = f"{tracking_uri.rstrip('/')}/api/2.0/mlflow/registered-models/alias?{query}"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return str(json.load(response)["model_version"]["version"])
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def uses_model_uri(deployment) -> bool:
    """추론 서버가 MLflow champion(MODEL_URI)을 읽게 설정돼 있는가. 아니면 재시작해도 모델이 안 바뀐다."""
    return any(env.name == "MODEL_URI" and env.value
               for container in deployment.spec.template.spec.containers for env in (container.env or []))


def restart_inference_server(**context) -> None:
    """champion 버전과 추론 서버 annotation이 다를 때만 재시작하고, 새 Pod가 준비될 때까지 기다린다."""
    from kubernetes import client, config

    version = champion_version(Variable.get("mlflow_tracking_uri"))
    if version is None:
        raise AirflowSkipException("MLflow에 pctr@champion이 아직 없다")
    config.load_incluster_config()
    apps = client.AppsV1Api()
    namespace = Variable.get("inference_namespace", default_var="inference")
    name = Variable.get("inference_deployment", default_var="inference-server")
    deployment = apps.read_namespaced_deployment(name, namespace)
    if not uses_model_uri(deployment):
        raise AirflowSkipException(f"{namespace}/{name}에 MODEL_URI가 없다. 재시작해도 champion v{version}을 읽지 않는다")
    serving = (deployment.spec.template.metadata.annotations or {}).get(VERSION_ANNOTATION)
    if serving == version and rollout_done(deployment):
        raise AirflowSkipException(f"{namespace}/{name}이 이미 champion v{version}을 서빙한다")
    if serving != version:
        # 재시작 annotation과 버전 annotation을 한 번에 바꾼다. 롤아웃이 실패해도 다음 실행이 다시 기다린다.
        now = datetime.now(timezone.utc).isoformat()
        apps.patch_namespaced_deployment(name, namespace, {"spec": {"template": {"metadata": {"annotations": {
            "kubectl.kubernetes.io/restartedAt": now, VERSION_ANNOTATION: version}}}}})
        print(f"{namespace}/{name} 재시작: champion v{serving} → v{version}")
    deadline = time.monotonic() + ROLLOUT_TIMEOUT_S
    while time.monotonic() < deadline:
        if rollout_done(apps.read_namespaced_deployment(name, namespace)):
            print(f"{namespace}/{name} 재시작 끝 (champion v{version})")
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
        # worker 풀(Spot)에서만 띄운다. 고정 노드에는 Airflow·Valkey·MLflow가 있다.
        node_selector={"pool": "worker"},
        tolerations=[k8s.V1Toleration(key="workload", operator="Equal", value="worker", effect="NoSchedule")],
        retries=2,   # Spot 노드는 중간에 회수될 수 있다
        # worker 풀은 평소 0대라 노드 생성·이미지 pull에 시간이 걸린다(generate_news와 같은 값)
        startup_timeout_seconds=600,
        execution_timeout=timedelta(hours=1),
        namespace=NAMESPACE,
        service_account_name=SERVICE_ACCOUNT,
        image_pull_policy="IfNotPresent",
        # 가짜 로그 103,244행 학습의 최대 메모리가 2.2GB였다(로컬, 쓰는 행만 행렬로 만든 뒤).
        container_resources=k8s.V1ResourceRequirements(
            requests={"cpu": "500m", "memory": "3Gi"},
            limits={"cpu": "2", "memory": "5Gi"},
        ),
        env_vars={
            "GCP_PROJECT_ID": "{{ var.value.gcp_project_id }}",
            "TRAIN_END": "{{ data_interval_end.isoformat() }}",
            "TRAIN_DAYS": "{{ var.value.get('train_days', '14') }}",
            "TRAINING_SET": "/tmp/training_set.parquet",
            "OMP_NUM_THREADS": "2",   # CPU limit 2와 맞춘다. 없으면 LightGBM이 노드 전체 코어 수만큼 스레드를 띄운다
            "TRAIN_INCLUDE_SIMULATOR": "{{ var.value.get('train_include_simulator', 'false') }}",
            "DATA_VALID_FROM": "{{ var.value.get('data_valid_from', '') }}",
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

    # 학습이 실패·skip돼도 돈다. champion과 서빙 버전이 같으면 skip한다.
    restart = PythonOperator(task_id="restart_inference_server", python_callable=restart_inference_server,
                             trigger_rule="all_done")

    check >> train >> restart
