# Feast 피처 스토어 (로컬 프로토타입)

> 상태: **[현재]** 로컬(parquet + SQLite)에서 끝까지 돈다. **[계획]** BigQuery(오프라인) + Valkey(온라인)로 옮긴다.

PCTR LightGBM 모델의 피처(일반 9 + 유저 임베딩 768 + 기사 임베딩 768)를 Feast로 정의한다.
학습 표와 추론 서버가 **같은 정의**에서 피처를 꺼내게 하는 것이 목적이다.

```
원천 표(parquet / [계획] BigQuery feat.*)
   ├─ get_historical_features(entity_df=노출 목록) → 학습 표 (point-in-time join)
   └─ materialize → 온라인 저장소(SQLite / [계획] Valkey)
                       └─ get_online_features(reader 1명 + 후보 기사 N건) → 추론 서버
```

## 실행 (WSL)

`uv`와 `requirements.txt`는 이 프로토타입에서만 쓴다. 저장소 전체의 패키지 관리 도구는 아직 [미정]이다.

```bash
cd features/feast
uv venv -p 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python prepare_sources.py --src <final-project/virtual_reader 경로>   # 원천 표 5개 (약 1.5분), --valid-from으로 검증 시작 시각 지정
cd feature_repo && ../.venv/bin/feast apply && cd ..                           # 정의 등록
.venv/bin/python build_training_set.py --ref <virtual_reader/features/features.npz>  # 학습 표 + 대조 (약 1분)
.venv/bin/python online_demo.py                                                # materialize + 온라인 조회, --now로 '지금' 지정
```

## 파일

| 파일 | 하는 일 |
| --- | --- |
| `prepare_sources.py` | 가짜 로그 → 원천 표 parquet 5개. [계획] Airflow 집계 단계가 BigQuery에 같은 표를 만든다 |
| `feature_repo/definitions.py` | entity 2개, feature view 4개, on-demand feature view 1개, feature service 1개 |
| `feature_repo/feature_store.yaml` | 로컬 설정 (file + sqlite) |
| `feature_repo/feature_store.gcp.yaml.example` | [계획] BigQuery + Valkey 설정 예시 (미실행) |
| `build_training_set.py` | `get_historical_features`로 학습 표를 만들고 `make_features.py` 결과와 대조 |
| `online_demo.py` | `materialize` → `get_online_features` → 같은 시각 오프라인 값과 비교 |

## 피처 정의

| feature view | entity | 원천 표 1행 = | 피처 |
| --- | --- | --- | --- |
| `article_profile` | article | 기사 1건 (시각 = 발행 시각) | `a_emb`(768), `category`, `published_ts` |
| `reader_profile` | reader | 클릭 1건 (그 직후 상태) | `u_emb`(768), `hist_clicks`, `last_click_ts` |
| `reader_activity` | reader | 유효 노출 1건 (그 직후 상태) | `hist_exposures` |
| `reader_article` | reader + article | 상태가 바뀐 사건 1건 | `already_opened`, `already_seen` |
| `request_context` (on-demand) | — | 요청마다 계산 | `article_age_h`, `hour`, `hours_since_last_click` |

- 피처 행 시각 = 사건 시각 + 1µs. 노출과 같은 시각의 사건은 그 노출에 붙지 않는다.
- `u_emb`는 클릭 때만 바뀐다. 타임 디케이 가중치 `0.5 ** (Δt/24h)`는 분자·분모에 같이 곱해져 약분된다.
- `position`은 요청값이다. 학습 땐 entity_df 열, 서빙 땐 요청에서 받는다.
- 행이 없으면 null이다. `hist_*`, `already_*`는 0으로, `u_emb`는 NaN(cold)으로 채운다 (`build_training_set.to_matrix`).
- ttl은 모두 0(무제한)이다. 피처가 전부 '지금까지' 누적값이라, 기간으로 자르면 오래 쉬다 온 독자가 0으로 학습된다.
- 같은 (entity 키, 시각) 행이 여러 개면 Feast가 어느 행을 고를지 정해져 있지 않아, 원천 표를 쓸 때 마지막 행만 남긴다.

## 결과 (2026-11-13, 가짜 로그 11/3~11/6 12시)

- 학습 표 40,763행 × 1,545열. `make_features.py` 결과와 모든 열이 **차이 0**으로 같다. cold 41.5%.
- materialize 약 3초. 독자 1명 + 후보 10건 온라인 조회 약 23ms(로컬 SQLite, 두 번째 호출부터).
- 같은 시각의 온라인 값과 오프라인 값이 같다 (training-serving skew 없음).

## 알아 둘 것

- Feast 0.68 file 오프라인 저장소는 feature view를 붙일 때마다 '지금까지 나온 join key + 시각'으로 중복을 지운다.
  다른 독자가 같은 기사를 같은 ms에 본 노출 2건이 1건으로 합쳐졌다(40,763행 중 1쌍).
  `build_training_set.historical()`이 그런 행을 따로 호출해 피한다. BigQuery 저장소에서도 같은지는 아직 확인하지 않았다.
  회피책이 못 막는 경우를 위해 `check_rows()`가 결과 행 수와 row_id를 항상 검사하고, 다르면 멈춘다.
- materialize 범위는 view의 ttl보다 짧으면 안 된다. 짧으면 오프라인엔 있고 온라인엔 없는 값이 생긴다.
  `online_demo.py`는 원천 표의 가장 이른 시각부터 전체를 쓴다. 서빙 DAG는 첫 적재만 전체, 이후 `materialize_incremental`로 한다.
- on-demand feature view는 Feast가 "experimental"로 표시한다. 오프라인에서 행이 많으면 느릴 수 있다.
