-- [파이프라인 구간: 정제] raw.served·raw.actions → clean.served·clean.actions (+ orphan, rejected)
-- 열과 규칙: 청사진 overview.html 인터페이스 규약 ⑤, yoonmo.html STEP 3.
-- {이름}은 transform/clean_events.py가 표 경로로 바꾼다. @이름은 쿼리 매개변수다.
--   @start, @end         처리할 내보내기 구간 시작의 범위 (source_file의 interval_start 문자열, UTC ISO ms)
--   @load_dates          그 구간들이 들어 있는 raw 파티션 (load_date)
--   @min_date            clean 쪽을 읽을 때 쓰는 가장 이른 파티션 날짜 (짝 찾기·중복 확인)
-- 모든 쓰기는 MERGE(없을 때만 넣기)라서 같은 구간을 다시 돌려도 행이 늘지 않는다.

CREATE TABLE IF NOT EXISTS `{clean_served}` (
  impression_id STRING NOT NULL, request_id STRING NOT NULL,
  reader_id STRING NOT NULL, article_id STRING NOT NULL, position INT64 NOT NULL,
  served_at TIMESTAMP NOT NULL, served_date DATE NOT NULL,
  score FLOAT64, model_version STRING, features_as_of TIMESTAMP,
  received_at TIMESTAMP
) PARTITION BY served_date CLUSTER BY reader_id;

CREATE TABLE IF NOT EXISTS `{clean_actions}` (
  event_id STRING NOT NULL, event_type STRING NOT NULL,
  event_time TIMESTAMP NOT NULL, event_date DATE NOT NULL,
  impression_id STRING NOT NULL, page_type STRING NOT NULL,
  duration_ms INT64, ended_at TIMESTAMP,
  reader_id STRING NOT NULL, article_id STRING NOT NULL, request_id STRING NOT NULL,
  position INT64 NOT NULL, served_at TIMESTAMP NOT NULL,
  is_bot BOOL NOT NULL, received_at TIMESTAMP,
  source STRING, schema_version STRING, measurement_version STRING
) PARTITION BY event_date CLUSTER BY reader_id, article_id;

CREATE TABLE IF NOT EXISTS `{clean_orphan_actions}` (
  event_id STRING NOT NULL, event_type STRING, event_time TIMESTAMP, event_date DATE NOT NULL,
  impression_id STRING, payload JSON, source_file STRING, first_seen_at TIMESTAMP
) PARTITION BY event_date;

CREATE TABLE IF NOT EXISTS `{clean_rejected_actions}` (
  event_id STRING NOT NULL, reason STRING NOT NULL, event_type STRING,
  event_time TIMESTAMP, event_date DATE NOT NULL, impression_id STRING,
  payload JSON, source_file STRING, rejected_at TIMESTAMP
) PARTITION BY event_date;

-- 1. 이번 구간의 raw 행. 같은 키가 여러 번 왔으면 먼저 적재된 것 하나만.
CREATE TEMP TABLE served_batch AS
SELECT * EXCEPT (rn) FROM (
  SELECT
    impression_id,
    JSON_VALUE(payload, '$.request_id') AS request_id,
    JSON_VALUE(payload, '$.reader_id') AS reader_id,
    JSON_VALUE(payload, '$.article_id') AS article_id,
    LAX_INT64(payload.position) AS position,
    served_at,
    LAX_FLOAT64(payload.score) AS score,
    JSON_VALUE(payload, '$.model_version') AS model_version,
    SAFE.TIMESTAMP(JSON_VALUE(payload, '$.features_as_of')) AS features_as_of,
    SAFE.TIMESTAMP(JSON_VALUE(payload, '$.received_at')) AS received_at,
    ROW_NUMBER() OVER (PARTITION BY impression_id ORDER BY loaded_at, source_file) AS rn
  FROM `{raw_served}`
  WHERE load_date IN UNNEST(@load_dates)
    AND REGEXP_EXTRACT(source_file, r'/interval_start=([^/]+)/') >= @start
    AND REGEXP_EXTRACT(source_file, r'/interval_start=([^/]+)/') < @end
    AND impression_id IS NOT NULL
)
WHERE rn = 1;

CREATE TEMP TABLE actions_batch AS
SELECT * EXCEPT (rn) FROM (
  SELECT
    LOWER(event_id) AS event_id, event_type, event_time, impression_id,
    JSON_VALUE(payload, '$.page_type') AS page_type,
    LAX_INT64(payload.duration_ms) AS duration_ms,
    SAFE.TIMESTAMP(JSON_VALUE(payload, '$.ended_at')) AS ended_at,
    SAFE.TIMESTAMP(JSON_VALUE(payload, '$.received_at')) AS received_at,
    JSON_VALUE(payload, '$.source') AS source,
    JSON_VALUE(payload, '$.schema_version') AS schema_version,
    JSON_VALUE(payload, '$.measurement_version') AS measurement_version,
    payload, source_file,
    ROW_NUMBER() OVER (PARTITION BY LOWER(event_id) ORDER BY loaded_at, source_file) AS rn
  FROM `{raw_actions}`
  WHERE load_date IN UNNEST(@load_dates)
    AND REGEXP_EXTRACT(source_file, r'/interval_start=([^/]+)/') >= @start
    AND REGEXP_EXTRACT(source_file, r'/interval_start=([^/]+)/') < @end
    AND event_id IS NOT NULL AND event_time IS NOT NULL
)
WHERE rn = 1;

-- 2. clean.served: impression_id가 없을 때만 넣는다.
MERGE `{clean_served}` T
USING served_batch S
ON T.impression_id = S.impression_id AND T.served_date >= @min_date
WHEN NOT MATCHED AND S.request_id IS NOT NULL AND S.reader_id IS NOT NULL
  AND S.article_id IS NOT NULL AND S.position IS NOT NULL AND S.served_at IS NOT NULL THEN
  INSERT (impression_id, request_id, reader_id, article_id, position, served_at, served_date,
          score, model_version, features_as_of, received_at)
  VALUES (S.impression_id, S.request_id, S.reader_id, S.article_id, S.position, S.served_at,
          DATE(S.served_at, 'Asia/Seoul'), S.score, S.model_version, S.features_as_of, S.received_at);

-- 3. 이번 구간 행동 + 앞서 짝이 없어 orphan으로 둔 행동을 함께 다시 본다.
CREATE TEMP TABLE action_candidates AS
SELECT event_id, event_type, event_time, impression_id, page_type, duration_ms, ended_at,
       received_at, source, schema_version, measurement_version, payload, source_file
FROM actions_batch
UNION ALL
SELECT o.event_id, o.event_type, o.event_time, o.impression_id,
       JSON_VALUE(o.payload, '$.page_type'), LAX_INT64(o.payload.duration_ms),
       SAFE.TIMESTAMP(JSON_VALUE(o.payload, '$.ended_at')),
       SAFE.TIMESTAMP(JSON_VALUE(o.payload, '$.received_at')),
       JSON_VALUE(o.payload, '$.source'), JSON_VALUE(o.payload, '$.schema_version'),
       JSON_VALUE(o.payload, '$.measurement_version'), o.payload, o.source_file
FROM `{clean_orphan_actions}` o
WHERE o.event_date >= @min_date
  AND o.event_id NOT IN (SELECT event_id FROM actions_batch);

CREATE TEMP TABLE joined AS
SELECT a.*, s.reader_id, s.article_id, s.request_id, s.position, s.served_at
FROM action_candidates a
LEFT JOIN (
  SELECT impression_id, reader_id, article_id, request_id, position, served_at
  FROM `{clean_served}` WHERE served_date >= @min_date
) s USING (impression_id);

-- 4. fraud click 기준 시각: 그 impression의 첫 exposure 시작 (이미 clean에 있는 것 포함), 없으면 served_at.
CREATE TEMP TABLE first_exposure AS
SELECT impression_id, MIN(event_time) AS first_exposure_at
FROM (
  SELECT impression_id, event_time FROM joined WHERE event_type = 'exposure'
  UNION ALL
  SELECT impression_id, event_time FROM `{clean_actions}`
  WHERE event_date >= @min_date AND event_type = 'exposure'
    AND impression_id IN (SELECT impression_id FROM joined)
)
GROUP BY impression_id;

CREATE TEMP TABLE judged AS
SELECT j.*,
  CASE
    WHEN COALESCE(j.event_type, '') NOT IN ('exposure', 'article_open', 'article_close', 'like', 'unlike')
      OR COALESCE(j.page_type, '') NOT IN ('list', 'detail')
      OR j.impression_id IS NULL THEN 'invalid_event'
    WHEN j.reader_id IS NULL THEN 'orphan'
    WHEN j.event_type = 'article_open'
      AND TIMESTAMP_DIFF(j.event_time, COALESCE(f.first_exposure_at, j.served_at), MICROSECOND) < 100000
      THEN 'fraud_click'
    ELSE 'ok'
  END AS verdict
FROM joined j LEFT JOIN first_exposure f USING (impression_id);

-- 5. 결과별로 넣는다.
MERGE `{clean_actions}` T
USING (SELECT * FROM judged WHERE verdict = 'ok') S
ON T.event_id = S.event_id AND T.event_date >= @min_date
WHEN NOT MATCHED THEN
  INSERT (event_id, event_type, event_time, event_date, impression_id, page_type, duration_ms,
          ended_at, reader_id, article_id, request_id, position, served_at, is_bot, received_at,
          source, schema_version, measurement_version)
  VALUES (S.event_id, S.event_type, S.event_time, DATE(S.event_time, 'Asia/Seoul'), S.impression_id,
          S.page_type, S.duration_ms, S.ended_at, S.reader_id, S.article_id, S.request_id,
          S.position, S.served_at, FALSE, S.received_at, S.source, S.schema_version,
          S.measurement_version);

MERGE `{clean_rejected_actions}` T
USING (SELECT * FROM judged WHERE verdict IN ('fraud_click', 'invalid_event')) S
ON T.event_id = S.event_id AND T.event_date >= @min_date
WHEN NOT MATCHED THEN
  INSERT (event_id, reason, event_type, event_time, event_date, impression_id, payload,
          source_file, rejected_at)
  VALUES (S.event_id, S.verdict, S.event_type, S.event_time, DATE(S.event_time, 'Asia/Seoul'),
          S.impression_id, S.payload, S.source_file, CURRENT_TIMESTAMP());

MERGE `{clean_orphan_actions}` T
USING (SELECT * FROM judged WHERE verdict = 'orphan') S
ON T.event_id = S.event_id AND T.event_date >= @min_date
WHEN NOT MATCHED THEN
  INSERT (event_id, event_type, event_time, event_date, impression_id, payload, source_file,
          first_seen_at)
  VALUES (S.event_id, S.event_type, S.event_time, DATE(S.event_time, 'Asia/Seoul'),
          S.impression_id, S.payload, S.source_file, CURRENT_TIMESTAMP());

-- 짝을 찾아 판정이 끝난 orphan은 지운다.
DELETE FROM `{clean_orphan_actions}`
WHERE event_date >= @min_date
  AND event_id IN (SELECT event_id FROM judged WHERE verdict != 'orphan');

SELECT verdict, COUNT(*) AS n FROM judged GROUP BY verdict ORDER BY verdict;
