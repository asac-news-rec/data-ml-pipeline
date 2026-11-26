-- [파이프라인 구간: 집계 → 피처] clean + feat 기사 표 → Feast 원천 표 4개.
-- features/feast/prepare_sources.py의 article_table()·build()와 같은 결과를 BigQuery에서 만든다.
-- {이름}은 transform/build_feast_sources.py가 표 경로로 바꾼다. 매번 전체를 다시 만든다(MVP).
-- @data_valid_from: 이 시각 이후 행동만 쓴다(시험 로그 빼기). 값이 없으면 1970-01-01이라 전부 쓴다.
--
-- 사건: 독자별로 노출(impression마다 첫 exposure, kind 0)과 클릭(article_open, kind 1)을 시간순으로 본다.
--       같은 시각이면 노출을 먼저 처리한다.
-- 피처 행 시각 = 사건 시각 + 1µs. 같은 (키, 시각)이면 사건 순서상 마지막 상태만 남긴다.
-- u_emb(t) = Σ wᵢeᵢ / Σ wᵢ, wᵢ = 0.5^((t - tᵢ)/24h). 분자·분모의 공통 인수를 약분하면
--           wᵢ = 2^((tᵢ - t₀)/24h)로 바꿀 수 있다(t₀ = 그 독자의 첫 클릭). 그래서 누적합으로 계산한다.

-- 0. 카테고리 코드: backend generate/news_category.csv 대분류의 가나다순 (로컬 학습 categories.json과 같다)
CREATE TEMP TABLE category_code AS
SELECT category, code
FROM UNNEST(['건강', '경제', '국제', '금융', '기업', '문화', '부동산', '사회', '스포츠', '연예',
             '정치', '증권', '테크·과학']) AS category WITH OFFSET AS code;

-- 1. 기사마다 가장 최근에 만든 임베딩 하나
CREATE TEMP TABLE article_vec AS
SELECT article_id, embedding
FROM `{article_embedding}`
QUALIFY ROW_NUMBER() OVER (PARTITION BY article_id ORDER BY created_at DESC) = 1;

CREATE OR REPLACE TABLE `{out_article_profile}` AS
SELECT m.article_id,
       m.published_at AS event_timestamp,       -- 기사 피처는 발행 시각부터 존재한다
       v.embedding AS a_emb,
       c.code AS category,
       UNIX_MICROS(m.published_at) / 1e6 AS published_ts
FROM `{article_meta}` m
JOIN article_vec v USING (article_id)
LEFT JOIN category_code c USING (category);

-- 2. 독자별 사건 목록
CREATE TEMP TABLE events AS
WITH exposures AS (
  SELECT reader_id, article_id, MIN(event_time) AS t, 0 AS kind, impression_id AS tie
  FROM `{clean_actions}`
  WHERE event_type = 'exposure' AND event_time >= @data_valid_from
  GROUP BY reader_id, article_id, impression_id
), clicks AS (
  SELECT reader_id, article_id, event_time AS t, 1 AS kind, event_id AS tie
  FROM `{clean_actions}`
  WHERE event_type = 'article_open' AND event_time >= @data_valid_from
)
SELECT *, ROW_NUMBER() OVER (PARTITION BY reader_id ORDER BY t, kind, tie) AS seq
FROM (SELECT * FROM exposures UNION ALL SELECT * FROM clicks);

-- 3. reader_activity: 노출마다 지금까지의 노출 수
CREATE OR REPLACE TABLE `{out_reader_activity}` AS
SELECT reader_id, event_timestamp, hist_exposures
FROM (
  SELECT reader_id,
         TIMESTAMP_ADD(t, INTERVAL 1 MICROSECOND) AS event_timestamp,
         COUNT(*) OVER (PARTITION BY reader_id ORDER BY seq) AS hist_exposures,
         seq
  FROM events WHERE kind = 0
)
QUALIFY ROW_NUMBER() OVER (PARTITION BY reader_id, event_timestamp ORDER BY seq DESC) = 1;

-- 4. reader_article: 사건마다 그 (독자, 기사)를 지금까지 열었나·봤나
CREATE OR REPLACE TABLE `{out_reader_article}` AS
SELECT reader_id, article_id, event_timestamp, already_opened, already_seen
FROM (
  SELECT reader_id, article_id,
         TIMESTAMP_ADD(t, INTERVAL 1 MICROSECOND) AS event_timestamp,
         MAX(kind) OVER w AS already_opened,
         MAX(1 - kind) OVER w AS already_seen,
         seq
  FROM events
  WINDOW w AS (PARTITION BY reader_id, article_id ORDER BY seq)
)
QUALIFY ROW_NUMBER() OVER (PARTITION BY reader_id, article_id, event_timestamp ORDER BY seq DESC) = 1;

-- 5. reader_profile: 클릭마다 u_emb·클릭 수·마지막 클릭 시각
CREATE TEMP TABLE clicks AS
SELECT e.reader_id, e.seq, e.t,
       POW(2, (UNIX_MICROS(e.t) - MIN(UNIX_MICROS(e.t)) OVER (PARTITION BY e.reader_id)) / 1e6
              / 3600 / 24) AS w,
       v.embedding
FROM events e
LEFT JOIN article_vec v USING (article_id)
WHERE e.kind = 1;

CREATE TEMP TABLE click_vec AS
SELECT reader_id, seq, dim,
       SUM(w * x) OVER (PARTITION BY reader_id, dim ORDER BY seq) AS num
FROM clicks, UNNEST(embedding) AS x WITH OFFSET AS dim;

CREATE OR REPLACE TABLE `{out_reader_profile}` AS
WITH per_click AS (
  SELECT reader_id, seq, t,
         COUNT(*) OVER (PARTITION BY reader_id ORDER BY seq) AS hist_clicks,
         SUM(IF(embedding IS NULL OR ARRAY_LENGTH(embedding) = 0, 0, w))
           OVER (PARTITION BY reader_id ORDER BY seq) AS den
  FROM clicks
), vec AS (
  SELECT reader_id, seq, ARRAY_AGG(num ORDER BY dim) AS num
  FROM click_vec GROUP BY reader_id, seq
)
SELECT p.reader_id,
       TIMESTAMP_ADD(p.t, INTERVAL 1 MICROSECOND) AS event_timestamp,
       IF(p.den > 0, ARRAY(SELECT n / p.den FROM UNNEST(v.num) AS n WITH OFFSET AS i ORDER BY i),
          NULL) AS u_emb,
       p.hist_clicks,
       UNIX_MICROS(p.t) / 1e6 AS last_click_ts
FROM per_click p
LEFT JOIN vec v USING (reader_id, seq)
QUALIFY ROW_NUMBER() OVER (PARTITION BY p.reader_id, event_timestamp ORDER BY p.seq DESC) = 1;

SELECT 'article_profile' AS t, COUNT(*) AS n FROM `{out_article_profile}`
UNION ALL SELECT 'reader_profile', COUNT(*) FROM `{out_reader_profile}`
UNION ALL SELECT 'reader_activity', COUNT(*) FROM `{out_reader_activity}`
UNION ALL SELECT 'reader_article', COUNT(*) FROM `{out_reader_article}`;
