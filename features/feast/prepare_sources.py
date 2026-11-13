"""가짜 로그(raw) → Feast가 읽는 원천 표(parquet) 5개를 만든다.

실서비스에서는 이 스크립트가 하는 일을 Airflow 정제·집계 단계가 BigQuery 표로 만든다.
여기서는 로컬 parquet으로 같은 모양의 표를 만든다.

실행:  .venv/bin/python prepare_sources.py --src <virtual_reader 폴더>
입력:  <src>/fake_logs/served, actions, articles.jsonl
       <src>/../article_embedding/data/article_embeddings.npy, article_ids.json
결과:  data/article_profile.parquet   기사 1건 = 1행 (a_emb 768, category, published_ts)
       data/reader_profile.parquet    클릭 1건 = 1행 (그 클릭 직후의 u_emb 768, hist_clicks, last_click_ts)
       data/reader_activity.parquet   노출 1건 = 1행 (그 노출 직후의 hist_exposures)
       data/reader_article.parquet    (독자, 기사) 상태가 바뀔 때마다 1행 (already_opened, already_seen)
       data/exposures.parquet         학습 표의 뼈대(entity_df). 유효 노출 1건 = 1행 + 라벨

피처 행의 event_timestamp = 사건 시각 + 1마이크로초.
Feast는 '피처 시각 <= 노출 시각'인 행 중 가장 최근 것을 붙인다(point-in-time join).
+1µs를 두면 노출과 같은 시각에 일어난 사건은 그 노출에 붙지 않는다.
make_features.py가 '같은 시각이면 노출을 먼저 처리'하는 규칙과 같은 결과다.

u_emb 스냅샷을 클릭 시각에만 남겨도 되는 이유:
  u_emb(t) = Σ w_i e_i / Σ w_i,  w_i = 0.5 ** ((t - t_i) / 24h)
  t가 늘면 모든 w_i에 같은 수(0.5 ** (Δt / 24h))가 곱해지고, 분자·분모에서 약분된다.
  그래서 u_emb는 새 클릭이 있을 때만 바뀐다. 노출 시각마다 다시 계산할 필요가 없다.
"""
import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data"
HALF_LIFE_H = 24.0
VALID_FROM = datetime(2026, 11, 6, tzinfo=timezone(timedelta(hours=9)))
EPS = pd.Timedelta(microseconds=1)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def ts(text):
    return datetime.fromisoformat(text).timestamp()


def to_time(seconds):
    # 로그 시각은 ms 단위. float 오차(…749000072)가 남지 않게 µs 정수로 바꾼다.
    return pd.Timestamp(round(seconds * 1_000_000), unit="us", tz="UTC")


def load(src):
    logs = src / "fake_logs"
    served = {}
    for path in sorted((logs / "served").rglob("*.jsonl")):
        for row in read_jsonl(path):
            served[row["impression_id"]] = row
    actions = []
    for path in sorted((logs / "actions").rglob("*.jsonl")):
        actions.extend(read_jsonl(path))
    articles = {a["article_id"]: a for a in read_jsonl(logs / "articles.jsonl")}
    emb_dir = src.parent / "article_embedding" / "data"
    ids = json.loads((emb_dir / "article_ids.json").read_text(encoding="utf-8"))
    vectors = np.load(emb_dir / "article_embeddings.npy").astype(np.float32)
    emb = {aid: vectors[i] for i, aid in enumerate(ids)}
    return served, actions, articles, emb


def article_table(articles, emb):
    categories = sorted({a["category"] for a in articles.values()})
    code = {c: i for i, c in enumerate(categories)}
    rows = []
    for a in articles.values():
        published = ts(a["published_at"])
        rows.append({
            "article_id": a["article_id"],
            # 기사 피처는 발행 시각부터 존재한다
            "event_timestamp": to_time(published),
            "a_emb": emb[a["article_id"]],
            "category": code[a["category"]],
            "published_ts": published,
        })
    return pd.DataFrame(rows), categories


def history_tables(served, actions):
    """독자별 사건을 시간순으로 따라가며, 상태가 바뀔 때마다 스냅샷 행을 남긴다."""
    opened_imps = {a["impression_id"] for a in actions if a["event_type"] == "article_open"}
    events = defaultdict(list)
    seen_exposure = set()
    for a in actions:
        imp = served.get(a["impression_id"])
        if imp is None:
            continue
        if a["event_type"] == "exposure" and a["impression_id"] not in seen_exposure:
            seen_exposure.add(a["impression_id"])
            events[imp["reader_id"]].append((ts(a["event_time"]), 0, imp))
        elif a["event_type"] == "article_open":
            events[imp["reader_id"]].append((ts(a["event_time"]), 1, imp))
    return events, opened_imps


def build(served, actions, emb):
    events, opened_imps = history_tables(served, actions)
    profile, activity, pair, exposures = [], [], [], []
    dim = len(next(iter(emb.values())))
    for reader_id, evs in events.items():
        evs.sort(key=lambda e: (e[0], e[1]))
        vec_sum, weight, t_ref = np.zeros(dim), 0.0, None
        clicks = n_exposures = 0
        opened, seen = set(), set()
        for t, kind, imp in evs:
            aid = imp["article_id"]
            if kind == 0:   # 유효 노출: 학습 표 1행 + 노출 수 스냅샷
                exposures.append({
                    "row_id": len(exposures),
                    "impression_id": imp["impression_id"],
                    "reader_id": reader_id,
                    "article_id": aid,
                    "event_timestamp": to_time(t),
                    "request_ts": t,
                    "position": imp["position"],
                    "label": int(imp["impression_id"] in opened_imps),
                    "split": "valid" if t >= VALID_FROM.timestamp() else "train",
                })
                n_exposures += 1
                seen.add(aid)
                activity.append({"reader_id": reader_id, "event_timestamp": to_time(t) + EPS,
                                 "hist_exposures": n_exposures})
            else:           # 클릭: 유저 임베딩 갱신
                if t_ref is not None and t > t_ref:
                    factor = 0.5 ** ((t - t_ref) / 3600 / HALF_LIFE_H)
                    vec_sum *= factor
                    weight *= factor
                t_ref = t
                vec_sum += emb[aid]
                weight += 1.0
                clicks += 1
                opened.add(aid)
                profile.append({"reader_id": reader_id, "event_timestamp": to_time(t) + EPS,
                                "u_emb": (vec_sum / weight).astype(np.float32),
                                "hist_clicks": clicks, "last_click_ts": t})
            pair.append({"reader_id": reader_id, "article_id": aid,
                         "event_timestamp": to_time(t) + EPS,
                         "already_opened": int(aid in opened), "already_seen": int(aid in seen)})
    return (pd.DataFrame(profile), pd.DataFrame(activity), pd.DataFrame(pair),
            pd.DataFrame(exposures))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True, help="virtual_reader 폴더")
    args = parser.parse_args()

    served, actions, articles, emb = load(args.src)
    article, categories = article_table(articles, emb)
    profile, activity, pair, exposures = build(served, actions, emb)

    DATA.mkdir(exist_ok=True)
    tables = {"article_profile": article, "reader_profile": profile,
              "reader_activity": activity, "reader_article": pair, "exposures": exposures}
    for name, df in tables.items():
        df.to_parquet(DATA / f"{name}.parquet", index=False)
        print(f"{name:16s} {len(df):>7,}행  {list(df.columns)}")
    (DATA / "categories.json").write_text(json.dumps(categories, ensure_ascii=False),
                                          encoding="utf-8")


if __name__ == "__main__":
    main()
