"""충전기 지킴이 - 1단계 고장 판정 + 허브(2025.9) 비교

사용법:  python -m guardian.judge
결과:    out/judge_YYYYMMDD_HHMM.csv  (판정된 충전기 목록)
         out/hub_compare.csv          (1년 전 허브 목록 대비 변화)
"""
import glob
import gzip
import json
import math
import os
from datetime import datetime

import pandas as pd

DATA = "data"
OUT = "out"
PUBLIC = ["JD", "ME", "JC", "JE", "JT"]          # 공공 운영기관
SLOW = {"02", "08"}                                # 완속 타입
STAT_NAME = {"0": "알수없음", "1": "통신이상", "2": "사용가능", "3": "충전중",
             "4": "운영중지", "5": "점검중", "6": "예약중", "9": "상태미확인"}


# ---------- 데이터 읽기 ----------
def _read_gz(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    return pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False) for f in files],
                     ignore_index=True)


def load_latest():
    """가장 최근 수집 시점의 상태 + 충전기 기본정보"""
    status = _read_gz(f"{DATA}/status/*.csv.gz")
    t = status["collected_at"].max()
    now = status[status["collected_at"] == t].drop_duplicates(["statId", "chgerId"])
    master = _read_gz(f"{DATA}/master/*.csv.gz")
    master = master.sort_values("collected_at").drop_duplicates(["statId", "chgerId"], keep="last")
    info = master[["statId", "chgerId", "statNm", "busiNm", "chgerType", "output",
                   "addr", "lat", "lng", "limitYn", "delYn"]]
    df = now.merge(info, on=["statId", "chgerId"], how="left")
    return df, datetime.strptime(t, "%Y-%m-%d %H:%M")


def _ts(s):
    return pd.to_datetime(s, format="%Y%m%d%H%M%S", errors="coerce")


# ---------- 1단계 판정 ----------
def judge(df, t):
    df = df.copy()
    df["fast"] = ~df["chgerType"].isin(SLOW)
    df["상태"] = df["stat"].map(STAT_NAME).fillna("?")
    df["상태유지_시간"] = ((t - _ts(df["statUpdDt"])).dt.total_seconds() / 3600).round(1)
    df["마지막충전_일"] = ((t - _ts(df["lastTsdt"])).dt.total_seconds() / 86400).round(1)
    df["현재충전_시간"] = ((t - _ts(df["nowTsdt"])).dt.total_seconds() / 3600).round(1)

    h = df["상태유지_시간"]
    rules = [
        (df["stat"].isin(["4", "5"]), "확정고장", "운영중지/점검중"),
        ((df["stat"] == "1") & (h >= 24), "추정고장", "통신이상 24시간 이상"),
        (df["stat"].isin(["0", "9"]) & (h >= 24), "추정고장", "상태미확인 24시간 이상"),
        ((df["stat"] == "3") & df["fast"] & (df["현재충전_시간"] >= 3), "상태고착", "급속인데 충전중 3시간 이상"),
        ((df["stat"] == "2") & (df["마지막충전_일"] >= 30) & (df["limitYn"] != "Y"),
         "숨은고장후보", "사용가능인데 30일 이상 충전 0건 (2단계에서 정밀화)"),
    ]
    df["판정"] = ""
    df["사유"] = ""
    for cond, label, why in rules:
        m = cond.fillna(False) & (df["판정"] == "")
        df.loc[m, "판정"] = label
        df.loc[m, "사유"] = why

    # 방치 기간: 고장 판정은 상태 유지 시간, 숨은고장후보는 마지막 충전 이후
    days = (h / 24).where(df["판정"] != "숨은고장후보", df["마지막충전_일"])
    df["방치_일"] = days.where(df["판정"] != "").round(1)
    df["방치등급"] = pd.cut(df["방치_일"], [-1, 7, 30, 365, 1e9],
                         labels=["", "지연(7일+)", "장기(30일+)", "심각(1년+)"]).astype(str).replace("nan", "")
    return df


# ---------- 허브 2025.9 비교 ----------
def _dist(a, b, c, d):
    return math.hypot((a - c) * 111000, (b - d) * 111000 * math.cos(math.radians(33.4)))


def hub_compare(df_all):
    hub = json.load(open(f"{DATA}/hub/stations.json", encoding="utf-8"))
    st = (df_all.assign(lat=pd.to_numeric(df_all["lat"], errors="coerce"),
                        lng=pd.to_numeric(df_all["lng"], errors="coerce"))
          .groupby("statId").agg(busiId=("busiId", "first"), statNm=("statNm", "first"),
                                 lat=("lat", "first"), lng=("lng", "first"), n=("chgerId", "count"))
          .reset_index())
    grid = {}
    for r in st.itertuples():
        if pd.notna(r.lat):
            grid.setdefault((round(r.lat * 200), round(r.lng * 200)), []).append(r)
    rows = []
    for h in hub:
        la, ln = float(h["latitude"]), float(h["longitude"])
        k = (round(la * 200), round(ln * 200))
        near = [r for dx in (-1, 0, 1) for dy in (-1, 0, 1) for r in grid.get((k[0] + dx, k[1] + dy), [])
                if _dist(la, ln, r.lat, r.lng) < 100]
        same = [r for r in near if r.busiId == h["provider"]]
        before = h["chargerCount"] + h["quickChargerCount"]
        now = sum(r.n for r in same)
        if same and now >= before:
            kind = "유지"
        elif same:
            kind = "대수감소"
        elif near:
            kind = "운영기관변경"
        else:
            kind = "소멸(100m내 없음)"
        rows.append(dict(허브_충전소=h["chargingPlace"], 운영기관=h["provider"], 대수_2025_9=before,
                         대수_현재=now, 변화=kind, 현재_충전소=", ".join(r.statNm for r in same or near),
                         운영시간=f'{h["startTime"]}~{h["endTime"]}', 휴무=h["holiday"],
                         주차료=h["parkingFeeFlag"], lat=la, lng=ln))
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    df_all, t = load_latest()
    df_all = df_all[df_all["delYn"] != "Y"]
    pub = judge(df_all[df_all["busiId"].isin(PUBLIC)], t)
    flagged = pub[pub["판정"] != ""].sort_values(["판정", "방치_일"], ascending=[True, False])
    cols = ["busiId", "statId", "chgerId", "statNm", "addr", "fast", "상태", "판정", "사유",
            "방치_일", "방치등급", "상태유지_시간", "마지막충전_일", "lat", "lng"]
    path = f"{OUT}/judge_{t:%Y%m%d_%H%M}.csv"
    flagged[cols].to_csv(path, index=False, encoding="utf-8-sig")

    print(f"기준 시각 {t:%Y-%m-%d %H:%M} · 공공 충전기 {len(pub)}대")
    print(pd.crosstab(pub["busiId"], pub["판정"].replace("", "정상"), margins=True).to_string())
    print("\n방치 등급 (판정된 것만):")
    print(flagged["방치등급"].replace("", "7일 이하").value_counts().to_string())
    print(f"\n저장: {path} ({len(flagged)}대)")

    hc = hub_compare(df_all)
    hc.to_csv(f"{OUT}/hub_compare.csv", index=False, encoding="utf-8-sig")
    print("\n허브(2025.9) 대비 변화 - 공공 운영기관:")
    print(pd.crosstab(hc["운영기관"], hc["변화"]).loc[[p for p in PUBLIC if p in set(hc["운영기관"])]].to_string())


if __name__ == "__main__":
    main()
