"""고장났수다 - 예측 고정과 채점

  python -m guardian.freeze freeze          # 지금 데이터로 예측을 뽑아 predictions/<기준시각>/ 에 저장 → 깃허브에 올려 고정
  python -m guardian.freeze score <폴더>    # 고정 시각부터 7일 동안의 실제 데이터로 채점 → <폴더>/score/ 에 저장

고정하는 것
  hidden_top20.csv   숨은 고장(B1·B2) 경보 상위 20대  ← 채점의 주인공
  repair_top20.csv   수리 트랙 경보 상위 20대(통신두절·운영불가 등, 이미 상태로 확인된 고장)
  control20.csv      비교군: '정상'으로 본 충전기 중 무작위 20대(시드 고정, 숨은 고장과 비슷한 이용량 조건)
  all_flags.csv      공공 충전기 전체 판정표
  SCORING.md         채점 기준 (고정 후 바꾸지 않음)
  manifest.json      데이터 기준 시각, 코드·파일 지문(md5)
"""
import hashlib
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from guardian.diagnose import DATA, ERR, PUBLIC, diagnose, load, sessions

ROOT = "predictions"
SEED = 20261109
N = 20
WINDOW_D = 7
QUICK_H = 24
CONTROL_NAME = {"적중-계속 멈춤": "계속 0회", "적중-운영사 확인": "고장 처리됨", "오탐": "정상 사용",
                "판단 보류": "판단 보류"}
BUSI = {"JD": "제주특별자치도", "ME": "환경부", "JC": "제주에너지공사", "JE": "제주전기자동차서비스", "JT": "제주테크노파크"}
COLS = ["순위", "statId", "chgerId", "statNm", "운영기관", "fast", "stat", "code", "판정", "grade", "why", "방치_일",
        "lam", "p_hid", "raw", "점수", "addr"]
NAME = {"A1": "운영불가", "A2": "통신두절", "A3": "간헐통신", "A4": "충전고착", "B1": "숨은고장(같은 충전소 비교)",
        "B2": "숨은고장(주변 비교)", "B3": "실패충전 반복"}

SCORING = """# 채점 기준 (11/9 17:30 고정, 이후 변경하지 않음)

시험 기간: 고정한 데이터 기준 시각부터 7일. 데이터는 30분마다 수집한 환경공단 API 상태.
'정상 충전' = 마지막 충전 시작 시각이 새로 바뀌고, 충전 길이가 5분 이상인 것.

## 숨은 고장 TOP 20과 비교군 20 (충전기마다 하나로 판정)
| 결과 | 기준 |
|---|---|
| 적중-계속 멈춤 | 7일 동안 정상 충전 0회, 비교 대상(같은 충전소의 같은 종류 충전기, 없으면 10km 안 가장 가까운 같은 종류 30대)은 정상 충전 1회 이상 |
| 적중-운영사 확인 | 기간 중 운영중지·점검중으로 바뀌거나 통신이상이 생김(정상 충전 재개 여부와 무관) |
| 오탐 | 고정 후 24시간 안에, 상태 변화 없이 정상 충전 |
| 판단 보류 | 24시간 뒤에 상태 변화 없이 정상 충전(원격 재부팅 수리인지 원래 정상인지 구분 불가) / 비교 대상 충전기도 7일 동안 0회 / 데이터 없음 |

적중률 = 적중 / (적중 + 오탐). 판단 보류는 분모에서 빼고 개수를 따로 보고한다.
비교군은 같은 기준으로 판정하되 이름을 '계속 0회 / 고장 처리됨 / 정상 사용 / 판단 보류'로 부르고, '계속 0회' 비율을 숨은 고장의 '적중-계속 멈춤' 비율과 비교한다.

## 수리 TOP 20 (이미 상태 신호로 확인된 고장)
적중을 따지지 않는다. 7일 뒤 '여전히 같은 문제로 방치'인 비율(방치 지속률)과 '정상 충전 재개(복구)' 비율을 보고한다.
"""


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _table(df):
    t = df.reset_index(drop=True).copy()
    t.insert(0, "순위", t.index + 1)
    t["운영기관"] = t["busiId"].map(BUSI)
    t["판정"] = t["code"].map(NAME).fillna("정상")
    return t[[c for c in COLS if c in t.columns]]


def freeze(root=ROOT):
    d = diagnose()
    a = d.attrs
    pub = d[d["busiId"].isin(PUBLIC)].copy()
    rep = pub[(pub["트랙"] == "수리")].sort_values("raw", ascending=False)
    hidden = rep[rep["code"].isin(["B1", "B2"]) & (rep["grade"] == "경보")].head(N)
    repair = rep[rep["grade"] == "경보"].head(N)
    lam_min = float(hidden["lam"].min()) if len(hidden) else 0.0
    pool = pub[(pub["code"] == "") & (pub["stat"] == "2") & (pub["limitYn"] != "Y") & pub["lam"].notna()
               & (pub["lam"] >= lam_min) & ~pub["statNm"].fillna("").str.contains("버스|차고지|회차지|여객")]
    pool = pool.sort_values(["statId", "chgerId"])
    rng = np.random.default_rng(SEED)
    control = pool.iloc[np.sort(rng.choice(len(pool), size=min(N, len(pool)), replace=False))]

    stamp = f"{a['now']:%Y%m%d_%H%M}"
    out = os.path.join(root, stamp)
    os.makedirs(out, exist_ok=True)
    files = {"hidden_top20.csv": _table(hidden), "repair_top20.csv": _table(repair),
             "control20.csv": _table(control), "all_flags.csv": _table(pub[pub["code"] != ""].sort_values("raw", ascending=False))}
    for name, t in files.items():
        t.to_csv(os.path.join(out, name), index=False, encoding="utf-8-sig")
    with open(os.path.join(out, "SCORING.md"), "w", encoding="utf-8") as f:
        f.write(SCORING)
    man = {"데이터_기준시각": f"{a['now']:%Y-%m-%d %H:%M}", "수집_시작": f"{a['t_first']:%Y-%m-%d %H:%M}",
           "스냅샷_수": int(a["n_snaps"]), "고정_실행시각": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "점수_기준값(score_ref)": a["score_ref"], "비교군_시드": SEED, "비교군_이용량_하한(lam)": lam_min,
           "비교군_후보수": int(len(pool)), "시험기간_일": WINDOW_D,
           "코드_md5": {p: _md5(os.path.join("guardian", p)) for p in ["diagnose.py", "judge.py", "freeze.py"]},
           "파일_md5": {n: _md5(os.path.join(out, n)) for n in [*files, "SCORING.md"]},
           "대수": {"숨은고장": len(hidden), "수리": len(repair), "비교군": len(control), "전체판정": len(files["all_flags.csv"])}}
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)
    return out, man


# ---------------- 채점 ----------------
def _refs(r, info, fast_of):
    """비교 대상: 같은 충전소의 같은 종류 충전기, 없으면 10km 안 가장 가까운 같은 종류 30대 (진단과 같은 방식)"""
    k = (r.statId, r.chgerId)
    same = info[(info["statId"] == r.statId) & (info["chgerId"] != r.chgerId) & (info["fast"] == fast_of.get(k))]
    if len(same):
        return set(zip(same["statId"], same["chgerId"])), "같은 충전소 다른 충전기"
    me = info[(info["statId"] == r.statId) & (info["chgerId"] == r.chgerId)]
    if me.empty or pd.isna(me["lat"].iloc[0]):
        return set(), "비교 대상 없음"
    cand = info[(info["statId"] != r.statId) & (info["fast"] == fast_of.get(k))].dropna(subset=["lat", "lng"])
    dist = np.hypot((cand["lat"] - me["lat"].iloc[0]) * 111.0, (cand["lng"] - me["lng"].iloc[0]) * 92.6)
    nb = cand[dist <= 10].assign(dist=dist).nsmallest(30, "dist")
    return set(zip(nb["statId"], nb["chgerId"])), f"주변 {len(nb)}대"


def _judge_one(r, s_win, succ, t0, t_end, fast_of, info):
    k = (r.statId, r.chgerId)
    st = s_win[(s_win["statId"] == r.statId) & (s_win["chgerId"] == r.chgerId)]
    if st.empty:
        return "판단 보류", "시험 기간 데이터 없음", None
    bad = st[st["stat"].isin(ERR | {"4", "5"})]
    first_bad = bad["t"].min() if len(bad) else pd.NaT
    mine = succ[(succ["statId"] == r.statId) & (succ["chgerId"] == r.chgerId)]
    first_ok = mine["start"].min() if len(mine) else pd.NaT
    refs, ref_name = _refs(r, info, fast_of)
    sib = succ[[(a, b) in refs for a, b in zip(succ["statId"], succ["chgerId"])]] if refs else succ.iloc[0:0]
    if pd.notna(first_bad):
        what = "운영중지·점검중" if bad["stat"].isin(["4", "5"]).any() else "통신이상"
        return "적중-운영사 확인", f"{first_bad:%m/%d %H:%M} {what}", first_ok
    if pd.isna(first_ok):
        if len(sib):
            return "적중-계속 멈춤", f"0회, 비교 대상({ref_name}) {len(sib)}회", None
        return "판단 보류", f"비교 대상({ref_name})도 0회", None
    if first_ok - t0 <= pd.Timedelta(hours=QUICK_H):
        return "오탐", f"{first_ok:%m/%d %H:%M} 정상 충전(24시간 이내)", first_ok
    return "판단 보류", f"{first_ok:%m/%d %H:%M} 정상 충전(24시간 이후, 상태 변화 없음)", first_ok


def score(folder):
    man = json.load(open(os.path.join(folder, "manifest.json"), encoding="utf-8"))
    t0 = pd.Timestamp(man["데이터_기준시각"])
    t_end = t0 + pd.Timedelta(days=WINDOW_D)
    s, info = load()
    s_win = s[(s["t"] > t0) & (s["t"] <= t_end)]
    last = s_win["t"].max() if len(s_win) else t0
    ss = sessions(s)
    succ = ss[(ss["start"] > t0) & (ss["start"] <= t_end) & ~ss["short"]]
    fast_of = {(x.statId, x.chgerId): bool(x.fast) for x in info.itertuples()}
    out = os.path.join(folder, "score")
    os.makedirs(out, exist_ok=True)
    res = {}
    for name in ["hidden_top20", "control20", "repair_top20"]:
        t = pd.read_csv(os.path.join(folder, f"{name}.csv"), dtype={"statId": str, "chgerId": str})
        rows = []
        for r in t.itertuples():
            g, why, first_ok = _judge_one(r, s_win, succ, t0, t_end, fast_of, info)
            if name == "control20":   # 비교군은 '고장 예측'이 아니므로 이름만 바꿔 부른다
                g = CONTROL_NAME.get(g, g)
            if name == "repair_top20":
                g = "복구(정상 충전 재개)" if pd.notna(first_ok) else "계속 방치"
                why = f"{first_ok:%m/%d %H:%M} 첫 정상 충전" if pd.notna(first_ok) else "7일 동안 정상 충전 없음"
            rows.append({**r._asdict(), "결과": g, "근거_채점": why})
        df = pd.DataFrame(rows).drop(columns=["Index"])
        df.to_csv(os.path.join(out, f"{name}_결과.csv"), index=False, encoding="utf-8-sig")
        res[name] = df["결과"].value_counts().to_dict()
    h = res["hidden_top20"]
    hit = sum(v for k, v in h.items() if k.startswith("적중"))
    fp = h.get("오탐", 0)
    c = res["control20"]
    summ = {"고정_데이터_기준": man["데이터_기준시각"], "채점_데이터_마지막": f"{last:%Y-%m-%d %H:%M}",
            "시험기간_완료": bool(last >= t_end - pd.Timedelta(minutes=30)),
            "숨은고장_결과": h, "숨은고장_적중률(%)": round(100 * hit / (hit + fp), 1) if hit + fp else None,
            "비교군_결과": c, "비교군_계속멈춤_비율(%)": round(100 * c.get("계속 0회", 0) / max(1, sum(c.values())), 1),
            "숨은고장_계속멈춤_비율(%)": round(100 * h.get("적중-계속 멈춤", 0) / max(1, sum(h.values())), 1),
            "수리TOP20_결과": res["repair_top20"]}
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, ensure_ascii=False, indent=2)
    return out, summ


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "freeze"
    if cmd == "freeze":
        o, m = freeze()
        print("고정 완료:", o)
        print(json.dumps(m, ensure_ascii=False, indent=2))
    elif cmd == "score":
        o, m = score(sys.argv[2])
        print("채점 완료:", o)
        print(json.dumps(m, ensure_ascii=False, indent=2))
