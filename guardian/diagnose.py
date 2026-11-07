"""고장났수다 - 고장 진단 v2 (평가서 4건 반영)

사용법:  python -m guardian.diagnose
결과:    out/diagnosis_YYYYMMDD_HHMM.csv   (충전기별 판정 + 근거 문장 + 점검 우선순위)

구조
  0. 품질 게이트   : 수집 공백·이력 없음·이용 제한·전용 충전기는 판정 보류/제외
  A. 공식 상태     : A1 운영불가 / A2 통신두절(6h 주의, 24h 경보) / A3 간헐 통신(플래핑) / A4 고착(장시간 점유와 구분: 급속 48h·완속 72h부터 경보, 주말 휴무 기관은 주말엔 주의까지)
  B. 숨은 고장     : B1 형제 비교 / B2 이웃 비교(이질성 반영) / B3 실패 세션 반복  → BH 다중검정
  C. 묶음          : 충전소 단위 동시 장애, 운영사 일괄 갱신 이벤트
  D. 트랙 분리     : 수리 트랙(우선순위 점수) / 행정 트랙(1년+ 미사용 = 철거·목록 정리)
기준 숫자의 근거는 수집 데이터 분포(세션 길이 99퍼센타일 등)이며 상단 상수에 모아 둔다.
"""
import json
import math
import os

import numpy as np
import pandas as pd

from guardian.judge import DATA, OUT, PUBLIC, SLOW, STAT_NAME, _read_gz

# ---------------- 기준 숫자 (근거는 설명서 참고) ----------------
ERR = {"1", "0", "9"}
SHORT_MIN = 5                 # 5분 미만 세션 = 실패 의심
COMM_WARN_H, COMM_ALARM_H = 6, 24
STUCK = {True: (4, 48), False: (24, 72)}   # 급속/완속 (주의, 경보) 시간 — 세션 길이 99%: 급속 3.6h, 완속 14.5h
#   경보를 12h→48h로 올린 이유: 관용차를 퇴근 후·주말에 꽂아 두면 충전 중 표시가 이어짐(점유≠고장).
#   48h는 금요일 저녁~일요일 저녁 주차를 넘는 길이.
FLAP_WINDOW_H, FLAP_MIN_SNAPS, FLAP_ERR_RATIO = 48, 20, 0.5
CV2_SIB = 0.16                # 형제끼리도 선호도 차이 ±40% 허용 (입구 쪽 기기 선호 등)
CV2_MIN = 0.16                # 이웃 이질성 하한
NB_NEIGHBORS = 30             # 이웃 비교: 가까운 정상 충전기 30대 (반경 자동 확장, 최대 10km)
P_ALARM, P_WATCH, FDR_Q = 1e-3, 1e-2, 0.05
GHOST_DAYS = 365
DAYS_CAP = 90                 # 수리 우선순위에서 방치 일수 상한
RESTRICTED = ("전용", "출입제한")
CONF = {"A1": 1.0, "A2": 0.9, "A3": 0.8, "A4": 0.7}


def _ts(s):
    return pd.to_datetime(s, format="%Y%m%d%H%M%S", errors="coerce")


def _pct(p):
    return "0.0001% 미만" if p < 1e-6 else f"{p * 100:.2g}%"


def _days(d):
    return "400일이 넘는 기간" if d >= 399.9 else f"{d:.1f}일"


def _dur(h):
    return f"{h / 24:.0f}일째" if h >= 48 else f"{h:.0f}시간째"


def _km(lat1, lng1, lat2, lng2):
    return np.hypot((lat1 - lat2) * 111.0, (lng1 - lng2) * 111.0 * math.cos(math.radians(33.4)))


# ---------------- 운영시간 (허브 862) ----------------
def _hh(s):
    try:
        h, m = str(s).split(":")[:2]
        return int(h) + int(m) / 60
    except ValueError:
        return None


def load_hours(st):
    """충전소별 (시작시, 종료시, 주말휴무, 전용여부, 설명) — 허브 100m 이내 매칭, 없으면 24시간"""
    hub = pd.DataFrame(json.load(open(f"{DATA}/hub/stations.json", encoding="utf-8")))
    hl, hg = hub["latitude"].astype(float).values, hub["longitude"].astype(float).values
    out = {}
    for r in st.itertuples():
        d = _km(r.lat, r.lng, hl, hg)
        i = int(np.argmin(d))
        if d[i] > 0.1:
            out[r.statId] = (0.0, 24.0, False, False, "허브 목록에 없음(24시간 가정)")
            continue
        h = hub.iloc[i]
        a, b = _hh(h["startTime"]), _hh(h["endTime"])
        if a is None or b is None or b <= a:
            a, b = 0.0, 24.0
        wk = any(k in str(h["holiday"]) for k in ("토", "일 미개방", "주말"))
        rs = any(k in str(h["holiday"]) for k in RESTRICTED)
        out[r.statId] = (a, b, wk, rs, f'{h["startTime"]}~{h["endTime"]} / {h["holiday"]}')
    return out


def open_days(t0, t1, hrs):
    """t0~t1 사이 실제로 열려 있던 시간(일 단위). 400일 넘게 지난 건 400일로 자름(어차피 확률≈0)."""
    a, b, wk, _, _ = hrs
    if pd.isna(t0) or t1 <= t0:
        return 0.0
    t0 = max(t0, t1 - pd.Timedelta(days=400))
    if a == 0 and b == 24 and not wk:
        return (t1 - t0).total_seconds() / 86400
    tot, day = 0.0, t0.normalize()
    while day <= t1:
        if not (wk and day.weekday() >= 5):
            s, e = day + pd.Timedelta(hours=a), day + pd.Timedelta(hours=b)
            lo, hi = max(s, t0), min(e, t1)
            if hi > lo:
                tot += (hi - lo).total_seconds()
        day += pd.Timedelta(days=1)
    return tot / 86400


# ---------------- 데이터 ----------------
def load():
    s = _read_gz(f"{DATA}/status/*.csv.gz")
    lp = f"{DATA}/latest.csv.gz"       # 30분마다 올라오는 최신 상태가 더 새로우면 덧붙임
    if os.path.exists(lp):
        lt = pd.read_csv(lp, dtype=str, keep_default_na=False)
        if len(lt) and lt["collected_at"].max() > s["collected_at"].max():
            s = pd.concat([s, lt], ignore_index=True)
    s["t"] = pd.to_datetime(s["collected_at"])
    s = s.sort_values("t").drop_duplicates(["statId", "chgerId", "t"])
    m = _read_gz(f"{DATA}/master/*.csv.gz").sort_values("collected_at") \
        .drop_duplicates(["statId", "chgerId"], keep="last")
    info = m[["statId", "chgerId", "statNm", "chgerType", "addr", "lat", "lng", "limitYn", "delYn"]].copy()
    info["lat"] = pd.to_numeric(info["lat"], errors="coerce")
    info["lng"] = pd.to_numeric(info["lng"], errors="coerce")
    info["fast"] = ~info["chgerType"].isin(SLOW)
    return s, info


def sessions(s):
    """스냅샷에서 관측된 충전 세션(마지막 충전 시작 시각이 바뀐 것) 목록과 길이"""
    x = s[s["lastTsdt"] != ""][["statId", "chgerId", "lastTsdt", "lastTedt", "t"]].copy()
    x["start"] = _ts(x["lastTsdt"])
    x["end"] = _ts(x["lastTedt"])
    x = x.sort_values("t").groupby(["statId", "chgerId", "start"], as_index=False) \
        .agg(end=("end", "last"), seen=("t", "first"))
    x["dur_min"] = (x["end"] - x["start"]).dt.total_seconds() / 60
    x.loc[x["dur_min"] < 0, "dur_min"] = np.nan
    x["short"] = x["dur_min"] < SHORT_MIN          # 길이 모름(충전 중 등)은 실패로 보지 않음
    return x


def _binom_tail(x, n, p):
    """P(X >= x), X ~ 이항(n, p)"""
    return float(sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(x, n + 1)))


def bh(p):
    """Benjamini-Hochberg q값"""
    p = np.asarray(p, float)
    n = len(p)
    if n == 0:
        return p
    o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(q, 1)
    return out


# ---------------- 진단 ----------------
def diagnose():
    s, info = load()
    t_first, now = s["t"].min(), s["t"].max()
    snaps = np.sort(s["t"].unique())
    gaps = np.diff(snaps).astype("timedelta64[m]").astype(int)
    cur = s[s["t"] == now].merge(info, on=["statId", "chgerId"], how="left")
    cur = cur[cur["delYn"] != "Y"].reset_index(drop=True)
    key = list(zip(cur["statId"], cur["chgerId"]))

    # 세션 / 성공 세션 / 수집 창 이용 횟수
    ss = sessions(s)
    ss_k = ss.set_index(["statId", "chgerId"])
    first_seen = s.groupby(["statId", "chgerId"])["t"].min()
    win = ss[ss["start"] > ss.set_index(["statId", "chgerId"]).index.map(first_seen).values]
    k_win = win.groupby(["statId", "chgerId"]).size()
    succ = ss[~ss["short"]].groupby(["statId", "chgerId"])["start"].max()
    anyl = ss.groupby(["statId", "chgerId"])["start"].agg(["min", "max"])
    cur["k"] = [int(k_win.get(k, 0)) for k in key]
    # 마지막 '성공' 충전: 성공 세션이 없으면 관측된 가장 이른 세션 시각(보수적)
    cur["last_ok"] = [succ.get(k, anyl["min"].get(k, pd.NaT)) if k in anyl.index else pd.NaT for k in key]
    cur["last_any"] = [anyl["max"].get(k, pd.NaT) if k in anyl.index else pd.NaT for k in key]

    # 현재 상태가 이어진 시간: max(지금-statUpdDt, 지금-스냅샷 기준 연속 시작)
    sg = s.merge(cur[["statId", "chgerId"]], on=["statId", "chgerId"])
    sg["cls"] = np.where(sg["stat"].isin(ERR), "E", sg["stat"])
    run_start = {}
    for k, g in sg.groupby(["statId", "chgerId"]):
        c = g["cls"].values
        i = len(c) - 1
        while i > 0 and c[i - 1] == c[-1]:
            i -= 1
        run_start[k] = g["t"].values[i]
    upd = _ts(cur["statUpdDt"])
    rs = pd.to_datetime(pd.Series([run_start.get(k) for k in key]))
    cur["state_h"] = np.fmax((now - upd).dt.total_seconds(), (now - rs).dt.total_seconds()) / 3600
    cur["charge_h"] = (now - _ts(cur["nowTsdt"])).dt.total_seconds() / 3600
    cur.loc[cur["charge_h"].isna() & (cur["stat"] == "3"), "charge_h"] = cur["state_h"]

    # 플래핑 (최근 48시간)
    recent = sg[sg["t"] > now - pd.Timedelta(hours=FLAP_WINDOW_H)]
    fl = recent.groupby(["statId", "chgerId"]).agg(
        n=("cls", "size"), err=("cls", lambda c: (c == "E").mean()),
        tr=("cls", lambda c: int((c.values[1:] != c.values[:-1]).sum())))
    ok48 = ss[(ss["start"] > now - pd.Timedelta(hours=FLAP_WINDOW_H)) & ~ss["short"]] \
        .groupby(["statId", "chgerId"]).size()

    # 허브 운영시간
    st = cur.groupby("statId").agg(lat=("lat", "first"), lng=("lng", "first")).dropna().reset_index()
    hours = load_hours(st)
    H24 = (0.0, 24.0, False, False, "허브 목록에 없음(24시간 가정)")
    cur["hours"] = [hours.get(x, H24) for x in cur["statId"]]

    # ---- A. 공식 상태 ----
    code = [""] * len(cur)
    grade = [""] * len(cur)
    why = [""] * len(cur)
    for i, r in cur.iterrows():
        fast = bool(r["fast"])
        if r["stat"] in ("4", "5"):
            if r["stat"] == "5" and r["state_h"] < 24:
                code[i], grade[i], why[i] = "A1", "주의", f"점검중 {r['state_h']:.0f}시간째 (진행 중일 수 있음)"
            else:
                code[i], grade[i], why[i] = "A1", "경보", f"{STAT_NAME[r['stat']]} {r['state_h'] / 24:.0f}일째"
        elif r["stat"] in ERR and r["state_h"] >= COMM_WARN_H:
            g = "경보" if r["state_h"] >= COMM_ALARM_H else "주의"
            code[i], grade[i], why[i] = "A2", g, f"{STAT_NAME[r['stat']]} {_dur(r['state_h'])} 연속"
        elif r["stat"] == "3" and r["charge_h"] >= STUCK[fast][0]:
            g = "경보" if r["charge_h"] >= STUCK[fast][1] else "주의"
            # 주말 휴무 기관(허브 기준)은 토·일과 월요일 9시 전에는 경보로 올리지 않음 (관용차 주말 주차)
            weekend = now.weekday() >= 5 or (now.weekday() == 0 and now.hour < 9)
            held = g == "경보" and r["hours"][2] and weekend
            if held:
                g = "주의"
            code[i], grade[i] = "A4", g
            why[i] = f"{'급속' if fast else '완속'} 충전중 {_dur(r['charge_h'])} " \
                     f"(정상 충전의 99%는 {3.6 if fast else 14.5}시간 이내)" + \
                     ("" if g == "경보" else ", 주말 휴무 기관이라 주말엔 주의로만" if held else ", 완충 후 장시간 주차 가능성")
        k = (r["statId"], r["chgerId"])
        if code[i] == "" and k in fl.index:
            f = fl.loc[k]
            if f["n"] >= FLAP_MIN_SNAPS and f["err"] >= FLAP_ERR_RATIO and f["tr"] >= 2 and ok48.get(k, 0) == 0:
                code[i], grade[i] = "A3", "경보"
                why[i] = f"최근 48시간 중 {f['err'] * 100:.0f}% 통신이상, 상태 {int(f['tr'])}번 바뀜, 정상 충전 0건"
    cur["code"], cur["grade"], cur["why"] = code, grade, why

    # ---- 0. 품질 게이트 (숨은 고장 검정 대상) ----
    cand = (cur["code"] == "") & (cur["stat"] == "2")
    cur["gate"] = ""
    cur.loc[cand & (cur["limitYn"] == "Y"), "gate"] = "이용 제한 충전기"
    cur.loc[cand & cur["hours"].map(lambda h: h[3]), "gate"] = "전용 충전기(허브)"
    cur.loc[cand & cur["last_ok"].isna(), "gate"] = "충전 이력 없음(신규·미사용)"
    test = cand & (cur["gate"] == "")

    # 열린 하루(24시간 환산)당 이용률: 운영시간이 다른 충전기도 같은 단위로 비교
    cur["w_open"] = [max(open_days(first_seen.get(k, now), now, h), 0.1) for k, h in zip(key, cur["hours"])]
    cur["rate"] = cur["k"] / cur["w_open"]

    # 정상 기준 집단: 지금 사용가능/충전중이고 7일 안에 성공 충전이 있었던 충전기
    pool = cur[cur["stat"].isin(["2", "3"]) & (cur["code"] == "") &
               ((now - cur["last_ok"]).dt.total_seconds() < 7 * 86400)]
    plat, plng = pool["lat"].values, pool["lng"].values
    short_p0 = {f: ss.merge(info[["statId", "chgerId", "fast"]], on=["statId", "chgerId"])
                .query("fast == @f")["short"].mean() for f in (True, False)}
    span = min(1.0, (now - t_first).total_seconds() / 86400)

    res = []
    for i, r in cur.iterrows():
        out = dict(p_sib=np.nan, p_nb=np.nan, lam=np.nan, p_b3=np.nan, alt=0, hold="", basis="", n_ref=0)
        if pd.isna(r["lat"]):
            res.append(out)
            continue
        d_all = _km(r["lat"], r["lng"], plat, plng)
        same_t = (pool["fast"] == r["fast"]).values
        me = ((pool["statId"] == r["statId"]) & (pool["chgerId"] == r["chgerId"])).values
        same_st = (pool["statId"] == r["statId"]).values
        out["alt"] = int(((d_all <= 1.0) & same_t & ~me).sum())
        sib = pool[same_t & same_st & ~me]
        idx = np.where(same_t & ~same_st & (d_all <= 10))[0]
        nb = pool.iloc[idx[np.argsort(d_all[idx])][:NB_NEIGHBORS]]
        grp, cv2 = (sib, CV2_SIB) if len(sib) else (nb, None)
        if len(grp) and (len(sib) or len(nb) >= 5):
            mu = grp["k"].sum() / grp["w_open"].sum()
            if cv2 is None:   # 이웃: 충전기마다 원래 이용량이 다른 정도(이질성)를 데이터로 추정
                rt = grp["rate"].values
                cv2 = (rt.var(ddof=1) - mu / grp["w_open"].mean()) / mu ** 2 if mu > 0 and len(rt) > 1 else CV2_MIN
                cv2 = float(np.clip(cv2, CV2_MIN, 4.0))
            out.update(lam=mu, basis="형제" if len(sib) else "이웃", n_ref=len(grp), cv2=cv2)
        if test[i] and pd.notna(out["lam"]) and out["lam"] > 0:
            T_open = open_days(r["last_ok"], now, r["hours"])
            T_cal = (now - r["last_ok"]).total_seconds() / 86400
            out["T_open"], out["T_cal"] = T_open, T_cal
            if T_cal >= 1:
                mu = out["lam"]
                shape = 1 / (1 / (grp["k"].sum() + 1) + out["cv2"])   # 추정 불확실성 + 이질성
                p = (shape / mu / (shape / mu + T_open)) ** shape      # 음이항: 0건일 확률
                out["p_sib" if out["basis"] == "형제" else "p_nb"] = p
                # 동시 활동 확인: 최근 24시간 기준 집단 이용이 평소의 절반 미만이면 보류 (태풍·연휴 등)
                rec = win[win["start"] > now - pd.Timedelta(hours=24)].merge(
                    grp[["statId", "chgerId"]], on=["statId", "chgerId"]).shape[0]
                if rec < 0.5 * mu * len(grp) * span:
                    out["hold"] = "기준 집단 이용 급감(보류)"
            mine = ss[(ss["statId"] == r["statId"]) & (ss["chgerId"] == r["chgerId"]) & ss["dur_min"].notna()]
            ns = int(mine["short"].sum())
            if len(mine) >= 3 and ns >= 3:
                out["p_b3"] = _binom_tail(ns, len(mine), short_p0[bool(r["fast"])])
                out["b3_txt"] = f"관측된 {len(mine)}번 충전 중 {ns}번이 5분 미만"
        res.append(out)
    R = pd.DataFrame(res)
    cur = pd.concat([cur, R], axis=1)

    # 숨은 고장 p: 형제 있으면 형제 비교 우선(운영시간·날씨가 상쇄됨), 없으면 이웃 비교
    cur["p_hid"] = cur["p_sib"].where(cur["p_sib"].notna(), cur["p_nb"])
    t = test & cur["p_hid"].notna() & (cur["hold"] == "")
    cur["q_hid"] = np.nan
    cur.loc[t, "q_hid"] = bh(cur.loc[t, "p_hid"].values)
    for i, r in cur[t].iterrows():
        who = f"같은 충전소 다른 {int(r['n_ref'])}대" if r["basis"] == "형제" else f"주변 비슷한 충전기 {int(r['n_ref'])}대"
        basis = f"{who}는 평균 하루 {r['lam']:.1f}번 쓰이는데, 이 기기는 열려 있던 {_days(r['T_open'])} 동안 0번"
        txt = f"{basis} (우연일 확률 {_pct(r['p_hid'])})"
        if r["p_hid"] < P_ALARM and r["q_hid"] <= FDR_Q:
            cur.loc[i, ["code", "grade", "why"]] = ["B1" if pd.notna(r["p_sib"]) else "B2", "경보", txt]
        elif r["p_hid"] < P_WATCH:
            cur.loc[i, ["code", "grade", "why"]] = ["B1" if pd.notna(r["p_sib"]) else "B2", "관찰", txt]
    b3 = test & cur["p_b3"].notna() & (cur["p_b3"] < P_ALARM)
    for i, r in cur[b3].iterrows():
        if cur.loc[i, "grade"] != "경보":
            cur.loc[i, ["code", "grade", "why"]] = ["B3", "경보", f"{r['b3_txt']} (평소엔 {short_p0[bool(r['fast'])] * 100:.0f}%)"]
    cur.loc[test & (cur["hold"] != "") & (cur["code"] == ""), "gate"] = cur["hold"]

    # ---- C. 묶음 ----
    cur["bundle"] = ""
    up_min = cur["statUpdDt"].str[:12]
    bat = cur[cur["code"].isin(["A1", "A2"])].groupby(up_min).agg(n=("statId", "size"), st=("statId", "nunique"))
    for u in bat[(bat["n"] >= 10) & (bat["st"] >= 5) & (bat.index.str.len() == 12)].index:
        cur.loc[(up_min == u) & cur["code"].isin(["A1", "A2"]), "bundle"] = \
            f"운영사 일괄 갱신 {u[:4]}-{u[4:6]}-{u[6:8]} {u[8:10]}:{u[10:12]} (실제 고장 시작은 더 이를 수 있음)"
    flagged = cur["grade"] == "경보"
    for sid, g in cur.groupby("statId"):
        if len(g) >= 2 and flagged[g.index].all():
            cur.loc[g.index, "bundle"] = (cur.loc[g.index, "bundle"] + " / 충전소 전체 동시 이상 → 전원·통신 설비 1건으로 점검").str.strip(" /")

    # ---- D. 트랙 + 우선순위 ----
    days = np.where(cur["code"].isin(["A1", "A2", "A3", "A4"]), cur["state_h"] / 24,
                    (now - cur["last_ok"]).dt.total_seconds() / 86400)
    cur["방치_일"] = pd.Series(days).round(1)
    since_use = (now - cur["last_any"]).dt.total_seconds() / 86400
    ghost = cur["code"].isin(["A1", "A2", "A3"]) & (cur["grade"] == "경보") & \
        ((since_use > GHOST_DAYS) | (cur["last_any"].isna() & (cur["방치_일"] > GHOST_DAYS)))
    cur["트랙"] = np.select([ghost, cur["grade"] != ""], ["행정(철거·목록 정리 검토)", "수리"], "")

    conf = cur["code"].map(CONF).astype(float)
    hid = cur["code"].isin(["B1", "B2", "B3"])
    lp = -np.log10(cur["p_hid"].fillna(cur["p_b3"]).clip(lower=1e-12))
    conf = conf.where(~hid, 0.5 + 0.4 * np.minimum(1, lp / 8))
    conf = conf * cur["grade"].map({"경보": 1.0, "주의": 0.6, "관찰": 0.4}).fillna(0)
    altb = np.select([cur["alt"] == 0, cur["alt"] <= 2], [1.5, 1.2], 1.0)
    lam = cur["lam"].fillna(cur.groupby("fast")["rate"].transform("mean"))
    raw = conf * (0.2 + lam) * (1 + np.log10(1 + np.minimum(cur["방치_일"].fillna(0).clip(lower=0), DAYS_CAP))) \
        * altb * np.where(cur["fast"], 1.3, 1.0)
    raw = raw.where(cur["트랙"] == "수리", 0)
    ref = np.nanpercentile(raw[raw > 0], 99) if (raw > 0).any() else 1   # 11/10 고정 시 상수로 박음
    cur["raw"] = raw
    cur["점수"] = np.minimum(100, 100 * raw / ref).round(1)
    cur.attrs.update(now=now, t_first=t_first, n_snaps=len(snaps), max_gap_min=int(gaps.max()) if len(gaps) else 0,
                     score_ref=float(ref), short_p0=short_p0)
    return cur


def main():
    os.makedirs(OUT, exist_ok=True)
    d = diagnose()
    a = d.attrs
    pub = d[d["busiId"].isin(PUBLIC)].copy()
    pub["판정"] = pub["code"].map({"A1": "운영불가", "A2": "통신두절", "A3": "간헐통신", "A4": "충전고착",
                                  "B1": "숨은고장(형제)", "B2": "숨은고장(이웃)", "B3": "실패세션반복"}).fillna("정상")
    print(f"수집 {a['t_first']:%m/%d %H:%M} ~ {a['now']:%m/%d %H:%M} · 스냅샷 {a['n_snaps']}개 · 최대 공백 {a['max_gap_min']}분")
    print("\n[등급별] 공공 충전기", len(pub), "대")
    print(pd.crosstab([pub["busiId"]], [pub["판정"], pub["grade"].replace("", "-")]).to_string())
    print("\n트랙:", pub["트랙"].value_counts().to_dict())
    print("판정 보류:", pub["gate"].replace("", np.nan).dropna().value_counts().to_dict())
    rep = pub[pub["트랙"] == "수리"].sort_values("raw", ascending=False).reset_index(drop=True)
    rep.insert(0, "순위", rep.index + 1)
    cols = ["순위", "점수", "busiId", "statNm", "chgerId", "fast", "판정", "grade", "why", "bundle", "방치_일",
            "lam", "basis", "alt", "p_hid", "q_hid", "statId", "addr", "lat", "lng"]
    adm = pub[pub["트랙"].str.startswith("행정")]
    path = f"{OUT}/diagnosis_{a['now']:%Y%m%d_%H%M}.csv"
    pd.concat([rep[cols], adm.assign(순위=np.nan)[cols]]).to_csv(path, index=False, encoding="utf-8-sig")
    print("\nTOP 15 (수리 트랙)")
    print(rep[["순위", "점수", "busiId", "statNm", "chgerId", "판정", "grade", "why"]].head(15).to_string(index=False))
    print(f"\n저장: {path}")
    return d


if __name__ == "__main__":
    main()
