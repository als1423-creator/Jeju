"""충전기 지킴이 - 화면 (Streamlit)

실행:  streamlit run app.py
"""
import glob
import os
import re

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from guardian.diagnose import DATA, PUBLIC, _read_gz, diagnose

st.set_page_config(page_title="충전기 지킴이", layout="wide", initial_sidebar_state="collapsed")

# ---------------- 표시 규칙 ----------------
BUSI = {"JD": "제주특별자치도", "ME": "환경부", "JC": "제주에너지공사",
        "JE": "제주전기자동차서비스", "JT": "제주테크노파크"}
NAME = {"A1": "운영불가", "A2": "통신두절", "A3": "간헐통신", "A4": "충전고착",
        "B1": "숨은고장", "B2": "숨은고장", "B3": "실패충전 반복"}
TONE = {"경보": "#D92D20", "주의": "#E8761E", "관찰": "#C9A000", "행정": "#7A5AF8", "정상": "#98A2B3",
        "사용가능": "#12B76A", "충전중": "#2E90FA", "확인불가": "#98A2B3"}
RGB = {"경보": [217, 45, 32, 230], "주의": [232, 118, 30, 220], "관찰": [201, 160, 0, 210],
       "행정": [122, 90, 248, 210], "정상": [152, 162, 179, 90],
       "사용가능": [18, 183, 106, 220], "충전중": [46, 144, 250, 210], "확인불가": [152, 162, 179, 120]}
ACTION = {
    "A1": "운영사 점검 기록 확인. 장기 점검이면 사유와 완료 예정일 확인",
    "A2": "통신 모뎀·라우터 전원 확인 후 원격 재부팅. 복구 안 되면 현장 출동",
    "A3": "통신 모듈·안테나 접촉 불량 점검 (끊김과 연결 반복)",
    "A4": "원격 충전 종료 처리 후 릴레이·종료 신호 점검. 완충 후 주차일 수 있음",
    "B1": "현장 시험 충전: 화면, 카드 결제, 커넥터, 출력 확인",
    "B2": "현장 시험 충전: 화면, 카드 결제, 커넥터, 출력 확인",
    "B3": "결제 모듈·차단기·절연 점검 (시작 직후 끊김 반복)",
}
RESULT = ["고장 확인·수리 완료", "고장 확인·부품 대기", "일반차 주차 등 점유", "공사·폐쇄", "정상 (오탐)"]
NOTE = {
    "gov": "숨은고장은 이용 기록으로 추정한 결과이며, 고장 확정이 아닙니다. 현장 확인 후 결과를 기록해 주세요.",
    "op": "숨은고장은 이용 기록으로 추정한 결과이며, 고장 확정이 아닙니다. 점검 후 결과를 회신해 주세요.",
    "cit": "'방문 전 확인 권장'은 데이터에 근거한 추정이며, 고장 확정이 아닙니다.",
}
JEJU = dict(lat=(33.10, 33.58), lng=(126.14, 126.98))   # 본섬 범위 (추자도는 따로 허용)

CSS = """
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css');
html, body, [class*="css"], .stMarkdown, .stDataFrame, button, input { font-family: Pretendard, 'Malgun Gothic', sans-serif; }
#MainMenu, footer, [data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none; }
header[data-testid="stHeader"] { background: transparent; height: 0; }
.block-container { padding-top: 1.6rem; max-width: 1240px; }
h1, h2, h3 { letter-spacing: -0.02em; }
.brand { font-size: 1.55rem; font-weight: 800; color: #101828; margin: 0; }
.brand-sub { color: #667085; font-size: .92rem; margin-top: .15rem; }
.note { background: #F9FAFB; border: 1px solid #EAECF0; border-radius: 10px; padding: .55rem .85rem;
        color: #475467; font-size: .86rem; margin: .4rem 0 1rem; }
.kpis { display: grid; grid-template-columns: 1fr 1fr; gap: .6rem; }
.kpi { border: 1px solid #EAECF0; border-radius: 12px; padding: .8rem .95rem; background: #fff; }
.kpi .l { color: #667085; font-size: .82rem; display: flex; align-items: center; gap: .4rem; }
.kpi .v { font-size: 1.7rem; font-weight: 800; color: #101828; line-height: 1.25; }
.kpi .s { color: #98A2B3; font-size: .76rem; }
.dot { width: .55rem; height: .55rem; border-radius: 50%; display: inline-block; }
.legend { color: #667085; font-size: .78rem; display: flex; gap: .9rem; margin-top: .3rem; flex-wrap: wrap; }
.sec { font-size: 1.12rem; font-weight: 700; color: #101828; margin: 1.4rem 0 .5rem; }
.wo-title { font-size: 1.02rem; font-weight: 700; color: #101828; margin-bottom: .2rem; }
.wo-id { color: #667085; font-size: .82rem; margin-bottom: .7rem; }
.kv { display: grid; grid-template-columns: 5.2rem 1fr; row-gap: .45rem; font-size: .9rem; }
.kv .k { color: #667085; } .kv .v { color: #101828; }
.badge { display: inline-block; padding: .05rem .5rem; border-radius: 999px; font-size: .78rem; font-weight: 600; color: #fff; }
.msg { background: #F2F4F7; border-radius: 14px; padding: .9rem 1.05rem; max-width: 640px; font-size: .9rem; color: #101828; }
.msg .h { font-weight: 700; margin-bottom: .1rem; }
.msg .t { color: #667085; font-size: .8rem; margin-bottom: .6rem; }
.msg .i { padding: .45rem 0; border-top: 1px solid #E4E7EC; }
.msg .i .r { color: #475467; font-size: .82rem; margin-top: .1rem; }
.card { border: 1px solid #EAECF0; border-radius: 12px; padding: .8rem 1rem; margin-bottom: .6rem; background: #fff; }
.card .n { font-weight: 700; color: #101828; } .card .a { color: #667085; font-size: .82rem; margin: .1rem 0 .45rem; }
.chips { display: flex; flex-wrap: wrap; gap: .35rem; }
.chip { border-radius: 999px; padding: .1rem .55rem; font-size: .78rem; background: #F2F4F7; color: #344054; }
.foot { border-top: 1px solid #EAECF0; margin-top: 2.5rem; padding-top: .9rem; color: #98A2B3; font-size: .76rem; line-height: 1.7; }
</style>
"""


# ---------------- 데이터 ----------------
def fix_coords(d):
    """제주 범위 밖 좌표 → 같은 주소의 정상 좌표로 교체, 없으면 지도에서 제외. 오류 목록 반환"""
    ok_main = d["lat"].between(*JEJU["lat"]) & d["lng"].between(*JEJU["lng"])
    ok_chuja = d["lat"].between(33.90, 34.02) & d["lng"].between(126.20, 126.40)
    ok = ok_main | ok_chuja
    bad = d[~ok & d["statId"].notna()].copy()
    good = d[ok].groupby("addr")[["lat", "lng"]].median()
    for i, r in bad.iterrows():
        if r["addr"] in good.index:
            d.loc[i, ["lat", "lng"]] = good.loc[r["addr"]].values
        else:
            d.loc[i, ["lat", "lng"]] = np.nan
    err = bad.drop_duplicates("statId")[["운영기관", "statNm", "statId", "addr", "lat", "lng"]]
    return d, err


def short_reason(r):
    code, why = r["code"], str(r["why"])
    why = re.sub(r"\s*\(우연일 확률[^)]*\)", "", why)
    if code in ("B1", "B2") and pd.notna(r.get("lam")):
        who = "같은 충전소" if code == "B1" else "주변"
        return f"{who} 충전기 하루 평균 {r['lam']:.1f}회 · 이 충전기 {days(r['방치_일'])}째 0회"
    if code == "A4":
        return re.sub(r" \(정상 충전의.*", "", why)
    return why


def days(x):
    if pd.isna(x):
        return "-"
    return f"{x * 24:.0f}시간" if x < 2 else f"{x:.0f}일"


@st.cache_data(ttl=1800, show_spinner="진단 계산 중입니다 (약 20초)")
def load():
    d = diagnose()
    attrs = dict(d.attrs)
    m = _read_gz(f"{DATA}/master/*.csv.gz").sort_values("collected_at") \
        .drop_duplicates(["statId", "chgerId"], keep="last")[["statId", "chgerId", "busiNm", "output"]]
    d = d.drop(columns=[c for c in ["busiNm", "output"] if c in d.columns]).merge(m, on=["statId", "chgerId"], how="left")
    d["판정"] = d["code"].map(NAME).fillna("정상")
    d["표시"] = np.where(d["트랙"].str.startswith("행정"), "행정", d["grade"].replace("", "정상"))
    normal = d["표시"] == "정상"   # 이상 없는 충전기는 현재 상태 색으로 (사용 가능 = 초록)
    d.loc[normal, "표시"] = d.loc[normal, "stat"].map({"2": "사용가능", "3": "충전중"}).fillna("확인불가")
    d["운영기관"] = d["busiId"].map(BUSI).fillna(d["busiNm"])
    d["종류"] = np.where(d["fast"], "급속", "완속")
    d["고유번호"] = d["statId"] + "-" + d["chgerId"]
    d["근거"] = d.apply(lambda r: short_reason(r) if r["code"] else "", axis=1)
    d["방치"] = d["방치_일"].apply(days)
    d, err = fix_coords(d)
    return d, attrs, err


@st.cache_data(ttl=3600)
def kepco_monthly(statId):
    f = glob.glob(f"{DATA}/kepco*.csv.gz") + glob.glob(f"{DATA}/kepco/*.csv.gz")
    if not f or not str(statId).startswith("KP"):
        return None
    k = pd.read_csv(f[0], dtype=str)
    k = k[k["충전소ID"].str.zfill(6) == statId[2:]]
    if k.empty:
        return None
    k["월"] = pd.to_datetime(k["충전시작시각"], errors="coerce").dt.strftime("%Y-%m")
    return k.groupby(["월", "충전기명"]).size().rename("횟수").reset_index()


# ---------------- 그리기 ----------------
def kpi(label, value, tone=None, sub=""):
    dot = f"<span class='dot' style='background:{TONE[tone]}'></span>" if tone else ""
    return f"<div class='kpi'><div class='l'>{dot}{label}</div><div class='v'>{value}</div><div class='s'>{sub}</div></div>"


GOV_LEGEND = [("경보", "경보"), ("주의", "주의"), ("관찰", "관찰"), ("행정", "철거·정리 검토"),
              ("사용가능", "사용 가능"), ("충전중", "충전 중")]
ALERT = {"경보", "주의", "관찰", "행정"}
CIT_LEGEND = [("사용가능", "사용 가능"), ("충전중", "충전 중"), ("관찰", "방문 전 확인"), ("경보", "이용 불가"),
              ("확인불가", "확인 불가")]


def map_deck(df, offline, height=430, focus=None, legend=GOV_LEGEND, big=ALERT):
    """focus=(위도, 경도, 이름) 이면 그곳을 확대하고 테두리로 강조"""
    df = df.dropna(subset=["lat", "lng"])[["lat", "lng", "표시", "statNm", "chgerId", "판정", "grade", "근거"]].copy()
    df["color"] = df["표시"].map(RGB)
    df["r"] = np.where(df["표시"].isin(big), 110, 45)
    df = df.sort_values("표시", key=lambda s: s.map({"정상": 0, "확인불가": 0, "충전중": 1, "사용가능": 2,
                                                   "관찰": 3, "주의": 4, "행정": 4, "경보": 5}))
    layer = pdk.Layer("ScatterplotLayer", df, get_position=["lng", "lat"], get_fill_color="color",
                      get_radius="r", radius_min_pixels=2.5, radius_max_pixels=9, pickable=True)
    tip = {"html": "<b>{statNm}</b> {chgerId}번<br/>{판정} {grade}<br/>{근거}",
           "style": {"fontSize": "12px", "backgroundColor": "#101828", "color": "white"}}
    layers = [layer]
    if focus:
        view = pdk.ViewState(latitude=focus[0], longitude=focus[1], zoom=15.2)
        ring = pd.DataFrame([{"lat": focus[0], "lng": focus[1]}])
        layers.append(pdk.Layer("ScatterplotLayer", ring, get_position=["lng", "lat"], get_radius=45,
                                filled=False, stroked=True, get_line_color=[29, 78, 216, 255],
                                line_width_min_pixels=3, radius_min_pixels=14))
    else:
        view = pdk.ViewState(latitude=33.37, longitude=126.55, zoom=8.75)
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, tooltip=tip,
                             map_style=None if offline else "light"), height=height, use_container_width=True)
    st.markdown("<div class='legend'>" + "".join(
        f"<span><span class='dot' style='background:{TONE[k]}'></span> {name}</span>" for k, name in legend) +
        ("<span><span class='dot' style='background:#fff;border:2px solid #1D4ED8'></span> 선택한 충전소</span>" if focus else "")
        + "</div>", unsafe_allow_html=True)


def usage_chart(d, r):
    """같은 충전소 충전기들의 수집 기간 충전 횟수 비교"""
    s = d[(d["statId"] == r["statId"]) & (d["fast"] == r["fast"])].copy()
    if len(s) < 2:
        return False
    s["충전기"] = s["chgerId"] + "번"
    s["구분"] = np.where(s["chgerId"] == r["chgerId"], "이 충전기", "같은 충전소")
    s = s[["충전기", "구분", "k"]].astype({"k": int})
    c = alt.Chart(s).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X("충전기:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("k:Q", title="충전 횟수"),
        color=alt.Color("구분:N", scale=alt.Scale(domain=["이 충전기", "같은 충전소"], range=["#D92D20", "#D0D5DD"]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["충전기", alt.Tooltip("k:Q", title="충전 횟수")]).properties(height=220)
    st.caption("수집 기간 중 같은 충전소 충전기별 충전 횟수")
    st.altair_chart(c, use_container_width=True)
    return True


def work_order_text(r):
    return "\n".join([
        f"[점검 지시서] {r['statNm']} {r['chgerId']}번 ({r['고유번호']})",
        f"운영기관: {r['운영기관']}", f"주소: {r['addr']}",
        f"충전기: {r['종류']} {r.get('output', '')}kW",
        f"판정: {r['판정']} ({r['grade']}), 방치 {r['방치']}",
        f"근거: {r['근거']}", f"권장 조치: {ACTION.get(r['code'], '현장 확인')}",
        *([f"참고: {r['bundle']}"] if r.get("bundle") else []),
        "", "현장 확인 결과 (하나 표시):", *[f"  [ ] {x}" for x in RESULT],
        "", "확인자:            확인일:"])


def work_order(d, r):
    with st.container(border=True):
        st.markdown(f"<div class='wo-title'>점검 지시서 · {r['statNm']} {r['chgerId']}번</div>"
                    f"<div class='wo-id'>고유번호 {r['고유번호']}</div>", unsafe_allow_html=True)
        a, b = st.columns([1.15, 1], gap="large")
        with a:
            badge = f"<span class='badge' style='background:{TONE.get(r['grade'] or '정상')}'>{r['grade']}</span>"
            rows = [("운영기관", r["운영기관"]), ("주소", r["addr"]),
                    ("충전기", f"{r['종류']} {r.get('output', '')}kW"),
                    ("판정", f"{r['판정']} {badge}"), ("방치", r["방치"]), ("근거", r["근거"]),
                    ("권장 조치", ACTION.get(r["code"], "현장 확인"))]
            if r.get("bundle"):
                rows.append(("참고", r["bundle"]))
            st.markdown("<div class='kv'>" + "".join(f"<div class='k'>{k}</div><div class='v'>{v}</div>" for k, v in rows)
                        + "</div>", unsafe_allow_html=True)
        with b:
            if not usage_chart(d, r):
                k = kepco_monthly(r["statId"])
                if k is not None:
                    st.caption("한전 기록: 이 충전소 월별 충전 횟수")
                    st.altair_chart(alt.Chart(k).mark_line(point=True).encode(
                        x=alt.X("월:N", title=None), y=alt.Y("횟수:Q", title=None), color=alt.Color("충전기명:N", title=None)
                    ).properties(height=220), use_container_width=True)
                else:
                    st.caption("같은 충전소에 비교할 충전기가 없습니다.")
        st.divider()
        c1, c2 = st.columns([3, 1], vertical_alignment="center")
        with c1:
            st.radio("현장 확인 결과 (점검 후 기록)", RESULT, index=None, horizontal=True, key=f"res_{r['고유번호']}")
        with c2:
            st.download_button("지시서 내려받기", work_order_text(r).encode("utf-8"),
                               f"점검지시서_{r['고유번호']}.txt", use_container_width=True, type="primary")


def pick_table(df, cols, key, height=360):
    """행을 누르면 선택된 행 번호를 돌려주는 표"""
    ev = st.dataframe(df[cols], hide_index=True, use_container_width=True, height=height,
                      on_select="rerun", selection_mode="single-row", key=key,
                      column_config={"순위": st.column_config.NumberColumn(width="small"),
                                     "충전기 번호": st.column_config.TextColumn(width="small",
                                                                         help="운영사가 충전소 안에서 붙인 번호"),
                                     "고유번호": st.column_config.TextColumn(help="환경부 통합 충전기 ID (충전소ID-충전기 번호)"),
                                     "근거": st.column_config.TextColumn(width="large")})
    sel = ev.selection.rows if ev and hasattr(ev, "selection") else []
    return sel[0] if sel else 0


def table_cols(df):
    t = df.rename(columns={"statNm": "충전소", "chgerId": "충전기 번호", "grade": "등급"})
    return t


# ---------------- 화면 ----------------
st.markdown(CSS, unsafe_allow_html=True)
d, attrs, coord_err = load()
pub = d[d["busiId"].isin(PUBLIC)]

h1, h2 = st.columns([5, 1.3], vertical_alignment="center")
with h1:
    st.markdown("<p class='brand'>충전기 지킴이</p><div class='brand-sub'>제주 공공 전기차 충전기의 숨은 고장과 장기 방치를 찾아 점검 순서를 정합니다</div>",
                unsafe_allow_html=True)
with h2:
    offline = st.toggle("오프라인 모드", value=False, help="인터넷이 없을 때 켜세요. 배경 지도만 꺼지고 나머지는 그대로 동작합니다.")

tab_gov, tab_op, tab_cit = st.tabs(["공공기관", "운영사", "도민"])

# ---- 공공기관 ----
with tab_gov:
    st.markdown(f"<div class='note'>{NOTE['gov']}</div>", unsafe_allow_html=True)
    sel = st.multiselect("운영기관", list(BUSI), default=["JD", "ME"], format_func=lambda x: BUSI[x],
                         label_visibility="collapsed", placeholder="운영기관 선택")
    v = pub[pub["busiId"].isin(sel)] if sel else pub
    m, k = st.columns([1.55, 1], gap="large")
    with m:
        map_deck(v, offline)
    with k:
        st.markdown("<div class='kpis'>" +
                    kpi("경보", int((v["표시"] == "경보").sum()), "경보", "바로 점검 필요") +
                    kpi("주의", int((v["표시"] == "주의").sum()), "주의", "지켜보며 확인") +
                    kpi("관찰", int((v["표시"] == "관찰").sum()), "관찰", "숨은고장 가능성") +
                    kpi("철거·정리 검토", int((v["표시"] == "행정").sum()), "행정", "1년 넘게 고장·미사용") +
                    "</div>" +
                    f"<div class='kpi' style='margin-top:.6rem'><div class='l'>대상 충전기</div><div class='v'>{len(v):,}대</div>"
                    f"<div class='s'>{', '.join(BUSI[x] for x in sel) if sel else '전체 공공 운영기관'}</div></div>",
                    unsafe_allow_html=True)

    st.markdown("<div class='sec'>오늘 점검할 곳</div>", unsafe_allow_html=True)
    rep = v[v["트랙"] == "수리"].sort_values("raw", ascending=False).reset_index(drop=True)
    rep.insert(0, "순위", rep.index + 1)
    only_alarm = st.toggle("경보만 보기", value=True, key="gov_alarm")
    show = rep[rep["grade"] == "경보"].reset_index(drop=True) if only_alarm else rep
    if len(show):
        t = table_cols(show)
        i = pick_table(t, ["순위", "충전소", "충전기 번호", "판정", "등급", "방치", "근거"], "gov_tbl")
        st.caption("행을 누르면 아래에 점검 지시서가 열립니다.")
        work_order(d, show.loc[i])
        out = t[["순위", "운영기관", "충전소", "충전기 번호", "고유번호", "종류", "판정", "등급", "방치", "근거",
                 "addr", "p_hid", "점수"]].rename(columns={"addr": "주소", "p_hid": "우연일 확률"})
        st.download_button("점검 목록 내려받기 (CSV)", out.to_csv(index=False).encode("utf-8-sig"), "점검목록.csv")
    else:
        st.info("해당하는 충전기가 없습니다.")

    adm = v[v["표시"] == "행정"]
    with st.expander(f"철거·목록 정리 검토 {len(adm)}대 · 고장 상태로 1년 넘게 미사용"):
        st.dataframe(table_cols(adm)[["운영기관", "충전소", "충전기 번호", "고유번호", "판정", "근거", "addr"]]
                     .rename(columns={"addr": "주소"}), hide_index=True, use_container_width=True)
    if len(coord_err):
        with st.expander(f"등록정보 오류 {len(coord_err)}건 · 좌표가 제주 밖으로 등록됨"):
            st.caption("같은 주소의 정상 좌표로 자동 보정했습니다. 운영사에 정정을 요청하세요.")
            st.dataframe(coord_err.rename(columns={"statNm": "충전소", "statId": "충전소ID", "addr": "주소",
                                                   "lat": "등록 위도", "lng": "등록 경도"}),
                         hide_index=True, use_container_width=True)

# ---- 운영사 ----
with tab_op:
    st.markdown(f"<div class='note'>{NOTE['op']}</div>", unsafe_allow_html=True)
    ops = pub.groupby("운영기관").size().sort_values(ascending=False).index.tolist()
    op = st.selectbox("운영사", ops, label_visibility="collapsed")
    mine = pub[(pub["운영기관"] == op) & (pub["트랙"] == "수리")].sort_values("raw", ascending=False).reset_index(drop=True)
    alarm = mine[mine["grade"] == "경보"].reset_index(drop=True)
    st.markdown("<div class='kpis' style='grid-template-columns:repeat(3,1fr)'>" +
                kpi("내 충전기", f"{(pub['운영기관'] == op).sum():,}대") +
                kpi("오늘 경보", len(alarm), "경보") +
                kpi("주의·관찰", int(mine["grade"].isin(["주의", "관찰"]).sum()), "주의") + "</div>",
                unsafe_allow_html=True)

    a, b = st.columns([1, 1], gap="large")
    with a:
        st.markdown("<div class='sec'>오늘 보낼 알림</div>", unsafe_allow_html=True)
        top = alarm.head(10)
        items = "".join(
            f"<div class='i'>{n}. {r.statNm} {r.chgerId}번 · <b>{r.판정}</b><div class='r'>{r.근거} · {r.고유번호}</div></div>"
            for n, r in enumerate(top.itertuples(), 1))
        st.markdown(f"<div class='msg'><div class='h'>[충전기 지킴이] {op} 점검 요청</div>"
                    f"<div class='t'>{attrs['now']:%m월 %d일 %H:%M} 기준 · 경보 {len(alarm)}건 중 상위 {len(top)}건</div>"
                    f"{items or '<div class=i>오늘 경보가 없습니다.</div>'}</div>", unsafe_allow_html=True)
        tok, chat = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
        if st.button("텔레그램으로 보내기", disabled=not (tok and chat),
                     help=None if tok and chat else "텔레그램 봇 설정 후 사용할 수 있습니다."):
            import urllib.parse
            import urllib.request
            text = "\n".join([f"[충전기 지킴이] {op} 점검 요청 ({attrs['now']:%m/%d %H:%M})"] +
                             [f"{n}. {r.statNm} {r.chgerId}번 · {r.판정}\n   {r.근거}" for n, r in enumerate(top.itertuples(), 1)])
            urllib.request.urlopen(f"https://api.telegram.org/bot{tok}/sendMessage?" +
                                   urllib.parse.urlencode({"chat_id": chat, "text": text}), timeout=15)
            st.success("보냈습니다.")
    with b:
        st.markdown("<div class='sec'>점검 목록</div>", unsafe_allow_html=True)
        all_g = st.toggle("주의·관찰도 보기", value=False, key="op_all",
                          help="주의는 대부분 완충 후 주차 등 일시적인 경우입니다.")
        lst = mine if all_g else alarm
        if len(lst):
            st.dataframe(table_cols(lst)[["충전소", "충전기 번호", "고유번호", "판정", "등급", "방치"]],
                         hide_index=True, use_container_width=True, height=420,
                         column_config={"충전기 번호": st.column_config.TextColumn(help="운영사가 충전소 안에서 붙인 번호"),
                                        "고유번호": st.column_config.TextColumn(help="환경부 통합 충전기 ID")})
            st.download_button("목록 내려받기 (CSV)", table_cols(lst)[["충전소", "충전기 번호", "고유번호", "종류", "판정", "등급",
                                                                       "방치", "근거", "addr"]].rename(columns={"addr": "주소"})
                               .to_csv(index=False).encode("utf-8-sig"), f"{op}_점검목록.csv")
        else:
            st.info("해당하는 충전기가 없습니다.")

# ---- 도민 ----
with tab_cit:
    st.markdown(f"<div class='note'>{NOTE['cit']}</div>", unsafe_allow_html=True)

    def status(r):
        if r["stat"] in ("4", "5") or r["code"] in ("A2", "A3"):
            return "이용 불가"
        if r["code"] in ("B1", "B2", "B3") and r["grade"] in ("경보", "관찰"):
            return "방문 전 확인"
        return {"2": "사용 가능", "3": "충전 중"}.get(r["stat"], "확인 불가")

    # 시민이 쓸 수 없는 곳(이용 제한, 버스 차고지)은 도민 화면에서 제외
    c = d[(d["limitYn"] != "Y") & ~d["statNm"].fillna("").str.contains("버스|차고지|회차지|여객")].copy()
    c["상태"] = c.apply(status, axis=1)
    c["표시"] = c["상태"].map({"이용 불가": "경보", "방문 전 확인": "관찰", "사용 가능": "사용가능",
                              "충전 중": "충전중", "확인 불가": "확인불가"})
    c["근거"] = np.where(c["상태"] == "방문 전 확인", "최근 " + c["방치"] + " 동안 정상 충전 기록 없음", c["상태"])
    c["판정"], c["grade"] = c["상태"], ""
    q = st.text_input("충전소 검색", placeholder="충전소 이름이나 지역 (예: 한림, 공항, 교래)", label_visibility="collapsed")
    hit = c[c["statNm"].fillna("").str.contains(q) | c["addr"].fillna("").str.contains(q)] if q else c
    focus_id = st.session_state.get("cit_focus")
    if focus_id is not None and focus_id not in set(hit["statId"]):
        focus_id = None
    focus = None
    if focus_id:
        f = hit[hit["statId"] == focus_id].dropna(subset=["lat", "lng"])
        if len(f):
            focus = (float(f["lat"].median()), float(f["lng"].median()), f["statNm"].iloc[0])
    mcol, lcol = st.columns([1.2, 1], gap="large")
    with mcol:
        if focus:
            fa, fb = st.columns([3, 1], vertical_alignment="center")
            fa.markdown(f"**{focus[2]}** 위치를 보고 있습니다")
            if fb.button("전체 지도", use_container_width=True, key="cit_reset"):
                st.session_state["cit_focus"] = None
                st.rerun()
        map_deck(hit, offline, height=520, focus=focus, legend=CIT_LEGEND, big={"사용가능", "충전중", "관찰", "경보"})
    order = ["사용 가능", "충전 중", "방문 전 확인", "이용 불가", "확인 불가"]
    g = hit.groupby(["statId", "statNm", "addr"]).agg(
        급속=("fast", "sum"), 전체=("chgerId", "size"), 이용=("k", "sum"),
        **{s: ("상태", lambda x, s=s: int((x == s).sum())) for s in order}).reset_index()
    g["완속"] = g["전체"] - g["급속"]
    with lcol:
        if q:
            st.markdown(f"<div class='sec' style='margin-top:0'>검색 결과 {len(g)}곳</div>", unsafe_allow_html=True)
            g = g.sort_values(["사용 가능", "전체"], ascending=False)
        else:
            st.markdown("<div class='sec' style='margin-top:0'>많이 이용하는 충전소 10곳</div>", unsafe_allow_html=True)
            st.caption("최근 수집 기간 충전 횟수 기준 · 지금 상태를 함께 보여 드립니다")
            g = g.sort_values("이용", ascending=False).head(10)
        tone = {"사용 가능": "#E7F6EC", "충전 중": "#EEF4FF", "방문 전 확인": "#FEF7E0", "이용 불가": "#FEE4E2", "확인 불가": "#F2F4F7"}
        with st.container(height=520, border=False):
            for _, r in g.head(30).iterrows():
                chips = "".join(f"<span class='chip' style='background:{tone[s]}'>{s} {int(r[s])}</span>"
                                for s in order if r[s])
                with st.container(border=True):
                    a, b = st.columns([4, 1.3], vertical_alignment="center")
                    a.markdown(f"<div class='n' style='font-weight:700'>{r['statNm']}</div>"
                               f"<div class='a' style='color:#667085;font-size:.82rem;margin:.1rem 0 .45rem'>{r['addr']}</div>"
                               f"<div class='chips'><span class='chip'>급속 {int(r['급속'])} · 완속 {int(r['완속'])} · 전체 {int(r['전체'])}</span></div>"
                               f"<div class='chips' style='margin-top:.35rem'>{chips}</div>", unsafe_allow_html=True)
                    on = r["statId"] == focus_id
                    if b.button("보는 중" if on else "지도에서 보기", key=f"go_{r['statId']}", use_container_width=True,
                                type="primary" if on else "secondary", disabled=on):
                        st.session_state["cit_focus"] = r["statId"]
                        st.rerun()
        if len(g) > 30:
            st.caption(f"외 {len(g) - 30}곳. 검색어를 더 좁혀 보세요.")

# ---------------- 하단 ----------------
st.markdown(
    f"<div class='foot'>기준 시각 {attrs['now']:%Y-%m-%d %H:%M} · 수집 기간 {attrs['t_first']:%m/%d}~{attrs['now']:%m/%d} "
    f"(30분 간격, {attrs['n_snaps']}회)<br>"
    "데이터: 한국환경공단 전기자동차 충전소 정보 API · 제주데이터허브 [인구, 생활] 전기자동차 충전소 정보(862, 2025.9.21 기준) · "
    "한국전력공사 전기차 충전소 충전량(2023.1~2026.9)<br>충전기 지킴이 · 제주 AI·빅데이터 활용 AI Agent 개발 챌린지</div>",
    unsafe_allow_html=True)
