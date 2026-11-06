"""충전기 지킴이 - 화면 (Streamlit)

실행:  streamlit run app.py
탭:    ① 공공기관  ② 운영사 알림  ③ 도민 지도
"""
import os
import glob

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from guardian.diagnose import diagnose, _read_gz, DATA, PUBLIC
from guardian.judge import hub_compare, load_latest

st.set_page_config(page_title="충전기 지킴이", page_icon="⚡", layout="wide")

BUSI = {"JD": "제주특별자치도", "ME": "환경부(한국환경공단)", "JC": "제주에너지공사",
        "JE": "제주전기자동차서비스", "JT": "제주테크노파크"}
NAME = {"A1": "운영불가", "A2": "통신두절", "A3": "간헐통신", "A4": "충전고착",
        "B1": "숨은고장(형제 비교)", "B2": "숨은고장(이웃 비교)", "B3": "실패충전 반복"}
COLOR = {"경보": [220, 38, 38, 220], "주의": [245, 140, 30, 210], "관찰": [235, 200, 40, 200],
         "행정": [120, 90, 160, 200], "정상": [40, 160, 110, 110]}
ACTION = {
    "A1": "운영사 점검 기록 확인 → 장기 점검이면 사유·완료 예정일 확인",
    "A2": "통신 모뎀·라우터 전원 확인, 원격 재부팅 → 안 되면 현장 출동",
    "A3": "통신 모듈·안테나 접촉 불량 점검 (끊겼다 붙었다 반복)",
    "A4": "원격 충전 종료 처리 → 릴레이 융착·종료 신호 유실 점검",
    "B1": "현장 시험 충전: 화면·카드 결제·커넥터·출력 확인",
    "B2": "현장 시험 충전: 화면·카드 결제·커넥터·출력 확인 (주변 충전기도 함께)",
    "B3": "결제 모듈·차단기·절연 상태 점검 (시작 직후 끊김 반복)",
}


# ---------------- 데이터 ----------------
@st.cache_data(ttl=1800, show_spinner="진단 계산 중… (약 20초)")
def load():
    d = diagnose()
    attrs = dict(d.attrs)
    m = _read_gz(f"{DATA}/master/*.csv.gz").sort_values("collected_at") \
        .drop_duplicates(["statId", "chgerId"], keep="last")[["statId", "chgerId", "busiNm", "output", "chgerType"]]
    d = d.drop(columns=[c for c in ["busiNm", "output"] if c in d.columns]).merge(
        m.drop(columns="chgerType"), on=["statId", "chgerId"], how="left")
    d["판정"] = d["code"].map(NAME).fillna("정상")
    d["표시"] = np.where(d["트랙"].str.startswith("행정"), "행정", d["grade"].replace("", "정상"))
    d["운영기관"] = d["busiId"].map(BUSI).fillna(d["busiNm"])
    d["급속"] = np.where(d["fast"], "급속", "완속")
    d["방치_일"] = d["방치_일"].round(1)
    return d, attrs


@st.cache_data(ttl=3600)
def load_hub():
    df_all, _ = load_latest()
    return hub_compare(df_all[df_all["delYn"] != "Y"])


@st.cache_data(ttl=1800)
def history(statId, chgerId):
    s = _read_gz(f"{DATA}/status/*.csv.gz")
    s = s[(s["statId"] == statId) & (s["chgerId"] == chgerId)].copy()
    s["시각"] = pd.to_datetime(s["collected_at"])
    return s.sort_values("시각")


@st.cache_data(ttl=3600)
def kepco_station(statId):
    f = glob.glob(f"{DATA}/kepco*.csv.gz") + glob.glob(f"{DATA}/kepco/*.csv.gz")
    if not f or not statId.startswith("KP"):
        return None
    k = pd.read_csv(f[0], dtype=str)
    k = k[k["충전소ID"].str.zfill(6) == statId[2:]]
    if k.empty:
        return None
    k["월"] = pd.to_datetime(k["충전시작시각"], errors="coerce").dt.to_period("M").astype(str)
    return k.groupby(["월", "충전기명"]).size().unstack(fill_value=0)


def deck(df, offline, radius=60):
    df = df.dropna(subset=["lat", "lng"]).copy()
    df["color"] = df["표시"].map(COLOR)
    df["r"] = np.where(df["표시"] == "정상", radius * 0.6, radius * 1.6)
    layer = pdk.Layer("ScatterplotLayer", df, get_position=["lng", "lat"], get_fill_color="color",
                      get_radius="r", radius_min_pixels=3, pickable=True)
    view = pdk.ViewState(latitude=33.38, longitude=126.55, zoom=9.2)
    tip = {"html": "<b>{statNm}</b> {chgerId}<br/>{판정} · {표시}<br/>{why}", "style": {"fontSize": "12px"}}
    return pdk.Deck(layers=[layer], initial_view_state=view, tooltip=tip,
                    map_style=None if offline else "light")


def work_order(r):
    return f"""### 점검 지시서 · {r['statNm']} {r['chgerId']}번
- **운영기관**: {r['운영기관']}
- **주소**: {r['addr']}
- **충전기**: {r['급속']} {r.get('output', '')}kW · ID {r['statId']}-{r['chgerId']}
- **판정**: {r['판정']} ({r['grade']}) · 방치 {r['방치_일']}일
- **근거**: {r['why']}
- **권장 조치**: {ACTION.get(r['code'], '현장 확인')}
{('- **참고**: ' + r['bundle']) if r.get('bundle') else ''}

**현장 확인 결과** (체크 후 회신): ☐ 고장 확인·수리 ☐ 고장 확인·부품 대기 ☐ 일반차 주차 등 점유 ☐ 공사·폐쇄 ☐ 정상(오탐)
"""


# ---------------- 화면 ----------------
d, attrs = load()
pub = d[d["busiId"].isin(PUBLIC)]

with st.sidebar:
    st.markdown("## ⚡ 충전기 지킴이")
    st.caption("제주 공공 전기차 충전기의 숨은 고장·장기 방치를 찾아 점검 순서를 정합니다.")
    st.markdown(f"**기준 시각** {attrs['now']:%Y-%m-%d %H:%M}  \n"
                f"**수집 기간** {attrs['t_first']:%m/%d} ~ {attrs['now']:%m/%d} · 스냅샷 {attrs['n_snaps']}개")
    offline = st.toggle("오프라인 모드 (배경 지도 끄기)", value=False)
    st.divider()
    st.caption("데이터: 한국환경공단 충전기 상태 API(30분 수집) · 제주데이터허브 전기자동차 충전소 정보(862) · "
               "한국전력공사 충전소 충전량(2023~2026.9)")

tab1, tab2, tab3 = st.tabs(["🏛️ 공공기관", "🔧 운영사 알림", "🚗 도민 지도"])

# ---- ① 공공기관 ----
with tab1:
    sel = st.multiselect("운영기관", list(BUSI), default=["JD", "ME"], format_func=lambda x: BUSI[x])
    v = pub[pub["busiId"].isin(sel)] if sel else pub
    c = st.columns(5)
    c[0].metric("대상 충전기", f"{len(v):,}대")
    c[1].metric("🔴 경보", int((v["표시"] == "경보").sum()))
    c[2].metric("🟠 주의", int((v["표시"] == "주의").sum()))
    c[3].metric("🟡 관찰", int((v["표시"] == "관찰").sum()))
    c[4].metric("🟣 철거·정리 검토", int((v["표시"] == "행정").sum()))

    st.pydeck_chart(deck(v, offline), use_container_width=True)

    st.subheader("점검 우선순위")
    rep = v[v["트랙"] == "수리"].sort_values("raw", ascending=False).reset_index(drop=True)
    rep.insert(0, "순위", rep.index + 1)
    show = rep[["순위", "점수", "운영기관", "statNm", "chgerId", "급속", "판정", "grade", "why", "방치_일", "alt"]] \
        .rename(columns={"statNm": "충전소", "chgerId": "충전기", "grade": "등급", "why": "근거",
                         "방치_일": "방치(일)", "alt": "1km 대체"})
    st.dataframe(show, hide_index=True, use_container_width=True, height=380)
    st.download_button("우선순위 목록 내려받기 (CSV)", show.to_csv(index=False).encode("utf-8-sig"),
                       "점검우선순위.csv", "text/csv")

    st.subheader("충전기 상세 · 점검 지시서")
    if len(rep):
        pick = st.selectbox("충전기 선택", rep.index,
                            format_func=lambda i: f"{i + 1}위 · {rep.loc[i, 'statNm']} {rep.loc[i, 'chgerId']} ({rep.loc[i, '판정']})")
        r = rep.loc[pick]
        a, b = st.columns([3, 2])
        with a:
            st.markdown(work_order(r))
            st.download_button("지시서 내려받기 (.md)", work_order(r).encode("utf-8"),
                               f"점검지시서_{r['statId']}_{r['chgerId']}.md")
        with b:
            h = history(r["statId"], r["chgerId"])
            if len(h):
                h["상태코드"] = h["stat"].astype(int)
                st.caption("수집 기간 상태 변화 (1 통신이상 · 2 사용가능 · 3 충전중 · 4 운영중지 · 5 점검중 · 9 미확인)")
                st.line_chart(h.set_index("시각")["상태코드"], height=200)
            k = kepco_station(r["statId"])
            if k is not None:
                st.caption("한전 기록: 이 충전소의 월별 충전 횟수")
                st.bar_chart(k.tail(18), height=200)

    with st.expander(f"🟣 철거·목록 정리 검토 대상 ({int((v['표시'] == '행정').sum())}대) — 고장 상태로 1년 넘게 미사용"):
        adm = v[v["표시"] == "행정"][["운영기관", "statNm", "chgerId", "판정", "why", "addr"]]
        st.dataframe(adm.rename(columns={"statNm": "충전소", "chgerId": "충전기", "why": "근거", "addr": "주소"}),
                     hide_index=True, use_container_width=True)

    with st.expander("📉 1년 전(허브 2025.9) 대비 사라지거나 줄어든 충전소"):
        hc = load_hub()
        hc = hc[hc["운영기관"].isin(sel or PUBLIC)]
        st.dataframe(pd.crosstab(hc["운영기관"].map(BUSI), hc["변화"]), use_container_width=True)
        st.dataframe(hc[hc["변화"] != "유지"].drop(columns=["lat", "lng"]), hide_index=True, use_container_width=True)

# ---- ② 운영사 알림 ----
with tab2:
    ops = pub.groupby("운영기관").size().sort_values(ascending=False).index.tolist()
    op = st.selectbox("운영사 선택", ops)
    mine = pub[(pub["운영기관"] == op) & (pub["트랙"] == "수리")].sort_values("raw", ascending=False)
    c = st.columns(3)
    c[0].metric("내 충전기", f"{(pub['운영기관'] == op).sum():,}대")
    c[1].metric("🔴 오늘 경보", int((mine["grade"] == "경보").sum()))
    c[2].metric("🟠 주의·관찰", int(mine["grade"].isin(["주의", "관찰"]).sum()))
    st.markdown("#### 오늘 보낼 알림 (미리보기)")
    top = mine[mine["grade"] == "경보"].head(10)
    lines = [f"⚡ [충전기 지킴이] {op} 점검 요청 {attrs['now']:%m/%d %H:%M}", f"경보 {len(mine[mine['grade'] == '경보'])}건 중 상위 {len(top)}건"]
    for i, r in enumerate(top.itertuples(), 1):
        lines.append(f"{i}. {r.statNm} {r.chgerId}번 · {r.판정} · {r.why}")
    msg = "\n".join(lines)
    st.code(msg, language=None)
    tok, chat = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if st.button("텔레그램으로 보내기 (시연)", disabled=not (tok and chat),
                 help=None if tok and chat else ".env에 TELEGRAM_TOKEN, TELEGRAM_CHAT_ID를 넣으면 활성화"):
        import urllib.parse, urllib.request
        urllib.request.urlopen(f"https://api.telegram.org/bot{tok}/sendMessage?" +
                               urllib.parse.urlencode({"chat_id": chat, "text": msg}), timeout=15)
        st.success("보냈습니다.")
    st.markdown("#### 전체 목록")
    st.dataframe(mine[["statNm", "chgerId", "급속", "판정", "grade", "why", "방치_일", "addr"]]
                 .rename(columns={"statNm": "충전소", "chgerId": "충전기", "grade": "등급", "why": "근거",
                                  "방치_일": "방치(일)", "addr": "주소"}), hide_index=True, use_container_width=True)
    st.download_button("목록 내려받기 (CSV)", mine.to_csv(index=False).encode("utf-8-sig"), f"{op}_점검목록.csv")

# ---- ③ 도민 지도 ----
with tab3:
    st.markdown("#### 가려는 충전소, 지금 쓸 수 있을까?")
    st.caption("공식 상태에 더해, '사용가능'으로 보이지만 최근 오래 쓰인 기록이 없는 충전기를 미리 알려 드립니다.")
    q = st.text_input("충전소 이름이나 지역 검색", placeholder="예: 한림, 공항, 송악산")
    allc = d.copy()

    def citizen(r):
        if r["stat"] in ("4", "5") or r["code"] in ("A2", "A3"):
            return "이용 불가 (공식 상태)"
        if r["code"] in ("B1", "B2", "B3") and r["grade"] in ("경보", "관찰"):
            return f"⚠️ 방문 전 확인 권장 · 최근 {r['방치_일']:.0f}일 정상 충전 기록 없음"
        if r["stat"] == "3":
            return "충전 중"
        if r["stat"] == "2":
            return "사용 가능"
        return "상태 확인 불가"

    if q:
        allc = allc[allc["statNm"].fillna("").str.contains(q) | allc["addr"].fillna("").str.contains(q)]
    allc["안내"] = allc.apply(citizen, axis=1)
    allc["표시"] = np.select([allc["안내"].str.startswith("이용 불가"), allc["안내"].str.startswith("⚠️")],
                           ["경보", "관찰"], "정상")
    allc["why"] = allc["안내"]
    st.pydeck_chart(deck(allc, offline, radius=50), use_container_width=True)
    g = allc.groupby(["statNm", "addr"]).agg(
        급속=("fast", "sum"), 전체=("chgerId", "size"),
        사용가능=("안내", lambda s: (s == "사용 가능").sum()),
        주의=("안내", lambda s: s.str.startswith("⚠️").sum())).reset_index()
    st.dataframe(g.rename(columns={"statNm": "충전소", "addr": "주소"}).sort_values("주의", ascending=False).head(200),
                 hide_index=True, use_container_width=True)
    st.caption("※ '방문 전 확인 권장'은 데이터에 근거한 추정이며 고장 확정이 아닙니다.")
