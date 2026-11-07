"""고장났수다 - 화면 (Streamlit)

실행:  streamlit run app.py
화면:  홈(표지) · 공공기관 · 운영사 · 도민  (위쪽 메뉴로 이동)
"""
import glob
import html
import os
import re

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from guardian.diagnose import DATA, PUBLIC, _read_gz, diagnose

LOGO_SVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -1.5 172 43" width="172" height="43"><path d="M10 2h20a8 8 0 0 1 8 8v16a8 8 0 0 1-8 8H15l-7.2 5.4c-.7.5-1.6 0-1.6-.8V33.6A8 8 0 0 1 2 26V10a8 8 0 0 1 8-8z" fill="#1D4ED8"/><path d="M22.6 7.5 12.8 20.6h6.4l-2.1 9.9 9.8-13.4h-6.5z" fill="#fff"/><circle cx="34" cy="6" r="5.2" fill="#F04438" stroke="#fff" stroke-width="2.4"/><path d="M53.429 10.118500000000001V12.9805H67.955C67.955 15.8965 67.901 19.3795 66.983 24.1045L70.574 24.4825C71.57300000000001 19.1635 71.57300000000001 15.6265 71.57300000000001 12.494499999999999V10.118500000000001ZM59.099000000000004 18.488500000000002V27.0745H51.107V29.9905H73.733V27.0745H62.717V18.488500000000002ZM86.90899999999999 23.2675C81.725 23.2675 78.431 25.1035 78.431 28.127499999999998C78.431 31.2055 81.725 33.0145 86.90899999999999 33.0145C92.09299999999999 33.0145 95.36 31.2055 95.36 28.127499999999998C95.36 25.1035 92.09299999999999 23.2675 86.90899999999999 23.2675ZM86.90899999999999 26.075499999999998C90.095 26.075499999999998 91.79599999999999 26.7235 91.79599999999999 28.127499999999998C91.79599999999999 29.5585 90.095 30.2335 86.90899999999999 30.2335C83.723 30.2335 82.02199999999999 29.5585 82.02199999999999 28.127499999999998C82.02199999999999 26.7235 83.723 26.075499999999998 86.90899999999999 26.075499999999998ZM75.974 9.659500000000001V12.5215H80.969C80.91499999999999 15.6535 79.106 18.8125 75.056 20.1625L76.838 22.9975C79.80799999999999 22.0255 81.779 20.054499999999997 82.886 17.5435C83.966 19.6495 85.829 21.2965 88.583 22.133499999999998L90.31099999999999 19.325499999999998C86.369 18.1645 84.64099999999999 15.3835 84.58699999999999 12.5215H89.501V9.659500000000001ZM91.472 8.0125V22.8355H95.06299999999999V16.679499999999997H98.384V13.7365H95.06299999999999V8.0125ZM115.77199999999999 7.985499999999998V21.7825H119.363V16.247500000000002H122.684V13.331499999999998H119.363V7.985499999999998ZM100.70599999999999 17.6515V20.5945H102.785C106.32199999999999 20.5945 110.18299999999999 20.3785 114.23299999999999 19.5415L113.82799999999999 16.6525C110.47999999999999 17.3275 107.267 17.5705 104.297 17.624499999999998V9.470500000000001H100.70599999999999ZM114.071 22.8085V23.9695C114.071 25.8055 113.261 27.9925 111.23599999999999 29.5045C109.26499999999999 28.0735 108.482 25.9945 108.482 23.9695V22.8085H104.999V23.9695C104.999 26.4265 103.622 28.9645 100.38199999999999 30.1255L102.16399999999999 32.7445C104.405 31.9075 105.89 30.3955 106.72699999999999 28.5325C107.53699999999999 30.476499999999998 108.96799999999999 32.1235 111.26299999999999 32.9875C113.53099999999999 32.0695 114.96199999999999 30.3955 115.77199999999999 28.4785C116.636 30.3955 118.121 31.9075 120.443 32.7445L122.225 30.1255C118.931 29.072499999999998 117.63499999999999 26.6155 117.63499999999999 23.9695V22.8085ZM133.42999999999998 8.714500000000001V9.9025C133.42999999999998 12.8185 130.56799999999998 16.0855 124.844 16.868499999999997L126.24799999999999 19.7575C130.595 19.0825 133.727 17.057499999999997 135.34699999999998 14.3845C136.94 17.057499999999997 140.072 19.0825 144.41899999999998 19.7575L145.82299999999998 16.868499999999997C140.099 16.0855 137.237 12.7915 137.237 9.9025V8.714500000000001ZM124.00699999999999 21.566499999999998V24.4825H133.42999999999998V33.0145H137.021V24.4825H146.63299999999998V21.566499999999998ZM164.26399999999998 7.958500000000001V33.0415H167.88199999999998V20.1895H171.5V17.2195H167.88199999999998V7.958500000000001ZM149.25199999999998 10.2805V27.0475H151.35799999999998C155.70499999999998 27.0475 159.134 26.9125 162.914 26.2375L162.563 23.2675C159.35 23.8345 156.40699999999998 24.0235 152.843 24.0775V13.1695H160.916V10.2805Z" fill="#101828"/></svg>'
MARK_SVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="-0.5 -1.5 43 43" width="43" height="43"><path d="M10 2h20a8 8 0 0 1 8 8v16a8 8 0 0 1-8 8H15l-7.2 5.4c-.7.5-1.6 0-1.6-.8V33.6A8 8 0 0 1 2 26V10a8 8 0 0 1 8-8z" fill="#1D4ED8"/><path d="M22.6 7.5 12.8 20.6h6.4l-2.1 9.9 9.8-13.4h-6.5z" fill="#fff"/><circle cx="34" cy="6" r="5.2" fill="#F04438" stroke="#fff" stroke-width="2.4"/></svg>'

st.set_page_config(page_title="고장났수다", page_icon=MARK_SVG, layout="wide", initial_sidebar_state="collapsed")

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
    "cit": "'방문 전 확인'은 데이터에 근거한 추정이며, 고장 확정이 아닙니다.",
}
JEJU = dict(lat=(33.10, 33.58), lng=(126.14, 126.98))   # 본섬 범위 (추자도는 따로 허용)

# 선 아이콘 (24px 격자, 직접 그림)
ICON = {
    "robot": '<rect x="4" y="8" width="16" height="12" rx="3"/><path d="M12 8V5"/><circle cx="12" cy="3.6" r="1.3"/>'
             '<circle cx="9" cy="13.6" r="1.2" fill="currentColor" stroke="none"/>'
             '<circle cx="15" cy="13.6" r="1.2" fill="currentColor" stroke="none"/><path d="M9.6 17.1h4.8M2.6 12.5v3.2M21.4 12.5v3.2"/>',
    "shield": '<path d="M12 2.8 19.6 5.7v5.5c0 4.6-3.2 8.4-7.6 10-4.4-1.6-7.6-5.4-7.6-10V5.7z"/><path d="m8.6 12.1 2.4 2.4 4.5-4.6"/>',
    "users": '<circle cx="12" cy="8.4" r="3.3"/><path d="M5.7 20.2c.5-3.5 3-5.8 6.3-5.8s5.8 2.3 6.3 5.8"/>'
             '<circle cx="5.5" cy="9.6" r="2.3"/><path d="M1.7 17.9c.3-2.3 1.8-3.8 3.9-4"/>'
             '<circle cx="18.5" cy="9.6" r="2.3"/><path d="M22.3 17.9c-.3-2.3-1.8-3.8-3.9-4"/>',
    "gov": '<path d="M3 9.4 12 4l9 5.4"/><path d="M5 10.5v7M9.7 10.5v7M14.3 10.5v7M19 10.5v7"/><path d="M3 20.3h18"/>',
    "op": '<path d="M14.6 3.6a5 5 0 0 0-5.3 6.6L3.7 15.8a2.2 2.2 0 0 0 3.1 3.1l5.6-5.6a5 5 0 0 0 6.6-5.3l-2.9 2.9-2.6-.5-.5-2.6z"/>',
    "cit": '<path d="M5 16.4v-4.2L7 7h10l2 5.2v4.2"/><path d="M3.6 16.4h16.8M5.2 12.2h13.6"/>'
           '<circle cx="7.6" cy="16.9" r="1.8"/><circle cx="16.4" cy="16.9" r="1.8"/>',
}


def icon(name, size=28):
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">{ICON[name]}</svg>')


# 표지 오른쪽 충전기 그림 (직접 그림, 특정 회사 제품 아님)
CHARGER_SVG = (
    '<svg class="charger" viewBox="0 0 300 540" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="전기차 충전기 그림">'
    '<defs>'
    '<linearGradient id="gb" x1="0" x2="1"><stop offset="0" stop-color="#30353D"/><stop offset=".55" stop-color="#1C2026"/><stop offset="1" stop-color="#101318"/></linearGradient>'
    '<linearGradient id="gp" x1="0" x2="1"><stop offset="0" stop-color="#343943"/><stop offset=".5" stop-color="#1B1F25"/><stop offset="1" stop-color="#0E1114"/></linearGradient>'
    '<linearGradient id="gs" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#15275A"/><stop offset="1" stop-color="#0A1633"/></linearGradient>'
    '<radialGradient id="gw"><stop offset="0" stop-color="#0B0F19" stop-opacity=".30"/><stop offset="1" stop-color="#0B0F19" stop-opacity="0"/></radialGradient>'
    '</defs>'
    '<ellipse cx="150" cy="514" rx="125" ry="15" fill="url(#gw)"/>'
    '<path d="M94 494h112l14 14H80z" fill="#15181D"/><rect x="80" y="506" width="140" height="8" rx="2" fill="#0B0D11"/>'
    '<rect x="128" y="270" width="44" height="228" rx="6" fill="url(#gp)"/>'
    '<rect x="171" y="302" width="17" height="36" rx="5" fill="#2B3038"/>'
    '<path d="M44 92 64 38q4-10 16-10h140q12 0 16 10l20 54z" fill="#1E2229"/>'
    '<rect x="40" y="88" width="220" height="30" rx="5" fill="#14171C"/>'
    '<text x="150" y="109" text-anchor="middle" font-size="15" font-weight="700" fill="#F2F4F7" letter-spacing="1.5" '
    'font-family="Pretendard, Malgun Gothic, sans-serif">전기차 충전소</text>'
    '<path d="M44 116h18v146q0 10-8 10h-6q-4 0-4-6z" fill="#181B21"/>'
    '<path d="M256 116h-18v146q0 10 8 10h6q4 0 4-6z" fill="#0F1216"/>'
    '<rect x="62" y="116" width="176" height="170" rx="10" fill="url(#gb)"/>'
    '<rect x="98" y="132" width="104" height="138" rx="12" fill="#D0D5DD"/>'
    '<rect x="104" y="138" width="92" height="126" rx="8" fill="url(#gs)"/>'
    '<circle cx="150" cy="176" r="17" fill="none" stroke="#32D583" stroke-width="3"/>'
    '<path d="m152 164-9 13h7l-2 11 10-14h-7z" fill="#32D583"/>'
    '<text x="150" y="216" text-anchor="middle" font-size="13" font-weight="700" fill="#32D583" '
    'font-family="Pretendard, Malgun Gothic, sans-serif">사용 가능</text>'
    '<rect x="122" y="230" width="56" height="5" rx="2.5" fill="#22407F"/><rect x="122" y="241" width="38" height="5" rx="2.5" fill="#22407F"/>'
    '<circle cx="150" cy="278" r="3" fill="#32D583"/>'
    '<g class="ping"><circle cx="200" cy="136" r="12" fill="#F04438" stroke="#fff" stroke-width="3"/>'
    '<path d="M200 129.5v8" stroke="#fff" stroke-width="2.8" stroke-linecap="round"/><circle cx="200" cy="142" r="1.7" fill="#fff"/></g>'
    '<path d="M197 286c18 44 26 94 10 144-12 36-48 40-46 0 2-38 36-58 24-94" fill="none" stroke="#0D0F12" '
    'stroke-width="7" stroke-linecap="round"/>'
    '<rect x="172" y="318" width="15" height="30" rx="5" fill="#E4E7EC" transform="rotate(-8 180 333)"/>'
    '</svg>'
)

CSS = """
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css');
html, body, [class*="css"], .stMarkdown, .stDataFrame, button, input { font-family: Pretendard, 'Malgun Gothic', sans-serif; }
#MainMenu, footer, [data-testid="stToolbarActions"], [data-testid="stMainMenu"], [data-testid="stDecoration"] { display: none !important; }
header[data-testid="stHeader"] { border-bottom: 1px solid #EAECF0; }
[data-testid="stMain"] { overflow-x: hidden; }
[data-testid="stMainBlockContainer"] { padding-top: 5.6rem; padding-bottom: 2rem; max-width: 1240px; }
h1, h2, h3 { letter-spacing: -0.02em; }

/* 위쪽 메뉴 오른쪽: 기준 시각 */
.st-key-topbar { position: fixed; top: 0; right: 1.6rem; height: 3.75rem; z-index: 999995;
                 width: auto !important; align-items: center; }
@media (max-width: 760px) { .st-key-topbar { display: none; } }
.live { display: inline-flex; align-items: center; gap: .45rem; font-size: .82rem; color: #475467; white-space: nowrap; }
.live i { width: .5rem; height: .5rem; border-radius: 50%; background: #12B76A; box-shadow: 0 0 0 3px #D1FADF; }

/* 화면 제목 */
.ph { display: flex; align-items: center; gap: .9rem; margin: 0 0 .4rem; }
.ph .pi { width: 44px; height: 44px; border-radius: 12px; display: flex; align-items: center; justify-content: center; flex: none; }
.ph .pt { font-size: 1.45rem; font-weight: 800; color: #101828; letter-spacing: -.02em; line-height: 1.2; }
.ph .pd { color: #475467; font-size: .9rem; margin-top: .15rem; }
.ph .pn { color: #98A2B3; font-size: .78rem; margin-top: .1rem; }

.kpis { display: grid; grid-template-columns: 1fr 1fr; gap: .6rem; }
.kpi { border: 1px solid #EAECF0; border-radius: 12px; padding: .8rem .95rem; background: #fff; }
.kpi .l { color: #667085; font-size: .82rem; display: flex; align-items: center; gap: .4rem; }
.kpi .v { font-size: 1.7rem; font-weight: 800; color: #101828; line-height: 1.25; }
.kpi .s { color: #98A2B3; font-size: .76rem; }
.dot { width: .55rem; height: .55rem; border-radius: 50%; display: inline-block; }
.legend { color: #667085; font-size: .78rem; display: flex; gap: .9rem; flex-wrap: wrap; }
.sec { font-size: 1.12rem; font-weight: 700; color: #101828; margin: 1.4rem 0 .5rem; }
.wo-title { font-size: 1.02rem; font-weight: 700; color: #101828; margin-bottom: .2rem; }
.wo-id { color: #667085; font-size: .82rem; margin-bottom: .7rem; }
.kv { display: grid; grid-template-columns: 5.2rem 1fr; row-gap: .45rem; font-size: .9rem; }
.kv .k { color: #667085; } .kv .v { color: #101828; }
.badge { display: inline-block; padding: .05rem .5rem; border-radius: 999px; font-size: .78rem; font-weight: 600; color: #fff; }
.msg { background: #F2F4F7; border-radius: 14px; padding: .9rem 1.05rem; max-width: 640px; font-size: .9rem;
       color: #101828; margin-bottom: 1.1rem; }
.msg .h { font-weight: 700; margin-bottom: .1rem; }
.msg .t { color: #667085; font-size: .8rem; margin-bottom: .6rem; }
.msg .i { padding: .45rem 0; border-top: 1px solid #E4E7EC; }
.msg .i .r { color: #475467; font-size: .82rem; margin-top: .1rem; }
.chips { display: flex; flex-wrap: wrap; gap: .35rem; }
.chip { border-radius: 999px; padding: .1rem .55rem; font-size: .78rem; background: #F2F4F7; color: #344054; }

/* 지도 오른쪽 아래 ⓘ 버튼: 지도 위에 겹쳐 놓기 */
[class*="st-key-mapinfo"] { margin-top: -3.9rem; position: relative; z-index: 5; padding-right: .6rem; pointer-events: none; }
[class*="st-key-mapinfo"] button { pointer-events: auto; background: rgba(255,255,255,.94); border-color: #D0D5DD;
                                   min-height: 2.1rem; padding: 0 .55rem; }
.mi { font-size: .84rem; color: #344054; line-height: 1.65; }
.mi b { color: #101828; }

/* 도민 목록: 왼쪽·아래 여백 */
.st-key-cit_list { padding: .15rem .55rem 1.2rem .7rem; }
.st-key-cit_list [data-testid="stVerticalBlockBorderWrapper"], .st-key-cit_list .stVerticalBlock { gap: .7rem; }
.cn { font-weight: 700; color: #101828; }
.ca { color: #667085; font-size: .82rem; margin: .1rem 0 .45rem; }

.foot { border-top: 1px solid #EAECF0; margin-top: 2.5rem; padding-top: .9rem; color: #98A2B3; font-size: .76rem; line-height: 1.7; }
</style>
"""

HOME_CSS = """
<style>
header[data-testid="stHeader"] { background: #EDEFF2; border-bottom: none; }
[data-testid="stMainBlockContainer"] { padding-top: 3.75rem; }
.hero { position: relative; margin: 0 calc(50% - 50vw); background: linear-gradient(180deg, #EDEFF2 0%, #F2F3F5 100%); overflow: hidden; }
.hero-in { position: relative; z-index: 2; max-width: 1180px; margin: 0 auto; padding: 2.2rem 2rem 0;
           display: grid; grid-template-columns: 1.1fr .9fr; align-items: center; min-height: 560px; }
.hero .h1 { font-size: clamp(2.8rem, 5.6vw, 4.6rem); font-weight: 800; letter-spacing: -.045em; line-height: 1.05;
           color: #0B0F19; margin: 0; padding: 0; }
.hero .lead { font-size: clamp(1.08rem, 1.7vw, 1.4rem); font-weight: 600; color: #1D2939; margin: 1rem 0 2.6rem; letter-spacing: -.02em; }
.feats { list-style: none; margin: 0; padding: 0; }
.feats li { display: flex; align-items: center; gap: 1rem; margin: 0 0 1.05rem; font-size: 1.04rem; color: #1D2939; }
.feats li svg { flex: none; color: #101828; }
.hero-art { position: relative; height: 560px; display: flex; justify-content: center; align-items: flex-end; }
.hero-art .charger { height: 540px; width: auto; position: relative; z-index: 2; margin-bottom: -6px; }
.hero-art .ping { animation: blink 1.6s ease-in-out infinite; }
@keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: .35; } }
.scan { position: absolute; z-index: 3; left: -6%; top: 57%; width: 250px; background: #fff; border-radius: 14px;
        padding: .8rem .95rem .85rem; box-shadow: 0 14px 36px rgba(16, 24, 40, .14); border: 1px solid #EAECF0; }
.scan .h { display: flex; align-items: center; gap: .45rem; font-size: .76rem; font-weight: 700; color: #D92D20; }
.scan .h i { width: .5rem; height: .5rem; border-radius: 50%; background: #F04438; box-shadow: 0 0 0 3px #FEE4E2; }
.scan .t { font-size: .95rem; font-weight: 700; color: #101828; margin: .3rem 0 .15rem; }
.scan .b { font-size: .8rem; color: #475467; line-height: 1.45; }
.scan .f { font-size: .72rem; color: #98A2B3; margin-top: .45rem; }
.wave { position: absolute; left: 0; bottom: -1px; width: 100%; height: 150px; z-index: 1; display: block; }
@media (max-width: 900px) {
  .hero-in { grid-template-columns: 1fr; min-height: 0; padding-top: 2.4rem; }
  .hero-art { height: 340px; margin-top: .5rem; justify-content: flex-end; padding-right: 6%; }
  .hero-art .charger { height: 320px; }
  .scan { left: 0; top: 57%; width: 205px; padding: .65rem .8rem; }
  .scan .b { font-size: .76rem; }
  .wave { height: 90px; }
}
.home-h { font-size: 1.15rem; font-weight: 800; color: #101828; margin: 1.6rem 0 .1rem; letter-spacing: -.02em; }
.home-s { color: #667085; font-size: .86rem; margin: 0 0 .4rem; }
.role .ic { width: 44px; height: 44px; border-radius: 12px; display: flex; align-items: center; justify-content: center; margin-bottom: .7rem; }
.role .ti { font-size: 1.08rem; font-weight: 800; color: #101828; }
.role .de { color: #475467; font-size: .87rem; line-height: 1.5; margin: .25rem 0 .8rem; min-height: 2.6rem; }
.role .nu { font-size: 1.6rem; font-weight: 800; color: #101828; line-height: 1.1; }
.role .nl { color: #667085; font-size: .78rem; }
.steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: .8rem; }
.step { background: #F9FAFB; border: 1px solid #EAECF0; border-radius: 12px; padding: .85rem 1rem; }
.step .n { font-size: .74rem; font-weight: 700; color: #1D4ED8; }
.step .t { font-weight: 700; color: #101828; margin: .15rem 0 .2rem; font-size: .95rem; }
.step .d { color: #667085; font-size: .82rem; line-height: 1.5; }
@media (max-width: 760px) { .steps { grid-template-columns: 1fr; } }
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


def data_sig():
    """데이터 파일이 바뀌면 다시 계산하도록 파일 크기·수정 시각을 묶은 값"""
    fs = sorted(glob.glob(f"{DATA}/status/*.csv.gz")) + glob.glob(f"{DATA}/latest.csv.gz")
    return tuple((f, os.path.getsize(f), int(os.path.getmtime(f))) for f in fs)


@st.cache_data(ttl=1800, max_entries=2, show_spinner="진단 계산 중입니다 (약 20초)")
def load(sig):
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
CIT_LEGEND = [("사용가능", "사용 가능"), ("충전중", "충전 중"), ("관찰", "방문 전 확인"), ("경보", "이용 불가")]
GOV_INFO = ("<b>경보</b> 바로 점검이 필요한 고장 · <b>주의</b> 지켜보며 확인 · <b>관찰</b> 숨은 고장 가능성 · "
            "<b>철거·정리 검토</b> 고장 상태로 1년 넘게 미사용")
CIT_INFO = ("<b>방문 전 확인</b> 최근 정상 충전 기록이 없거나 상태 신호가 잠시 끊긴 곳 · "
            "<b>이용 불가</b> 운영 중지·점검 중이거나 6시간 넘게 신호가 끊긴 곳")


PAGE_TONE = {"gov": ("#EEF4FF", "#1D4ED8"), "op": ("#FEF0E7", "#E8761E"), "cit": ("#E7F6EC", "#12B76A")}


def page_head(title, ic, desc, note):
    bg, fg = PAGE_TONE[ic]
    st.markdown(f"<div class='ph'><div class='pi' style='background:{bg};color:{fg}'>{icon(ic, 22)}</div>"
                f"<div><div class='pt'>{title}</div><div class='pd'>{desc}</div><div class='pn'>※ {note}</div></div></div>",
                unsafe_allow_html=True)


def map_deck(df, key, height=430, focus=None, legend=GOV_LEGEND, big=ALERT, info=GOV_INFO):
    """focus=(위도, 경도, 이름) 이면 그곳을 확대하고 테두리로 강조"""
    df = df.dropna(subset=["lat", "lng"])[["lat", "lng", "표시", "statNm", "chgerId", "판정", "grade", "근거"]].copy()
    df["color"] = df["표시"].map(RGB)
    df["r"] = np.where(df["표시"].isin(big), 110, 45)
    df = df.sort_values("표시", key=lambda s: s.map({"정상": 0, "확인불가": 0, "충전중": 1, "사용가능": 2,
                                                   "관찰": 3, "주의": 4, "행정": 4, "경보": 5}))
    layers = [pdk.Layer("ScatterplotLayer", df, get_position=["lng", "lat"], get_fill_color="color",
                            get_radius="r", radius_min_pixels=2.5, radius_max_pixels=9, pickable=True)]
    tip = {"html": "<b>{statNm}</b> {chgerId}번<br/>{판정} {grade}<br/>{근거}",
           "style": {"fontSize": "12px", "backgroundColor": "#101828", "color": "white"}}
    if focus:
        view = pdk.ViewState(latitude=focus[0], longitude=focus[1], zoom=15.2)
        ring = pd.DataFrame([{"lat": focus[0], "lng": focus[1]}])
        layers.append(pdk.Layer("ScatterplotLayer", ring, get_position=["lng", "lat"], get_radius=45,
                                filled=False, stroked=True, get_line_color=[29, 78, 216, 255],
                                line_width_min_pixels=3, radius_min_pixels=14))
    else:
        view = pdk.ViewState(latitude=33.37, longitude=126.55, zoom=8.75)
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, tooltip=tip,
                             map_style="light"), height=height, width="stretch")
    # 지도 오른쪽 아래 ⓘ : 지도 보는 법
    with st.container(key=f"mapinfo_{key}", horizontal=True, horizontal_alignment="right"):
        with st.popover("", icon=":material/info:", help="지도 보는 법"):
            st.markdown(
                "<div class='mi'><b>점 하나 = 충전기 1대</b>. 확인이 필요한 곳은 크게 표시합니다.<br>"
                f"{info}<br>점에 마우스를 올리면 충전소 이름과 판정 근거가 보입니다.<br>"
                f"기준 {attrs['now']:%m월 %d일 %H:%M} · 30분마다 새 데이터로 갱신<br>"
                "배경 지도 © CARTO, © OpenStreetMap 기여자</div>", unsafe_allow_html=True)
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
    st.altair_chart(c, width="stretch")
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
                    ).properties(height=220), width="stretch")
                else:
                    st.caption("같은 충전소에 비교할 충전기가 없습니다.")
        st.divider()
        c1, c2 = st.columns([3, 1], vertical_alignment="center")
        with c1:
            st.radio("현장 확인 결과 (점검 후 기록)", RESULT, index=None, horizontal=True, key=f"res_{r['고유번호']}")
        with c2:
            st.download_button("지시서 내려받기", work_order_text(r).encode("utf-8"),
                               f"점검지시서_{r['고유번호']}.txt", width="stretch", type="primary", on_click="ignore")


def pick_table(df, cols, key, height=360):
    """표의 아무 칸이나 한 번 누르면 그 행 번호를 돌려준다 (체크박스 없음)"""
    kw = dict(hide_index=True, width="stretch", height=height, on_select="rerun", key=key,
              column_config={"순위": st.column_config.NumberColumn(width="small"),
                             "충전기 번호": st.column_config.TextColumn(width="small",
                                                                 help="운영사가 충전소 안에서 붙인 번호"),
                             "고유번호": st.column_config.TextColumn(help="환경부 통합 충전기 ID (충전소ID-충전기 번호)"),
                             "근거": st.column_config.TextColumn(width="large")})
    try:
        ev = st.dataframe(df[cols], selection_mode="single-cell", **kw)
    except Exception:   # 칸 선택을 지원하지 않는 옛 버전이면 행 선택(체크박스)으로
        ev = st.dataframe(df[cols], selection_mode="single-row", **kw)
    sel = getattr(ev, "selection", None)
    cells = getattr(sel, "cells", None) if sel is not None else None
    rows = [c[0] for c in cells] if cells else list(getattr(sel, "rows", None) or [])
    i = int(rows[0]) if rows else 0
    return i if 0 <= i < len(df) else 0


def table_cols(df):
    return df.rename(columns={"statNm": "충전소", "chgerId": "충전기 번호", "grade": "등급"})


def hidden_example():
    """표지 카드에 보여 줄 실제 숨은고장 경보 1건 (점검 순서 1등)"""
    b = pub[(pub["트랙"] == "수리") & pub["code"].isin(["B1", "B2"]) & (pub["grade"] == "경보")]
    return b.sort_values("raw", ascending=False).iloc[0] if len(b) else None


# ---------------- 화면 ----------------
def home():
    st.html(HOME_CSS)
    ex = hidden_example()
    if ex is not None:
        card = (f"<div class='scan'><div class='h'><i></i>AI 진단 · 숨은 고장 의심</div>"
                f"<div class='t'>{html.escape(str(ex['statNm']))} {html.escape(str(ex['chgerId']))}번</div>"
                f"<div class='b'>화면은 '사용 가능'이지만 {html.escape(str(ex['근거']))}</div>"
                f"<div class='f'>{attrs['now']:%m월 %d일 %H:%M} 실제 진단 결과</div></div>")
    else:
        card = ("<div class='scan'><div class='h'><i></i>AI 진단 · 숨은 고장 의심</div>"
                "<div class='t'>멀쩡한 척하는 충전기</div>"
                "<div class='b'>화면은 '사용 가능'이지만 며칠째 충전이 0회</div></div>")
    feats = [("robot", "숨은 고장을 찾아주는 AI 에이전트"), ("shield", "전기차 충전기 관리자의 든든한 조력자"),
             ("users", "사용자들의 불편함을 해소")]
    st.markdown(
        "<div class='hero'><div class='hero-in'><div class='hero-text'>"
        "<div class='h1'>고장났수다</div><p class='lead'>고장난 전기차 충전기를 찾아주는 AI 에이전트</p>"
        "<ul class='feats'>" + "".join(f"<li>{icon(i)}<span>{t}</span></li>" for i, t in feats) + "</ul></div>"
        f"<div class='hero-art'>{CHARGER_SVG}{card}</div></div>"
        "<svg class='wave' viewBox='0 0 1440 150' preserveAspectRatio='none'>"
        "<path d='M0 118C220 84 430 70 700 82s520 26 740-48V150H0z' fill='#fff'/></svg></div>",
        unsafe_allow_html=True)

    rep = pub[pub["트랙"] == "수리"]
    n_alarm = int((rep["grade"] == "경보").sum())
    c = d[(d["limitYn"] != "Y") & ~d["statNm"].fillna("").str.contains("버스|차고지|회차지|여객")]
    roles = [
        (PAGES["gov"], "공공기관", "gov", "#EEF4FF", "#1D4ED8",
         "제주 공공 충전기를 지도 한 장으로 보고, 오늘 점검할 곳과 점검 지시서를 바로 뽑습니다.",
         f"{len(pub):,}대", "진단 중인 공공 충전기"),
        (PAGES["op"], "운영사", "op", "#FEF0E7", "#E8761E",
         "내 충전기 중 손볼 곳만 골라 점검 순서대로 알림을 보냅니다.",
         f"{n_alarm}건", "오늘 점검 요청 (경보)"),
        (PAGES["cit"], "도민", "cit", "#E7F6EC", "#12B76A",
         "가기 전에 충전소 상태를 확인하고, 헛걸음할 곳은 미리 피합니다.",
         f"{int((c['stat'] == '2').sum()):,}대", "지금 사용 가능"),
    ]
    st.markdown("<div class='home-h'>누구를 위한 서비스인가요</div>"
                "<p class='home-s'>쓰는 사람에 따라 화면이 나뉩니다. 위쪽 메뉴로도 이동할 수 있어요.</p>",
                unsafe_allow_html=True)
    cols = st.columns(3, gap="medium")
    for col, (page, title, ic, bg, fg, desc, num, lab) in zip(cols, roles):
        with col, st.container(border=True):
            st.markdown(f"<div class='role'><div class='ic' style='background:{bg};color:{fg}'>{icon(ic, 24)}</div>"
                        f"<div class='ti'>{title}</div><div class='de'>{desc}</div>"
                        f"<div class='nu'>{num}</div><div class='nl'>{lab}</div></div>", unsafe_allow_html=True)
            st.page_link(page, label=f"{title} 화면 열기", icon=":material/arrow_forward:")

    st.markdown(
        "<div class='home-h'>어떻게 찾나요</div><p class='home-s'>멀쩡해 보이는 충전기도 이용 기록으로 고장을 찾아냅니다.</p>"
        "<div class='steps'>"
        "<div class='step'><div class='n'>1단계</div><div class='t'>30분마다 상태 수집</div>"
        "<div class='d'>환경부 API로 제주 충전기 전부의 상태와 마지막 충전 시각을 모읍니다.</div></div>"
        "<div class='step'><div class='n'>2단계</div><div class='t'>이웃 충전기와 비교</div>"
        "<div class='d'>같은 충전소·주변 충전기는 잘 쓰이는데 혼자만 오래 안 쓰이면 숨은 고장으로 봅니다.</div></div>"
        "<div class='step'><div class='n'>3단계</div><div class='t'>점검 순서와 알림</div>"
        "<div class='d'>이용량, 방치 기간, 주변 대체 충전기를 따져 먼저 고칠 곳부터 알려 줍니다.</div></div>"
        "</div>", unsafe_allow_html=True)


def gov_page():
    page_head("공공기관", "gov", "제주 공공 충전기의 고장·방치 현황을 보고 오늘 점검할 곳을 정합니다.", NOTE["gov"])
    opts = [b for b in BUSI if (pub["busiId"] == b).any()]   # 관리 충전기가 없는 기관은 빼기
    sel = st.pills("운영기관", opts, selection_mode="multi", default=opts, format_func=lambda x: BUSI[x],
                   label_visibility="collapsed", key="gov_busi")
    v = pub[pub["busiId"].isin(sel)] if sel else pub
    m, k = st.columns([1.55, 1], gap="large")
    with m:
        map_deck(v, "gov")
    with k:
        st.markdown("<div class='kpis'>" +
                    kpi("경보", int((v["표시"] == "경보").sum()), "경보", "바로 점검 필요") +
                    kpi("주의", int((v["표시"] == "주의").sum()), "주의", "지켜보며 확인") +
                    kpi("관찰", int((v["표시"] == "관찰").sum()), "관찰", "숨은고장 가능성") +
                    kpi("철거·정리 검토", int((v["표시"] == "행정").sum()), "행정", "1년 넘게 고장·미사용") +
                    "</div>" +
                    f"<div class='kpi' style='margin-top:.6rem'><div class='l'>대상 충전기</div><div class='v'>{len(v):,}대</div>"
                    f"<div class='s'>{'전체 공공 운영기관' if not sel or len(sel) == len(opts) else ', '.join(BUSI[x] for x in sel)}</div></div>",
                    unsafe_allow_html=True)

    st.markdown("<div class='sec'>오늘 점검할 곳</div>", unsafe_allow_html=True)
    rep = v[v["트랙"] == "수리"].sort_values("raw", ascending=False).reset_index(drop=True)
    rep.insert(0, "순위", rep.index + 1)
    only_alarm = st.toggle("경보만 보기", value=True, key="gov_alarm")
    show = rep[rep["grade"] == "경보"].reset_index(drop=True) if only_alarm else rep
    if len(show):
        t = table_cols(show)
        i = pick_table(t, ["순위", "충전소", "충전기 번호", "판정", "등급", "방치", "근거"], "gov_tbl")
        st.caption("표에서 아무 칸이나 한 번 누르면 그 충전기의 점검 지시서가 아래에 열립니다.")
        work_order(d, show.loc[i])
        out = t[["순위", "운영기관", "충전소", "충전기 번호", "고유번호", "종류", "판정", "등급", "방치", "근거",
                 "addr", "점수"]].rename(columns={"addr": "주소"})
        st.download_button("점검 목록 내려받기 (CSV)", out.to_csv(index=False).encode("utf-8-sig"), "점검목록.csv",
                           on_click="ignore")
    else:
        st.info("해당하는 충전기가 없습니다.")

    adm = v[v["표시"] == "행정"]
    with st.expander(f"철거·목록 정리 검토 {len(adm)}대 · 고장 상태로 1년 넘게 미사용"):
        st.dataframe(table_cols(adm)[["운영기관", "충전소", "충전기 번호", "고유번호", "판정", "근거", "addr"]]
                     .rename(columns={"addr": "주소"}), hide_index=True, width="stretch")
    if len(coord_err):
        with st.expander(f"등록정보 오류 {len(coord_err)}건 · 좌표가 제주 밖으로 등록됨"):
            st.caption("같은 주소의 정상 좌표로 자동 보정했습니다. 운영사에 정정을 요청하세요.")
            st.dataframe(coord_err.rename(columns={"statNm": "충전소", "statId": "충전소ID", "addr": "주소",
                                                   "lat": "등록 위도", "lng": "등록 경도"}),
                         hide_index=True, width="stretch")


def op_page():
    page_head("운영사", "op", "내 충전기 중 손볼 곳을 점검 순서대로 정리해 알림으로 보냅니다.", NOTE["op"])
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
        st.markdown(f"<div class='msg'><div class='h'>[고장났수다] {op} 점검 요청</div>"
                    f"<div class='t'>{attrs['now']:%m월 %d일 %H:%M} 기준 · 경보 {len(alarm)}건 중 상위 {len(top)}건</div>"
                    f"{items or '<div class=i>오늘 경보가 없습니다.</div>'}</div>", unsafe_allow_html=True)
        tok, chat = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
        if st.button("텔레그램으로 보내기", icon=":material/send:", disabled=not (tok and chat),
                     help=None if tok and chat else "텔레그램 봇 설정 후 사용할 수 있습니다."):
            import urllib.parse
            import urllib.request
            text = "\n".join([f"[고장났수다] {op} 점검 요청 ({attrs['now']:%m/%d %H:%M})"] +
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
                         hide_index=True, width="stretch", height=420,
                         column_config={"충전기 번호": st.column_config.TextColumn(help="운영사가 충전소 안에서 붙인 번호"),
                                        "고유번호": st.column_config.TextColumn(help="환경부 통합 충전기 ID")})
            st.download_button("목록 내려받기 (CSV)", table_cols(lst)[["충전소", "충전기 번호", "고유번호", "종류", "판정", "등급",
                                                                       "방치", "근거", "addr"]].rename(columns={"addr": "주소"})
                               .to_csv(index=False).encode("utf-8-sig"), f"{op}_점검목록.csv", on_click="ignore")
        else:
            st.info("해당하는 충전기가 없습니다.")


def cit_status(r):
    """도민에게 보여 줄 상태 4가지"""
    if r["stat"] in ("4", "5") or r["code"] in ("A2", "A3"):
        return "이용 불가"
    if r["code"] in ("B1", "B2", "B3") and r["grade"] in ("경보", "관찰"):
        return "방문 전 확인"
    return {"2": "사용 가능", "3": "충전 중"}.get(r["stat"], "방문 전 확인")   # 6시간 미만 신호 끊김·예약 중


def cit_reason(r):
    if r["상태"] != "방문 전 확인":
        return r["상태"]
    if r["code"] in ("B1", "B2", "B3"):
        return f"최근 {r['방치']} 동안 정상 충전 기록 없음"
    return "예약 중" if r["stat"] == "6" else "상태 신호가 잠시 끊김"


def cit_page():
    page_head("도민", "cit", "가기 전에 충전소의 지금 상태를 확인하세요.", NOTE["cit"])
    # 시민이 쓸 수 없는 곳(이용 제한, 버스 차고지)은 도민 화면에서 제외
    c = d[(d["limitYn"] != "Y") & ~d["statNm"].fillna("").str.contains("버스|차고지|회차지|여객")].copy()
    c["상태"] = c.apply(cit_status, axis=1)
    c["표시"] = c["상태"].map({"이용 불가": "경보", "방문 전 확인": "관찰", "사용 가능": "사용가능", "충전 중": "충전중"})
    c["근거"] = c.apply(cit_reason, axis=1)
    c["판정"], c["grade"] = c["상태"], ""
    q = st.text_input("충전소 검색", placeholder="충전소 이름이나 지역 (예: 한림, 공항, 교래)", label_visibility="collapsed")
    hit = c[c["statNm"].fillna("").str.contains(q, regex=False) | c["addr"].fillna("").str.contains(q, regex=False)] if q else c
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
            if fb.button("전체 지도", width="stretch", key="cit_reset"):
                st.session_state["cit_focus"] = None
                st.rerun()
        map_deck(hit, "cit", height=520, focus=focus, legend=CIT_LEGEND, big={"사용가능", "충전중", "관찰", "경보"},
                 info=CIT_INFO)
    order = ["사용 가능", "충전 중", "방문 전 확인", "이용 불가"]
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
        tone = {"사용 가능": "#E7F6EC", "충전 중": "#EEF4FF", "방문 전 확인": "#FEF7E0", "이용 불가": "#FEE4E2"}
        with st.container(height=520, border=False, key="cit_list"):
            for _, r in g.head(30).iterrows():
                chips = "".join(f"<span class='chip' style='background:{tone[s]}'>{s} {int(r[s])}</span>"
                                for s in order if r[s])
                with st.container(border=True):
                    a, b = st.columns([4, 1.4], gap="medium", vertical_alignment="center")
                    a.markdown(f"<div class='cn'>{r['statNm']}</div><div class='ca'>{r['addr']}</div>"
                               f"<div class='chips'><span class='chip'>급속 {int(r['급속'])} · 완속 {int(r['완속'])} · 전체 {int(r['전체'])}</span></div>"
                               f"<div class='chips' style='margin-top:.35rem'>{chips}</div>", unsafe_allow_html=True)
                    on = r["statId"] == focus_id
                    if b.button("보는 중" if on else "지도에서 보기", key=f"go_{r['statId']}", width="stretch",
                                type="primary" if on else "secondary", disabled=on):
                        st.session_state["cit_focus"] = r["statId"]
                        st.rerun()
        if len(g) > 30:
            st.caption(f"외 {len(g) - 30}곳. 검색어를 더 좁혀 보세요.")


# ---------------- 실행 ----------------
st.logo(LOGO_SVG, size="large")
PAGES = {
    "home": st.Page(home, title="홈", icon=":material/home:", url_path="home", default=True),
    "gov": st.Page(gov_page, title="공공기관", icon=":material/account_balance:", url_path="gov"),
    "op": st.Page(op_page, title="운영사", icon=":material/engineering:", url_path="operator"),
    "cit": st.Page(cit_page, title="도민", icon=":material/electric_car:", url_path="citizen"),
}
nav = st.navigation(list(PAGES.values()), position="top")
st.html(CSS)
d, attrs, coord_err = load(data_sig())
pub = d[d["busiId"].isin(PUBLIC)]

nav.run()

st.markdown(
    f"<div class='foot'>기준 시각 {attrs['now']:%Y-%m-%d %H:%M} · 수집 기간 {attrs['t_first']:%m/%d}~{attrs['now']:%m/%d} "
    f"(30분 간격, {attrs['n_snaps']}회)<br>"
    "데이터: 한국환경공단 전기자동차 충전소 정보 API · 제주데이터허브 [인구, 생활] 전기자동차 충전소 정보(862, 2025.9.21 기준) · "
    "한국전력공사 전기차 충전소 충전량(2023.1~2026.9)<br>고장났수다 · 제주 AI·빅데이터 활용 AI Agent 개발 챌린지</div>",
    unsafe_allow_html=True)

# 위쪽 메뉴 오른쪽에 고정 (CSS .st-key-topbar)
with st.container(key="topbar", horizontal=True, vertical_alignment="center"):
    st.markdown(f"<span class='live'><i></i>{attrs['now']:%m/%d %H:%M} 기준 · 30분마다 갱신</span>",
                unsafe_allow_html=True, width="content")
