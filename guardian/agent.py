"""고장났수다 - AI 에이전트 (점검 담당자 비서)

하는 일
  1. 질문을 받으면 도구(아래 TOOLS)를 골라 진단 데이터에서 직접 계산하고, 그 결과로만 답한다.
  2. 답 속 숫자를 도구 결과와 하나씩 맞춰 본다(숫자 대조). 맞지 않는 숫자가 있으면 한 번 고쳐 쓰게 하고,
     그래도 안 맞으면 경고와 함께 원래 계산 결과를 보여 준다.
  3. 점검 현황 보고서를 만든다. 표와 숫자는 코드가 계산하고, LLM은 첫머리 요약 문단만 쓴다.
  4. 운영사 점검 요청서 초안을 쓴다. 실제 발송은 담당자가 확인 버튼을 눌러야 한다.

LLM: OpenAI Chat Completions(도구 호출). 키는 환경변수 OPENAI_API_KEY, 모델은 OPENAI_MODEL(기본 gpt-4.1-mini).
키가 없으면 정해진 질문 버튼만 동작한다(도구 실행 + 정해진 문장).
"""
import json
import os
import re
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

from guardian.diagnose import DATA, PUBLIC

MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
API_URL = "https://api.openai.com/v1/chat/completions"
MAX_STEPS = 5
BUSI = {"JD": "제주특별자치도", "ME": "환경부", "JC": "제주에너지공사", "JE": "제주전기자동차서비스",
        "JT": "제주테크노파크"}
GRADE_HELP = "경보=바로 점검, 주의=지켜보며 확인(일시적일 수 있음), 관찰=숨은 고장 가능성"
EXCLUDE = "버스|차고지|회차지|여객"
STOP = {"급속", "완속", "충전기", "충전소", "어디", "어디야", "어디에", "지금", "사용", "가능", "가능한", "쓸", "수", "있는", "근처", "알려줘", "찾아줘", "상태"}


# ---------------- 공통 ----------------
def cit_status(r):
    """도민에게 보여 줄 상태 4가지"""
    if r["stat"] in ("4", "5") or r["code"] in ("A2", "A3"):
        return "이용 불가"
    if r["code"] in ("B1", "B2", "B3") and r["grade"] in ("경보", "관찰"):
        return "방문 전 확인"
    return {"2": "사용 가능", "3": "충전 중"}.get(r["stat"], "방문 전 확인")


def _op(df, operator):
    """운영기관 이름(일부)이나 코드로 거르기"""
    if not operator:
        return df
    o = str(operator).strip()
    codes = [k for k, v in BUSI.items() if o.upper() == k or o in v or v in o]
    return df[df["busiId"].isin(codes)] if codes else df.iloc[0:0]


def _rows(df, cols, limit):
    out = df[cols].head(limit).copy()
    for c in out.columns:
        if out[c].dtype.kind == "f":
            out[c] = out[c].round(1)
    return out.fillna("").to_dict("records")


def _repair(d):
    return d[d["busiId"].isin(PUBLIC) & (d["트랙"] == "수리")].sort_values("raw", ascending=False)


# ---------------- 도구 ----------------
def t_overview(d, attrs, operator=None):
    pub = _op(d[d["busiId"].isin(PUBLIC)], operator)
    rep = pub[pub["트랙"] == "수리"]
    by = rep.groupby(["판정", "grade"]).size().rename("대수").reset_index() \
        .rename(columns={"grade": "등급"}).sort_values("대수", ascending=False)
    return {"기준시각": f"{attrs['now']:%Y-%m-%d %H:%M}",
            "대상": BUSI.get(str(operator).upper(), operator) if operator else "공공 운영기관 전체",
            "공공충전기_대수": int(len(pub)),
            "경보": int((rep["grade"] == "경보").sum()), "주의": int((rep["grade"] == "주의").sum()),
            "관찰": int((rep["grade"] == "관찰").sum()),
            "철거정리검토_1년넘게미사용": int((pub["트랙"].str.startswith("행정")).sum()),
            "판정별": by.to_dict("records"),
            "운영기관별_경보": rep[rep["grade"] == "경보"].groupby("운영기관").size().sort_values(ascending=False).to_dict()}


def t_priority(d, attrs, operator=None, grade=None, code=None, limit=10):
    r = _op(_repair(d), operator).reset_index(drop=True)
    r.insert(0, "순위", r.index + 1)
    if grade is None:   # 판정을 콕 집어 물으면 경보·주의·관찰 모두, 아니면 경보만
        grade = "전체" if code else "경보"
    if grade and grade != "전체":
        r = r[r["grade"] == grade]
    if code:
        r = r[r["판정"].str.contains(str(code)) | (r["code"] == code)]
    limit = int(min(max(limit or 10, 1), 30))
    return {"기준시각": f"{attrs['now']:%Y-%m-%d %H:%M}", "조건": {"운영기관": operator or "전체", "등급": grade or "전체"},
            "해당_대수": int(len(r)), "등급별_대수": r["grade"].value_counts().to_dict(),
            "목록": _rows(r.rename(columns={"statNm": "충전소", "chgerId": "번호", "grade": "등급"}),
                        ["순위", "충전소", "번호", "운영기관", "판정", "등급", "방치", "근거", "고유번호"], limit)}


def t_search(d, attrs, query, limit=8):
    q = str(query).strip()
    c = d[(d["limitYn"] != "Y") & ~d["statNm"].fillna("").str.contains(EXCLUDE)].copy()
    # 질문 말투(어디야, 지금, 쓸 수 있는…)는 빼고, 급속·완속은 종류 조건으로 쓴다
    words = [re.sub(r"(에서|으로|에|의|은|는|이|가|을|를|로)$", "", w) for w in re.split(r"[\s?,.!]+", q) if w not in STOP]
    words = [w for w in words if len(w) >= 2 and w not in STOP]
    if "급속" in q:
        c = c[c["fast"]]
    elif "완속" in q:
        c = c[~c["fast"]]
    if not words:
        return {"검색어": q, "결과": "충전소 이름이나 지역을 넣어 주세요"}
    m = pd.Series(True, index=c.index)
    for w in words:
        m &= c["statNm"].fillna("").str.contains(w, regex=False) | c["addr"].fillna("").str.contains(w, regex=False)
    hit = c[m]
    if hit.empty:
        return {"검색어": q, "결과": "일치하는 충전소 없음"}
    hit = hit.assign(상태=hit.apply(cit_status, axis=1))
    out = []
    groups = sorted(hit.groupby(["statId", "statNm", "addr"]), key=lambda kv: -(kv[1]["상태"] == "사용 가능").sum())
    for (sid, nm, ad), g in groups[: int(limit)]:
        out.append({"충전소": nm, "주소": ad, "충전소ID": sid,
                    "급속": int(g["fast"].sum()), "완속": int((~g["fast"]).sum()),
                    **{s: int((g["상태"] == s).sum()) for s in ["사용 가능", "충전 중", "방문 전 확인", "이용 불가"]},
                    "문제있는_충전기": [f"{x.chgerId}번 {x.판정}({x.grade}) {x.근거}" for x in g.itertuples() if x.code]})
    return {"기준시각": f"{attrs['now']:%Y-%m-%d %H:%M}", "검색어": q, "충전소_수": int(hit["statId"].nunique()), "결과": out}


def t_detail(d, attrs, station, charger_no=None):
    s = d[d["statNm"].fillna("").str.contains(str(station), regex=False) | (d["statId"] == station)]
    if charger_no:
        s = s[s["chgerId"].str.lstrip("0") == str(charger_no).lstrip("0")]
    if s.empty:
        return {"결과": "해당 충전기 없음", "충전소": station, "번호": charger_no}
    s = s.sort_values(["statNm", "chgerId"]).head(20)
    last = pd.to_datetime(s["lastTsdt"], format="%Y%m%d%H%M%S", errors="coerce")
    s = s.assign(마지막충전=last.dt.strftime("%Y-%m-%d %H:%M").fillna("기록 없음"),
                 현재상태=s.apply(cit_status, axis=1), 수집기간_충전횟수=s["k"].astype(int))
    return {"기준시각": f"{attrs['now']:%Y-%m-%d %H:%M}",
            "충전기": _rows(s.rename(columns={"statNm": "충전소", "chgerId": "번호", "grade": "등급"}),
                         ["충전소", "번호", "종류", "운영기관", "현재상태", "판정", "등급", "방치", "근거", "마지막충전",
                          "수집기간_충전횟수", "고유번호"], 20)}


def t_neglect(d, attrs, operator=None, limit=15):
    a = _op(d[d["busiId"].isin(PUBLIC) & d["트랙"].str.startswith("행정")], operator)
    last = pd.to_datetime(a["lastTsdt"], format="%Y%m%d%H%M%S", errors="coerce")
    a = a.assign(마지막충전=last.dt.strftime("%Y-%m-%d").fillna("기록 없음"))
    return {"기준시각": f"{attrs['now']:%Y-%m-%d %H:%M}", "해당_대수": int(len(a)),
            "설명": "고장 상태로 1년 넘게 충전 기록이 없는 충전기. 수리보다 철거·목록 정리 검토 대상",
            "목록": _rows(a.rename(columns={"statNm": "충전소", "chgerId": "번호"}),
                        ["충전소", "번호", "운영기관", "마지막충전", "판정", "근거", "고유번호"], int(limit))}


def t_hub_change(d, attrs):
    h = pd.DataFrame(json.load(open(f"{DATA}/hub/stations.json", encoding="utf-8")))
    h["n"] = h["chargerCount"].astype(int) + h["quickChargerCount"].astype(int)
    before = h.groupby("provider")["n"].sum()
    now = d[d["busiId"].isin(PUBLIC)].groupby("busiId").size()
    rows = []
    for k in PUBLIC:
        b, n = int(before.get(k, 0)), int(now.get(k, 0))
        rows.append({"운영기관": BUSI[k], "2025년9월_허브": b, "현재": n, "증감": n - b,
                     "증감률(%)": round((n - b) / b * 100, 1) if b else None})
    return {"비교": "제주데이터허브 [인구, 생활] 전기자동차 충전소 정보(862, 2025.9.21 기준) 대비 현재(환경공단 API)",
            "기준시각": f"{attrs['now']:%Y-%m-%d %H:%M}", "운영기관별": rows}


def t_operator_message(d, attrs, operator):
    r = _op(_repair(d), operator)
    if r.empty and not _op(d, operator).shape[0]:
        return {"결과": "해당 운영기관 없음", "운영기관": operator}
    a = r[r["grade"] == "경보"].head(10)
    name = BUSI.get(str(operator).upper(), a["운영기관"].iloc[0] if len(a) else operator)
    lines = [f"[고장났수다] {name} 점검 요청 ({attrs['now']:%m/%d %H:%M} 기준)",
             f"경보 {int((r['grade'] == '경보').sum())}건 중 우선순위 상위 {len(a)}건입니다."]
    lines += [f"{i}. {x.statNm} {x.chgerId}번 ({x.고유번호}) · {x.판정}\n   {x.근거}" for i, x in enumerate(a.itertuples(), 1)]
    lines.append("점검 후 결과(수리 완료 / 부품 대기 / 점유 / 오탐)를 회신해 주세요.")
    return {"운영기관": name, "경보_건수": int((r["grade"] == "경보").sum()), "초안": "\n".join(lines),
            "주의": "발송은 담당자가 확인한 뒤에만 합니다."}


TOOL_FUNCS = {"overview": t_overview, "priority_list": t_priority, "search_station": t_search,
              "charger_detail": t_detail, "long_neglect": t_neglect, "hub_change": t_hub_change,
              "operator_message": t_operator_message}

_OP = {"type": "string", "description": "운영기관 이름 또는 코드 (제주특별자치도/JD, 환경부/ME, 제주에너지공사/JC, 제주전기자동차서비스/JE). 없으면 전체"}
TOOLS = [
    {"type": "function", "function": {"name": "overview", "description": "공공 충전기 전체(또는 운영기관별) 고장 현황 숫자",
                                      "parameters": {"type": "object", "properties": {"operator": _OP}}}},
    {"type": "function", "function": {"name": "priority_list", "description": "점검 우선순위 목록. 오늘 점검할 곳, 경보 목록, 특정 판정(통신두절·숨은고장 등) 목록",
                                      "parameters": {"type": "object", "properties": {
                                          "operator": _OP,
                                          "grade": {"type": "string", "enum": ["경보", "주의", "관찰", "전체"], "description": "생략하면 판정(code)을 지정한 경우 전체, 아니면 경보"},
                                          "code": {"type": "string", "description": "판정 이름 일부: 운영불가, 통신두절, 간헐통신, 충전고착, 숨은고장, 실패충전"},
                                          "limit": {"type": "integer"}}}}},
    {"type": "function", "function": {"name": "search_station", "description": "충전소 이름이나 지역(읍면동, 장소)으로 찾아 지금 상태 요약. 도민 질문(어디서 충전할 수 있나)에 사용",
                                      "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
                                                     "required": ["query"]}}},
    {"type": "function", "function": {"name": "charger_detail", "description": "특정 충전소(와 번호)의 충전기별 상태, 판정, 근거, 마지막 충전 시각",
                                      "parameters": {"type": "object", "properties": {"station": {"type": "string"}, "charger_no": {"type": "string"}},
                                                     "required": ["station"]}}},
    {"type": "function", "function": {"name": "long_neglect", "description": "고장 상태로 1년 넘게 미사용인 충전기(철거·목록 정리 검토 대상)",
                                      "parameters": {"type": "object", "properties": {"operator": _OP, "limit": {"type": "integer"}}}}},
    {"type": "function", "function": {"name": "hub_change", "description": "제주데이터허브(2025.9) 대비 운영기관별 충전기 수 변화",
                                      "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "operator_message", "description": "운영사에 보낼 점검 요청서 초안 작성(발송은 하지 않음)",
                                      "parameters": {"type": "object", "properties": {"operator": _OP}, "required": ["operator"]}}},
]


def run_tool(d, attrs, name, args):
    f = TOOL_FUNCS.get(name)
    if not f:
        return {"오류": f"없는 도구 {name}"}
    try:
        return f(d, attrs, **(args or {}))
    except TypeError as e:
        return {"오류": f"인자 오류: {e}"}


# ---------------- 숫자 대조 ----------------
NUM = re.compile(r"(?<![0-9A-Za-z.])\d+(?:[.,]\d+)*(?![0-9A-Za-z])")   # 한글 단위(대, 일, 회)는 붙어 있어도 숫자로 읽음


def _nums(text):
    out = set()
    for m in NUM.findall(str(text)):
        v = m.replace(",", "")
        out.add(v)
        if "." in v:
            out.add(v.rstrip("0").rstrip("."))
    return out


def check_numbers(answer, sources):
    """답 속 숫자가 도구 결과(또는 질문)에 있는지. 1~10의 작은 수(순번 등)는 넘어간다"""
    allowed = set()
    for s in sources:
        allowed |= _nums(s)
    bad = []
    for n in _nums(answer):
        try:
            if float(n) <= 10 and float(n).is_integer():
                continue
        except ValueError:
            pass
        if n not in allowed:
            bad.append(n)
    return sorted(set(bad))


# ---------------- LLM ----------------
def api_key():
    return os.environ.get("OPENAI_API_KEY", "").strip()


def _chat(messages, key, tools=True):
    body = {"model": MODEL, "messages": messages, "temperature": 0.2}
    if tools:
        body["tools"] = TOOLS
    req = urllib.request.Request(API_URL, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)["choices"][0]["message"]
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"OpenAI 오류 {e.code}: {e.read().decode('utf-8', 'ignore')[:200]}") from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"OpenAI에 연결하지 못했습니다: {e.reason}") from None


def system_prompt(attrs):
    return (
        "너는 '고장났수다'의 AI 에이전트다. 제주 공공 전기차 충전기의 고장과 장기 방치를 찾아 점검 담당자(공무원·운영사)와 도민을 돕는다.\n"
        f"데이터 기준 시각: {attrs['now']:%Y-%m-%d %H:%M} (30분마다 수집한 환경공단 API + 제주데이터허브 862 + 한전 충전 기록).\n"
        "규칙:\n"
        "1. 숫자·충전소 이름·날짜는 반드시 도구를 호출해서 얻은 결과에서만 쓴다. 기억이나 추측으로 숫자를 만들지 않는다.\n"
        "2. 도구 결과에 없는 내용이면 '데이터에서 확인할 수 없습니다'라고 답한다.\n"
        f"3. 등급 뜻: {GRADE_HELP}. 숨은 고장은 이용 기록으로 추정한 것이지 고장 확정이 아니다.\n"
        "4. 한국어로, 짧고 쉽게. 목록은 마크다운 표나 번호 목록. 첫 문장에 답(숫자)을 먼저.\n"
        "5. 운영사 점검 요청서는 초안만 만든다. 발송은 담당자가 운영사 화면에서 확인 후 직접 한다고 안내한다.\n"
        "6. 개수를 답할 때는 등급별(경보·주의·관찰) 대수도 함께 말한다.\n"
        "7. 충전기를 고치거나 상태를 바꾸는 일은 할 수 없다. 할 수 있는 일은 조회·계산·문서 작성이다.")


def ask(d, attrs, question, history=None, key=None):
    """질문 → (답, 도구 기록, 숫자 대조 결과). 키가 없으면 RuntimeError"""
    key = key or api_key()
    if not key:
        raise RuntimeError("OPENAI_API_KEY가 없습니다")
    msgs = [{"role": "system", "content": system_prompt(attrs)}]
    for h in (history or [])[-6:]:
        msgs.append({"role": h["role"], "content": h["content"]})
    msgs.append({"role": "user", "content": question})
    trace, sources = [], [question]
    answer = ""
    for _ in range(MAX_STEPS):
        m = _chat(msgs, key)
        calls = m.get("tool_calls") or []
        if not calls:
            answer = m.get("content") or ""
            break
        msgs.append({"role": "assistant", "content": m.get("content"), "tool_calls": calls})
        for c in calls:
            name = c["function"]["name"]
            try:
                args = json.loads(c["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            res = run_tool(d, attrs, name, args)
            txt = json.dumps(res, ensure_ascii=False, default=str)
            sources.append(txt)
            trace.append({"도구": name, "입력": args, "결과": res})
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": txt[:12000]})
    else:
        answer = "도구를 여러 번 불렀지만 답을 정리하지 못했습니다. 질문을 조금 더 구체적으로 해 주세요."
    bad = check_numbers(answer, sources)
    if bad:   # 한 번 고쳐 쓰게 한다
        msgs += [{"role": "assistant", "content": answer},
                 {"role": "user", "content": f"답에 도구 결과에 없는 숫자가 있습니다: {', '.join(bad)}. "
                                             "도구 결과에 있는 숫자만 써서 다시 답하세요. 모르면 모른다고 하세요."}]
        answer = _chat(msgs, key, tools=False).get("content") or answer
        bad = check_numbers(answer, sources)
    return answer, trace, bad


# ---------------- 키 없이 쓰는 질문 버튼 ----------------
PRESETS = {
    "오늘 경보 목록": ("priority_list", {"grade": "경보", "limit": 10}),
    "운영기관별 현황": ("overview", {}),
    "1년 넘게 방치된 곳": ("long_neglect", {"limit": 15}),
    "1년 전(허브)과 비교": ("hub_change", {}),
}


def table_md(rows, cols=None):
    if not rows:
        return "_해당 없음_"
    cols = cols or list(rows[0].keys())
    esc = lambda v: str(v).replace("|", "/").replace("\n", " ")
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] +
                     ["| " + " | ".join(esc(r.get(c, "")) for c in cols) + " |" for r in rows])


def preset_answer(d, attrs, label):
    name, args = PRESETS[label]
    res = run_tool(d, attrs, name, args)
    if name == "priority_list":
        txt = (f"지금 공공 충전기 경보는 **{res['해당_대수']}대**입니다. 점검 순서 상위 {len(res['목록'])}곳입니다.\n\n" +
               table_md(res["목록"], ["순위", "충전소", "번호", "운영기관", "판정", "방치", "근거"]))
    elif name == "overview":
        txt = (f"공공 충전기 **{res['공공충전기_대수']:,}대** 중 경보 **{res['경보']}대**, 주의 {res['주의']}대, "
               f"관찰 {res['관찰']}대, 철거·정리 검토 {res['철거정리검토_1년넘게미사용']}대입니다.\n\n**운영기관별 경보**\n\n" +
               table_md([{"운영기관": k, "경보": v} for k, v in res["운영기관별_경보"].items()]) +
               "\n\n**판정별**\n\n" + table_md(res["판정별"]))
    elif name == "long_neglect":
        txt = (f"고장 상태로 1년 넘게 충전 기록이 없는 충전기는 **{res['해당_대수']}대**입니다. "
               "수리보다 철거·목록 정리를 검토할 대상입니다.\n\n" + table_md(res["목록"]))
    else:
        txt = (f"{res['비교']}입니다.\n\n" + table_md(res["운영기관별"]))
    return txt, [{"도구": name, "입력": args, "결과": res}]


# ---------------- 점검 현황 보고서 ----------------
def build_report(d, attrs, operator=None):
    """표와 숫자는 코드가 계산. 반환: (마크다운, LLM에 줄 요약 재료)"""
    ov = t_overview(d, attrs, operator)
    pr = t_priority(d, attrs, operator, "경보", None, 10)
    ng = t_neglect(d, attrs, operator, 10)
    hub = t_hub_change(d, attrs)
    pub = _op(d[d["busiId"].isin(PUBLIC)], operator)
    rep = pub[pub["트랙"] == "수리"]
    cross = pd.crosstab(rep["운영기관"], rep["판정"]) if len(rep) else pd.DataFrame()
    new = rep[(rep["grade"] == "경보") & (rep["방치_일"] < 1)]
    title = f"제주 공공 전기차 충전기 점검 현황 보고서 ({ov['대상']})"
    md = [f"# {title}", f"기준 시각 {ov['기준시각']} · 고장났수다 AI 에이전트 자동 작성", "",
          "## 1. 요약", "{{SUMMARY}}", "",
          f"- 대상 공공 충전기 **{ov['공공충전기_대수']:,}대**",
          f"- 경보 **{ov['경보']}대** (이 중 최근 24시간 안에 새로 생긴 경보 {len(new)}대), 주의 {ov['주의']}대, 관찰 {ov['관찰']}대",
          f"- 고장 상태로 1년 넘게 미사용(철거·정리 검토) **{ov['철거정리검토_1년넘게미사용']}대**", "",
          "## 2. 오늘 점검할 곳 (경보, 점검 순서 상위 10)",
          table_md(pr["목록"], ["순위", "충전소", "번호", "운영기관", "판정", "방치", "근거"]), "",
          "## 3. 운영기관별 판정 현황 (경보·주의·관찰 합계)",
          table_md(cross.reset_index().to_dict("records")) if len(cross) else "_해당 없음_", "",
          "## 4. 철거·목록 정리 검토 (1년 넘게 미사용)",
          table_md(ng["목록"], ["충전소", "번호", "운영기관", "마지막충전", "판정", "근거"]), "",
          "## 5. 1년 전(제주데이터허브 2025.9)과 비교",
          table_md(hub["운영기관별"]), "",
          "## 6. 판정 기준과 한계",
          "- 운영불가: 운영중지·점검중 상태. 통신두절: 상태 신호가 6시간(주의)·24시간(경보) 넘게 끊김.",
          "- 충전고착: 급속 48시간·완속 72시간 넘게 '충전 중'(그 전은 주의). 주말 휴무 기관은 주말에 주의로만 표시.",
          "- 숨은 고장: 화면은 '사용 가능'인데, 같은 충전소·주변 충전기의 평소 이용량으로 볼 때 우연히 이렇게 오래 0회일 확률이 0.1% 미만.",
          "- 숨은 고장은 이용 기록으로 추정한 결과이며 고장 확정이 아니다. 현장 확인 결과를 기록해 정확도를 높인다.",
          "- 데이터: 한국환경공단 전기자동차 충전소 정보 API(30분 간격 수집), 제주데이터허브 862, 한국전력공사 충전 기록(기준 숫자 검증)."]
    facts = json.dumps({"개요": ov, "새경보_24시간": len(new), "상위경보": pr["목록"][:5], "1년전비교": hub["운영기관별"],
                        "철거정리검토": ng["해당_대수"]}, ensure_ascii=False, default=str)
    return "\n".join(md), facts


def report_summary(facts, key=None):
    """보고서 첫머리 요약 3문장. 키가 없거나 숫자가 안 맞으면 None"""
    key = key or api_key()
    if not key:
        return None
    msgs = [{"role": "system", "content": "너는 공공기관 보고서 작성 보조다. 주어진 JSON의 숫자만 써서 한국어 3문장 요약을 쓴다. "
                                          "가장 시급한 것, 눈에 띄는 변화, 권고 한 가지 순서. 추측 금지, 과장 금지."},
            {"role": "user", "content": facts}]
    txt = _chat(msgs, key, tools=False).get("content") or ""
    return None if check_numbers(txt, [facts]) else txt.strip()


def render_report(md, summary):
    s = summary or "아래 표는 코드가 진단 데이터에서 직접 계산한 결과입니다. (LLM 요약 없음)"
    return md.replace("{{SUMMARY}}", s)


def report_html(md):
    """브라우저에서 열어 인쇄(PDF 저장)할 수 있는 단순 HTML"""
    import html as h
    out, table = [], []

    def flush():
        if table:
            rows = [r for r in table if not re.match(r"^\|(-+\|)+$", r.replace(" ", ""))]
            cells = [[h.escape(c.strip()) for c in r.strip("|").split("|")] for r in rows]
            out.append("<table><tr>" + "".join(f"<th>{c}</th>" for c in cells[0]) + "</tr>" +
                       "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in cells[1:]) + "</table>")
            table.clear()
    for line in md.splitlines():
        if line.startswith("|"):
            table.append(line)
            continue
        flush()
        t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", h.escape(line))
        if line.startswith("# "):
            out.append(f"<h1>{t[2:]}</h1>")
        elif line.startswith("## "):
            out.append(f"<h2>{t[3:]}</h2>")
        elif line.startswith("- "):
            out.append(f"<li>{t[2:]}</li>")
        elif line.strip():
            out.append(f"<p>{t}</p>")
    flush()
    css = ("body{font-family:'Malgun Gothic',sans-serif;max-width:900px;margin:32px auto;color:#101828;font-size:14px;line-height:1.6}"
           "h1{font-size:22px}h2{font-size:17px;margin-top:28px;border-bottom:1px solid #EAECF0;padding-bottom:4px}"
           "table{border-collapse:collapse;width:100%;margin:8px 0;font-size:12.5px}th,td{border:1px solid #D0D5DD;padding:4px 6px;text-align:left}"
           "th{background:#F2F4F7}li{margin:2px 0}")
    return f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><title>점검 현황 보고서</title><style>{css}</style></head><body>{''.join(out)}</body></html>"
