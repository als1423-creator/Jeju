"""제주데이터허브 「[인구, 생활] 전기자동차 충전소 정보」(862) 내려받기

결과: data/hub/stations.json  (앱과 진단이 읽는 파일)
필요한 것: 환경변수 HUB_KEY (제주데이터허브 API 키, 저장소에는 두지 않음)
"""
import json
import os
import urllib.parse
import urllib.request

KEY = os.environ.get("HUB_KEY", "").strip()
if not KEY:
    raise SystemExit("환경변수 HUB_KEY가 없습니다. .env.example을 참고하세요.")
URL = "https://open.jejudatahub.net/api/proxy/atDab6t8218btaa122b26DDtbatD86t1/" + KEY

rows, page = [], 1
while True:
    q = urllib.parse.urlencode({"number": page, "limit": 100})
    with urllib.request.urlopen(URL + "?" + q, timeout=60) as r:
        d = json.loads(r.read().decode("utf-8"))
    rows += d["data"]
    if not d.get("hasMore"):
        break
    page += 1

os.makedirs("data/hub", exist_ok=True)
with open("data/hub/stations.json", "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, separators=(",", ":"))
print("총", d.get("totCnt", len(rows)), "곳 중", len(rows), "곳 저장 → data/hub/stations.json")
