"""제주 전기차 충전기 상태 자동 수집기 (한국환경공단 전기자동차 충전소 정보 API)

실행할 때마다 제주(zcode=50) 충전기 상태를 한 번 찍어서 저장합니다.
- data/status/YYYY-MM-DD.csv.gz : 30분마다 쌓이는 상태 기록 (급속 + 공공기관 충전기)
- data/master/YYYY-MM-DD.csv.gz : 하루 한 번 저장하는 충전기 기본 정보 (전체)

필요한 것: 환경변수 KECO_KEY (공공데이터포털 '일반 인증키(Decoding)')
"""
import csv
import gzip
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

URL = "https://apis.data.go.kr/B552584/EvCharger/getChargerInfo"
KST = timezone(timedelta(hours=9))
SLOW_TYPES = {"02", "08"}
PUBLIC_BUSI = {"ME", "JD", "JC", "JE", "JT"}

STATUS_COLS = ["collected_at", "statId", "chgerId", "busiId", "stat",
               "statUpdDt", "lastTsdt", "lastTedt", "nowTsdt"]
MASTER_COLS = ["collected_at", "statId", "chgerId", "statNm", "busiId", "bnm",
               "busiNm", "chgerType", "output", "addr", "lat", "lng",
               "zscode", "kind", "kindDetail", "limitYn", "delYn",
               "stat", "statUpdDt", "lastTsdt", "lastTedt", "nowTsdt"]


def fetch_page(key, page, rows=9999, retries=3):
    params = {"serviceKey": key, "pageNo": page, "numOfRows": rows,
              "zcode": "50", "dataType": "JSON"}
    url = URL + "?" + urllib.parse.urlencode(params)
    last_err = None
    for i in range(retries):
        text = ""
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                text = r.read().decode("utf-8")
            data = json.loads(text)
            if str(data.get("resultCode", "00")) != "00":
                raise RuntimeError(f"API 오류: {data.get('resultCode')} {data.get('resultMsg')}")
            items = data.get("items") or {}
            item = items.get("item", []) if isinstance(items, dict) else items
            if isinstance(item, dict):
                item = [item]
            return int(data.get("totalCount", 0)), item
        except json.JSONDecodeError:
            last_err = RuntimeError("JSON이 아닌 응답 (키 미활성/키 오류일 수 있음): " + text[:200])
        except Exception as e:
            last_err = e
        time.sleep(5 * (i + 1))
    raise last_err


def fetch_all(key):
    total, rows = fetch_page(key, 1)
    page = 1
    while len(rows) < total:
        page += 1
        _, more = fetch_page(key, page)
        if not more:
            break
        rows.extend(more)
    return total, rows


def append_gz(path, cols, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with gzip.open(path, "at", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


def main():
    key = os.environ.get("KECO_KEY", "").strip()
    if not key:
        sys.exit("환경변수 KECO_KEY가 없습니다.")
    now = datetime.now(KST)
    stamp = now.strftime("%Y-%m-%d %H:%M")
    day = now.strftime("%Y-%m-%d")

    total, rows = fetch_all(key)
    for r in rows:
        r["collected_at"] = stamp

    keep = [r for r in rows
            if str(r.get("chgerType", "")) not in SLOW_TYPES
            or r.get("busiId") in PUBLIC_BUSI]
    append_gz(f"data/status/{day}.csv.gz", STATUS_COLS, keep)

    master = f"data/master/{day}.csv.gz"
    if not os.path.exists(master):
        append_gz(master, MASTER_COLS, rows)

    print(f"[{stamp}] 제주 전체 {total}대 중 {len(rows)}대 수신, 상태 저장 {len(keep)}대")


if __name__ == "__main__":
    main()
