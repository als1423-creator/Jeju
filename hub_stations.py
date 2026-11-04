import csv, json, os, urllib.parse, urllib.request
KEY = os.environ["HUB_KEY"]
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
cols = sorted({k for r in rows for k in r})
with open("data/hub/stations.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader(); w.writerows(rows)
print("총", d["totCnt"], "곳 중", len(rows), "곳 저장")
