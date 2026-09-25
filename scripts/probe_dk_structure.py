#!/usr/bin/env python3
import json
from pathlib import Path
from src.draftkings_direct import _fetch_json, LEAGUE_IDS, V5_BASE_URL

out={}
for sport, league_id in LEAGUE_IDS.items():
    if sport not in {"wnba","mlb"}:
        continue
    rec={"league_id":league_id}
    try:
        data=_fetch_json(V5_BASE_URL.format(league_id=league_id))
        group=data.get("eventGroup") or data
        rec["event_count"]=len(group.get("events") or data.get("events") or [])
        cats=[]
        for cat in group.get("offerCategories") or []:
            cats.append({
                "id": cat.get("offerCategoryId") or cat.get("categoryId") or cat.get("id"),
                "name": cat.get("name"),
                "subcategories":[{
                    "id": d.get("subcategoryId") or d.get("offerSubcategoryId") or d.get("id"),
                    "name": d.get("name"),
                    "has_inline_offers": bool((d.get("offerSubcategory") or {}).get("offers")),
                } for d in (cat.get("offerSubcategoryDescriptors") or [])]
            })
        rec["categories"]=cats
    except Exception as exc:
        rec["error"]=f"{type(exc).__name__}: {exc}"
    out[sport]=rec

Path("data/dk_structure_probe.json").write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
