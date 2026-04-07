import asyncio, json, time, aiohttp
from datetime import datetime, timezone, timedelta

POLY_GAMMA_API = "https://gamma-api.polymarket.com"

async def test_find_market():
    now_utc = datetime.now(timezone.utc)
    params = {
        "end_date_min": now_utc.isoformat(),
        "end_date_max": (now_utc + timedelta(minutes=7)).isoformat(),
        "closed": "false",
        "limit": 20,
        "order": "endDate",
        "ascending": "true",
    }
    print(f"Query params: {json.dumps(params, indent=2)}")
    
    async with aiohttp.ClientSession() as s:
        async with s.get(f"{POLY_GAMMA_API}/markets", params=params) as r:
            print(f"Status: {r.status}")
            data = await r.json()
    
    for m in data:
        slug = m.get("slug", "")
        q = m.get("question", "")
        if "btc-updown-5m" in slug:
            clob_ids = json.loads(m.get("clobTokenIds", "[]"))
            outcomes = json.loads(m.get("outcomes", "[]"))
            prices = json.loads(m.get("outcomePrices", "[]"))
            print(f"\n  FOUND BTC 5-MIN MARKET!")
            print(f"  Question: {q}")
            print(f"  Slug: {slug}")
            print(f"  End: {m.get('endDate','')}")
            print(f"  conditionId: {m.get('conditionId','')[:40]}")
            for i, out in enumerate(outcomes):
                print(f"  {out}: tokenId={clob_ids[i][:25]}... price={prices[i]}")
            print(f"  orderMinSize: {m.get('orderMinSize')}")
            print(f"  acceptingOrders: {m.get('acceptingOrders')}")
            return True
    
    print("No BTC 5-min market found in current window")
    print(f"Markets found: {[m.get('slug','')[:30] for m in data[:5]]}")
    return False

asyncio.run(test_find_market())
