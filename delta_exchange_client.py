from __future__ import annotations

import hashlib
import hmac
import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DeltaEnvironment:
    rest_base: str
    ws_base: str


ENVIRONMENTS = {
    "production": DeltaEnvironment(
        rest_base="https://api.india.delta.exchange/v2",
        ws_base="wss://socket.india.delta.exchange",
    ),
    "testnet": DeltaEnvironment(
        rest_base="https://cdn-ind.testnet.deltaex.org/v2",
        ws_base="wss://socket-ind.testnet.deltaex.org",
    ),
}


class DeltaRestClient:
    def __init__(self, api_key: str, api_secret: str, environment: str = "production", user_agent: str = "python-delta-client") -> None:
        if environment not in ENVIRONMENTS:
            raise ValueError(f"Unsupported environment: {environment}")
        self.api_key = api_key
        self.api_secret = api_secret
        self.environment = ENVIRONMENTS[environment]
        self.user_agent = user_agent

    @staticmethod
    def _sign(secret: str, message: str) -> str:
        return hmac.new(secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).hexdigest()

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
        auth: bool = False,
    ) -> dict[str, Any]:
        query_string = ""
        if params:
            query_string = "?" + urllib.parse.urlencode(params)
        body = json.dumps(payload, separators=(",", ":")) if payload else ""
        url = f"{self.environment.rest_base}{path}{query_string}"
        headers = {
            "Accept": "application/json",
            "User-Agent": self.user_agent,
        }
        if body:
            headers["Content-Type"] = "application/json"
        if auth:
            timestamp = str(int(time.time()))
            signature_payload = method.upper() + timestamp + path + query_string + body
            signature = self._sign(self.api_secret, signature_payload)
            headers["api-key"] = self.api_key
            headers["timestamp"] = timestamp
            headers["signature"] = signature

        request = urllib.request.Request(url=url, data=body.encode("utf-8") if body else None, headers=headers, method=method.upper())
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))

    def get_products(self) -> dict[str, Any]:
        return self._request("GET", "/products")

    def get_product(self, symbol: str) -> dict[str, Any] | None:
        response = self.get_products()
        products = response.get("result") or response.get("success") or response
        if isinstance(products, list):
            for product in products:
                if product.get("symbol") == symbol:
                    return product
        elif isinstance(response.get("result"), list):
            for product in response["result"]:
                if product.get("symbol") == symbol:
                    return product
        return None

    def get_candles(self, symbol: str, resolution: str, start_s: int, end_s: int) -> dict[str, Any]:
        return self._request(
            "GET",
            "/history/candles",
            params={"symbol": symbol, "resolution": resolution, "start": start_s, "end": end_s},
        )

    def place_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/orders", payload=payload, auth=True)

    def get_open_positions(self) -> dict[str, Any]:
        return self._request("GET", "/positions/margined", auth=True)

    def get_open_orders(self, product_id: int | None = None) -> dict[str, Any]:
        params = {"product_id": product_id} if product_id is not None else None
        return self._request("GET", "/orders", params=params, auth=True)

    def cancel_order(self, order_id: int) -> dict[str, Any]:
        return self._request("DELETE", f"/orders/{order_id}", auth=True)
