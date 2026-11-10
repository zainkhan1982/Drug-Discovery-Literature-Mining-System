"""Base HTTP client with retry, rate limiting, and timeouts for literature APIs."""

import time
from typing import Optional

import httpx


class BaseLiteratureClient:
    """Base client with retry/backoff, rate limiting, and request timeouts."""

    def __init__(
        self,
        base_url: str,
        rate_limit: int = 3,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        email: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.rate_limit = rate_limit  # requests per second
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.email = email
        self.api_key = api_key
        self._last_request_time: Optional[float] = None
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={
                    "User-Agent": "DrugDiscoveryLitMining/1.0 (local research tool)",
                    "Accept": "application/json, application/xml",
                },
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def _rate_limit(self) -> None:
        """Enforce requests-per-second rate limiting."""
        if self._last_request_time is not None:
            elapsed = time.monotonic() - self._last_request_time
            min_interval = 1.0 / self.rate_limit if self.rate_limit > 0 else 0
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

    async def _request_with_retry(
        self,
        method: str,
        url: str,
        params: Optional[dict] = None,
        data: Optional[dict] = None,
        headers: Optional[dict] = None,
    ) -> httpx.Response:
        """Make an HTTP request with exponential backoff retry."""
        client = await self._get_client()
        last_exception: Optional[Exception] = None

        for attempt in range(self.max_retries):
            await self._rate_limit()
            try:
                response = await client.request(
                    method, url, params=params, data=data, headers=headers
                )
                self._last_request_time = time.monotonic()

                # Retry on server errors (5xx) and rate limiting (429)
                if response.status_code in (429, 500, 502, 503, 504):
                    wait = self.backoff_factor * (2 ** attempt)
                    await httpx.AsyncClient().aclose()  # no-op, just for type
                    import asyncio

                    await asyncio.sleep(wait)
                    continue

                return response

            except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as exc:
                last_exception = exc
                wait = self.backoff_factor * (2 ** attempt)
                import asyncio

                await asyncio.sleep(wait)
                continue

        if last_exception:
            raise last_exception
        # Should not reach here
        raise RuntimeError(f"Request failed after {self.max_retries} retries: {url}")

    async def get(
        self,
        path: str,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
    ) -> httpx.Response:
        url = f"{self.base_url}/{path.lstrip('/')}"
        return await self._request_with_retry("GET", url, params=params, headers=headers)

    async def post(
        self,
        path: str,
        data: Optional[dict] = None,
        headers: Optional[dict] = None,
    ) -> httpx.Response:
        url = f"{self.base_url}/{path.lstrip('/')}"
        return await self._request_with_retry("POST", url, data=data, headers=headers)
