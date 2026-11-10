"""Europe PMC REST client for literature search and metadata retrieval."""

import xml.etree.ElementTree as ET
from typing import Any, Optional

from app.clients.base import BaseLiteratureClient
from app.logging import logger


class EuropePMCClient(BaseLiteratureClient):
    """Client for Europe PMC REST web services."""

    def __init__(
        self,
        base_url: str = "https://www.ebi.ac.uk/europepmc/webservices/rest/",
        rate_limit: int = 10,
        timeout: float = 30.0,
    ):
        super().__init__(
            base_url=base_url,
            rate_limit=rate_limit,
            timeout=timeout,
        )

    async def search(
        self,
        query: str,
        page_size: int = 100,
        page: int = 1,
        date_range: Optional[dict] = None,
        publication_types: Optional[list[str]] = None,
        open_access: Optional[bool] = None,
    ) -> dict[str, Any]:
        """Search Europe PMC. Returns hit count and result list."""
        params: dict[str, Any] = {
            "query": query,
            "format": "json",
            "pageSize": page_size,
            "page": page,
            "resultType": "core",
        }

        if date_range:
            # Europe PMC supports FIRST_PDATE:[start TO end]
            params[
                "query"
            ] = f"({query}) AND FIRST_PDATE:[{date_range.get('start', '')} TO {date_range.get('end', '')}]"
        if publication_types:
            type_query = " OR ".join(
                f"PUB_TYPE:\"{pt}\"" for pt in publication_types
            )
            params["query"] = f"({params['query']}) AND ({type_query})"
        if open_access is True:
            params["query"] = f"({params['query']}) AND OPEN_ACCESS:Y"
        elif open_access is False:
            params["query"] = f"({params['query']}) AND NOT OPEN_ACCESS:Y"

        response = await self.get("search", params=params)
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("resultList", {}).get("result", []):
            # Extract full-text URLs
            full_text_urls = []
            if item.get("pmcid"):
                full_text_urls.append(
                    f"https://www.ebi.ac.uk/europepmc/webservices/rest/{item['pmcid']}/fullTextXML"
                )
            if item.get("doi"):
                full_text_urls.append(f"https://doi.org/{item['doi']}")

            # Determine access status
            is_open_access = item.get("isOpenAccess", "N") == "Y"
            in_pmc = bool(item.get("pmcid"))

            if in_pmc and is_open_access:
                access_status = "open_access_available"
            elif is_open_access:
                access_status = "open_access_available"
            else:
                access_status = "abstract_only"

            results.append(
                {
                    "pmid": item.get("pmid", ""),
                    "pmcid": item.get("pmcid", ""),
                    "doi": item.get("doi", ""),
                    "title": item.get("title", ""),
                    "authors": item.get("authorString", ""),
                    "journal": item.get("journalTitle", ""),
                    "publication_date": item.get("firstPublicationDate", ""),
                    "abstract": item.get("abstractText", ""),
                    "source": "europepmc",
                    "url": f"https://europepmc.org/article/{item.get('pmid', item.get('pmcid', ''))}",
                    "full_text_urls": full_text_urls,
                    "is_open_access": is_open_access,
                    "in_pmc": in_pmc,
                    "access_status": access_status,
                    "cited_by_count": int(item.get("citedByCount", 0)),
                }
            )

        return {
            "hit_count": data.get("hitCount", 0),
            "page": data.get("page", page),
            "page_size": data.get("pageSize", page_size),
            "papers": results,
        }

    async def fetch_full_text_xml(self, pmcid: str) -> tuple[str | None, str | None]:
        """Fetch full-text XML for a PMCID. Returns (xml_text, error)."""
        if not pmcid:
            return None, "No PMCID available"

        url = f"{self.base_url}{pmcid}/fullTextXML"
        try:
            response = await self.get(url)
            if response.status_code == 200:
                return response.text, None
            elif response.status_code == 404:
                return None, "Full text not found"
            else:
                return None, f"HTTP {response.status_code}"
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to fetch full text for %s: %s", pmcid, exc)
            return None, str(exc)

    async def search_and_fetch(
        self,
        query: str,
        page_size: int = 100,
        page: int = 1,
        date_range: Optional[dict] = None,
        publication_types: Optional[list[str]] = None,
        open_access: Optional[bool] = None,
    ) -> dict[str, Any]:
        """Search Europe PMC with metadata (abstracts included in core result type)."""
        return await self.search(
            query=query,
            page_size=page_size,
            page=page,
            date_range=date_range,
            publication_types=publication_types,
            open_access=open_access,
        )
