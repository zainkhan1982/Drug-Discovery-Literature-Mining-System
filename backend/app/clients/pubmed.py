"""PubMed E-utilities client for literature search and metadata retrieval."""

import xml.etree.ElementTree as ET
from typing import Any, Optional

from app.clients.base import BaseLiteratureClient
from app.logging import logger


class PubMedClient(BaseLiteratureClient):
    """Client for PubMed E-utilities (ESearch, ESummary, EFetch)."""

    def __init__(
        self,
        base_url: str,
        email: str = "dds-local@example.com",
        api_key: Optional[str] = None,
        rate_limit: int = 3,
        timeout: float = 30.0,
    ):
        super().__init__(
            base_url=base_url,
            rate_limit=rate_limit,
            timeout=timeout,
            email=email,
            api_key=api_key,
        )

    def _default_params(self) -> dict:
        params = {"email": self.email, "retmode": "json"}
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    async def search(
        self,
        term: str,
        retmax: int = 100,
        retstart: int = 0,
        date_range: Optional[str] = None,
        publication_types: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Search PubMed using ESearch. Returns ID list and count."""
        params = self._default_params()
        params.update(
            {
                "db": "pubmed",
                "term": term,
                "retmax": retmax,
                "retstart": retstart,
                "sort": "relevance",
            }
        )
        if date_range:
            params["datetype"] = "pdat"
            params["mindate"] = date_range.get("start", "")
            params["maxdate"] = date_range.get("end", "")
        if publication_types:
            type_query = " OR ".join(
                f"\"{pt}\"[Publication Type]" for pt in publication_types
            )
            params["term"] = f"({params['term']}) AND ({type_query})"

        response = await self.get("esearch.fcgi", params=params)
        response.raise_for_status()
        data = response.json()

        esearch_result = data.get("esearchresult", {})
        return {
            "count": int(esearch_result.get("count", 0)),
            "id_list": esearch_result.get("idlist", []),
            "retstart": int(esearch_result.get("retstart", 0)),
            "retmax": int(esearch_result.get("retmax", 0)),
        }

    async def fetch_metadata(self, pmid_list: list[str]) -> list[dict[str, Any]]:
        """Fetch metadata for PMIDs using ESummary."""
        if not pmid_list:
            return []

        params = self._default_params()
        params.update({"db": "pubmed", "id": ",".join(pmid_list), "retmode": "json"})

        response = await self.get("esummary.fcgi", params=params)
        response.raise_for_status()
        data = response.json()

        results = []
        result_data = data.get("result", {})
        for pmid in pmid_list:
            item = result_data.get(pmid, {})
            # Extract DOI from articleids
            doi = None
            for aid in item.get("articleids", []):
                if aid.get("idtype") == "doi":
                    doi = aid.get("value")
                    break

            results.append(
                {
                    "pmid": item.get("uid", ""),
                    "doi": doi,
                    "title": item.get("title", ""),
                    "authors": [
                        a.get("name", "") for a in item.get("authors", [])
                    ],
                    "journal": item.get("fulljournalname", ""),
                    "publication_date": item.get("pubdate", ""),
                    "source": "pubmed",
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{item.get('uid', '')}/",
                }
            )
        return results

    async def fetch_abstracts(self, pmid_list: list[str]) -> dict[str, str]:
        """Fetch abstracts for PMIDs using EFetch (XML). Returns PMID -> abstract text."""
        if not pmid_list:
            return {}

        params = self._default_params()
        params.update({"db": "pubmed", "id": ",".join(pmid_list), "retmode": "xml"})

        response = await self.get("efetch.fcgi", params=params)
        response.raise_for_status()

        abstracts: dict[str, str] = {}
        try:
            root = ET.fromstring(response.text)
            for article in root.findall(".//PubmedArticle"):
                pmid_elem = article.find(".//PMID")
                pmid = pmid_elem.text if pmid_elem is not None else ""

                abstract_parts = []
                for abstract_text in article.findall(".//AbstractText"):
                    label = abstract_text.get("Label", "")
                    text = abstract_text.text or ""
                    if label:
                        abstract_parts.append(f"{label}: {text}")
                    else:
                        abstract_parts.append(text)

                if pmid and abstract_parts:
                    abstracts[pmid] = " ".join(abstract_parts)
        except ET.ParseError as exc:
            logger.warning("Failed to parse PubMed EFetch XML: %s", exc)

        return abstracts

    async def search_and_fetch(
        self,
        term: str,
        retmax: int = 100,
        retstart: int = 0,
        date_range: Optional[dict] = None,
        publication_types: Optional[list[str]] = None,
        fetch_abstracts: bool = True,
    ) -> dict[str, Any]:
        """Search PubMed and fetch metadata for all results."""
        search_result = await self.search(
            term=term,
            retmax=retmax,
            retstart=retstart,
            date_range=date_range,
            publication_types=publication_types,
        )

        id_list = search_result.get("id_list", [])
        metadata = await self.fetch_metadata(id_list)

        # Enrich with abstracts if requested
        if fetch_abstracts and id_list:
            abstracts = await self.fetch_abstracts(id_list)
            for item in metadata:
                pmid = item.get("pmid", "")
                if pmid in abstracts:
                    item["abstract"] = abstracts[pmid]

        return {
            "count": search_result.get("count", 0),
            "retstart": search_result.get("retstart", 0),
            "retmax": search_result.get("retmax", 0),
            "papers": metadata,
        }
