import os
from typing import List

from tavily import TavilyClient


def get_tavily_client() -> TavilyClient:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY environment variable is not set. "
            "Set it to enable web search capabilities."
        )
    return TavilyClient(api_key=api_key)


def web_search(query: str, max_results: int = 5) -> List[dict]:
    """
    Search the web using Tavily for external context such as:
      - CMS compliance rules
      - Medicaid roster submission standards
      - Provider organization details
      - Meanings of validation failures
    """
    client = get_tavily_client()
    response = client.search(query=query, max_results=max_results)
    # The tavily client already returns a structured dict; normalize to list-of-dicts for the agent.
    if isinstance(response, dict) and "results" in response:
        return response["results"]
    return response  # type: ignore[return-value]

