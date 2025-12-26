import aiohttp
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from custom_components.local_openai import LOGGER


class WeaviateClient:
    """Weaviate API Client."""

    def __init__(
        self, hass: HomeAssistant, host: str, api_key: str | None, class_name: str
    ) -> None:
        """Initialize the weaviate client."""
        self._aiohttp = async_get_clientsession(hass=hass)
        self._host = host
        self._api_key = api_key
        self._class_name = class_name.lower().capitalize()

    async def near_text(self, query: str, threshold: float, limit: int):
        """Query weaviate for vector similarity."""
        query_obj = {
            "query": f"""
            {{
              Get {{
                {self._class_name}(
                  nearText: {{
                    concepts: ["{query}"],
                    certainty: {threshold},
                  }},
                  limit: {limit},
                ) {{
                  content
                  _additional {{
                    certainty
                  }}
                }}
              }}
            }}
            """
        }
        try:
            headers = {"Content-Type": "application/json"}

            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"

            async with self._aiohttp.post(
                f"{self._host}/v1/graphql", json=query_obj, headers=headers
            ) as resp:
                resp.raise_for_status()
                result = await resp.json()
                return result.get("data", {}).get("Get", {}).get(self._class_name, [])
        except aiohttp.ClientError as err:
            LOGGER.warning("Error communicating with Weaviate API: %s", err)
            return []
