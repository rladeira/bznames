"""Module for downloading and parsing Brazilian name dataset from IBGE API."""

import json
import pathlib
import string
from typing import Any, Literal

import requests

IBGE_CENSUS_2010_NAMES_URL = "https://servicodados.ibge.gov.br/api/v1/censos/nomes/ranking"
DEFAULT_TIMEOUT_SECONDS = 10
_API_MAX_LIMIT_PROXY = 1_000_000
BZ_NAMES_FILE_PREFIX = "bz_names"


def fetch_census_2010_name_data(
    count: int | Literal["all"] = "all",
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    """Retrieve raw name ranking statistics from the official IBGE API (Censo 2010).

    Args:
        count: The number of names to retrieve, or "all" to retrieve all.
        timeout: Request timeout in seconds.

    Returns:
        A list of dictionaries representing name frequencies.
    """
    quantity = _API_MAX_LIMIT_PROXY if count == "all" else count

    response = requests.get(
        IBGE_CENSUS_2010_NAMES_URL,
        params={"qtd": quantity},
        timeout=timeout,
    )
    response.raise_for_status()

    return response.json()


def clean_ibge_data(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize names by converting to lowercase and stripping punctuation.

    Args:
        data: Raw data list of dicts.

    Returns:
        Cleaned name data with lowercase normalized names.
    """
    remove_punct = str.maketrans("", "", string.punctuation)

    return [
        {
            "name": x["nome"].lower().translate(remove_punct),
            "freq": x["freq"],
        }
        for x in data
    ]


def load_ibge_name_data(
    count: int | Literal["all"] = "all",
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    """Load name dataset from local cache, downloading and caching it if not cached.

    Args:
        count: The number of names to retrieve if downloading.
        timeout: Request timeout in seconds if downloading.

    Returns:
        A list of cleaned name/frequency dictionaries.
    """
    suffix = "" if count == "all" else f"_{count}"
    filename = f"{BZ_NAMES_FILE_PREFIX}{suffix}.json"

    package_base = pathlib.Path(__file__).resolve().parents[2]
    cache_path = package_base / "data" / filename

    if cache_path.exists():
        with cache_path.open("r") as f:
            return json.load(f)
    else:
        data = fetch_census_2010_name_data(count=count, timeout=timeout)
        data = clean_ibge_data(data)

        if not cache_path.parent.exists():
            cache_path.parent.mkdir()

        with cache_path.open("w") as f:
            json.dump(data, f)

        return data
