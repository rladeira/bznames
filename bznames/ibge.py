import json
import os
import string
from typing import Any

import requests


def get_name_data_from_ibge() -> list[dict[str, Any]]:
    url = "https://servicodados.ibge.gov.br/api/v1/censos/nomes/ranking?qtd=%s"

    response = requests.get(url % 1_000_000)
    response.raise_for_status()

    return response.json()


def clean_ibge_data(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    remove_punct = str.maketrans("", "", string.punctuation)

    return [
        {
            "name": x["nome"].lower().translate(remove_punct),
            "freq": x["freq"],
        }
        for x in data
    ]


def load_ibge_name_data() -> list[dict[str, Any]]:
    if os.path.exists("data/bz_names.json"):
        with open("data/bz_names.json", "r") as f:
            return json.load(f)
    else:
        data = get_name_data_from_ibge()
        data = clean_ibge_data(data)

        if not os.path.exists("data"):
            os.mkdir("data")

        with open("data/bz_names.json", "w") as f:
            json.dump(data, f)

        return data
