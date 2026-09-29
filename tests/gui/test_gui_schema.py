from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from metbit_server.session import Session
from metbit_server.stages import resolve_schema
from metbit_server.stages.base import Stage, option_values


class Tagged(BaseModel):
    value: Optional[float] = Field(None, title="Value")
    target: str = Field("", json_schema_extra={"options": "numeric_columns"})


TAGGED = Stage(id="tagged", title="Tagged", params=Tagged, run=lambda p, s, prog: None)


def test_optional_fields_flatten_to_their_plain_type():
    prop = resolve_schema(TAGGED, Session(["import"]))["properties"]["value"]
    assert prop["type"] == "number"
    assert "anyOf" not in prop
    assert prop["title"] == "Value"


def test_numeric_columns_option(run_all, csv_path, tmp_path):
    df = pd.read_csv(csv_path)
    df.insert(2, "Age", np.linspace(20, 70, 12))
    df.insert(3, "Site", ["x", "y"] * 6)
    path = tmp_path / "age.csv"
    df.to_csv(path, index=False)
    s = run_all(("import", {"path": str(path)}))
    assert option_values(s, "numeric_columns") == ["Age"]
    prop = resolve_schema(TAGGED, s)["properties"]["target"]
    assert (prop["enum"], prop["default"]) == (["Age"], "Age")


def test_numeric_columns_none_when_absent(run_all):
    assert option_values(run_all(), "numeric_columns") is None
    assert option_values(Session(["import"]), "numeric_columns") is None
