import pandas as pd

from src.storage import load_parquet_or_empty, upsert_dataframe


def test_load_parquet_or_empty_returns_empty_for_missing(tmp_path):
    path = tmp_path / "missing.parquet"
    df = load_parquet_or_empty(path)
    assert df.empty


def test_upsert_dataframe_deduplicates():
    existing = pd.DataFrame({"id": ["a", "b"], "value": [1, 2]})
    new = pd.DataFrame({"id": ["b", "c"], "value": [3, 4]})
    merged = upsert_dataframe(existing, new, keys=["id"], sort_columns=["id"])
    assert len(merged) == 3
    assert merged.loc[merged["id"] == "b", "value"].iloc[0] == 3
