import pandas as pd
import pytest

from metbit.utility import lazypair


def test_lazypair_builds_pair_indices_and_datasets():
    df = pd.DataFrame(
        {"Class": ["A", "A", "B", "C"], "value": [1, 2, 3, 4]},
        index=[10, 11, 12, 13],
    )

    lp = lazypair(df, "Class")

    assert set(lp.names) == {"A_vs_B", "A_vs_C", "B_vs_C"}
    pair_lookup = dict(zip(lp.names, lp.index_))
    assert pair_lookup["A_vs_B"] == [10, 11, 12]

    datasets = lp.get_dataset()
    assert len(datasets) == 3
    assert all("Class" in subset.columns for subset in datasets)
    assert set(datasets[0]["Class"].unique()).issubset({"A", "B"})


def test_lazypair_requires_multiple_groups():
    df = pd.DataFrame({"Class": ["A", "A"]})
    with pytest.raises(ValueError):
        lazypair(df, "Class")


def test_lazypair_missing_column_raises_key_error():
    df = pd.DataFrame({"Group": ["A", "B"]})
    with pytest.raises(KeyError):
        lazypair(df, "Class")
