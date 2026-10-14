import pandas as pd
import pytest

from src.churn.data import clean_types, encode_target, normalize_columns, prepare


@pytest.mark.parametrize(
    "values, expected_positive",
    [
        (["Attrited Customer", "Existing Customer", "Existing Customer"], "Attrited Customer"),
        (["Existing Customer", "Attrited Customer"], "Attrited Customer"),
        (["No", "Yes", "No"], "Yes"),
        ([0, 1, 1], "1"),
        ([True, False], "True"),
        (["churned", "active"], "churned"),
    ],
)
def test_encode_target_finds_the_churn_class(values, expected_positive):
    y, positive = encode_target(pd.Series(values, name="target"))
    assert positive == expected_positive
    assert y.tolist() == [int(str(v) == expected_positive) for v in values]


def test_encode_target_rejects_unknown_labels():
    with pytest.raises(ValueError, match="positive_label"):
        encode_target(pd.Series(["A", "B", "A"], name="target"))


def test_encode_target_with_explicit_positive_label():
    y, positive = encode_target(pd.Series(["A", "B", "A"], name="target"), positive_label="B")
    assert positive == "B"
    assert y.tolist() == [0, 1, 0]
    with pytest.raises(ValueError, match="not found"):
        encode_target(pd.Series(["A", "B"], name="target"), positive_label="C")


def test_encode_target_rejects_missing_and_multiclass():
    with pytest.raises(ValueError, match="missing"):
        encode_target(pd.Series(["Yes", None, "No"], name="target"))
    with pytest.raises(ValueError, match="exactly 2"):
        encode_target(pd.Series(["Yes", "No", "Maybe"], name="target"))


def test_normalize_columns_collapses_whitespace():
    df = normalize_columns(pd.DataFrame(columns=["Call  Failure", " Status ", "Charge \t Amount"]))
    assert list(df.columns) == ["Call Failure", "Status", "Charge Amount"]


def test_clean_types_turns_numeric_text_with_blanks_into_numbers():
    df = clean_types(pd.DataFrame({"charges": ["10.5", " ", "3"], "plan": ["A", "", "B"]}))
    assert pd.api.types.is_numeric_dtype(df["charges"])
    assert df["charges"].isna().tolist() == [False, True, False]
    assert df["plan"].isna().tolist() == [False, True, False]


def test_duplicates_are_removed_before_the_split():
    base = pd.DataFrame({"x": range(50), "churn": [0, 1] * 25})
    df = pd.concat([base, base.iloc[:10]], ignore_index=True)  # 10 exact duplicates
    data = prepare(df, "churn")
    assert data.n_rows_raw == 60
    assert data.n_duplicates_removed == 10
    assert len(data.df) == 50
    assert set(data.train_idx).isdisjoint(data.test_idx)
    assert len(data.train_idx) + len(data.test_idx) == 50
    train_rows = set(map(tuple, data.df.loc[data.train_idx].to_numpy()))
    test_rows = set(map(tuple, data.df.loc[data.test_idx].to_numpy()))
    assert train_rows.isdisjoint(test_rows)


def test_iranian_sample_duplicates_do_not_cross_the_split(iranian_data):
    assert iranian_data.n_duplicates_removed > 0
    train = iranian_data.df.loc[iranian_data.train_idx]
    test = iranian_data.df.loc[iranian_data.test_idx]
    assert pd.merge(train, test, how="inner").empty
