LEAKAGE_FLAGS = {"target_copy", "likely_leakage", "missing_reveals_target"}


def _flags(profile, name):
    return next(set(c["flags"]) for c in profile["columns"] if c["name"] == name)


def test_telco_leaks_are_flagged_and_excluded(telco_profile):
    excluded = telco_profile["suggested_excluded"]
    for col in ("Churn Label", "Churn Reason", "Churn Score"):
        assert _flags(telco_profile, col) & LEAKAGE_FLAGS, col
        assert col in excluded
    assert "target_copy" in _flags(telco_profile, "Churn Label")
    # Churn Reason is empty exactly for non-churners: its missingness gives the label away.
    assert "missing_reveals_target" in _flags(telco_profile, "Churn Reason")


def test_telco_identifiers_and_locations_are_excluded(telco_profile):
    excluded = telco_profile["suggested_excluded"]
    for col in ("CustomerID", "Count", "Country", "State", "City", "Zip Code", "Lat Long", "Latitude", "Longitude"):
        assert col in excluded, col
    assert "Contract" in telco_profile["suggested_features"]
    assert "Tenure Months" in telco_profile["suggested_features"]


def test_iranian_strong_but_legitimate_columns_are_not_flagged(iranian_profile):
    for col in ("Status", "Customer Value"):
        assert not _flags(iranian_profile, col) & LEAKAGE_FLAGS, col
        assert col in iranian_profile["suggested_features"]
    assert iranian_profile["suggested_excluded"] == {}


def test_profile_is_computed_on_the_training_split(telco_data, telco_profile):
    assert telco_profile["n_train"] == len(telco_data.train_idx)
    assert telco_profile["computed_on"] == "training split only"
    assert telco_data.target not in [c["name"] for c in telco_profile["columns"]]
