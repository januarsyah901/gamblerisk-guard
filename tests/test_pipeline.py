import pytest


@pytest.mark.skip(reason="Pending model artifacts and data/reference_profiles.csv")
def test_labels_match_training_notebook():
    """Pipeline labels for the 1,111 reference profiles must match the notebook 100%."""
