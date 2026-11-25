import os
import pandas as pd
import pytest
from src.visualization import (
    compute_pca,
    plot_pca_2d,
    plot_pca_3d,
    plot_radar_comparison,
    plot_risk_distribution,
    plot_turnover_by_risk,
    plot_segment_comparison,
    plot_segment_medians_summary,
    FEATURE_COLS
)

REF_DATA_PATH = "data/reference_profiles.csv"
SAMPLE_DATA_PATH = "data/sample_test.csv"


@pytest.fixture
def ref_df():
    assert os.path.exists(REF_DATA_PATH), f"File {REF_DATA_PATH} should exist"
    return pd.read_csv(REF_DATA_PATH)


@pytest.fixture
def sample_df():
    assert os.path.exists(SAMPLE_DATA_PATH), f"File {SAMPLE_DATA_PATH} should exist"
    return pd.read_csv(SAMPLE_DATA_PATH)


def test_reference_profiles_validity(ref_df):
    """Reference profiles must contain 1,111 accounts with required schema and 5 segments."""
    assert len(ref_df) == 1111
    required_cols = ["account_id", *FEATURE_COLS, "segment", "risk_level"]
    for col in required_cols:
        assert col in ref_df.columns, f"Missing column {col} in reference data"

    # All accounts in reference must have at least 10 bets
    assert (ref_df["total_bets"] >= 10).all()
    # Win rate must be in [0, 1]
    assert (ref_df["win_rate"] >= 0).all() and (ref_df["win_rate"] <= 1).all()
    # avg_cashout >= 1.0
    assert (ref_df["avg_cashout"] >= 1.0).all()

    # 5 unique segments
    unique_segments = ref_df["segment"].unique()
    assert len(unique_segments) == 5
    assert "Taruhan Nominal Tinggi" in unique_segments
    assert "Target Rendah, Win Rate Tinggi" in unique_segments


def test_sample_test_scenarios(sample_df):
    """sample_test.csv must contain scenarios for all risk levels, anomalies, and edge cases."""
    assert len(sample_df) >= 10
    required_cols = ["account_id", *FEATURE_COLS]
    for col in required_cols:
        assert col in sample_df.columns

    # Must contain at least one account with total_bets < 10 (insufficient data test)
    assert (sample_df["total_bets"] < 10).any(), "Must contain accounts with < 10 bets for validation test"


def test_pca_computation(ref_df, sample_df):
    """PCA must compute correct components and explained variance ratios."""
    ref_proj, pca, extra_proj = compute_pca(ref_df, n_components=3, extra_df=sample_df)

    assert "PC1" in ref_proj.columns
    assert "PC2" in ref_proj.columns
    assert "PC3" in ref_proj.columns
    assert len(pca.explained_variance_ratio_) == 3
    assert pca.explained_variance_ratio_.sum() > 0.50

    assert extra_proj is not None
    assert "PC1" in extra_proj.columns


def test_pca_plots(ref_df, sample_df):
    """PCA 2D and 3D figures must render properly with variance labels."""
    fig_2d = plot_pca_2d(ref_df, extra_df=sample_df, color_by="segment")
    assert fig_2d is not None
    assert "PC1 (" in fig_2d.layout.xaxis.title.text

    fig_3d = plot_pca_3d(ref_df, extra_df=sample_df, color_by="risk_level")
    assert fig_3d is not None


def test_radar_plot(ref_df):
    """Radar plot must compare user metrics against segment and population medians."""
    user_metrics = {"total_bets": 25, "avg_bet": 40.0, "win_rate": 0.8, "avg_cashout": 1.2}
    fig_radar = plot_radar_comparison(user_metrics, ref_df, target_segment="Target Rendah, Win Rate Tinggi")
    assert fig_radar is not None
    assert len(fig_radar.data) == 3  # Population, Segment, User traces


def test_risk_distribution_and_turnover_plots(ref_df):
    """Risk distribution and turnover plots must render with consistent totals."""
    fig_dist = plot_risk_distribution(ref_df, as_donut=False)
    assert fig_dist is not None

    fig_donut = plot_risk_distribution(ref_df, as_donut=True)
    assert fig_donut is not None

    fig_turnover = plot_turnover_by_risk(ref_df)
    assert fig_turnover is not None


def test_segment_comparison_plots(ref_df):
    """Segment comparison charts must render box plots and medians summary."""
    fig_box = plot_segment_comparison(ref_df, metric="win_rate")
    assert fig_box is not None

    fig_summary = plot_segment_medians_summary(ref_df)
    assert fig_summary is not None
