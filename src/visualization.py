"""Plotly charts for GambleRisk Guard:
- PCA scatter (2D/3D) with explained variance ratio on axis labels
- Radar comparison chart (account vs segment median vs population median)
- Risk distribution (count & proportion)
- Estimated financial turnover per risk level (sum of total_bets * avg_bet)
- Segment profile comparison across behavioral features
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = ["total_bets", "avg_bet", "win_rate", "avg_cashout"]
LOG_COLS = ["total_bets", "avg_bet", "avg_cashout"]

# Consistent color palette for risk levels & segments
RISK_COLOR_MAP = {
    "Rendah": "#10B981",          # Emerald green
    "Perlu Dipantau": "#F59E0B",  # Amber / yellow
    "Tinggi": "#EF4444",          # Red
    "Review manual": "#8B5CF6",   # Purple
    "Data belum cukup": "#9CA3AF" # Gray
}

SEGMENT_COLOR_MAP = {
    "Target Rendah, Win Rate Tinggi": "#10B981",
    "Frekuensi Tinggi, Target Rendah": "#3B82F6",
    "Taruhan Nominal Tinggi": "#F59E0B",
    "Target Menengah": "#F97316",
    "Target Cashout Tinggi": "#EF4444",
    "Anomali": "#8B5CF6"
}


def prepare_features_for_pca(df: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    """Applies log1p to skewed features and standardizes them for PCA."""
    df_copy = df[FEATURE_COLS].copy()
    for col in LOG_COLS:
        df_copy[col] = np.log1p(np.maximum(df_copy[col], 0))
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_copy)
    return scaled_features, scaler


def compute_pca(
    ref_df: pd.DataFrame,
    n_components: int = 3,
    extra_df: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, PCA, Optional[pd.DataFrame]]:
    """Fits PCA on reference profiles and projects reference (and optional extra) accounts.

    Returns:
        ref_projected: DataFrame with PCA coordinates and metadata
        pca_model: Fitted PCA object with explained_variance_ratio_
        extra_projected: DataFrame of extra points if provided
    """
    scaled_ref, scaler = prepare_features_for_pca(ref_df)
    pca = PCA(n_components=n_components, random_state=42)
    ref_pca_coords = pca.fit_transform(scaled_ref)

    ref_projected = ref_df.copy()
    ref_projected["PC1"] = ref_pca_coords[:, 0]
    ref_projected["PC2"] = ref_pca_coords[:, 1]
    if n_components >= 3:
        ref_projected["PC3"] = ref_pca_coords[:, 2]

    extra_projected = None
    if extra_df is not None and not extra_df.empty:
        # Check required columns
        valid_mask = extra_df[FEATURE_COLS].notnull().all(axis=1)
        if valid_mask.any():
            extra_valid = extra_df[valid_mask].copy()
            extra_features = extra_valid[FEATURE_COLS].copy()
            for col in LOG_COLS:
                extra_features[col] = np.log1p(np.maximum(extra_features[col], 0))
            scaled_extra = scaler.transform(extra_features)
            extra_pca_coords = pca.transform(scaled_extra)

            extra_projected = extra_valid.copy()
            extra_projected["PC1"] = extra_pca_coords[:, 0]
            extra_projected["PC2"] = extra_pca_coords[:, 1]
            if n_components >= 3:
                extra_projected["PC3"] = extra_pca_coords[:, 2]

    return ref_projected, pca, extra_projected


def plot_pca_2d(
    ref_df: pd.DataFrame,
    extra_df: Optional[pd.DataFrame] = None,
    color_by: str = "segment"
) -> go.Figure:
    """Generates a 2D interactive PCA scatter plot with explained variance on axis labels."""
    ref_proj, pca, extra_proj = compute_pca(ref_df, n_components=2, extra_df=extra_df)

    var_pc1 = pca.explained_variance_ratio_[0] * 100
    var_pc2 = pca.explained_variance_ratio_[1] * 100

    color_discrete_map = SEGMENT_COLOR_MAP if color_by == "segment" else RISK_COLOR_MAP

    fig = px.scatter(
        ref_proj,
        x="PC1",
        y="PC2",
        color=color_by if color_by in ref_proj.columns else "segment",
        color_discrete_map=color_discrete_map,
        hover_data={
            "account_id": True,
            "total_bets": True,
            "avg_bet": ":.2f",
            "win_rate": ":.3f",
            "avg_cashout": ":.2f",
            "segment": True,
            "risk_level": True,
            "PC1": False,
            "PC2": False
        },
        opacity=0.65,
        title=f"Proyeksi Klaster PCA 2D (Kumulatif: {var_pc1 + var_pc2:.1f}% varians)",
        labels={
            "PC1": f"PC1 ({var_pc1:.1f}% Varians)",
            "PC2": f"PC2 ({var_pc2:.1f}% Varians)",
            "segment": "Segmen Perilaku",
            "risk_level": "Level Risiko"
        }
    )

    fig.update_traces(marker=dict(size=7))

    # Overlay user accounts if present
    if extra_proj is not None and not extra_proj.empty:
        fig.add_trace(
            go.Scatter(
                x=extra_proj["PC1"],
                y=extra_proj["PC2"],
                mode="markers",
                name="Akun Pengguna (Audit/Simulasi)",
                marker=dict(
                    symbol="star",
                    size=13,
                    color="#DC2626",
                    line=dict(color="#FFFFFF", width=1.5)
                ),
                text=extra_proj.apply(
                    lambda r: f"ID: {r.get('account_id', 'User')}<br>Bets: {r.get('total_bets')}<br>Avg Bet: {r.get('avg_bet')}<br>Win Rate: {r.get('win_rate')}<br>Cashout: {r.get('avg_cashout')}",
                    axis=1
                ),
                hoverinfo="text"
            )
        )

    fig.update_layout(
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=60, b=80),
        height=550
    )
    return fig


def plot_pca_3d(
    ref_df: pd.DataFrame,
    extra_df: Optional[pd.DataFrame] = None,
    color_by: str = "segment"
) -> go.Figure:
    """Generates a 3D interactive PCA scatter plot with explained variance on axis labels."""
    ref_proj, pca, extra_proj = compute_pca(ref_df, n_components=3, extra_df=extra_df)

    var_pc1 = pca.explained_variance_ratio_[0] * 100
    var_pc2 = pca.explained_variance_ratio_[1] * 100
    var_pc3 = pca.explained_variance_ratio_[2] * 100
    total_var = var_pc1 + var_pc2 + var_pc3

    color_discrete_map = SEGMENT_COLOR_MAP if color_by == "segment" else RISK_COLOR_MAP

    fig = px.scatter_3d(
        ref_proj,
        x="PC1",
        y="PC2",
        z="PC3",
        color=color_by if color_by in ref_proj.columns else "segment",
        color_discrete_map=color_discrete_map,
        hover_data={
            "account_id": True,
            "total_bets": True,
            "avg_bet": ":.2f",
            "win_rate": ":.3f",
            "avg_cashout": ":.2f",
            "segment": True,
            "risk_level": True,
            "PC1": False,
            "PC2": False,
            "PC3": False
        },
        opacity=0.75,
        title=f"Proyeksi Klaster PCA 3D (Kumulatif: {total_var:.1f}% varians)",
        labels={
            "PC1": f"PC1 ({var_pc1:.1f}%)",
            "PC2": f"PC2 ({var_pc2:.1f}%)",
            "PC3": f"PC3 ({var_pc3:.1f}%)",
            "segment": "Segmen Perilaku",
            "risk_level": "Level Risiko"
        }
    )

    fig.update_traces(marker=dict(size=4))

    # Overlay user accounts if present
    if extra_proj is not None and not extra_proj.empty:
        fig.add_trace(
            go.Scatter3d(
                x=extra_proj["PC1"],
                y=extra_proj["PC2"],
                z=extra_proj["PC3"],
                mode="markers",
                name="Akun Pengguna (Audit/Simulasi)",
                marker=dict(
                    symbol="diamond",
                    size=8,
                    color="#EF4444",
                    line=dict(color="#FFFFFF", width=1.5)
                ),
                text=extra_proj.apply(
                    lambda r: f"ID: {r.get('account_id', 'User')}<br>Bets: {r.get('total_bets')}<br>Avg Bet: {r.get('avg_bet')}<br>Win Rate: {r.get('win_rate')}<br>Cashout: {r.get('avg_cashout')}",
                    axis=1
                ),
                hoverinfo="text"
            )
        )

    fig.update_layout(
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        margin=dict(l=10, r=10, t=50, b=40),
        height=620
    )
    return fig


def plot_radar_comparison(
    user_metrics: Dict[str, float],
    ref_df: pd.DataFrame,
    target_segment: Optional[str] = None
) -> go.Figure:
    """Generates a radar chart comparing a single user account vs segment median vs population median.

    Values are normalized between 0 and 1 using population percentiles (P1-P99)
    for a balanced and intuitive polygon comparison.
    """
    # 4 behavioral metrics
    metrics = ["total_bets", "avg_bet", "win_rate", "avg_cashout"]
    labels = ["Frekuensi (total_bets)", "Nominal (avg_bet)", "Win Rate", "Pengali (avg_cashout)"]

    # Calculate population bounds (P5 - P95) for robust min-max normalization
    bounds = {}
    pop_medians = {}
    for m in metrics:
        p_min = float(np.percentile(ref_df[m], 5))
        p_max = float(np.percentile(ref_df[m], 95))
        if p_max <= p_min:
            p_max = p_min + 1.0
        bounds[m] = (p_min, p_max)
        pop_medians[m] = float(ref_df[m].median())

    # Segment medians
    if target_segment and target_segment in ref_df["segment"].values:
        seg_sub = ref_df[ref_df["segment"] == target_segment]
        seg_medians = {m: float(seg_sub[m].median()) for m in metrics}
        seg_label = f"Median Segmen ({target_segment})"
    else:
        seg_medians = pop_medians
        seg_label = "Median Segmen Terdekat"

    def normalize(val: float, m: str) -> float:
        low, high = bounds[m]
        # apply log scale for skewed features before normalizing
        if m in LOG_COLS:
            val_log = np.log1p(max(val, 0))
            low_log = np.log1p(max(low, 0))
            high_log = np.log1p(max(high, 0))
            norm = (val_log - low_log) / (high_log - low_log) if high_log > low_log else 0.5
        else:
            norm = (val - low) / (high - low)
        return float(np.clip(norm, 0.05, 1.0))

    user_norm = [normalize(user_metrics.get(m, 0.0), m) for m in metrics]
    seg_norm = [normalize(seg_medians[m], m) for m in metrics]
    pop_norm = [normalize(pop_medians[m], m) for m in metrics]

    # Close polygons
    user_norm += [user_norm[0]]
    seg_norm += [seg_norm[0]]
    pop_norm += [pop_norm[0]]
    radar_labels = labels + [labels[0]]

    fig = go.Figure()

    # Population Median
    fig.add_trace(go.Scatterpolar(
        r=pop_norm,
        theta=radar_labels,
        fill='toself',
        fillcolor='rgba(156, 163, 175, 0.15)',
        line=dict(color='#9CA3AF', dash='dash', width=1.5),
        name='Median Populasi (1.111 Akun)'
    ))

    # Segment Median
    fig.add_trace(go.Scatterpolar(
        r=seg_norm,
        theta=radar_labels,
        fill='toself',
        fillcolor='rgba(59, 130, 246, 0.2)',
        line=dict(color='#2563EB', width=2),
        name=seg_label
    ))

    # User Profile
    fig.add_trace(go.Scatterpolar(
        r=user_norm,
        theta=radar_labels,
        fill='toself',
        fillcolor='rgba(239, 68, 68, 0.3)',
        line=dict(color='#DC2626', width=2.5),
        name='Profil Akun Uji'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1.05],
                tickvals=[0.25, 0.5, 0.75, 1.0],
                ticktext=["Rendah", "Menengah", "Tinggi", "Ekstrem"]
            )
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        template="plotly_white",
        margin=dict(l=40, r=40, t=30, b=60),
        height=450
    )
    return fig


def plot_risk_distribution(
    df: pd.DataFrame,
    as_donut: bool = False
) -> go.Figure:
    """Generates distribution chart of accounts by risk level."""
    if "risk_level" not in df.columns:
        raise ValueError("DataFrame harus memiliki kolom 'risk_level'")

    risk_counts = df["risk_level"].value_counts().reset_index()
    risk_counts.columns = ["risk_level", "count"]
    risk_counts["percentage"] = (risk_counts["count"] / len(df)) * 100

    order = ["Rendah", "Perlu Dipantau", "Tinggi", "Review manual", "Data belum cukup"]
    risk_counts["order"] = risk_counts["risk_level"].apply(lambda x: order.index(x) if x in order else 99)
    risk_counts = risk_counts.sort_values("order")

    if as_donut:
        fig = px.pie(
            risk_counts,
            values="count",
            names="risk_level",
            color="risk_level",
            color_discrete_map=RISK_COLOR_MAP,
            hole=0.45,
            title="Proporsi Akun per Level Risiko"
        )
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Jumlah: %{value:,} akun<br>Proporsi: %{percent:.1%}'
        )
    else:
        fig = px.bar(
            risk_counts,
            x="risk_level",
            y="count",
            color="risk_level",
            color_discrete_map=RISK_COLOR_MAP,
            text=risk_counts.apply(lambda r: f"{r['count']:,} ({r['percentage']:.1f}%)", axis=1),
            title="Distribusi Jumlah Akun per Level Risiko",
            labels={"risk_level": "Level Risiko", "count": "Jumlah Akun"}
        )
        fig.update_traces(textposition='outside')

    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        height=400,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def plot_turnover_by_risk(df: pd.DataFrame) -> go.Figure:
    """Calculates and visualizes estimated turnover (sum of total_bets * avg_bet) per risk level."""
    df_calc = df.copy()
    if "turnover" not in df_calc.columns:
        df_calc["turnover"] = df_calc["total_bets"] * df_calc["avg_bet"]

    grouped = df_calc.groupby("risk_level", observed=False)["turnover"].agg(["sum", "count"]).reset_index()
    total_sum = grouped["sum"].sum()
    grouped["pct_turnover"] = (grouped["sum"] / total_sum * 100) if total_sum > 0 else 0

    order = ["Rendah", "Perlu Dipantau", "Tinggi", "Review manual", "Data belum cukup"]
    grouped["order"] = grouped["risk_level"].apply(lambda x: order.index(x) if x in order else 99)
    grouped = grouped.sort_values("order")

    fig = px.bar(
        grouped,
        x="risk_level",
        y="sum",
        color="risk_level",
        color_discrete_map=RISK_COLOR_MAP,
        text=grouped.apply(lambda r: f"{r['pct_turnover']:.1f}%<br>({r['sum']:,.0f} bits)", axis=1),
        title="Estimasi Perputaran Dana per Level Risiko (Σ total_bets × avg_bet)",
        labels={"risk_level": "Level Risiko", "sum": "Estimasi Perputaran Dana (bits)"}
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        height=420,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def plot_segment_comparison(ref_df: pd.DataFrame, metric: str = "win_rate") -> go.Figure:
    """Generates comparison charts (Box plot) across the 5 segments for a selected metric."""
    metric_labels = {
        "win_rate": "Win Rate (Rasio Menang)",
        "avg_cashout": "Rata-rata Target Pengali (Cashout Multiplier)",
        "avg_bet": "Rata-rata Nominal Taruhan (bits)",
        "total_bets": "Total Frekuensi Taruhan (total_bets)"
    }
    label = metric_labels.get(metric, metric)

    fig = px.box(
        ref_df,
        x="segment",
        y=metric,
        color="segment",
        color_discrete_map=SEGMENT_COLOR_MAP,
        title=f"Perbandingan Distribusi Metrik: {label}",
        labels={"segment": "Segmen Perilaku", metric: label},
        points="outliers"
    )

    if metric in LOG_COLS and ref_df[metric].max() > 500:
        fig.update_yaxes(type="log")

    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        xaxis_tickangle=-25,
        height=480,
        margin=dict(l=40, r=40, t=50, b=100)
    )
    return fig


def plot_segment_medians_summary(ref_df: pd.DataFrame) -> go.Figure:
    """Generates a multi-bar comparison chart summarizing normalized median profiles across 5 segments."""
    metrics = ["total_bets", "avg_bet", "win_rate", "avg_cashout"]
    summary_data = []

    for seg in ref_df["segment"].unique():
        sub = ref_df[ref_df["segment"] == seg]
        for m in metrics:
            summary_data.append({
                "segment": seg,
                "metric": m,
                "median_val": float(sub[m].median())
            })
    summ_df = pd.DataFrame(summary_data)

    # Human-friendly labels
    metric_display = {
        "total_bets": "Total Bets",
        "avg_bet": "Avg Bet (bits)",
        "win_rate": "Win Rate",
        "avg_cashout": "Avg Cashout (x)"
    }
    summ_df["metric_name"] = summ_df["metric"].map(metric_display)

    fig = px.bar(
        summ_df,
        x="metric_name",
        y="median_val",
        color="segment",
        barmode="group",
        color_discrete_map=SEGMENT_COLOR_MAP,
        title="Ringkasan Nilai Median Metrik per Segmen",
        labels={"metric_name": "Metrik Perilaku", "median_val": "Nilai Median", "segment": "Segmen"}
    )
    fig.update_layout(
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5),
        height=480,
        margin=dict(l=40, r=40, t=50, b=100)
    )
    return fig
