import os
import pandas as pd
import streamlit as st

from src.visualization import (
    plot_pca_2d,
    plot_pca_3d,
    plot_radar_comparison,
    plot_risk_distribution,
    plot_turnover_by_risk,
    plot_segment_comparison,
    plot_segment_medians_summary,
    FEATURE_COLS
)

st.set_page_config(
    page_title="Analytics Dashboard | GambleRisk Guard",
    page_icon="📊",
    layout="wide"
)

# --- HEADER ---
st.title("📊 Analytics Dashboard (Modul 3)")
st.caption("Eksplorasi visual peta segmen perilaku taruhan, proyeksi PCA, dan estimasi eksposur risiko keuangan.")

# --- DATA LOADING ---
DATA_PATH = "data/reference_profiles.csv"

@st.cache_data
def load_reference_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    else:
        st.error(f"File data referensi tidak ditemukan di `{DATA_PATH}`.")
        return pd.DataFrame()

ref_df = load_reference_data()

if ref_df.empty:
    st.stop()

# Ensure turnover column exists
if "turnover" not in ref_df.columns:
    ref_df["turnover"] = ref_df["total_bets"] * ref_df["avg_bet"]

# Check for session state accounts from Module 1 or Module 2
extra_accounts = None
if "audit_results" in st.session_state and isinstance(st.session_state["audit_results"], pd.DataFrame):
    extra_accounts = st.session_state["audit_results"]
elif "simulator_result" in st.session_state and isinstance(st.session_state["simulator_result"], dict):
    extra_accounts = pd.DataFrame([st.session_state["simulator_result"]])

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("⚙️ Filter & Opsi Analisis")

    # Filter Segmen
    all_segments = sorted(ref_df["segment"].unique().tolist())
    selected_segments = st.multiselect(
        "Pilih Segmen Perilaku:",
        options=all_segments,
        default=all_segments
    )

    # Filter Level Risiko
    all_risks = ["Rendah", "Perlu Dipantau", "Tinggi"]
    selected_risks = st.multiselect(
        "Pilih Level Risiko:",
        options=all_risks,
        default=all_risks
    )

    st.markdown("---")
    st.subheader("🎯 Opsi Overlay Akun")
    
    use_sample_overlay = st.checkbox(
        "Tampilkan Sampel Akun Uji (Overlay)",
        value=False,
        help="Menampilkan 15 titik akun uji dari data/sample_test.csv pada proyeksi PCA"
    )

    if use_sample_overlay:
        sample_path = "data/sample_test.csv"
        if os.path.exists(sample_path):
            extra_accounts = pd.read_csv(sample_path)
            st.success(f"Memuat {len(extra_accounts)} akun sampel pengujian.")
        else:
            st.warning("File `data/sample_test.csv` belum ditemukan.")

# Filter reference dataframe based on selection
filtered_ref_df = ref_df[
    ref_df["segment"].isin(selected_segments) &
    ref_df["risk_level"].isin(selected_risks)
]

if filtered_ref_df.empty:
    st.warning("⚠️ Tidak ada data yang sesuai dengan kombinasi filter yang dipilih.")
    st.stop()

# --- TOP KPI METRICS ---
total_accounts = len(filtered_ref_df)
total_turnover = filtered_ref_df["turnover"].sum()
high_risk_count = len(filtered_ref_df[filtered_ref_df["risk_level"] == "Tinggi"])
high_risk_pct = (high_risk_count / total_accounts * 100) if total_accounts > 0 else 0

# Calculate volume proportion of "Taruhan Nominal Tinggi"
high_stake_turnover = filtered_ref_df[filtered_ref_df["segment"] == "Taruhan Nominal Tinggi"]["turnover"].sum()
high_stake_pct = (high_stake_turnover / total_turnover * 100) if total_turnover > 0 else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("👥 Total Akun Terfilter", f"{total_accounts:,} akun")
col2.metric("💰 Estimasi Perputaran Dana", f"{total_turnover:,.0f} bits")
col3.metric("🔴 Proporsi Risiko Tinggi", f"{high_risk_pct:.1f}% ({high_risk_count} akun)")
col4.metric("⚖️ Porsi Volume Nominal Tinggi", f"{high_stake_pct:.1f}% volume")

st.markdown("---")

# --- TABS LAYOUT ---
tab_pca, tab_distribution, tab_segments, tab_radar = st.tabs([
    "🌐 Proyeksi Klaster PCA (2D & 3D)",
    "📊 Distribusi Risiko & Perputaran Dana",
    "🔍 Profil Karakteristik Segmen",
    "🎯 Radar Benchmark Profil"
])

# === TAB 1: PCA PROJECTION ===
with tab_pca:
    st.subheader("Proyeksi PCA (Principal Component Analysis)")
    st.markdown(
        """
        Visualisasi reduksi dimensi dari 4 fitur perilaku taruhan (`total_bets`, `avg_bet`, `win_rate`, `avg_cashout`) 
        setelah transformasi logaritmik dan standarisasi. Label sumbu menunjukkan persentase *explained variance ratio*.
        """
    )

    col_ctrl1, col_ctrl2 = st.columns([1, 1])
    with col_ctrl1:
        dim_choice = st.radio("Dimensi Proyeksi:", ["2D Scatter", "3D Interactive"], horizontal=True)
    with col_ctrl2:
        color_choice = st.radio("Pewarnaan Berdasarkan:", ["Segmen Perilaku", "Level Risiko"], horizontal=True)

    color_field = "segment" if color_choice == "Segmen Perilaku" else "risk_level"

    with st.spinner("Merender proyeksi PCA..."):
        if dim_choice == "2D Scatter":
            fig_pca = plot_pca_2d(
                ref_df=filtered_ref_df,
                extra_df=extra_accounts if (extra_accounts is not None and len(extra_accounts) > 0) else None,
                color_by=color_field
            )
            st.plotly_chart(fig_pca, use_container_width=True)
        else:
            fig_pca = plot_pca_3d(
                ref_df=filtered_ref_df,
                extra_df=extra_accounts if (extra_accounts is not None and len(extra_accounts) > 0) else None,
                color_by=color_field
            )
            st.plotly_chart(fig_pca, use_container_width=True)

    st.info(
        "💡 **Keterangan:** Titik referensi menggambarkan 1.111 profil pemain baseline dari data latih. "
        "Bintang merah (jika opsi overlay aktif) menandai posisi akun baru hasil pengujian / simulasi."
    )

# === TAB 2: DISTRIBUSI RISIKO & PERPUTARAN DANA ===
with tab_distribution:
    st.subheader("Distribusi Risiko & Estimasi Eksposur Finansial")
    st.markdown(
        "Analisis perbandingan antara **jumlah akun** dan **estimasi perputaran dana (turnover)** per level risiko."
    )

    col_dist1, col_dist2 = st.columns(2)

    with col_dist1:
        chart_type = st.segmented_control("Format Tampilan Akun:", ["Bar Chart", "Donut Chart"], default="Donut Chart")
        fig_dist = plot_risk_distribution(filtered_ref_df, as_donut=(chart_type == "Donut Chart"))
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_dist2:
        fig_turnover = plot_turnover_by_risk(filtered_ref_df)
        st.plotly_chart(fig_turnover, use_container_width=True)

    # Summary Data Table
    st.markdown("#### 📋 Tabel Rekapitulasi per Level Risiko")
    summary_table = filtered_ref_df.groupby("risk_level", observed=False).agg(
        Jumlah_Akun=("account_id", "count"),
        Total_Perputaran_Bits=("turnover", "sum"),
        Rata_rata_Bet=("avg_bet", "median"),
        Median_Win_Rate=("win_rate", "median"),
        Median_Cashout=("avg_cashout", "median")
    ).reset_index()

    summary_table["% Akun"] = (summary_table["Jumlah_Akun"] / len(filtered_ref_df) * 100).round(1)
    summary_table["% Volume"] = (summary_table["Total_Perputaran_Bits"] / filtered_ref_df["turnover"].sum() * 100).round(1)

    st.dataframe(
        summary_table.rename(columns={
            "risk_level": "Level Risiko",
            "Jumlah_Akun": "Jumlah Akun",
            "Total_Perputaran_Bits": "Est. Total Turnover (bits)",
            "Rata_rata_Bet": "Median Avg Bet",
            "Median_Win_Rate": "Median Win Rate",
            "Median_Cashout": "Median Cashout (x)"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.warning(
        "📌 **Catatan Analis:** Segmen *'Taruhan Nominal Tinggi'* (Level: Perlu Dipantau) menyumbang sebagian besar "
        "perputaran dana (~88%). Oleh karena itu, akun di segmen ini dipantau ketat dari sisi mitigasi likuiditas/eksposur modal, "
        "bukan semata-mata karena persentase kerugian."
    )

# === TAB 3: PROFIL SEGMEN ===
with tab_segments:
    st.subheader("Karakteristik & Perbandingan Antar Segmen")
    st.markdown("Menganalisis sebaran nilai dari masing-masing fitur perilaku di antara kelima segmen model.")

    metric_choice = st.selectbox(
        "Pilih Metrik Perilaku untuk Dianalisis:",
        options=["win_rate", "avg_cashout", "avg_bet", "total_bets"],
        format_func=lambda x: {
            "win_rate": "Win Rate (Rasio Kemenangan)",
            "avg_cashout": "Target Pengali Pencairan (avg_cashout)",
            "avg_bet": "Nominal Rata-rata Taruhan (avg_bet dalam bits)",
            "total_bets": "Frekuensi Bermain (total_bets)"
        }[x]
    )

    fig_box = plot_segment_comparison(filtered_ref_df, metric=metric_choice)
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("#### 📊 Profil Median Keseluruhan Segmen")
    fig_summary = plot_segment_medians_summary(filtered_ref_df)
    st.plotly_chart(fig_summary, use_container_width=True)

# === TAB 4: RADAR BENCHMARK ===
with tab_radar:
    st.subheader("Radar Benchmark Profil Akun vs Segmen")
    st.markdown(
        "Bandingkan profil akun (masukan interaktif atau dari sampel) terhadap rata-rata median populasi dan segmen target."
    )

    col_rad_ctrl, col_rad_chart = st.columns([1, 2])

    with col_rad_ctrl:
        st.markdown("**Parameter Akun yang Diuji:**")
        test_bets = st.number_input("Total Bets (Frekuensi)", min_value=1, max_value=2000, value=25)
        test_bet = st.number_input("Average Bet (bits)", min_value=1.0, max_value=50000.0, value=45.0, step=10.0)
        test_wr = st.slider("Win Rate", min_value=0.0, max_value=1.0, value=0.75, step=0.01)
        test_co = st.number_input("Average Cashout Multiplier (x)", min_value=1.0, max_value=50.0, value=1.25, step=0.1)

        test_segment = st.selectbox("Bandingkan terhadap Segmen:", options=all_segments)

    with col_rad_chart:
        user_input_dict = {
            "total_bets": test_bets,
            "avg_bet": test_bet,
            "win_rate": test_wr,
            "avg_cashout": test_co
        }
        fig_radar = plot_radar_comparison(user_input_dict, ref_df, target_segment=test_segment)
        st.plotly_chart(fig_radar, use_container_width=True)

# --- FOOTER ---
st.markdown("---")
st.caption(
    "GambleRisk Guard • Visualisasi & Analisis Data • Dibangun sesuai spesifikasi PRD & Rencana Kerja Tim SVPL261503."
)
