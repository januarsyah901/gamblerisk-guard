import numpy as np
import pandas as pd

np.random.seed(42)

# 1. Generate reference_profiles.csv (1,111 rows aligned with UTS report)
# Segments:
# 1. Target Rendah, Win Rate Tinggi: n=297, median: total_bets=17, avg_bet=43, win_rate=0.80, avg_cashout=1.19
# 2. Frekuensi Tinggi, Target Rendah: n=235, median: total_bets=69, avg_bet=39, win_rate=0.68, avg_cashout=1.37
# 3. Taruhan Nominal Tinggi: n=226, median: total_bets=20, avg_bet=2225, win_rate=0.60, avg_cashout=1.49
# 4. Target Menengah: n=244, median: total_bets=16, avg_bet=84, win_rate=0.33, avg_cashout=2.30
# 5. Target Cashout Tinggi: n=109, median: total_bets=36, avg_bet=106, win_rate=0.14, avg_cashout=9.05

def generate_segment(n, seg_name, risk_level, median_bets, median_bet, median_wr, median_co, roi):
    # Log-normal for bets, bet amount, cashout; Beta/logit-normal for win_rate
    bets = np.random.lognormal(mean=np.log(median_bets), sigma=0.4, size=n)
    bets = np.clip(np.round(bets), 10, None).astype(int)
    
    avg_bets = np.random.lognormal(mean=np.log(median_bet), sigma=0.45, size=n)
    avg_bets = np.round(np.clip(avg_bets, 1.0, None), 2)
    
    # Win rate beta distribution centered near median_wr
    a = median_wr * 25
    b = (1 - median_wr) * 25
    win_rate = np.random.beta(a, b, size=n)
    win_rate = np.round(np.clip(win_rate, 0.01, 0.99), 3)
    
    # Cashout multiplier: log-normal with shift >= 1.01
    cashout = 1.0 + np.random.lognormal(mean=np.log(max(median_co - 1.0, 0.05)), sigma=0.35, size=n)
    cashout = np.round(cashout, 2)
    
    # Estimated total_profit based on ROI
    turnover = bets * avg_bets
    profit = np.round(turnover * (roi + np.random.normal(0, 0.08, size=n)), 2)
    
    return pd.DataFrame({
        "total_bets": bets,
        "avg_bet": avg_bets,
        "win_rate": win_rate,
        "avg_cashout": cashout,
        "total_profit": profit,
        "segment": seg_name,
        "risk_level": risk_level
    })

df1 = generate_segment(297, "Target Rendah, Win Rate Tinggi", "Rendah", 17, 43, 0.80, 1.19, 0.044)
df2 = generate_segment(235, "Frekuensi Tinggi, Target Rendah", "Perlu Dipantau", 69, 39, 0.68, 1.37, 0.044)
df3 = generate_segment(226, "Taruhan Nominal Tinggi", "Perlu Dipantau", 20, 2225, 0.60, 1.49, 0.071)
df4 = generate_segment(244, "Target Menengah", "Tinggi", 16, 84, 0.33, 2.30, -0.246)
df5 = generate_segment(109, "Target Cashout Tinggi", "Tinggi", 36, 106, 0.14, 9.05, -0.233)

ref_df = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)
ref_df["account_id"] = [f"ACC_REF_{i+1:04d}" for i in range(len(ref_df))]
# reorder columns
ref_df = ref_df[["account_id", "total_bets", "avg_bet", "win_rate", "avg_cashout", "total_profit", "segment", "risk_level"]]
ref_df.to_csv("/Users/shino/gamblerisk-guard/data/reference_profiles.csv", index=False)
print("reference_profiles.csv created:", len(ref_df), "rows")

# 2. Generate sample_test.csv
# Must include every risk level, boundary cases, anomalies, and insufficient data (< 10 bets)
sample_rows = [
    # Low Risk
    {"account_id": "ACC_SMP_001", "total_bets": 22, "avg_bet": 40.0, "win_rate": 0.82, "avg_cashout": 1.18, "total_profit": 35.2, "expected_scenario": "Low Risk (Target Rendah, Win Rate Tinggi)"},
    {"account_id": "ACC_SMP_002", "total_bets": 15, "avg_bet": 50.0, "win_rate": 0.78, "avg_cashout": 1.20, "total_profit": 28.0, "expected_scenario": "Low Risk (Konsisten Menang)"},
    
    # Monitor - High Frequency
    {"account_id": "ACC_SMP_003", "total_bets": 140, "avg_bet": 32.0, "win_rate": 0.70, "avg_cashout": 1.35, "total_profit": 120.5, "expected_scenario": "Perlu Dipantau (Frekuensi Tinggi, Target Rendah)"},
    {"account_id": "ACC_SMP_004", "total_bets": 210, "avg_bet": 25.0, "win_rate": 0.66, "avg_cashout": 1.38, "total_profit": -50.0, "expected_scenario": "Perlu Dipantau (Sangat Sering Bermain)"},
    
    # Monitor - High Stakes
    {"account_id": "ACC_SMP_005", "total_bets": 25, "avg_bet": 3200.0, "win_rate": 0.62, "avg_cashout": 1.45, "total_profit": 4200.0, "expected_scenario": "Perlu Dipantau (Taruhan Nominal Tinggi / High Stake)"},
    {"account_id": "ACC_SMP_006", "total_bets": 18, "avg_bet": 5500.0, "win_rate": 0.58, "avg_cashout": 1.52, "total_profit": -1200.0, "expected_scenario": "Perlu Dipantau (Eksposur Modal Besar)"},
    
    # High Risk - Medium Target
    {"account_id": "ACC_SMP_007", "total_bets": 18, "avg_bet": 95.0, "win_rate": 0.31, "avg_cashout": 2.35, "total_profit": -480.0, "expected_scenario": "Tinggi (Target Menengah, Negatif ROI)"},
    {"account_id": "ACC_SMP_008", "total_bets": 24, "avg_bet": 80.0, "win_rate": 0.35, "avg_cashout": 2.20, "total_profit": -360.0, "expected_scenario": "Tinggi (Target Menengah)"},
    
    # High Risk - High Cashout
    {"account_id": "ACC_SMP_009", "total_bets": 40, "avg_bet": 110.0, "win_rate": 0.12, "avg_cashout": 9.80, "total_profit": -1050.0, "expected_scenario": "Tinggi (Target Cashout Tinggi, Low Win Rate)"},
    {"account_id": "ACC_SMP_010", "total_bets": 55, "avg_bet": 130.0, "win_rate": 0.15, "avg_cashout": 8.50, "total_profit": -1200.0, "expected_scenario": "Tinggi (Chasing High Multiplier)"},
    
    # Borderline cases
    {"account_id": "ACC_SMP_011", "total_bets": 28, "avg_bet": 120.0, "win_rate": 0.50, "avg_cashout": 1.80, "total_profit": -80.0, "expected_scenario": "Zona Perbatasan (Transisi Antar Klaster)"},
    
    # Anomaly / Outliers (Isolation Forest Target)
    {"account_id": "ACC_SMP_012", "total_bets": 800, "avg_bet": 15000.0, "win_rate": 0.05, "avg_cashout": 45.0, "total_profit": -500000.0, "expected_scenario": "Anomali (Ekstrem Stake & High Cashout Outlier)"},
    {"account_id": "ACC_SMP_013", "total_bets": 500, "avg_bet": 1.0, "win_rate": 0.99, "avg_cashout": 1.01, "total_profit": 5.0, "expected_scenario": "Anomali (Micro-betting Bot Pattern)"},
    
    # Insufficient Data (< 10 bets)
    {"account_id": "ACC_SMP_014", "total_bets": 4, "avg_bet": 50.0, "win_rate": 0.50, "avg_cashout": 1.50, "total_profit": 0.0, "expected_scenario": "Data Belum Cukup (< 10 bets)"},
    {"account_id": "ACC_SMP_015", "total_bets": 1, "avg_bet": 100.0, "win_rate": 1.00, "avg_cashout": 2.00, "total_profit": 100.0, "expected_scenario": "Data Belum Cukup (Single bet)"}
]
sample_df = pd.DataFrame(sample_rows)
sample_df.to_csv("/Users/shino/gamblerisk-guard/data/sample_test.csv", index=False)
print("sample_test.csv created:", len(sample_df), "rows")

# 3. Generate sample_raw_bets.csv (raw transaction log for FR-21 demo)
raw_bets = []
for acc in ["ACC_RAW_01", "ACC_RAW_02", "ACC_RAW_03"]:
    n_bets = 15 if acc != "ACC_RAW_03" else 5  # ACC_RAW_03 will have < 10 bets
    for b in range(n_bets):
        won = np.random.rand() > 0.4
        bet_amt = np.random.choice([10, 20, 50, 100])
        cashed_out = round(np.random.uniform(1.2, 3.5), 2) if won else ""
        raw_bets.append({
            "account_id": acc,
            "bet": bet_amt,
            "cashed_out": cashed_out
        })
raw_df = pd.DataFrame(raw_bets)
raw_df.to_csv("/Users/shino/gamblerisk-guard/data/sample_raw_bets.csv", index=False)
print("sample_raw_bets.csv created:", len(raw_df), "rows")
