# GambleRisk Guard

Behavioral risk profiling for online betting accounts. GambleRisk Guard segments accounts by how they bet (frequency, stake size, win rate, and cash-out target) and maps each segment to a risk level that an analyst can review.

> **Status:** in development.

## What it does

- **Single Profile Simulator:** enter one account's betting metrics and get its behavioral segment, risk level, confidence score, and a comparison radar chart.
- **Batch CSV Auditor:** upload many accounts (aggregated metrics or raw bet logs), get a labeled table, and download the results as CSV.
- **Analytics Dashboard:** PCA scatter plots (2D/3D) with explained variance, risk distribution, and segment comparison.

Outputs are indications for human review, not diagnoses.

## How it works

```text
input → validation → log1p → StandardScaler → Isolation Forest (anomaly flag)
      → K-Means (k=5, segment) → risk level + recommendation
      → GMM predict_proba (confidence / borderline flag)
```

The model was trained on 1,111 player profiles (accounts with at least 10 bets) derived from 50,000 bets on the Bustabit crash game. Features: `total_bets`, `avg_bet`, `win_rate`, `avg_cashout`. Profit is deliberately excluded from the features and used only to interpret segments.

| Risk level | Segment |
|:---|:---|
| 🟢 Low | Low target, high win rate |
| 🟡 Monitor | High frequency, low target |
| 🟡 Monitor | High stake |
| 🔴 High | Medium target |
| 🔴 High | High cash-out target |
| ⚪ Review | Anomaly (Isolation Forest) or out-of-range input |

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Run the tests:

```bash
pytest
```

## Project structure

```text
app.py            Streamlit entry point
pages/            App modules (simulator, batch auditor, dashboard)
src/              Preprocessing, inference, and visualization logic
models/           Trained artifacts and segment/reference metadata
notebooks/        Model training and benchmarking
data/             Reference profiles and demo data
tests/            Pipeline consistency and input validation tests
docs/             Product requirements (PRD.md)
```

## Limitations

- The model works in the behavioral feature space of the training data (stakes in bits, cash-out multipliers). Applying it to bank or e-wallet transactions requires retraining on that domain, not just renaming columns.
- Clusters are descriptive segments with weak-to-moderate separation (Silhouette ≈ 0.26). Relationships between behavior and losses are associations, not causal claims.
