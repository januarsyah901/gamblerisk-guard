"""Input validation, aggregation, and feature transformation.

Must reproduce the training notebook exactly:
- Features: total_bets, avg_bet, win_rate, avg_cashout.
- log1p on total_bets, avg_bet, avg_cashout; win_rate stays as is.
- avg_cashout = mean cash-out multiplier over winning bets; 1.0 for accounts that never won.
- Accounts with fewer than MIN_BETS bets are out of scope ("insufficient data").
"""

FEATURE_COLS = ["total_bets", "avg_bet", "win_rate", "avg_cashout"]
LOG_COLS = ["total_bets", "avg_bet", "avg_cashout"]
MIN_BETS = 10

AGGREGATED_REQUIRED_COLS = ["account_id", *FEATURE_COLS]
RAW_REQUIRED_COLS = ["account_id", "bet", "cashed_out"]

# TODO (Role A): validate_aggregated(df), aggregate_raw_bets(df), transform(df) -> scaled array
