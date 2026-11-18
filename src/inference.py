"""Model loading, prediction, risk level mapping, and recommendations.

Cluster IDs in kmeans.pkl are arbitrary; always map them through models/segment_map.json.
K-Means is the primary label. GMM provides a confidence score only (borderline if max prob < 0.6).
Isolation Forest flags anomalies for manual review; flagged accounts are never dropped.
"""

BORDERLINE_THRESHOLD = 0.6

# TODO (Role A): load_artifacts(), predict(df) -> DataFrame with segment, risk_level,
# confidence, is_borderline, is_anomaly, recommendation
