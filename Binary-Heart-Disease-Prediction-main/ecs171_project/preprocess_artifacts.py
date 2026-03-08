from dataclasses import dataclass
from typing import Dict, List
from sklearn.preprocessing import StandardScaler


@dataclass
class PreprocessArtifacts:
    medians: Dict[str, float]
    modes: Dict[str, int]
    scaler: StandardScaler
    scale_cols: List[str]
    feature_columns: List[str]
    dummy_columns: List[str]

    # For age_group bins (fixed)
    age_bins: List[int]
