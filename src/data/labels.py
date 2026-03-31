import pandas as pd
from pathlib import Path
from typing import Dict, List
from collections import Counter


def load_labels_for_region(region_id: str, data_root: str) -> Dict[int, str]:
    """Load ground truth labels for a region."""
    labels_path = Path(data_root) / region_id / f"{region_id}.labels.csv"
    if not labels_path.exists():
        return {}
    df = pd.read_csv(labels_path)
    return dict(zip(df['CELL_ID'], df['LABEL']))


def get_roi_true_label(roi_cell_ids: List[int], labels_dict: Dict[int, str]) -> str:
    """Get ground truth label for an ROI using majority voting."""
    roi_labels = [labels_dict.get(cid) for cid in roi_cell_ids if cid in labels_dict]
    if not roi_labels:
        return "Unknown"
    label_counts = Counter(roi_labels)
    return label_counts.most_common(1)[0][0]
