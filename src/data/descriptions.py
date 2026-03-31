from pathlib import Path
from typing import Dict, Tuple, Any, List, Union
import pandas as pd
import numpy as np


def generate_roi_descriptions(
    region_id: str,
    root_dir: Union[str, Path],
    target_rois: List[Dict[str, Any]],
    include_biomarker: bool = True,
    include_cell_type: bool = True,
    include_morphology: bool = False,
    verbose: bool = False
) -> Dict[Tuple[str, str], str]:
    """
    Generate ROI descriptions for a region (simplified version).
    """
    root_dir = Path(root_dir)

    if not target_rois:
        raise ValueError("target_rois cannot be empty")

    for roi in target_rois:
        if 'cell_ids' not in roi:
            raise ValueError("Each ROI must contain 'cell_ids' key")
        if 'patch_name' not in roi:
            raise ValueError("Each ROI must contain 'patch_name' key")

    # Load cell type data
    cell_types_path = root_dir / region_id / f"{region_id}.cell_types.csv"
    if not cell_types_path.exists():
        raise FileNotFoundError(f"Cell types file not found: {cell_types_path}")
    cell_types_df = pd.read_csv(cell_types_path)

    # Load expression data if needed
    cell_bm_df = None
    if include_biomarker:
        expr_path = root_dir / region_id / f"{region_id}.expression.csv"
        if expr_path.exists():
            cell_bm_df = pd.read_csv(expr_path)

    descriptions = {}

    for roi in target_rois:
        patch_name = roi['patch_name']
        cell_ids = roi['cell_ids'] if isinstance(roi['cell_ids'], list) else [roi['cell_ids']]

        # Filter to cells in this ROI
        roi_cells = cell_types_df[cell_types_df['CELL_ID'].isin(cell_ids)]

        summary_parts = []

        # Cell type summary
        if include_cell_type and not roi_cells.empty:
            cell_type_counts = roi_cells['ANNOTATION_LABEL'].value_counts(normalize=True)
            top_types = cell_type_counts.head(3)
            ct_summary = "Cell type composition: " + ", ".join(
                [f"{label} ({pct:.1%})" for label, pct in zip(top_types.index, top_types.values)]
            )
            summary_parts.append(ct_summary)

        # Biomarker summary (simplified)
        if include_biomarker and cell_bm_df is not None:
            roi_bm = cell_bm_df[cell_bm_df['CELL_ID'].isin(cell_ids)]
            if not roi_bm.empty and 'VALUE' in roi_bm.columns:
                # Get mean expression per biomarker
                numeric_cols = roi_bm.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    top_bm = roi_bm[numeric_cols].mean().nlargest(5)
                    bm_summary = "Highly expressed biomarkers: " + ", ".join([f"{col}" for col in top_bm.index])
                    summary_parts.append(bm_summary)

        description = "\n".join(summary_parts)
        descriptions[(region_id, patch_name)] = description

    if verbose:
        print(f"Generated {len(descriptions)} ROI descriptions")

    return descriptions
