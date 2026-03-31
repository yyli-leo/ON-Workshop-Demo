import pickle
from pathlib import Path
from typing import Dict, Tuple


def load_simulation_data(sota_dir: str) -> tuple:
    """Load cached SOTA results from simulation mode."""
    sota_path = Path(sota_dir)

    with open(sota_path / "visual_reports.pkl", 'rb') as f:
        visual_results = pickle.load(f)
    with open(sota_path / "omics_reports.pkl", 'rb') as f:
        omics_results = pickle.load(f)
    with open(sota_path / "final_interpretations.pkl", 'rb') as f:
        interp_results = pickle.load(f)
    with open(sota_path / "roi_manifest.pkl", 'rb') as f:
        sota_manifest = pickle.load(f)

    # Build manifest from SOTA data
    manifest = []
    for item in sota_manifest:
        manifest.append({
            "key": item["key"],
            "region_id": item["region_id"],
            "patch_name": item["patch_name"],
            "cell_ids": item.get("cell_ids", []),
            "num_cells": item.get("num_cells", 0),
        })

    return visual_results, omics_results, interp_results, manifest


def load_registry_data(registry_path: str) -> tuple:
    """Load registry for real mode execution."""
    with open(registry_path, 'rb') as f:
        registry = pickle.load(f)

    # Build manifest from registry
    manifest = []
    for region_id, shard in registry.shards.items():
        for (feat_id, spatial_id), roi_indices in shard.cluster_center_indices.items():
            for roi_idx in roi_indices:
                roi = shard.rois[roi_idx]
                manifest.append({
                    "key": (region_id, roi.patch_name),
                    "region_id": region_id,
                    "roi_obj": roi,
                    "patch_name": roi.patch_name,
                    "cell_ids": roi.cell_ids,
                    "num_cells": len(roi.cell_ids),
                })

    return registry, manifest
