import pickle
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from src.data.descriptions import generate_roi_descriptions
from src.processing.llm import call_llm_with_retry


def run_omics_profiler(manifest: list, model, config: dict,
                       manager, prompts: dict) -> dict:
    """Run OmicsProfiler stage."""
    print("=" * 60)
    print("STAGE 2: OmicsProfiler")
    print("=" * 60)

    results = {}

    # Generate descriptions
    print("\nGenerating ROI descriptions...")
    region_targets = {}
    for r in manifest:
        rid = r["region_id"]
        region_targets.setdefault(rid, []).append({
            "cell_ids": r["cell_ids"],
            "patch_name": r["patch_name"],
        })

    all_descriptions = {}
    for rid, targets in region_targets.items():
        descs = generate_roi_descriptions(
            region_id=rid, root_dir=config["data_root"],
            target_rois=targets, include_biomarker=True,
            include_cell_type=True, include_morphology=False, verbose=False,
        )
        all_descriptions.update(descs)

    print(f"Generated {len(all_descriptions)} descriptions")

    # LLM analysis
    def _run_omics(roi):
        key = roi["key"]
        desc = all_descriptions.get(key, "")
        if not desc:
            return key, "ERROR_EMPTY_DESC"

        prompt_text = manager.load_prompt(config["omics_prompt_tpl"], user_input=desc)
        return key, call_llm_with_retry(model, prompt_text, prompts['omics_system'], config)

    with ThreadPoolExecutor(max_workers=config["max_workers"]) as exe:
        futs = {exe.submit(_run_omics, r): r["key"] for r in manifest}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Omics"):
            key, result = fut.result()
            results[key] = result

    n_ok = sum(1 for v in results.values() if not v.startswith("ERROR"))
    print(f"\nOmicsProfiler: {n_ok} OK")

    # Save results
    output_dir = Path(config['workshop_root']) / "outputs" / config["run_id"]
    with open(output_dir / "omics_reports.pkl", 'wb') as f:
        pickle.dump(results, f)
    print(f"  💾 Saved: {output_dir / 'omics_reports.pkl'}")

    return results
