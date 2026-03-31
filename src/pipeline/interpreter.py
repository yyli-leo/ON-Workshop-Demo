import pickle
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from src.processing.llm import call_llm_with_retry
from src.pipeline.visual import run_visual_profiler
from src.pipeline.omics import run_omics_profiler


def run_interpreter(manifest: list, visual_results: dict, omics_results: dict,
                    model, config: dict, manager, prompts: dict) -> dict:
    """Run OmicsInterpreter stage."""
    print("=" * 60)
    print("STAGE 3: OmicsInterpreter")
    print("=" * 60)

    results = {}

    def _run_interp(roi):
        key = roi["key"]
        vis_rpt = visual_results.get(key, "SKIPPED_NO_IMAGES")
        omi_rpt = omics_results.get(key, "ERROR_NO_REPORT")

        vis_ok = not vis_rpt.startswith("ERROR") and not vis_rpt.startswith("SKIPPED")
        omi_ok = not omi_rpt.startswith("ERROR")
        if not vis_ok and not omi_ok:
            return key, "ERROR_NO_VALID_REPORTS"

        prompt_text = manager.load_prompt(
            config["interpreter_prompt_tpl"],
            visual_report=vis_rpt, omics_report=omi_rpt,
        )
        return key, call_llm_with_retry(model, prompt_text, prompts['interpreter_system'], config)

    with ThreadPoolExecutor(max_workers=config["max_workers"]) as exe:
        futs = {exe.submit(_run_interp, r): r["key"] for r in manifest}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Interpret"):
            key, result = fut.result()
            results[key] = result

    n_ok = sum(1 for v in results.values() if not v.startswith("ERROR"))
    print(f"\nOmicsInterpreter: {n_ok} OK")

    # Save results
    output_dir = Path(config['workshop_root']) / "outputs" / config["run_id"]
    with open(output_dir / "final_interpretations.pkl", 'wb') as f:
        pickle.dump(results, f)
    print(f"  💾 Saved: {output_dir / 'final_interpretations.pkl'}")

    return results


def run_pipeline(manifest: list, model, config: dict,
                 registry, manager, prompts: dict) -> tuple:
    """Run the complete three-stage pipeline."""
    visual_results = run_visual_profiler(manifest, model, config, registry, prompts)
    omics_results = run_omics_profiler(manifest, model, config, manager, prompts)
    interp_results = run_interpreter(manifest, visual_results, omics_results,
                                     model, config, manager, prompts)

    return visual_results, omics_results, interp_results
