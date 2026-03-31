import io
import pickle
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from PIL import Image

from src.processing.image import (
    load_rendered_image_bytes,
    prepare_images_for_llm,
    cleanup_temp_images,
)
from src.processing.llm import call_llm_with_retry


def run_visual_profiler(manifest: list, model, config: dict,
                        registry, prompts: dict) -> dict:
    """Run VisualProfiler stage."""
    print("=" * 60)
    print("STAGE 1: VisualProfiler")
    print("=" * 60)

    results = {}

    # Pre-load rendered image bytes (thread-safe)
    rendered_image_bytes = {}
    data_root = Path(config['data_root'])

    for region_id in registry.shards.keys():
        try:
            rendered_image_bytes[region_id] = load_rendered_image_bytes(region_id, data_root)
            print(f"  Loaded rendered image bytes: {region_id}")
        except FileNotFoundError as e:
            print(f"  ❌ {e}")
            rendered_image_bytes[region_id] = None

    def _run_visual(roi):
        """Process single ROI through VisualProfiler."""
        key = roi["key"]
        region_id = roi["region_id"]

        if region_id not in rendered_image_bytes or rendered_image_bytes[region_id] is None:
            return key, "SKIPPED_NO_RENDERED_IMAGE"

        img_data = None
        try:
            # Create fresh Image from bytes in each thread
            rendered_image = Image.open(io.BytesIO(rendered_image_bytes[region_id]))

            img_data = prepare_images_for_llm(
                rendered_image,
                roi["roi_obj"],
                scale_factor=config["full_image_scale_factor"]
            )

            payload = [
                {"type": "image", "path": img_data['full_path']},
                {"type": "image", "path": img_data['small_path']},
                {"type": "text", "text": prompts['visual_user']},
            ]

            result = call_llm_with_retry(model, payload, prompts['visual_system'], config)
            return key, result

        except Exception as e:
            return key, f"ERROR: {e}"

        finally:
            if img_data:
                cleanup_temp_images(img_data)

    with ThreadPoolExecutor(max_workers=config["max_workers"]) as exe:
        futures = {exe.submit(_run_visual, r): r for r in manifest}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Visual"):
            key, result = fut.result()
            results[key] = result

    n_ok = sum(1 for v in results.values() if not v.startswith("ERROR") and not v.startswith("SKIPPED"))
    print(f"\nVisualProfiler: {n_ok} OK")

    # Save results
    output_dir = Path(config['workshop_root']) / "outputs" / config["run_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "visual_reports.pkl", 'wb') as f:
        pickle.dump(results, f)
    print(f"  💾 Saved: {output_dir / 'visual_reports.pkl'}")

    return results
