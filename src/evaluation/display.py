from collections import Counter, defaultdict

from src.data.labels import load_labels_for_region, get_roi_true_label
from src.evaluation.evaluator import SpatialOmicsEvaluator


def evaluate_stage(results_dict: dict, stage_name: str, manifest: list,
                   config: dict, evaluator: SpatialOmicsEvaluator,
                   classes: list) -> dict:
    """Evaluate model results against true labels (Top-1 only)."""
    print("\n" + "=" * 60)
    print(f"EVALUATION: {stage_name}")
    print("=" * 60)

    # Group ROIs by region
    regions = defaultdict(list)
    for roi in manifest:
        regions[roi["region_id"]].append(roi)

    y_true = []
    y_pred = []

    # Process each region
    for region_id, region_rois in regions.items():
        labels_dict = load_labels_for_region(region_id, config["data_root"])
        for roi in region_rois:
            key = roi["key"]
            true_label = get_roi_true_label(roi["cell_ids"], labels_dict)

            # Skip labels not in CLASSES
            if true_label not in classes:
                continue

            # Get prediction
            report = results_dict.get(key, "")

            if report and report != "SKIPPED_NO_IMAGES" and not report.startswith("ERROR"):
                pred_structures = evaluator.parse_structures(report)
                top1_pred = pred_structures[0] if pred_structures else "No prediction"
            else:
                top1_pred = "No prediction"

            y_true.append(true_label)
            y_pred.append(top1_pred)

    # Compute metrics using evaluator
    y_pred_formatted = [[p] if p != "No prediction" else [] for p in y_pred]
    metrics = evaluator.compute_metrics(y_true, y_pred_formatted)

    # Print summary
    print(f"\n📊 {stage_name} Results:")
    print(f"  Accuracy: {metrics['acc@1']:.4f}")
    print(f"  Precision (Macro): {metrics['top1']['macro']['precision']:.4f}")
    print(f"  Recall (Macro): {metrics['top1']['macro']['recall']:.4f}")
    print(f"  F1 Score (Macro): {metrics['top1']['macro']['f1']:.4f}")

    # Per-class metrics
    print(f"\n📈 Per-Class Metrics (Top-1):")
    print(f"{'Class':<20} | Precision | Recall   | F1 Score | Support")
    print("-" * 70)

    true_counts = Counter(y_true)

    for cls in classes:
        if cls in metrics['top1']['per_class']:
            p = metrics['top1']['per_class'][cls]['precision']
            r = metrics['top1']['per_class'][cls]['recall']
            f1 = metrics['top1']['per_class'][cls]['f1']
            support = true_counts.get(cls, 0)
            print(f"  {cls:<20} | {p:.3f}     | {r:.3f}    | {f1:.3f}    | {support}")
        else:
            print(f"  {cls:<20} | N/A       | N/A      | N/A      | 0")

    return metrics


def evaluate_all(visual_results: dict, omics_results: dict, interp_results: dict,
                 manifest: list, config: dict, evaluator: SpatialOmicsEvaluator,
                 classes: list) -> tuple:
    """Evaluate all three stages and print comprehensive analysis."""
    print("\n" + "=" * 60)
    print("EVALUATING THREE-STAGE PIPELINE")
    print("=" * 60)

    visual_metrics = evaluate_stage(visual_results, "VisualProfiler", manifest,
                                    config, evaluator, classes)
    omics_metrics = evaluate_stage(omics_results, "OmicsProfiler", manifest,
                                   config, evaluator, classes)
    interp_metrics = evaluate_stage(interp_results, "OmicsInterpreter", manifest,
                                    config, evaluator, classes)

    print_comprehensive_analysis(visual_metrics, omics_metrics, interp_metrics, classes)

    return visual_metrics, omics_metrics, interp_metrics


def print_comprehensive_analysis(visual_metrics: dict, omics_metrics: dict,
                                 interp_metrics: dict, classes: list):
    """Print comprehensive analysis of all 3 agents."""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE ANALYSIS")
    print("=" * 60)

    # Individual Agent Performance
    print("\n📊 Individual Agent Performance:")
    print(f"{'Agent':<20} | Accuracy | Precision | Recall   | F1 (Macro)")
    print("-" * 70)
    print(f"{'VisualProfiler':<20} | {visual_metrics['acc@1']:.3f}    | {visual_metrics['top1']['macro']['precision']:.3f}      | {visual_metrics['top1']['macro']['recall']:.3f}      | {visual_metrics['top1']['macro']['f1']:.3f}")
    print(f"{'OmicsProfiler':<20} | {omics_metrics['acc@1']:.3f}    | {omics_metrics['top1']['macro']['precision']:.3f}      | {omics_metrics['top1']['macro']['recall']:.3f}      | {omics_metrics['top1']['macro']['f1']:.3f}")
    print(f"{'OmicsInterpreter':<20} | {interp_metrics['acc@1']:.3f}    | {interp_metrics['top1']['macro']['precision']:.3f}      | {interp_metrics['top1']['macro']['recall']:.3f}      | {interp_metrics['top1']['macro']['f1']:.3f}")

    # Improvement Analysis
    print("\n📈 OmicsInterpreter Improvement:")
    vs_vis_acc = interp_metrics['acc@1'] - visual_metrics['acc@1']
    vs_vis_f1 = interp_metrics['top1']['macro']['f1'] - visual_metrics['top1']['macro']['f1']
    print(f"  vs VisualProfiler:")
    print(f"    ΔAccuracy: {vs_vis_acc:+.3f} ({vs_vis_acc*100:+.1f}%)")
    print(f"    ΔF1 Score: {vs_vis_f1:+.3f} ({vs_vis_f1*100:+.1f}%)")

    vs_omics_acc = interp_metrics['acc@1'] - omics_metrics['acc@1']
    vs_omics_f1 = interp_metrics['top1']['macro']['f1'] - omics_metrics['top1']['macro']['f1']
    print(f"  vs OmicsProfiler:")
    print(f"    ΔAccuracy: {vs_omics_acc:+.3f} ({vs_omics_acc*100:+.1f}%)")
    print(f"    ΔF1 Score: {vs_omics_f1:+.3f} ({vs_omics_f1*100:+.1f}%)")

    # Per-Class Breakdown
    print("\n📉 Per-Class F1 Scores:")
    print(f"{'Class':<20} | VisualProfiler | OmicsProfiler | OmicsInterpreter")
    print("-" * 70)
    for cls in classes:
        v_f1 = visual_metrics['top1']['per_class'].get(cls, {}).get('f1', 0)
        o_f1 = omics_metrics['top1']['per_class'].get(cls, {}).get('f1', 0)
        i_f1 = interp_metrics['top1']['per_class'].get(cls, {}).get('f1', 0)
        print(f"  {cls:<20} | {v_f1:.3f}         | {o_f1:.3f}         | {i_f1:.3f}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    best_acc = max([
        ("VisualProfiler", visual_metrics['acc@1']),
        ("OmicsProfiler", omics_metrics['acc@1']),
        ("OmicsInterpreter", interp_metrics['acc@1']),
    ], key=lambda x: x[1])

    best_f1 = max([
        ("VisualProfiler", visual_metrics['top1']['macro']['f1']),
        ("OmicsProfiler", omics_metrics['top1']['macro']['f1']),
        ("OmicsInterpreter", interp_metrics['top1']['macro']['f1']),
    ], key=lambda x: x[1])

    print(f"\nBest Accuracy: {best_acc[0]} ({best_acc[1]:.3f})")
    print(f"Best F1 Score: {best_f1[0]} ({best_f1[1]:.3f})")

    if best_acc[0] == "OmicsInterpreter" and best_f1[0] == "OmicsInterpreter":
        print("\n✅ OmicsInterpreter achieves the best performance on both metrics!")
