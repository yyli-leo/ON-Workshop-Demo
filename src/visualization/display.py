from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

from src.data.labels import load_labels_for_region, get_roi_true_label
from src.evaluation.evaluator import SpatialOmicsEvaluator
from src.evaluation.display import evaluate_all


def parse_roi_coordinates(patch_name: str) -> tuple:
    """Parse ROI bounding box coordinates from patch_name."""
    parts = patch_name.split('-')
    if len(parts) == 4:
        x0, x1, y0, y1 = map(int, parts)
        return (x0, x1, y0, y1)
    return None


def load_rendered_image_for_vis(region_id: str, data_root: str, scale_factor=0.25):
    """Load and downscale rendered image for a region."""
    rendered_path = Path(data_root) / region_id / f"{region_id}_rendered.png"

    if not rendered_path.exists():
        return None

    img = Image.open(rendered_path)
    new_width = int(img.width * scale_factor)
    new_height = int(img.height * scale_factor)
    img_scaled = img.resize((new_width, new_height), resample=Image.Resampling.BILINEAR)

    return img_scaled


def add_bounding_boxes_to_image(image, selected_rois, scale_factor=0.25):
    """Add red bounding boxes with number labels above the boxes."""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)

    try:
        font_large = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
    except:
        try:
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except:
            font_large = ImageFont.load_default()

    for idx, (label, roi) in enumerate(selected_rois, 1):
        coords = parse_roi_coordinates(roi["patch_name"])
        if coords:
            x0, x1, y0, y1 = coords

            x0_scaled = int(x0 * scale_factor)
            x1_scaled = int(x1 * scale_factor)
            y0_scaled = int(y0 * scale_factor)
            y1_scaled = int(y1 * scale_factor)

            draw.rectangle([x0_scaled, y0_scaled, x1_scaled, y1_scaled],
                          outline='red', width=4)

            text = str(idx)
            bbox = draw.textbbox((0, 0), text, font=font_large)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            label_x = (x0_scaled + x1_scaled) // 2 - text_width // 2
            label_y = y0_scaled - text_height - 8

            padding = 5
            draw.rectangle([label_x - padding, label_y - padding,
                          label_x + text_width + padding, label_y + text_height + padding],
                         fill='red', outline='red')

            draw.text((label_x, label_y), text, fill='white', font=font_large)

    return img_copy


def display_roi_reports(selected_rois, interp_results):
    """Display OmicsInterpreter reports for selected ROIs."""
    from IPython.display import HTML, display

    report_html = "<div style='margin-top: 20px;'>"

    for idx, (label, roi) in enumerate(selected_rois, 1):
        key = roi['key']
        report = interp_results.get(key, "No report available")

        report_escaped = report.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

        report_html += f"""
        <div style='margin-bottom: 25px; padding: 15px; border-left: 4px solid red; background-color: #f9f9f9;'>
            <div style='display: flex; align-items: center; margin-bottom: 10px;'>
                <span style='background-color: red; color: white; width: 30px; height: 30px;
                             border-radius: 50%; display: inline-flex; align-items: center;
                             justify-content: center; font-weight: bold; margin-right: 10px;'>{idx}</span>
                <div>
                    <strong style='font-size: 14px;'>ROI {idx}: {roi['patch_name']}</strong><br>
                    <span style='color: #666; font-size: 12px;'>Label: {label} | Cells: {roi['num_cells']}</span>
                </div>
            </div>
            <div style='font-size: 11px; line-height: 1.6; color: #333; white-space: pre-wrap;'>"""

        report_html += f"""<strong>OmicsInterpreter Analysis:</strong>\n{report_escaped}\n</div>\n        </div>\n        """

    report_html += "</div>"
    display(HTML(report_html))


def show_visualization(manifest: list, interp_results: dict, config: dict,
                       classes: list):
    """Display visualization with ROI reports for each region."""
    SCALE_FACTOR = 0.25
    regions = sorted(list(set([r["region_id"] for r in manifest])))

    for region_id in regions:
        print("=" * 80)
        print(f"REGION: {region_id}")
        print("=" * 80)

        # Select ROIs by label
        region_rois = [r for r in manifest if r["region_id"] == region_id]

        roi_labels = {}
        for roi in region_rois:
            labels_dict = load_labels_for_region(region_id, config["data_root"])
            true_label = get_roi_true_label(roi["cell_ids"], labels_dict)
            roi_labels[roi["key"]] = true_label

        label_rois = defaultdict(list)
        for roi in region_rois:
            label = roi_labels.get(roi["key"])
            if label and label in classes:
                label_rois[label].append(roi)

        selected = []
        for label in classes:
            if label in label_rois and label_rois[label]:
                sorted_rois = sorted(label_rois[label], key=lambda r: r.get("num_cells", 0), reverse=True)
                selected.append((label, sorted_rois[0]))

        if not selected:
            print(f"No ROIs found for {region_id}")
            continue

        print(f"Selected {len(selected)} ROIs (one per tissue type)")

        # Load rendered image
        rendered_img = load_rendered_image_for_vis(region_id, config["data_root"], scale_factor=SCALE_FACTOR)

        if rendered_img is None:
            print(f"Could not load rendered image for {region_id}")
            continue

        img_with_boxes = add_bounding_boxes_to_image(rendered_img, selected, SCALE_FACTOR)

        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        ax.imshow(img_with_boxes)
        ax.set_title(f'{region_id}', fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        plt.show()

        display_roi_reports(selected, interp_results)
        print("\n" * 3)

    print("✅ Visualization complete!")


def show_results(manifest: list, visual_results: dict, omics_results: dict,
                 interp_results: dict, config: dict,
                 evaluator: SpatialOmicsEvaluator, classes: list):
    """
    Show evaluation metrics and visualization.

    This is the main entry point for displaying results after pipeline execution.
    """
    # Evaluate all stages
    visual_metrics, omics_metrics, interp_metrics = evaluate_all(
        visual_results, omics_results, interp_results, manifest,
        config, evaluator, classes
    )

    # Show visualization
    show_visualization(manifest, interp_results, config, classes)

    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETE")
    print("=" * 60)
