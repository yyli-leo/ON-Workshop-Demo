# ROI Interpretation Pipeline - Workshop Demo

A user-friendly demonstration of a three-stage AI pipeline for analyzing kidney tissue regions using spatial omics data.

## Quick Start

1. **Download the dataset** from [Google Drive](https://drive.google.com/file/d/1w7QZkybXoBLn_dJ-Vw1u-eP4oriXf6FL/view?usp=drive_link)
   - Extract to `data/DKD/` directory
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   # Or install core dependencies only:
   # pip install pyyaml python-dotenv pillow pandas matplotlib tqdm jinja2
   ```
3. **Open the notebook**: `demo.ipynb`
4. **Set your configuration** in Step 1:
   - Add your API key (for real mode)
   - Choose simulation mode for fast demo
5. **Run all cells**: `Runtime` → `Run All`
6. **View results**: Evaluation metrics and tissue visualizations

## What This Pipeline Does

The pipeline analyzes kidney tissue regions through three AI stages:

| Stage | Input | Output |
|-------|-------|--------|
| **VisualProfiler** | Microscopy images | Visual tissue structure identification |
| **OmicsProfiler** | Biomarker/cell type data | Molecular tissue structure identification |
| **OmicsInterpreter** | Both above sources | Final integrated interpretation |

## Configuration Options

### Simulation Mode (Recommended for First Run)
```python
SIMULATE = True
```
- Uses cached results from previous runs
- No API calls required
- Fast execution (~1 second)
- 50 ROIs from SOTA dataset

### Real Mode (Full Pipeline)
```python
SIMULATE = False
API_KEY = "your-api-key-here"
```
- Runs complete LLM pipeline
- Requires valid API key
- Execution time varies (5-30 minutes depending on ROIs)
- Supports any model: `gemini-2.5-flash-lite`, `gemini-2.5-pro`, `claude-sonnet-4-5`

## File Structure

```
OmicsNavigator_Demo_0401/
├── demo.ipynb              # User-facing notebook (start here!)
├── helper.py               # Thin facade preserving public API
├── config.yaml             # Pipeline configuration
├── README.md               # This file
├── CLAUDE.md               # Developer documentation for Claude Code
├── src/                    # Modular implementation
│   ├── __init__.py         # Package re-exports
│   ├── data_models.py      # Backward-compatibility stub
│   ├── config/
│   │   └── loader.py       # Configuration loading, ExperimentManager
│   ├── models/
│   │   └── wrappers.py     # LLM model wrappers (Gemini, Claude, OpenAI, Grok)
│   ├── evaluation/
│   │   ├── evaluator.py    # SpatialOmicsEvaluator class
│   │   └── display.py      # Evaluation metrics and display
│   ├── data/
│   │   ├── loading.py      # Registry and simulation data loading
│   │   ├── labels.py       # Ground truth label loading
│   │   ├── descriptions.py # ROI description generation
│   │   └── data_models.py  # Data model definitions
│   ├── processing/
│   │   ├── image.py        # Image processing utilities
│   │   └── llm.py          # LLM calling with retry logic
│   ├── pipeline/
│   │   ├── visual.py       # VisualProfiler stage
│   │   ├── omics.py        # OmicsProfiler stage
│   │   └── interpreter.py  # OmicsInterpreter stage + pipeline orchestrator
│   └── visualization/
│       └── display.py      # Visualization and results display
├── prompts/                # Jinja2 prompt templates
│   └── s255/               # Stage-specific templates
├── data/                   # Spatial omics data (download from Google Drive)
│   └── DKD/
│       ├── s255_c*/        # Sample regions (CSVs + rendered images)
│       ├── s255_pivot_ROIs_registry.pkl
│       ├── s255_pivot_ROIs_registry_labels.pkl
│       └── cached_results/  # SOTA demo results (simulation mode)
└── outputs/                # Pipeline run outputs
```

## Understanding the Output

### Evaluation Metrics

- **Accuracy**: Percentage of correct predictions
- **Precision**: Of all predictions for a class, how many were correct
- **Recall**: Of all actual instances of a class, how many were found
- **F1 Score**: Harmonic mean of precision and recall

### Tissue Classes

The pipeline classifies ROIs into 5 tissue types:
- Proximal tubules
- Distal tubules
- Glomeruli
- Blood vessel
- Interstitium

### Visualization

- **Red boxes**: ROIs analyzed by the pipeline
- **Numbers**: Reference to detailed reports below each image
- **Scale**: Images shown at 25% resolution to fit display

## Troubleshooting

### "Module Not Found" Errors

Ensure you have required dependencies:
```bash
pip install -r requirements.txt
```

### "API Key Not Set" Warning

Set your API key in Step 1 of the notebook:
```python
API_KEY = "your-actual-api-key"
```

### Data Not Found

Download the dataset from [Google Drive](https://drive.google.com/file/d/1w7QZkybXoBLn_dJ-Vw1u-eP4oriXf6FL/view?usp=drive_link) and extract it to the `data/DKD/` directory.

### Simulation Mode Shows Old Results

This is expected! Simulation mode uses cached results from `data/DKD/cached_results/`.

### Slow Execution in Real Mode

- Reduce `max_workers` in `config.yaml` (try 1, 2 or 5)
- Use faster models like `gemini-3.1-flash-lite-preview`
- Reduce dataset size by using a different registry

## Advanced Usage

### Modifying Configuration

Edit `config.yaml` to change:
- Data paths
- Number of parallel workers
- Image scale factor
- Prompt templates
- Model selection

### Project Architecture

The codebase is organized into modular sub-packages under `src/`:
- **Stateless pure functions** in sub-modules receive all data via explicit parameters
- **Global state** (`_config`, `_evaluator`, `_manager`, `_prompts`, `_registry`) is managed only by `helper.py` facade
- **Thread-safe** image processing using `ThreadPoolExecutor` with thread-local PIL Image copies

### Running with Different Data

1. Prepare your registry pickle file with `SpatialPivotRegistry` structure
2. Update `registry_path` in `config.yaml`
3. Ensure rendered images exist for each region (`{region_id}_rendered.png`)

## Technical Details

### Data Format

- **Registry**: Pickle file with `SpatialPivotRegistry` structure containing:
  - `RegionShardData`: Per-region ROI data with features and clustering
  - `RegionOfInterest`: ROI bounding boxes and cell IDs
  - `ClusterIdentity`: Two-stage clustering assignments
- **Images**: Pre-rendered PNG files per region (7-channel multiplexed immunofluorescence)
- **Labels**: CSV files with CELL_ID → LABEL mappings for evaluation

### Prompt Templates

Jinja2 templates in `prompts/s255/`:
- System instructions: Define AI role and behavior
- User prompts: Structure the analysis request with format requirements

### Data Models

Located in `src/data/data_models.py`:
- `RegionOfInterest`: Bounding box + enclosed cells
- `ClusterIdentity`: Two-stage clustering assignment
- `RegionShardData`: Multi-modal features + clustering metadata
- `SpatialPivotRegistry`: Top-level registry for all regions