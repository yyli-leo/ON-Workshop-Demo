"""
ROI Interpretation Pipeline - Facade Module

Thin wrapper that manages global state and delegates to src/ sub-modules.
The public API used by demo.ipynb is preserved exactly.
"""

import os
from pathlib import Path

from src.config.loader import load_config, get_config, get_classes, ExperimentManager
from src.models.wrappers import get_model_wrapper
from src.evaluation.evaluator import SpatialOmicsEvaluator
from src.data.loading import load_simulation_data, load_registry_data
from src.pipeline.interpreter import run_pipeline as _run_pipeline
from src.visualization.display import show_results as _show_results

# ---- Global state (managed by this facade only) ----
_evaluator = None
_manager = None
_prompts = {}
_registry = None


def _init_evaluator(config: dict):
    """Initialize evaluator from config."""
    global _evaluator
    if _evaluator is not None:
        return
    _evaluator = SpatialOmicsEvaluator(classes=config.get('classes', []))


def _init_modules_for_real_mode(config: dict):
    """Initialize manager and prompts for real mode."""
    global _manager, _prompts
    workshop_root = Path(config['workshop_root'])
    _manager = ExperimentManager(workshop_root)
    _prompts = {
        'visual_system': _manager.load_prompt(config['visual_system_tpl']),
        'visual_user': _manager.load_prompt(config['visual_prompt_tpl']),
        'omics_system': _manager.load_prompt(config['omics_system_tpl']),
        'interpreter_system': _manager.load_prompt(config['interpreter_system_tpl']),
    }


def load_data(config: dict, simulate: bool = True) -> tuple:
    """
    Load data based on mode.

    Args:
        config: Configuration dictionary from load_config()
        simulate: If True, load cached SOTA results. If False, load registry for real execution.

    Returns:
        Tuple of (manifest, visual_results, omics_results, interp_results)
    """
    _init_evaluator(config)

    if simulate:
        visual, omics, interp, manifest = load_simulation_data(config['sota_demo_dir'])
        print(f"Loaded {len(manifest)} ROIs from simulation cache")
        return manifest, visual, omics, interp
    else:
        global _registry
        _registry, manifest = load_registry_data(config['registry_path'])
        print(f"Loaded {len(manifest)} ROIs from registry")
        return manifest, {}, {}, {}


def initialize_model(model_name: str, api_key: str, config: dict):
    """
    Initialize model for real mode execution.

    Args:
        model_name: Name of the LLM model to use
        api_key: API key for the model provider
        config: Configuration dictionary

    Returns:
        Initialized model wrapper instance
    """
    _init_modules_for_real_mode(config)
    config['model_name'] = model_name
    config['api_key'] = api_key
    if model_name.lower().startswith('gemini'):
        os.environ['GOOGLE_API_KEY'] = api_key
    return get_model_wrapper(model_name)


def run_pipeline(manifest: list, model, config: dict) -> tuple:
    """
    Run the complete three-stage pipeline.

    Args:
        manifest: List of ROI dictionaries
        model: Initialized model wrapper
        config: Configuration dictionary

    Returns:
        Tuple of (visual_results, omics_results, interp_results)
    """
    return _run_pipeline(manifest, model, config,
                         _registry, _manager, _prompts)


def show_results(manifest: list, visual_results: dict, omics_results: dict,
                 interp_results: dict, config: dict):
    """
    Show evaluation metrics and visualization.

    This is the main entry point for displaying results after pipeline execution.

    Args:
        manifest: List of ROI dictionaries
        visual_results: Results from VisualProfiler stage
        omics_results: Results from OmicsProfiler stage
        interp_results: Results from OmicsInterpreter stage
        config: Configuration dictionary
    """
    _show_results(manifest, visual_results, omics_results, interp_results,
                  config, _evaluator, config.get('classes', []))
