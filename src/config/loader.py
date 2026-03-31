import os
import sys
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

try:
    from jinja2 import Environment, FileSystemLoader
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

# Module-level state
_config = None
_classes = None


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config.yaml file

    Returns:
        Configuration dictionary
    """
    global _config, _classes

    # Detect workshop root from config file location
    config_file = Path(config_path).resolve()
    workshop_root = config_file.parent
    parent_root = workshop_root.parent

    # Add parent root to path for data models
    if str(parent_root) not in sys.path:
        sys.path.insert(0, str(parent_root))

    # Load YAML config
    import yaml
    with open(config_path, 'r') as f:
        _config = yaml.safe_load(f)

    # Add auto-detected paths
    _config['workshop_root'] = str(workshop_root)
    _config['parent_root'] = str(parent_root)

    # Convert relative paths to absolute
    for key in ['data_root', 'registry_path', 'sota_demo_dir']:
        if key in _config and _config[key]:
            p = Path(_config[key])
            if not p.is_absolute():
                _config[key] = str(workshop_root / _config[key])

    # Store classes globally
    _classes = _config.get('classes', [])

    # Add run_id if not present
    if 'run_id' not in _config:
        _config['run_id'] = datetime.now().strftime("roi_pipeline_%Y%m%d_%H%M")

    # Load environment variables
    env_file = workshop_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    if os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.environ.get("GEMINI_API_KEY")

    return _config


def get_config() -> dict:
    """Get the current configuration."""
    return _config


def get_classes() -> list:
    """Get the current classes list."""
    return _classes


class ExperimentManager:
    """Manage experiment prompts and results."""

    def __init__(self, project_root: Path):
        self.root = project_root
        if JINJA2_AVAILABLE:
            self.prompt_env = Environment(loader=FileSystemLoader(self.root / "prompts"))
        else:
            self.prompt_env = None

    def load_prompt(self, template_name: str, **kwargs) -> str:
        """Load and render Jinja2 template."""
        if self.prompt_env is None:
            raise ImportError("jinja2 is required for prompt templates. Install with: pip install jinja2")
        template = self.prompt_env.get_template(template_name)
        return template.render(**kwargs)
