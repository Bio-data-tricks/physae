"""
I/O utilities for saving configurations and model outputs.
"""
import json
import yaml
from pathlib import Path


def save_config(config: dict, path: str):
    """
    Save configuration dictionary to file (JSON or YAML based on extension).

    Args:
        config: Configuration dictionary.
        path: Path to save the configuration.
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    if path.endswith('.json'):
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
    elif path.endswith(('.yaml', '.yml')):
        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    else:
        raise ValueError("Config file must be .json or .yaml/.yml")


def load_config(path: str) -> dict:
    """
    Load configuration dictionary from file.

    Args:
        path: Path to the configuration file.

    Returns:
        Configuration dictionary.
    """
    with open(path, 'r') as f:
        if path.endswith('.json'):
            return json.load(f)
        elif path.endswith(('.yaml', '.yml')):
            return yaml.safe_load(f)
        else:
            raise ValueError("Config file must be .json or .yaml/.yml")
