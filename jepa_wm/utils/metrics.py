"""
Experiment Metric Logging and Persistence Utilities.
"""

from typing import Any, Dict
import json
import os
import yaml


def save_metrics_json(metrics: Dict[str, Any], filepath: str) -> None:
    """Saves metrics dictionary to a JSON file."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def save_config_yaml(config: Dict[str, Any], filepath: str) -> None:
    """Saves experiment configuration to a YAML file."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)
