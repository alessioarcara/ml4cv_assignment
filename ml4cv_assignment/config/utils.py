from typing import Any, Dict, List

from loguru import logger
from pydantic import FilePath, validate_call

from ml4cv_assignment.utils.io import read_yaml


# Taken from repository:
# https://github.com/alessioarcara/SoccerAI/blob/main/soccerai/training/trainer_config.py
def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge dict `b` into dict `a`
    """
    result = a.copy()
    for k, v in b.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


@validate_call
def get_experiment_config(config_paths: List[FilePath]) -> Dict[str, Any]:
    """
    Loads and merges multiple YAML configuration files into a single configuration dictionary.
    Later files in the list override keys from earlier ones.
    """

    merged_config: Dict[str, Any] = {}

    logger.info(f"📄 Building config from {len(config_paths)} files:")

    for path in config_paths:
        logger.info(f"    -> Loading: {path.name}")
        config = read_yaml(path)
        merged_config = _deep_merge(merged_config, config)

    return merged_config
