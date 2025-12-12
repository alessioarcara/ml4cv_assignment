from typing import Dict

from loguru import logger

from ml4cv_assignment.utils.typings import StateDict


def remap_state_dict_keys(
    state_dict: StateDict, remap_rules: Dict[str, str]
) -> StateDict:
    """
    Remaps keys in a state_dict according to replacement rules.
    """
    if not remap_rules:
        return state_dict

    logger.info(f"Remapping checkpoint keys with rules: {remap_rules}")
    keys = list(state_dict.keys())

    for k in keys:
        new_key = k

        for old, new in remap_rules.items():
            if old in new_key:
                new_key = new_key.replace(old, new)

        if new_key != k:
            state_dict[new_key] = state_dict.pop(k)
            logger.debug(f"Renamed: {k} -> {new_key}")

    return state_dict
