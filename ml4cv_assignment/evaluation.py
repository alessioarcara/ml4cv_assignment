from typing import Dict, Literal

from pydantic import validate_call

from ml4cv_assignment.config.config import Config
from ml4cv_assignment.training.trainer import Trainer
from ml4cv_assignment.utils.misc import fix_random
from ml4cv_assignment.utils.typings import Stage


@validate_call
def evaluate_split(
    cfg: Config,
    split: Literal["val", "test"] = "test",
) -> Dict[str, float]:
    """
    Helper function to run the model evaluation on the specified dataset split.

    Args:
        config: Configuration object containing all settings.
        split: Dataset split to evaluate on ('val' or 'test').

    Returns:
        A dictionary containing the computed evaluation metrics.
    """
    fix_random(cfg.seed)

    trainer = Trainer(
        config=cfg.training,
        model=cfg.model,
        val_loader=cfg.val_dataloader,
        test_loader=cfg.test_dataloader,
        experiment_raw_config={},
    )

    trainer._on_eval_start()
    results = trainer.eval(stage=Stage.TEST if split == "test" else Stage.VAL)

    return results
