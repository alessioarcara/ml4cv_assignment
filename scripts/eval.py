import argparse

from ml4cv_assignment.config import Config
from ml4cv_assignment.training.trainer import Trainer
from ml4cv_assignment.utils.misc import display_eval_results, fix_random
from ml4cv_assignment.utils.typings import Stage


def main(config_path: str):
    cfg = Config.load(config_path)
    fix_random(cfg.seed)

    trainer = Trainer(
        config=cfg.training,
        model=cfg.model,
        test_loader=cfg.test_dataloader,
    )

    display_eval_results(trainer.eval(Stage.TEST), title="Test Set Evaluation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate segmentation model on Street Hazards test set"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="./configs/base.yaml",
        help="Path to the YAML config file",
    )
    args = parser.parse_args()
    main(args.config_path)
