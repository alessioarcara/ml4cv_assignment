import argparse

# from loguru import logger
from ml4cv_assignment.config.config import Config
from ml4cv_assignment.models.mask2former import Mask2Former

# from ml4cv_assignment.models.model import build_model
from ml4cv_assignment.training.trainer import Trainer
from ml4cv_assignment.utils.misc import fix_random


def main(config_path: str) -> None:
    cfg = Config.load(config_path)
    fix_random(cfg.seed)
    train_loader = cfg.train_dataloader
    # img_h, img_w = next(iter(train_loader))[0].shape[-2:]

    # logger.info("img_h: {}, img_w: {}", img_h, img_w)

    run_name = "test_m2former"

    # model_config = {
    #    "model": {
    #        "encoder_name": "resnet18d",
    #        "d": 128,
    #        "stride": 16,
    #    },
    #    "training": {
    #        "img_height": img_h,
    #        "img_width": img_w,
    #    },
    # }

    model = Mask2Former()

    trainer = Trainer(
        config=cfg.training,
        model=model,
        run_name=run_name,
        train_loader=train_loader,
        val_loader=cfg.val_dataloader,
    )
    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train segmentation model on Street Hazards"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="./configs/base.yaml",
        help="Path to the YAML config file",
    )
    args = parser.parse_args()
    main(args.config_path)
