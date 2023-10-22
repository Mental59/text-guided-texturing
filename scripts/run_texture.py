import pyrallis
import wandb
import dataclasses

from src.configs.train_config import TrainConfig
from src.training.trainer import TEXTure


@pyrallis.wrap()
def main(cfg: TrainConfig):
    wandb.init(
        project='text-guided-texturing',
        config=dataclasses.asdict(cfg),
        dir='./artifacts'
    )

    try:
        trainer = TEXTure(cfg)
        if cfg.log.eval_only:
            trainer.full_eval()
        else:
            trainer.paint()
    finally:
        wandb.finish()


if __name__ == '__main__':
    main()
