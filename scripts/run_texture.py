import pyrallis
import wandb
import dataclasses

import gc
import torch

from src.configs.train_config import TrainConfig
from src.training.trainer import TEXTure


@pyrallis.wrap()
def main(cfg: TrainConfig):
    # run_experiments(cfg)
    # run_exp1(cfg)
    one_run(cfg)


def run_exp1(cfg: TrainConfig):
    for train_grid_size, eval_grid_size in [(700, 512), (1200, 1024), (2200, 2048)]:
        cfg.render.train_grid_size = train_grid_size
        cfg.render.eval_grid_size = eval_grid_size
        one_run(cfg)


def run_experiments(cfg: TrainConfig):
    for guidance_scale in [4.0, 7.5, 10.0, 15.0]:
        for lr in [1e-1, 1e-2, 1e-4]:
            for z_update_thr in [0.1, 0.2, 0.4, 0.8]:
                cfg.guide.guidance_scale = guidance_scale
                cfg.optim.lr = lr
                cfg.guide.z_update_thr = z_update_thr
                one_run(cfg)


def one_run(cfg: TrainConfig):
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
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == '__main__':
    main()
