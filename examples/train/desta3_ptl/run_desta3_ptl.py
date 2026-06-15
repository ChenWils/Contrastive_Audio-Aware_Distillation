import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig, OmegaConf
from desta.collections.ptl_modules.desta3_ptl import DeSTA3PTLModule
# from desta.collections.ptl_modules.desta3_distill_ptl import DistillationPTLModule
from desta.collections.ptl_modules.wilz_CoD_distill import CoD_DistillPTLModule
from desta.collections.ptl_modules.wilz_contrastive_distill import ContrastiveDistillPTLModule
from desta.collections.ptl_modules.contrastive_GKD_distill import ContrastiveGKDPTLModule
from pytorch_lightning.loggers import WandbLogger
import logging
from lulutils import get_unique_filepath
from pytorch_lightning.callbacks import RichModelSummary, ModelCheckpoint
import os
from pytorch_lightning.callbacks import LearningRateMonitor
from desta.collections.utils.callbacks import PushToHubCallback
import torch
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.model_summary import ModelSummary


class CustomStrategy(DDPStrategy):
    def load_model_state_dict(self, checkpoint, strict=False):
        if checkpoint is None:
            return
        self.lightning_module.load_state_dict(checkpoint["state_dict"], strict=False)



@hydra.main(config_path="conf", config_name="251113_wilz_contrastive_distill")
def main(cfg: DictConfig):
    pl.seed_everything(42)
    os.makedirs(cfg.exp_dir, exist_ok=True)
    cfg.resume_from_checkpoint = cfg.resume_from_checkpoint if cfg.resume_from_checkpoint != "null" else None
    cfg.init_from_pretrained_weights = cfg.init_from_pretrained_weights if cfg.init_from_pretrained_weights != "null" else None
    assert cfg.resume_from_checkpoint is None or cfg.init_from_pretrained_weights is None, "Cannot provide both resume_from_checkpoint and init_from_pretrained_weights"

    root_logger = logging.getLogger()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    for handler in root_logger.handlers:
        handler.setFormatter(formatter)
    logging.info(f"Config: {cfg}")

    working_dir = os.getcwd()

    # wandb logger
    if hasattr(cfg, "wandb") and cfg.project != "debug":
        wandb_logger = WandbLogger(
            project=cfg.project,
            save_dir=working_dir,
            log_model=False,
            name=cfg.name,
        )
    else:
        wandb_logger = None
    
    if cfg.ptl_module == "distill":
        ptl_module = DistillationPTLModule(cfg)
    elif cfg.ptl_module == "wilz_distill":
        ptl_module = CoD_DistillPTLModule(cfg)
    elif cfg.ptl_module == "wilz_contrastive_distill":
        ptl_module = ContrastiveDistillPTLModule(cfg)
    elif cfg.ptl_module == "contrastive_GKD_distill":
        ptl_module = ContrastiveGKDPTLModule(cfg)
    else:
        ptl_module = DeSTA3PTLModule(cfg)
    
    if cfg.init_from_pretrained_weights is not None:
        logging.info(
            ptl_module.load_state_dict(torch.load(cfg.init_from_pretrained_weights)["state_dict"], strict=False)
        )

    logging.info(f"PTL Module:\n{ptl_module}")

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/',
        filename='{epoch}-{step}',
        save_top_k=-1,
        every_n_epochs=1,
        verbose=True,
    )

    trainer = pl.Trainer(
        logger=wandb_logger,
        default_root_dir=working_dir,
        # strategy=CustomStrategy(),
        callbacks=[
            checkpoint_callback,
            RichModelSummary(max_depth=4),
            LearningRateMonitor(logging_interval="step"),
            PushToHubCallback(cfg, repo_id="kehanlu/my-cool-model"),
        ],
        fast_dev_run=(cfg.project == "debug"),
        **cfg.trainer
    )
    
    summary = ModelSummary(ptl_module, max_depth=5)
    logging.info(summary)

    OmegaConf.save(ptl_module.cfg, (f"{cfg.exp_dir}/config.yaml"))

    trainer.fit(
        ptl_module, 
        ckpt_path=cfg.resume_from_checkpoint
    )


if __name__ == "__main__":
    main()