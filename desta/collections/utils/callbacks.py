from pytorch_lightning.callbacks import Callback
from pytorch_lightning import Trainer, LightningModule
from huggingface_hub import HfApi
import os
import logging
class PushToHubCallback(Callback):
    def __init__(self, cfg, repo_id: str):
        self.cfg = cfg
        self.repo_id = repo_id
        self.api = HfApi(token=os.getenv("HF_TOKEN"))
        self.revision = "my_exps"

        self.exp_dir = cfg.exp_dir
        self.root_dir = os.getenv("ROOT_DIR")

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        pass
        # try:
        #     self.api.create_branch(
        #         repo_id=self.repo_id,
        #         repo_type="model",
        #         branch=self.revision,
        #         exist_ok=True
        #     )
        #     self.api.upload_folder(
        #         folder_path=self.exp_dir,
        #         repo_id=self.repo_id,
        #         repo_type="model",
        #         revision=self.revision,
        #         path_in_repo=self.exp_dir.replace(f"{self.root_dir}/my_exps", "").lstrip("/"),
        #         allow_patterns=["*.ckpt", "*.json", "*.yaml", "*.log", "*.safetensors", "*.jsonl"],
        #     )
        # except Exception as e:
        #     logging.error(f"Error uploading to Hugging Face: {e}")

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule):
        try:
            pl_module.model.save_pretrained(f"{self.exp_dir}/hf_models/epoch-{pl_module.current_epoch}")
            pl_module.model.config.save_pretrained(f"{self.exp_dir}/hf_models/epoch-{pl_module.current_epoch}")
            pl_module.tokenizer.save_pretrained(f"{self.exp_dir}/hf_models/epoch-{pl_module.current_epoch}")

            self.on_train_epoch_end(trainer, pl_module)
        except Exception as e:
            logging.error(f"Error saving to Hugging Face: {e}")
        
        