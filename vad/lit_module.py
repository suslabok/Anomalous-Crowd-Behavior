import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from vad.models import build_model


class VADLitModule(pl.LightningModule):
    """Training wrapper. Loss terms (L_recon, L_mem, L_compact) get added in Phase 5."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Plain dict/list/number types only: DictConfig objects inside a checkpoint cannot be
        # loaded by torch>=2.6 with its default weights_only=True.
        cfg_plain = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else dict(cfg)
        self.save_hyperparameters({"cfg": cfg_plain})
        self.model = build_model(cfg)

    def forward(self, clip):
        return self.model(clip)

    def _step(self, batch, stage):
        pred = self(batch["clip"])
        loss = F.mse_loss(pred, batch["target"])
        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True, on_step=(stage == "train"))
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr,
                                 weight_decay=self.cfg.train.weight_decay)