import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from vad.models import build_model
from vad.models.losses import gradient_loss


class VADLitModule(pl.LightningModule):
    """Training wrapper. L_pred = intensity (MSE) + gradient loss. L_recon / L_mem / L_compact come in Phase 5."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Plain dict/list/number types only: DictConfig objects inside a checkpoint cannot be
        # loaded by torch>=2.6 with its default weights_only=True.
        cfg_plain = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else dict(cfg)
        self.save_hyperparameters({"cfg": cfg_plain})
        self.model = build_model(cfg)

    def forward(self, clip, flow=None):
        return self.model(clip, flow)

    def _step(self, batch, stage):
        pred = self(batch["clip"], batch.get("flow"))
        l_int = F.mse_loss(pred, batch["target"])
        l_grad = gradient_loss(pred, batch["target"])
        loss = l_int + self.cfg.train.gdl_weight * l_grad
        on_step = stage == "train"
        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True, on_step=on_step)
        self.log(f"{stage}/l_int", l_int, on_epoch=True, on_step=False)
        self.log(f"{stage}/l_grad", l_grad, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr,
                                 weight_decay=self.cfg.train.weight_decay)
