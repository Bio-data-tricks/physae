"""Training orchestration for PhysAE."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import torch

from physae.config import ExperimentConfig
from physae.metrics import summarize_metrics
from physae.models import PhysaeModel
from physae.utils import configure_logging, create_run_directory, seed_everything, select_device

log = logging.getLogger(__name__)


class Trainer:
    """Minimal trainer handling fit/evaluate/test loops."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        seed_everything(config.seed)
        output_dir = create_run_directory(config.logging.output_dir, config.logging.run_name)
        self.run_dir = output_dir
        configure_logging(output_dir)
        self.device = select_device(config.trainer.device)
        log.info("Run directory initialised at %s", output_dir)
        self._wandb_run = None
        if config.logging.enable_wandb:
            try:
                import wandb

                self._wandb_run = wandb.init(
                    project=config.logging.project,
                    name=config.logging.run_name,
                    dir=str(output_dir),
                    reinit=True,
                    config={"experiment_name": config.experiment_name},
                )
            except Exception as exc:  # pragma: no cover - optional dependency
                log.warning("W&B initialisation failed: %s", exc)

    def _move_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.to(self.device) for k, v in batch.items()}

    def _log_metrics(self, metrics: Dict[str, torch.Tensor], step: int, *, prefix: str) -> None:
        summary = summarize_metrics(metrics)
        log.info("%s step %s metrics=%s", prefix, step, summary)
        if self._wandb_run is not None:
            self._wandb_run.log({f"{prefix}/{k}": v for k, v in summary.items()}, step=step)

    def fit(self, model: PhysaeModel, datamodule) -> Path:
        model = model.to(self.device)
        datamodule.setup("fit")
        optimiser = model.configure_optimiser()
        optimiser.zero_grad()
        global_step = 0
        for epoch in range(self.config.trainer.max_epochs):
            log.info("Epoch %s/%s", epoch + 1, self.config.trainer.max_epochs)
            model.train()
            for batch in datamodule.train_dataloader():
                batch = self._move_batch(batch)
                loss, metrics = model.compute_loss(batch)
                loss.backward()
                if self.config.trainer.gradient_clip_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.trainer.gradient_clip_norm)
                optimiser.step()
                optimiser.zero_grad()
                global_step += 1
                if global_step % self.config.trainer.log_every_n_steps == 0:
                    self._log_metrics(metrics, global_step, prefix="train")
            val_metrics = self.evaluate(model, datamodule)
            if self._wandb_run is not None:
                self._wandb_run.log({f"val/{k}": v for k, v in val_metrics.items()}, step=global_step)
        ckpt_path = self.run_dir / "model.ckpt"
        model.save(str(ckpt_path))
        log.info("Saved checkpoint to %s", ckpt_path)
        if self._wandb_run is not None:
            self._wandb_run.save(str(ckpt_path))  # pragma: no cover - optional dependency
            self._wandb_run.finish()  # pragma: no cover - optional dependency
        return ckpt_path

    def evaluate(self, model: PhysaeModel, datamodule) -> Dict[str, float]:
        model.eval()
        metrics_total: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for batch in datamodule.val_dataloader():
                batch = self._move_batch(batch)
                _, metrics = model.compute_loss(batch)
                for key, value in metrics.items():
                    metrics_total.setdefault(key, torch.tensor(0.0, device=self.device))
                    metrics_total[key] += value
        num_batches = max(len(datamodule.val_dataloader()), 1)
        averaged = {k: (v / num_batches) for k, v in metrics_total.items()}
        summary = summarize_metrics(averaged)
        log.info("Validation metrics: %s", summary)
        return summary

    def test(self, model: PhysaeModel, datamodule) -> Dict[str, float]:
        model.eval()
        datamodule.setup("test")
        metrics_total: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for batch in datamodule.test_dataloader():
                batch = self._move_batch(batch)
                _, metrics = model.compute_loss(batch)
                for key, value in metrics.items():
                    metrics_total.setdefault(key, torch.tensor(0.0, device=self.device))
                    metrics_total[key] += value
        num_batches = max(len(datamodule.test_dataloader()), 1)
        averaged = {k: (v / num_batches) for k, v in metrics_total.items()}
        summary = summarize_metrics(averaged)
        log.info("Test metrics: %s", summary)
        return summary


__all__ = ["Trainer"]
