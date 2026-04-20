"""PithTrain logging module."""

import logging
import os
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Generator, Optional

import wandb
from wandb.sdk.wandb_run import Run as WandbRun

from pithtrain.config import SlottedDefault


class StdoutLogger(logging.Logger):
    """Logger that prints to standard output."""

    def __init__(self, name: str, level: int = 0):
        super().__init__(name, level)
        handler = logging.StreamHandler(sys.stdout)
        fmt = "%(asctime)s | %(levelname)s | %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
        formatter = logging.Formatter(fmt, datefmt)
        handler.setFormatter(formatter)
        self.addHandler(handler)

    def info(self, *args, rank: int = 0, **kwargs):
        """
        Log an info message only if the current process rank matches the specified rank.
        This is useful in distributed settings to avoid duplicate logs.

        Parameters
        ----------
        rank : int
            The rank of the process that should log the message. Defaults to 0. If "RANK"
            is not in the environment, all ranks will log. If rank is negative, all ranks
            will log.
        """
        if "RANK" not in os.environ or rank < 0 or int(os.environ["RANK"]) == rank:
            super().info(*args, **kwargs)


@dataclass(init=False, slots=True)
class LoggingWandbCfg(SlottedDefault):
    """Configuration for logging with Weights & Biases."""

    entity: str
    """The username or team name the runs are logged to."""

    project: str
    """The name of the project under which this run will be logged."""

    name: str
    """A short display name for this run, which appears in the UI to help you identify it."""

    group: Optional[str] = None
    """A group name to organize related runs together."""


@dataclass(init=False, slots=True)
class LoggingCfg(SlottedDefault):
    """Configuration for logging."""

    wandb: Optional[LoggingWandbCfg] = None
    """Configuration for logging with Weights & Biases."""


@dataclass(init=False, slots=True)
class LoggingCtx(SlottedDefault):
    """Context for logging."""

    stdout: StdoutLogger
    """Logger that prints to standard output."""

    wandb: Optional[WandbRun] = None
    """Weights & Biases run."""


def setup_stdout(cfg: LoggingCfg, ctx: LoggingCtx) -> None:
    """Setup the stdout logger."""
    logger = StdoutLogger("pithtrain", logging.INFO)
    ctx.stdout = logger


def setup_wandb(cfg: LoggingCfg, ctx: LoggingCtx) -> None:
    """Setup the WandB run."""
    if ctx.wandb is not None:
        return
    if cfg.wandb is None:
        return
    if os.environ.get("RANK", "0") != "0":
        return
    kwargs = asdict(cfg.wandb)
    kwargs["resume"] = "allow"
    kwargs["dir"] = os.environ.get("WANDB_DIR", "/tmp/wandb")
    ctx.wandb = wandb.init(**kwargs)
    # Define the metrics for monitoring.
    ctx.wandb.define_metric("train/step", hidden=True)
    ctx.wandb.define_metric("train/cross-entropy-loss", step_metric="train/step")
    ctx.wandb.define_metric("train/load-balance-loss", step_metric="train/step")
    ctx.wandb.define_metric("train/learning-rate", step_metric="train/step")
    ctx.wandb.define_metric("train/gradient-norm", step_metric="train/step")
    ctx.wandb.define_metric("infra/tokens-per-second", step_metric="train/step")
    ctx.wandb.define_metric("infra/peak-gpu-memory", step_metric="train/step")


def activate_wandb(cfg: object, ctx: object) -> None:
    """
    Lazily initialize the WandB run and upload config.

    Intended to be called from the training loop right before the first
    ``wandb.log()``, so that a run is only created after a training step fully
    succeeds.  The double-init guard in ``setup_wandb`` makes repeated calls a
    no-op.
    """
    assert hasattr(cfg, "logging") and isinstance(cfg.logging, LoggingCfg)
    assert hasattr(ctx, "logging") and isinstance(ctx.logging, LoggingCtx)
    setup_wandb(cfg.logging, ctx.logging)
    if ctx.logging.wandb is not None:
        config = {}
        for section in ("distributed", "training"):
            if hasattr(cfg, section):
                config[section] = getattr(cfg, section).to_json_dict()
        ctx.logging.wandb.config.update(config)


@contextmanager
def logging_context(cfg: object, ctx: object) -> Generator[LoggingCtx, None, None]:
    """Context manager for logging."""
    assert hasattr(cfg, "logging") and isinstance(cfg.logging, LoggingCfg)
    assert hasattr(ctx, "logging") and isinstance(ctx.logging, LoggingCtx)
    setup_stdout(cfg.logging, ctx.logging)
    yield ctx.logging
