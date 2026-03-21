import os
import copy
import math
import pytorch_lightning as pl

from IEFT.config import ex
from IEFT.modules import ViLTransformerSS
from IEFT.datamodules.multitask_datamodule import MTDataModule


def _as_int_num_gpus(num_gpus_cfg):
    if num_gpus_cfg is None:
        return 0
    if isinstance(num_gpus_cfg, int):
        return num_gpus_cfg
    if isinstance(num_gpus_cfg, (list, tuple)):
        return len(num_gpus_cfg)
    try:
        return int(num_gpus_cfg)
    except Exception:
        return 0


def _save_stable_last_checkpoint(trainer, checkpoint_dir: str):
    """
    Save exactly one stable last.ckpt at the end of training.
    This avoids relying on Lightning's automatic last checkpoint behavior.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    stable_last = os.path.join(checkpoint_dir, "last.ckpt")
    trainer.save_checkpoint(stable_last)
    print(f"[INFO] Stable last checkpoint saved to: {stable_last}")


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    num_gpus_int = _as_int_num_gpus(_config.get("num_gpus", 0))
    num_nodes = int(_config.get("num_nodes", 1))
    use_ddp = (num_gpus_int > 1) or (num_nodes > 1)

    dm = MTDataModule(_config, dist=use_ddp)
    model = ViLTransformerSS(_config)
    exp_name = f'{_config["exp_name"]}'

    os.makedirs(_config["log_dir"], exist_ok=True)

    load_name = (
        os.path.basename(_config["load_path"])
        if _config.get("load_path")
        else "none"
    )
    if load_name.endswith(".ckpt"):
        load_name = load_name[:-5]

    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=f'{exp_name}_seed{_config["seed"]}_from_{load_name}',
    )

    checkpoint_dir = os.path.join(logger.log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [lr_callback]

    per_gpu = int(_config["per_gpu_batchsize"])
    effective_gpus = max(1, num_gpus_int)
    denom = max(1, per_gpu * effective_gpus * max(1, num_nodes))
    grad_steps = max(1, int(_config["batch_size"]) // denom)

    max_steps = _config.get("max_steps", None)
    if isinstance(max_steps, str) and max_steps.strip().lower() in ("none", ""):
        max_steps = None

    if max_steps is None:
        try:
            dm.setup("fit")
            train_dl = dm.train_dataloader()
            steps_per_epoch = len(train_dl)
            max_epoch = int(_config["max_epoch"])
            est_steps = (
                math.ceil((steps_per_epoch * max_epoch) / grad_steps)
                if steps_per_epoch > 0
                else 0
            )
            max_steps = None if est_steps <= 0 else est_steps
        except Exception:
            max_steps = None

    if isinstance(max_steps, int) and max_steps <= 1:
        _config["warmup_steps"] = 0

    max_epochs = int(_config["max_epoch"]) if max_steps is None else 1000

    trainer_kwargs = dict(
        gpus=_config["num_gpus"],
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        benchmark=True,
        deterministic=True,
        max_epochs=max_epochs,
        max_steps=max_steps,
        callbacks=callbacks,
        logger=logger,
        checkpoint_callback=False,   # important for this older Lightning version
        prepare_data_per_node=False,
        replace_sampler_ddp=False,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=10,
        flush_logs_every_n_steps=10,
        resume_from_checkpoint=_config.get("resume_from", None),
        weights_summary="top",
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
    )

    if use_ddp:
        trainer_kwargs["accelerator"] = "ddp"

    trainer = pl.Trainer(**trainer_kwargs)

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm)
        _save_stable_last_checkpoint(trainer, checkpoint_dir)
    else:
        trainer.test(model, datamodule=dm)