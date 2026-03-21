import torch
import warnings

from transformers.optimization import AdamW
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

from IEFT.modules.objectives import compute_irtr_recall
from IEFT.gadgets.my_metrics import Accuracy, VQAScore, Scalar


def _safe_add_scalar(pl_module, tag: str, value, step: int):
    try:
        logger = getattr(pl_module, "logger", None)
        if logger is None:
            return
        exp = getattr(logger, "experiment", None)
        if exp is None:
            return
        if hasattr(exp, "add_scalar"):
            exp.add_scalar(tag, value, step)
    except Exception:
        pass


def set_metrics(pl_module):
    for split in ["train", "val"]:
        for k, v in pl_module.hparams.config["loss_names"].items():
            if v < 1:
                continue

            if k == "vqa":
                setattr(pl_module, f"{split}_vqa_score", VQAScore())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())

            elif k == "nlvr2":
                if split == "train":
                    setattr(pl_module, f"train_{k}_accuracy", Accuracy())
                    setattr(pl_module, f"train_{k}_loss", Scalar())
                else:
                    setattr(pl_module, f"dev_{k}_accuracy", Accuracy())
                    setattr(pl_module, f"dev_{k}_loss", Scalar())
                    setattr(pl_module, f"test_{k}_accuracy", Accuracy())
                    setattr(pl_module, f"test_{k}_loss", Scalar())

            elif k == "irtr":
                setattr(pl_module, f"{split}_irtr_loss", Scalar())

            elif k in ("mppd", "mpfr"):
                setattr(pl_module, f"{split}_{k}_loss", Scalar())

            elif k == "itm":
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
                setattr(pl_module, f"{split}_{k}_wpa_loss", Scalar())

            else:
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())


def epoch_wrapup(pl_module):
    phase = "train" if pl_module.training else "val"
    the_metric = 0

    # Recall computation (safe)
    if pl_module.hparams.config.get("get_recall_metric", False) and not pl_module.training:
        try:
            (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10), results = compute_irtr_recall(pl_module)

            if isinstance(results, dict) and results.get("skipped", False):
                reason = results.get("reason", "unknown")
                json_path = results.get("json", "")
                print("################  Recall Skipped  ################")
                print(f"Reason: {reason} | json: {json_path} | step: {pl_module.global_step}")
                print("")
            else:
                all_mean = (ir_r1 + ir_r5 + ir_r10 + tr_r1 + tr_r5 + tr_r10) / 6.0
                mean = round(all_mean.item(), 2)

                ir_r1_v = round(ir_r1.item(), 2)
                ir_r5_v = round(ir_r5.item(), 2)
                ir_r10_v = round(ir_r10.item(), 2)
                tr_r1_v = round(tr_r1.item(), 2)
                tr_r5_v = round(tr_r5.item(), 2)
                tr_r10_v = round(tr_r10.item(), 2)

                print("################  Original Evaluation Metrics  ################")
                print(
                    f"ir_top1: {ir_r1_v}%",
                    f", ir_top5: {ir_r5_v}%",
                    f", ir_top10: {ir_r10_v}%",
                    f", tr_top1: {tr_r1_v}%",
                    f", tr_top5: {tr_r5_v}%",
                    f", tr_top10: {tr_r10_v}%",
                    f" and this is mean: {mean}%",
                    pl_module.global_step,
                )
                print("")

                _safe_add_scalar(pl_module, "recalls/ir_r1", ir_r1_v, pl_module.global_step)
                _safe_add_scalar(pl_module, "recalls/ir_r5", ir_r5_v, pl_module.global_step)
                _safe_add_scalar(pl_module, "recalls/ir_r10", ir_r10_v, pl_module.global_step)
                _safe_add_scalar(pl_module, "recalls/tr_r1", tr_r1_v, pl_module.global_step)
                _safe_add_scalar(pl_module, "recalls/tr_r5", tr_r5_v, pl_module.global_step)
                _safe_add_scalar(pl_module, "recalls/tr_r10", tr_r10_v, pl_module.global_step)

                # selection metric: keep your original logic
                the_metric += (ir_r1_v + tr_r1_v)

        except Exception as e:
            warnings.warn(f"[epoch_wrapup] recall computation failed: {repr(e)}")

    # Standard loss/acc epoch metrics
    for loss_name, v in pl_module.hparams.config["loss_names"].items():
        if v < 1:
            continue

        value = 0

        if loss_name == "vqa":
            value = getattr(pl_module, f"{phase}_{loss_name}_score").compute()
            pl_module.log(f"{loss_name}/{phase}/score_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_score").reset()

            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()

        elif loss_name == "nlvr2":
            if phase == "train":
                value = getattr(pl_module, f"train_{loss_name}_accuracy").compute()
                pl_module.log(f"{loss_name}/train/accuracy_epoch", value)
                getattr(pl_module, f"train_{loss_name}_accuracy").reset()

                pl_module.log(
                    f"{loss_name}/train/loss_epoch",
                    getattr(pl_module, f"train_{loss_name}_loss").compute(),
                )
                getattr(pl_module, f"train_{loss_name}_loss").reset()
            else:
                value = getattr(pl_module, f"dev_{loss_name}_accuracy").compute()
                pl_module.log(f"{loss_name}/dev/accuracy_epoch", value)
                getattr(pl_module, f"dev_{loss_name}_accuracy").reset()

                pl_module.log(
                    f"{loss_name}/dev/loss_epoch",
                    getattr(pl_module, f"dev_{loss_name}_loss").compute(),
                )
                getattr(pl_module, f"dev_{loss_name}_loss").reset()

                value = getattr(pl_module, f"test_{loss_name}_accuracy").compute()
                pl_module.log(f"{loss_name}/test/accuracy_epoch", value)
                getattr(pl_module, f"test_{loss_name}_accuracy").reset()

                pl_module.log(
                    f"{loss_name}/test/loss_epoch",
                    getattr(pl_module, f"test_{loss_name}_loss").compute(),
                )
                getattr(pl_module, f"test_{loss_name}_loss").reset()

        elif loss_name == "irtr":
            pl_module.log(
                f"{loss_name}/{phase}/irtr_loss_epoch",
                getattr(pl_module, f"{phase}_irtr_loss").compute(),
            )
            getattr(pl_module, f"{phase}_irtr_loss").reset()

        elif loss_name in ("mppd", "mpfr"):
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()

        elif loss_name == "itm":
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()

            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()

            pl_module.log(
                f"{loss_name}/{phase}/wpa_loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_wpa_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_wpa_loss").reset()

        else:
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()

            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()

        the_metric += value

    pl_module.log(f"{phase}/the_metric", the_metric)


def check_non_acc_grad(pl_module):
    if pl_module.token_type_embeddings.weight.grad is None:
        return True
    grad = pl_module.token_type_embeddings.weight.grad
    return (grad.sum() == 0).item()


def set_task(pl_module):
    pl_module.current_tasks = [k for k, v in pl_module.hparams.config["loss_names"].items() if v >= 1]
    return


def set_schedule(pl_module):
    lr = pl_module.hparams.config["learning_rate"]
    wd = pl_module.hparams.config["weight_decay"]

    no_decay = [
        "bias",
        "LayerNorm.bias",
        "LayerNorm.weight",
        "norm.bias",
        "norm.weight",
        "norm1.bias",
        "norm1.weight",
        "norm2.bias",
        "norm2.weight",
    ]
    head_names = ["vqa_classifier", "nlvr2_classifier"]
    lr_mult = pl_module.hparams.config["lr_mult"]
    end_lr = pl_module.hparams.config["end_lr"]
    decay_power = pl_module.hparams.config["decay_power"]
    optim_type = pl_module.hparams.config["optim_type"]

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay) and not any(bb in n for bb in head_names)
            ],
            "weight_decay": wd,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay) and not any(bb in n for bb in head_names)
            ],
            "weight_decay": 0.0,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay) and any(bb in n for bb in head_names)
            ],
            "weight_decay": wd,
            "lr": lr * lr_mult,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay) and any(bb in n for bb in head_names)
            ],
            "weight_decay": 0.0,
            "lr": lr * lr_mult,
        },
    ]

    if optim_type == "adamw":
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98))
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
    elif optim_type == "sgd":
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optim_type: {optim_type}")

    if pl_module.trainer.max_steps is None:
        max_steps = (
            len(pl_module.trainer.datamodule.train_dataloader())
            * pl_module.trainer.max_epochs
            // pl_module.trainer.accumulate_grad_batches
        )
    else:
        max_steps = pl_module.trainer.max_steps

    warmup_steps = pl_module.hparams.config["warmup_steps"]
    if isinstance(warmup_steps, float):
        warmup_steps = int(max_steps * warmup_steps)

    if decay_power == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )
    else:
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
            lr_end=end_lr,
            power=decay_power,
        )

    sched = {"scheduler": scheduler, "interval": "step"}
    return [optimizer], [sched]