import logging
import os
from collections import defaultdict
from datetime import datetime
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from model import SlotModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from utils import (Batch, Common, ExperimentConfig, Lang,
                   get_dataloaders_and_lang)


def calculate_loss(logits, sample: Batch, lang: Lang):
    targets = sample.y_slots.view(-1)
    ignore_index = lang.slot2id["pad"]

    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    logits = logits.view(-1, len(lang.slot2id))

    loss = criterion(logits, targets)

    return loss


def train_loop(
    data: DataLoader,
    optimizer: torch.optim.Optimizer,
    model: SlotModel,
    lang: Lang,
    scheduler,
    grad_clip: bool,
):
    """
    Basic training loop

    Args:
        data: Dataloader
        optimizer: torch.optim.SGD or NTAvSGD
        model: torch.nn.Module
        lang: Lang object containing slot2id and pad_token
        scheduler
        grad_clip: Whether to apply gradient clipping
    Returns:
        loss_array: List of losses for each batch
    """

    model.train()
    loss_array = []
    sample: Batch

    batch_tqdm = tqdm(enumerate(data), desc=f"Batch | Loss: {0:.4f}", leave=False)

    sample: Batch
    for _, sample in batch_tqdm:
        optimizer.zero_grad()  # Zeroing the gradient
        slot_logits = model(sample.utterances, sample.attention_masks)

        loss = calculate_loss(slot_logits, sample, lang)

        batch_tqdm.set_description(f"Batch | Loss: {loss.item():.4f}")
        loss_array.append(loss.item())

        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler:
            scheduler.step()

    return loss_array


def evaluate_ts_simple(gold_sequences, pred_sequences):
    """
    Simplified TS evaluation for nested lists (e.g., [['O', 'T'], ['O']]).

    Args:
        gold_sequences (list of list): List of gold sequences, each sequence is a list of tags.
        pred_sequences (list of list): List of predicted sequences, each sequence is a list of tags.

    Returns: macro_f1, micro_p, micro_r, micro_f1

    Inspired by [script](https://github.com/lixin4ever/E2E-TBSA/blob/master/evals.py).

    """
    counts = defaultdict(lambda: {"tp": 0, "gold": 0, "pred": 0})

    # Flatten and iterate over all tags
    gold_tags = [tag for seq in gold_sequences for tag in seq]
    pred_tags = [tag for seq in pred_sequences for tag in seq]

    for g, p in zip(gold_tags, pred_tags):
        if g == p:
            counts[g]["tp"] += 1
        counts[g]["gold"] += 1  # Actual count
        counts[p]["pred"] += 1  # Predicted count

    # Macro F1 (average per-class F1)
    macro_f1 = 0.0
    n_classes = 0
    for label in counts:
        tp = counts[label]["tp"]
        precision = tp / (counts[label]["pred"] + 1e-4)
        recall = tp / (counts[label]["gold"] + 1e-4)
        f1 = 2 * precision * recall / (precision + recall + 1e-4)
        macro_f1 += f1
        n_classes += 1
    macro_f1 /= n_classes if n_classes else 1

    # Micro F1 (global)
    total_tp = sum(counts[label]["tp"] for label in counts)
    total_gold = sum(counts[label]["gold"] for label in counts)
    total_pred = sum(counts[label]["pred"] for label in counts)

    micro_p = total_tp / (total_pred + 1e-4)
    micro_r = total_tp / (total_gold + 1e-4)
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r + 1e-4)

    return {
        "macro f1": macro_f1,
        "micro p": micro_p,
        "micro r": micro_r,
        "micro f1": micro_f1,
    }


def eval_loop(model: SlotModel, dataloader: DataLoader, lang: Lang):
    model.eval()
    total_loss = []

    all_true_slots = []
    all_pred_slots = []

    with torch.no_grad():
        sample: Batch
        for _, sample in enumerate(dataloader):
            slot_logits = model(sample.utterances, sample.attention_masks)
            # print(sample.utterances.shape) # torch.Size([16, 26])

            loss = calculate_loss(slot_logits, sample, lang)
            total_loss.append(loss.cpu().item())

            # Extract predictions
            slot_preds = torch.argmax(slot_logits, dim=2).cpu()

            # print(slot_logits.shape) # torch.Size([16, 29, 3])
            # print(slot_preds.shape) # torch.Size([16, 29])

            true_slots = sample.y_slots.cpu()

            for i in range(sample.utterances.size(0)):

                tmp_ref = []
                tmp_hyp = []
                for j in range(sample.slots_len[i]):

                    if sample.y_slots[i][j] == lang.pad_token:
                        continue

                    tmp_ref.append(lang.id2slot[true_slots[i][j].item()])
                    tmp_hyp.append(lang.id2slot[slot_preds[i][j].item()])

                all_true_slots.append(tmp_ref)
                all_pred_slots.append(tmp_hyp)

    evaluated = evaluate_ts_simple(all_true_slots, all_pred_slots)

    ot_f1 = evaluated["micro f1"]
    ot_precision = evaluated["micro p"]
    ot_recall = evaluated["micro r"]

    return ot_f1, ot_precision, ot_recall, total_loss


def run_experiment(
    train_loader: DataLoader,
    dev_loader: DataLoader,
    test_loader: DataLoader,
    lang: Lang,
    experiment_config: ExperimentConfig,
    device: torch.device,
    writer: SummaryWriter,
    file_name: str = "",
):
    """
    Run a single experiment with the given configuration.
    It also creates a TensorBoard writer and logs hyperparams + results.

    Args are self-explanatory and typed.
    """

    model = SlotModel(slot_len=len(lang.slot2id)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=experiment_config.lr)

    patience = experiment_config.patience
    best_model_state: Optional[dict] = None
    top_score = -np.inf

    if experiment_config.scheduler:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=(
                int(0.1 * experiment_config.n_epochs)
                if experiment_config.n_epochs > 4
                else 4
            ),
            num_training_steps=experiment_config.n_epochs * len(train_loader),
        )
    else:
        scheduler = None

    for x in tqdm(range(1, experiment_config.n_epochs)):
        loss = train_loop(
            train_loader,
            optimizer,
            model,
            lang,
            scheduler=scheduler,
            grad_clip=experiment_config.grad_clip,
        )

        if experiment_config.log_inner:
            writer.add_scalar("loss/train", np.asarray(loss).mean(), x)

        if x % 1 == 0:
            ot_f1, ot_precision, ot_recall, loss_dev = eval_loop(
                model=model, dataloader=dev_loader, lang=lang
            )

            if experiment_config.log_inner:
                writer.add_scalar("loss/dev", np.asarray(loss_dev).mean(), x)
                writer.add_scalar("dev/f1", ot_f1, x)
                writer.add_scalar("dev/precision", ot_precision, x)
                writer.add_scalar("dev/recall", ot_recall, x)

            best_score = (ot_f1 + ot_precision + ot_recall) / 3

            # For decreasing the patience you can also use the average between slot f1 and intent accuracy
            if best_score > top_score:
                top_score = best_score
                # Here you should save the model
                patience = experiment_config.patience
                with torch.no_grad():
                    best_model_state = model.state_dict()
            else:
                patience -= 1
            if patience <= 0:  # Early stopping with patience
                break  # Not nice but it keeps the code clean

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

        ot_f1, ot_precision, ot_recall, loss_dev = eval_loop(
            model=model, dataloader=test_loader, lang=lang
        )
        if experiment_config.log_inner:
            model_path = os.path.join(file_name, "model.pt")
            torch.save(model.state_dict(), model_path)

        return ot_f1, ot_precision, ot_recall


def experiment_launcher(
    experiment_config: List[ExperimentConfig], common: Common, device: torch.device
):
    """
    Launch experiments given the list of experiments.

    Args are self-explanatory and typed.
    """

    train_loader, dev_loader, test_loader, lang = get_dataloaders_and_lang(
        common, device=device
    )

    for experiment in experiment_config:

        file_name = os.path.join(
            "runs",
            (f"{datetime.now().strftime('%Y%m%d%H%M%S')}-{experiment.name}"),
        )
        writer = SummaryWriter(file_name)

        hparams = {
            "n_runs": experiment.n_runs,
            "epochs": experiment.n_epochs,
            "lr": experiment.lr,
            "grad_clip": experiment.grad_clip,
            "scheduler": experiment.scheduler,
        }

        # Metrics (required by add_hparams, even if dummy values)
        metrics = {
            "hparam/dummy_metric": 0.0,  # Placeholder
        }

        writer.add_hparams(hparams, metrics)

        f1, precision, recall = [], [], []
        for run in range(experiment.n_runs):
            logging.info(
                f"Running experiment with config: {str(experiment)} - Run {run + 1}"
            )

            ot_f1, ot_precision, ot_recall = run_experiment(  # type: ignore
                train_loader,
                dev_loader,
                test_loader,
                lang,
                experiment,
                device,
                writer,
                file_name,
            )

            f1.append(ot_f1)
            precision.append(ot_precision)
            recall.append(ot_recall)

            experiment.log_inner = False  # Disable inner logging after the first run

        f1s = np.array(f1)
        precisions = np.array(precision)
        recalls = np.array(recall)

        writer.add_scalar("results/f1_mean", f1s.mean())
        writer.add_scalar("results/f1_std", f1s.std())
        writer.add_scalar("results/precision_mean", precisions.mean())
        writer.add_scalar("results/precision_std", precisions.std())
        writer.add_scalar("results/recall_mean", recalls.mean())
        writer.add_scalar("results/recall_std", recalls.std())
