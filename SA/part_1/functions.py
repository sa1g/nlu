# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
import logging
import os
from datetime import datetime
from typing import List, Optional
from sklearn.metrics import classification_report


import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from tqdm import tqdm

from model import SlotModel
from utils import (
    PAD_TOKEN,
    Batch,
    Common,
    ExperimentConfig,
    Lang,
    get_dataloaders_and_lang,
)
from collections import Counter
from transformers import get_linear_schedule_with_warmup


def calculate_loss(logits, sample: Batch, lang: Lang):
    # Flatten labels: shape (B*T,)
    targets = sample.y_slots.view(-1)

    # Count label frequencies (excluding ignore_index)
    ignore_index = lang.slot2id["pad"]
    valid_targets = targets[targets != ignore_index]

    num_classes = len(lang.id2slot)
    label_freq = torch.bincount(valid_targets, minlength=num_classes).float()

    # Avoid division by zero by adding epsilon
    epsilon = 1e-6
    weights = 1.0 / (label_freq + epsilon)
    weights = weights + 1.0  # your original `+1` logic
    weights = weights.to(logits.device)

    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=ignore_index)

    # Reshape logits to (B*T, num_classes)
    logits = logits.view(-1, num_classes)

    loss = criterion(logits, targets)
    return loss


# def calculate_loss(logits, sample: Batch, lang: Lang):
#     sent_count = Counter(sample.y_slots.flatten().tolist())

#     sent_weights = (
#         torch.tensor([1 / sent_count[x] + 1 for x in lang.id2slot.keys()])
#         .float()
#         .to(logits.device)
#     )
#     criterion = nn.CrossEntropyLoss(
#         weight=sent_weights, ignore_index=lang.slot2id["pad"]
#     )

#     loss = criterion(logits.view(-1, len(lang.id2slot)), sample.y_slots.view(-1))

#     return loss


def train_loop(
    data: DataLoader,
    optimizer: torch.optim.Optimizer,
    model: SlotModel,
    lang: Lang,
    scheduler,
    grad_clip: bool,
):
    model.train()
    loss_array = []
    sample: Batch

    batch_tqdm = tqdm(enumerate(data), desc=f"Batch | Loss: {0:.4f}", leave=False)

    sample: Batch
    for _, sample in batch_tqdm:
        optimizer.zero_grad()  # Zeroing the gradient
        slot_logits = model(sample.utterances, sample.attention_masks)

        # loss_fn = torch.nn.CrossEntropyLoss(ignore_index=lang.slot2id["pad"])
        # loss = loss_fn(slot_logits.view(-1, len(lang.slot2id)), sample.y_slots.view(-1))

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


def tag2ts(ts_tag_sequence):
    """
    Transform ts tag sequence to target spans
    :param ts_tag_sequence: tag sequence with 'T' and 'O'
    :return: List of (start, end) tuples for target spans
    """
    n_tags = len(ts_tag_sequence)
    ts_sequence = []
    beg, end = -1, -1
    for i in range(n_tags):
        ts_tag = ts_tag_sequence[i]
        if ts_tag == "T":
            if beg == -1:
                beg = i
            end = i
        elif ts_tag == "O" and beg != -1:
            ts_sequence.append((beg, end))
            beg, end = -1, -1
    if beg != -1:
        ts_sequence.append((beg, end))
    return ts_sequence


def match_ts(gold_ts_sequence, pred_ts_sequence):
    """
    Calculate the number of correctly predicted target spans
    :param gold_ts_sequence: gold standard target spans
    :param pred_ts_sequence: predicted target spans
    :return: hit_count, gold_count, pred_count
    """
    hit_count = 0
    gold_count = len(gold_ts_sequence)
    pred_count = len(pred_ts_sequence)

    for t in pred_ts_sequence:
        if t in gold_ts_sequence:
            hit_count += 1

    return hit_count, gold_count, pred_count


def evaluate_ts(gold_ts, pred_ts):
    """
    Evaluate the model performance for the binary tagging task
    :param gold_ts: gold standard ts tags
    :param pred_ts: predicted ts tags
    :return: Precision, Recall, F1 scores

    Adapted from:
    https://github.com/lixin4ever/E2E-TBSA/blob/master/evals.py#L51
    """
    assert len(gold_ts) == len(pred_ts)
    n_samples = len(gold_ts)

    n_tp_ts, n_gold_ts, n_pred_ts = 0, 0, 0

    for i in range(n_samples):
        g_ts_sequence = tag2ts(ts_tag_sequence=gold_ts[i])
        p_ts_sequence = tag2ts(ts_tag_sequence=pred_ts[i])

        hit_ts_count, gold_ts_count, pred_ts_count = match_ts(
            gold_ts_sequence=g_ts_sequence, pred_ts_sequence=p_ts_sequence
        )

        n_tp_ts += hit_ts_count
        n_gold_ts += gold_ts_count
        n_pred_ts += pred_ts_count

    precision = float(n_tp_ts) / (n_pred_ts + 1e-4)
    recall = float(n_tp_ts) / (n_gold_ts + 1e-4)
    f1_score = 2 * precision * recall / (precision + recall + 1e-4)

    return precision, recall, f1_score


def eval_loop(model: SlotModel, dataloader: DataLoader, lang: Lang):
    model.eval()
    total_loss = []

    all_true_slots = []
    all_pred_slots = []

    with torch.no_grad():
        data: Batch
        for i, data in enumerate(dataloader):
            x = data.utterances
            attention_mask = data.attention_masks
            slots = data.y_slots
            slots_len = data.slots_len

            slot_logits = model(x, attention_mask)

            # loss_fn = torch.nn.CrossEntropyLoss(ignore_index=lang.slot2id["pad"])
            # loss = loss_fn(
            #     slot_logits.view(-1, len(lang.slot2id)), slots.view(-1)
            # ).item()

            loss = calculate_loss(slot_logits, data, lang)
            total_loss.append(loss.cpu().item())

            # Extract predictions
            slot_preds = torch.argmax(slot_logits, dim=2).cpu()
            true_slots = slots.cpu()

            for i in range(x.size(0)):
                seq_len = slots_len[i]
                tmp_ref = [
                    lang.id2slot[true_slots[i][j].item()] for j in range(seq_len)
                ]
                tmp_hyp = [
                    lang.id2slot[slot_preds[i][j].item()] for j in range(seq_len)
                ]

                all_true_slots.append(tmp_ref)
                all_pred_slots.append(tmp_hyp)

    ot_precision, ot_recall, ot_f1 = evaluate_ts(all_true_slots, all_pred_slots)

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

    for i in train_loader:
        pass
    exit()

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
