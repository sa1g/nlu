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

from conll import evaluate
from model import IntentSlotModel
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


def calculate_loss(
    batch: Batch,
    lang: Lang,
    slot_logits: torch.Tensor,
    intent_logits: torch.Tensor,
):
    intent_count = Counter(batch.intents.tolist())
    intent_weights = (
        torch.tensor([1 / (intent_count[x] + 1) for x in lang.id2intent.keys()])
        .float()
        .to(intent_logits.device)
    )
    criterion_intents = nn.CrossEntropyLoss(weight=intent_weights)
    loss_intent = criterion_intents(intent_logits, batch.intents)

    slot_count = Counter(batch.y_slots.flatten().tolist())
    slot_weights = (
        torch.tensor([1 / (slot_count[x] + 1) for x in lang.id2slot.keys()])
        .float()
        .to(slot_logits.device)
    )
    criterion_slots = torch.nn.CrossEntropyLoss(
        weight=slot_weights, ignore_index=lang.pad_token  # type: ignore
    )

    loss_slot = criterion_slots(
        slot_logits.view(-1, len(lang.id2slot)), batch.y_slots.view(-1)
    )

    loss = loss_intent + loss_slot

    return loss


def train_loop(
    data: DataLoader,
    optimizer: torch.optim.Optimizer,
    model: IntentSlotModel,
    lang: Lang,
    scheduler,
    grad_clip: bool,
):
    model.train()
    loss_array = []
    sample: Batch

    batch_tqdm = tqdm(enumerate(data), desc=f"Batch | Loss: {0:.4f}", leave=False)

    for _, sample in batch_tqdm:
        optimizer.zero_grad()  # Zeroing the gradient
        intent_logits, slot_logits = model(sample.utterances, sample.attention_masks)

        loss = calculate_loss(sample, lang, slot_logits, intent_logits)

        batch_tqdm.set_description(f"Batch | Loss: {loss.item():.4f}")
        loss_array.append(loss.item())

        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler:
            scheduler.step()

    return loss_array


def eval_loop(model: IntentSlotModel, data: DataLoader, lang: Lang):
    model.eval()
    loss_array = []

    ref_intents = []
    hyp_intents = []

    ref_slots = []
    hyp_slots = []

    with torch.no_grad():  # It used to avoid the creation of computational graph
        sample: Batch
        for sample in data:
            intent_logits, slot_logits = model(
                sample.utterances, sample.attention_masks
            )

            loss = calculate_loss(
                batch=sample,
                lang=lang,
                slot_logits=slot_logits,
                intent_logits=intent_logits,
            )
            loss_array.append(loss.item())

            # Intent inference
            out_intents = [
                lang.id2intent[x] for x in torch.argmax(intent_logits, dim=1).tolist()
            ]
            gt_intents = [lang.id2intent[x] for x in sample.intents.tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)

            # Slot inference
            output_slots = torch.argmax(slot_logits, dim=2)

            for i in range(sample.utterances.size(0)):
                sequence_length = int(sample.slots_len[i].item())

                utterance = lang.tokenizer.tokenize(
                    lang.tokenizer.decode(
                        sample.utterances[i].cpu().tolist(),
                        include_special_tokens=False,
                    )
                )[:sequence_length]

                tmp_ref = []
                tmp_hyp = []
                for j in range(sequence_length):
                    if sample.y_slots[i][j].item() == lang.pad_token:
                        # Skip padding tokens
                        continue

                    tmp_ref.append(
                        (utterance[j], lang.id2slot[sample.y_slots[i][j].item()])
                    )
                    tmp_hyp.append(
                        (utterance[j], lang.id2slot[output_slots[i][j].item()])
                    )

                ref_slots.append(tmp_ref)
                hyp_slots.append(tmp_hyp)

    # Compute the F1 score for the slots
    try:
        f1_slot = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        print("\nWarning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        f1_slot = {"total": {"f": 0}}

    print(f1_slot["total"]["f"])

    accuracy_intention = classification_report(  # type: ignore
        ref_intents,
        hyp_intents,
        output_dict=True,
        zero_division=False,
    )["accuracy"]

    print(f"Intent accuracy: {accuracy_intention:.4f}")

    return float(f1_slot["total"]["f"]), float(accuracy_intention), loss_array


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

    model = IntentSlotModel(
        slot_len=len(lang.slot2id), intent_len=len(lang.intent2id)
    ).to(device)

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
            results_dev, intent_res, loss_dev = eval_loop(
                model=model, data=dev_loader, lang=lang
            )

            if experiment_config.log_inner:
                writer.add_scalar("loss/dev", np.asarray(loss_dev).mean(), x)
                writer.add_scalar("dev/slot_f1_dev", results_dev, x)
                writer.add_scalar("dev/intent_acc_dev", intent_res, x)  # type: ignore

            best_score = (results_dev + intent_res) / 2

            # For decreasing the patience you can also use the average between slot f1 and intent accuracy
            if (results_dev + intent_res) / 2 > top_score:
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

        results_test, intent_test, _ = eval_loop(
            model=model, data=test_loader, lang=lang
        )
        if experiment_config.log_inner:
            model_path = os.path.join(file_name, "model.pt")
            torch.save(model.state_dict(), model_path)

        return results_test, intent_test  # type: ignore


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

        f1, accuracy = [], []
        for run in range(experiment.n_runs):
            logging.info(
                f"Running experiment with config: {str(experiment)} - Run {run + 1}"
            )

            f1_run, acc_run = run_experiment(  # type: ignore
                train_loader,
                dev_loader,
                test_loader,
                lang,
                experiment,
                device,
                writer,
                file_name,
            )
            f1.append(f1_run)
            accuracy.append(acc_run)

            experiment.log_inner = False  # Disable inner logging after the first run

        slot_f1s = np.asarray(f1)
        intent_accs = np.asarray(accuracy)

        writer.add_scalar("results/test_slot_f1_mean", slot_f1s.mean())
        writer.add_scalar("results/test_slot_f1_std", slot_f1s.std())
        writer.add_scalar("results/test_intent_acc_mean", intent_accs.mean())
        writer.add_scalar("results/test_intent_acc_std", intent_accs.std())
