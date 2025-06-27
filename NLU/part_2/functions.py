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
    scheduler: bool,
    grad_clip: bool,
):
    if scheduler:
        raise NotImplementedError()
    if grad_clip:
        raise NotImplementedError()

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
        optimizer.step()

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
            output_slots = torch.argmax(slot_logits, dim=1)

            # torch.Size([16, 25, 130])
            # torch.Size([16, 130])
            # print(slot_logits.shape)
            # print(output_slots.shape)

            for i in range(sample.utterances.size(0)):
                sequence_length = int(sample.slots_len[i].item())

                utterance = lang.tokenizer.tokenize(
                    lang.tokenizer.decode(
                        sample.utterances[i].cpu().tolist(),
                        include_special_tokens=False,
                    )
                )

                tmp_ref = [
                    (utterance[j], lang.id2slot[sample.y_slots[i][j].item()])
                    for j in range(sequence_length)
                ]

                tmp_hyp = [
                    (utterance[j], lang.id2slot[output_slots[i][j].item()])
                    for j in range(sequence_length)
                ]

                # print(tmp_ref)
                # print(tmp_hyp)

                ref_slots.append(tmp_ref)
                hyp_slots.append(tmp_hyp)

        # f1_slot = evaluate(ref_slots, hyp_slots)

        # print(f1_slot["total"]["f"])

        accuracy_intention = classification_report(  # type: ignore
            ref_intents,
            hyp_intents,
            output_dict=True,
            zero_division=False,
        )["accuracy"]

        print(f"Intent accuracy: {accuracy_intention:.4f}")

        exit(23)

    #         for id_seq, seq in enumerate(output_slots):
    #             length = sample.slots_len.tolist()[id_seq]
    #             utt =

    #             utt_ids = sample.utterances[id_seq][:length].tolist()
    #             gt_ids = sample.y_slots[id_seq].tolist()
    #             gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
    #             utterance = [lang.id2word[elem] for elem in utt_ids]
    #             to_decode = seq[:length].tolist()
    #             ref_slots.append(
    #                 [(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)]
    #             )
    #             tmp_seq = []
    #             for id_el, elem in enumerate(to_decode):
    #                 tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
    #             hyp_slots.append(tmp_seq)
    # try:
    #     results = evaluate(ref_slots, hyp_slots)
    # except Exception as ex:
    #     # Sometimes the model predicts a class that is not in REF
    #     print("Warning:", ex)
    #     ref_s = set([x[1] for x in ref_slots])
    #     hyp_s = set([x[1] for x in hyp_slots])
    #     print(hyp_s.difference(ref_s))
    #     results = {"total": {"f": 0}}

    # report_intent = classification_report(
    #     ref_intents, hyp_intents, zero_division=False, output_dict=True
    # )
    # return results, report_intent, loss_array


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
    best_f1 = 0

    for x in tqdm(range(1, experiment_config.n_epochs)):
        loss = train_loop(
            train_loader,
            optimizer,
            model,
            lang,
            scheduler=experiment_config.scheduler,
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
                writer.add_scalar("dev/slot_f1_dev", results_dev["total"]["f"], x)
                writer.add_scalar("dev/intent_acc_dev", intent_res["accuracy"], x)  # type: ignore

            f1 = results_dev["total"]["f"]

            # For decreasing the patience you can also use the average between slot f1 and intent accuracy
            if f1 > best_f1:
                best_f1 = f1
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
            test_loader, criterion_slots, criterion_intents, model, lang
        )
        if experiment_config.log_inner:
            model_path = os.path.join(file_name, "model.pt")
            torch.save(model.state_dict(), model_path)

    return results_test["total"]["f"], intent_test["accuracy"]  # type: ignore


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

            f1_run, acc_run = run_experiment(
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
