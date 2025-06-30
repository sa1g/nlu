# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
import logging
import os
from datetime import datetime
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from conll import evaluate
from model import ModelIAS
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from tqdm import tqdm
from utils import (PAD_TOKEN, Common, ExperimentConfig, Lang,
                   get_dataloaders_and_lang)


def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        torch.nn.init.xavier_uniform_(
                            param[idx * mul : (idx + 1) * mul]
                        )
                elif "weight_hh" in name:
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        torch.nn.init.orthogonal_(param[idx * mul : (idx + 1) * mul])
                elif "bias" in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)


def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad()  # Zeroing the gradient
        slots, intent = model(sample["utterances"], sample["slots_len"])
        loss_intent = criterion_intents(intent, sample["intents"])
        loss_slot = criterion_slots(slots, sample["y_slots"])
        loss = loss_intent + loss_slot  # In joint training we sum the losses.
        # Is there another way to do that?
        loss_array.append(loss.item())
        loss.backward()  # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()  # Update the weights
    return loss_array


def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    model.eval()
    loss_array = []

    ref_intents = []
    hyp_intents = []

    ref_slots = []
    hyp_slots = []
    # softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad():  # It used to avoid the creation of computational graph
        for sample in data:
            slots, intents = model(sample["utterances"], sample["slots_len"])
            loss_intent = criterion_intents(intents, sample["intents"])
            loss_slot = criterion_slots(slots, sample["y_slots"])
            loss = loss_intent + loss_slot
            loss_array.append(loss.item())
            # Intent inference
            # Get the highest probable class
            out_intents = [
                lang.id2intent[x] for x in torch.argmax(intents, dim=1).tolist()
            ]
            gt_intents = [lang.id2intent[x] for x in sample["intents"].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)

            # Slot inference
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = sample["slots_len"].tolist()[id_seq]
                utt_ids = sample["utterance"][id_seq][:length].tolist()
                gt_ids = sample["y_slots"][id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                utterance = [lang.id2word[elem] for elem in utt_ids]
                to_decode = seq[:length].tolist()
                ref_slots.append(
                    [(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)]
                )
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)
    try:
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"total": {"f": 0}}

    report_intent = classification_report(
        ref_intents, hyp_intents, zero_division=False, output_dict=True
    )
    return results, report_intent, loss_array


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

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)

    model = ModelIAS(
        experiment_config.hid_size,
        out_slot,
        out_int,
        experiment_config.emb_size,
        vocab_len,
        n_layer=experiment_config.n_layer,
        pad_index=PAD_TOKEN,
        bidirectional=experiment_config.bidirectional,
        emb_dropout=experiment_config.emb_dropout,
        out_dropout=experiment_config.out_dropout,
    ).to(device)
    model.apply(init_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=experiment_config.lr)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss()  # Because we do not have the pad token

    patience = experiment_config.patience
    best_model_state: Optional[dict] = None
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_f1 = 0

    for x in tqdm(range(1, experiment_config.n_epochs)):
        loss = train_loop(
            train_loader,
            optimizer,
            criterion_slots,
            criterion_intents,
            model,
            clip=experiment_config.clip,
        )
        if experiment_config.log_inner:
            writer.add_scalar("loss/train", np.asarray(loss).mean(), x)

        if x % 5 == 0:  # We check the performance every 5 epochs
            sampled_epochs.append(x)
            losses_train.append(np.asarray(loss).mean())
            results_dev, intent_res, loss_dev = eval_loop(
                dev_loader, criterion_slots, criterion_intents, model, lang
            )
            losses_dev.append(np.asarray(loss_dev).mean())

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
        writer.add_scalar("hparams/hid_size", experiment.hid_size)
        writer.add_scalar("hparams/emd_size", experiment.emb_size)
        writer.add_scalar("hparams/n_layer", experiment.n_layer)
        writer.add_scalar("hparams/lr", experiment.lr)
        writer.add_scalar("hparams/clip", experiment.clip)
        writer.add_scalar("hparams/n_epochs", experiment.n_epochs)
        writer.add_scalar("hparams/patience", experiment.patience)
        writer.add_scalar("hparams/bidirectional", experiment.bidirectional)
        writer.add_scalar("hparams/out_dropout", experiment.out_dropout)
        writer.add_scalar("hparams/emb_dropout", experiment.emb_dropout)
        writer.add_text("hparams/optim", experiment.optim.__name__)

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
