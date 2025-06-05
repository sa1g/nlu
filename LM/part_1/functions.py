# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

import math
import os
from datetime import datetime
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from tqdm import tqdm

from utils import (Common, ExperimentConfig, Lang, get_dataloaders_and_lang,
                   logging)


def train_loop(data, optimizer, criterion, model, clip=5):
    model.train()
    loss_array = []
    number_of_tokens = []

    for sample in data:
        optimizer.zero_grad()  # Zeroing the gradient
        output = model(sample["source"])
        loss = criterion(output, sample["target"])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward()  # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid explosioning gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()  # Update the weights

    return sum(loss_array) / sum(number_of_tokens)


def eval_loop(data, eval_criterion, model):
    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []
    # softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad():  # It used to avoid the creation of computational graph
        for sample in data:
            output = model(sample["source"])
            loss = eval_criterion(output, sample["target"])
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])

    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return


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


def run_experiment(
    train_loader: DataLoader,
    dev_loader: DataLoader,
    test_loader: DataLoader,
    lang: Lang,
    experiment_config: ExperimentConfig,
    device: torch.device,
):

    file_name = os.path.join(
        "runs", (f"{datetime.now().strftime('%Y%m%d%H%M%S')}-{experiment_config.name}")
    )
    writer = SummaryWriter(log_dir=file_name)
    writer.add_scalar("hparams/hid_size", experiment_config.hid_size)
    writer.add_scalar("hparams/emd_size", experiment_config.emb_size)
    writer.add_scalar("hparams/lr", experiment_config.lr)
    writer.add_scalar("hparams/clip", experiment_config.clip)
    writer.add_scalar("hparams/n_epochs", experiment_config.n_epochs)
    writer.add_scalar("hparams/patience", experiment_config.patience)
    writer.add_scalar("hparams/dropout_embedding", experiment_config.dropout_embedding)
    writer.add_scalar("hparams/dropout_output", experiment_config.dropout_output)
    writer.add_text("hparams/model", experiment_config.model_type.__name__)
    writer.add_text("hparams/optim", experiment_config.optim.__name__)

    vocal_len = len(lang.word2id)

    model = experiment_config.model_type(
        emb_size=experiment_config.emb_size,
        hidden_size=experiment_config.hid_size,
        output_size=vocal_len,
        pad_index=lang.word2id["<pad>"],
        out_dropout=experiment_config.dropout_output,
        emb_dropout=experiment_config.dropout_embedding,
        n_layers=1,
    ).to(device)
    model.apply(init_weights)

    optimizer = experiment_config.optim(model.parameters(), lr=experiment_config.lr)
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(
        ignore_index=lang.word2id["<pad>"], reduction="sum"
    )

    patience = experiment_config.patience
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model_state: Optional[dict] = None

    pbar = tqdm(range(1, experiment_config.n_epochs))
    # If the PPL is too high try to change the learning rate
    for epoch in pbar:
        loss = train_loop(
            train_loader, optimizer, criterion_train, model, experiment_config.clip
        )

        writer.add_scalar("loss/train", loss, epoch)

        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            writer.add_scalar("loss/dev", loss_dev, epoch)
            writer.add_scalar("ppl/dev", ppl_dev)
            losses_dev.append(np.asarray(loss_dev).mean())
            pbar.set_description("PPL: %f" % ppl_dev)
            if ppl_dev < best_ppl:  # the lower, the better
                best_ppl = ppl_dev
                with torch.no_grad():
                    best_model_state = model.state_dict()
                patience = 3
            else:
                patience -= 1

            if patience <= 0:  # Early stopping with patience
                break  # Not nice but it keeps the code clean

    if best_model_state is not None:
        best_model = experiment_config.model_type(
            emb_size=experiment_config.emb_size,
            hidden_size=experiment_config.hid_size,
            output_size=vocal_len,
            pad_index=lang.word2id["<pad>"],
            out_dropout=experiment_config.dropout_output,
            emb_dropout=experiment_config.dropout_embedding,
            n_layers=1,
        )
        best_model.load_state_dict(best_model_state)
        best_model.to(device)

        final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)

        writer.add_scalar("ppl/test", final_ppl)
        writer.add_scalar("ppl/dev_final", best_ppl)

        # Save the model
        model_path = os.path.join(file_name, "model.pt")
        torch.save(best_model_state, model_path)  # type: ignore


def experiments_launcher(
    experiment_config: List[ExperimentConfig], common: Common, device: torch.device
):
    train_loader, dev_loader, test_loader, lang = get_dataloaders_and_lang(
        common, device
    )

    for experiment in experiment_config:
        logging.info(f"Running experiment with config: {str(experiment)}")

        run_experiment(
            train_loader,
            dev_loader,
            test_loader,
            lang,
            experiment,
            device,
        )
