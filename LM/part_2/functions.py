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
from utils import Common, ExperimentConfig, Lang, get_dataloaders_and_lang, logging

class NTAvSGD(torch.optim.SGD):
    def __init__(
        self,
        params,
        lr=1e-3,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        rnn_modules=None,
    ):
        super().__init__(params, lr, momentum, dampening, weight_decay, nesterov)
        self.avg_params = []
        self.is_triggered = False
        self.rnn_modules = rnn_modules or []

    def initialize_avg_params(self):
        self.avg_params = []
        for group in self.param_groups:
            avg_group = {}
            for param in group["params"]:
                if param.requires_grad:
                    avg_group[param] = param.clone().detach()
            self.avg_params.append(avg_group)

    def update_avg_params(self):
        for avg_group, group in zip(self.avg_params, self.param_groups):
            for param in group["params"]:
                if param.requires_grad:
                    avg_group[param].mul_(0.5).add_(param.data, alpha=0.5)

    def step(self, closure=None):  # type: ignore
        loss = None
        if closure is not None:
            loss = closure()

        # Standard SGD step
        super().step(closure)

        # Initialize avg_params on first step
        if not self.avg_params:
            self.initialize_avg_params()

        # Update avg_params if triggered
        if self.is_triggered:
            self.update_avg_params()
            self.swap_params_with_avg()
            self.flatten_rnn_parameters()

    def trigger(self, state=True):
        self.is_triggered = state

    def swap_params_with_avg(self):
        if self.avg_params:
            for avg_group, group in zip(self.avg_params, self.param_groups):
                for param in group["params"]:
                    if param.requires_grad:
                        param.data, avg_group[param] = avg_group[param], param.data

    def flatten_rnn_parameters(self):
        for rnn_module in self.rnn_modules:
            rnn_module.flatten_parameters()

    def state_dict(self):
        state_dict = super(NTAvSGD, self).state_dict()
        state_dict["avg_params"] = [
            {k: v.clone() for k, v in avg_group.items()}
            for avg_group in self.avg_params
        ]
        state_dict["is_triggered"] = self.is_triggered
        return state_dict

    def load_state_dict(self, state_dict):
        self.is_triggered = state_dict.pop("is_triggered")
        avg_params = state_dict.pop("avg_params")
        self.avg_params = [
            {k: v.clone() for k, v in avg_group.items()} for avg_group in avg_params
        ]
        super().load_state_dict(state_dict)



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
        weight_tying=experiment_config.weight_tying,
    ).to(device)
    model.apply(init_weights)

    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(
        ignore_index=lang.word2id["<pad>"], reduction="sum"
    )

    optimizer: torch.optim.SGD | NTAvSGD = (
        experiment_config.optim(model.parameters(), lr=experiment_config.lr)
    )

    patience = experiment_config.patience
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model_state: Optional[dict] = None
    best_model_average: Optional[dict] = None

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
            writer.add_scalar("ppl/dev", ppl_dev, epoch)
            losses_dev.append(np.asarray(loss_dev).mean())
            pbar.set_description("PPL: %f" % ppl_dev)

            if ppl_dev < best_ppl:  # the lower, the better
                best_ppl = ppl_dev
                with torch.no_grad():
                    if "T" in optimizer.__dict__:
                        optimizer.average() # type: ignore
                        best_model_average = model.state_dict()
                        optimizer.restore()
                    best_model_state = model.state_dict()
                patience = experiment_config.patience
            else:
                patience -= 1

            if patience <= 0:  # Early stopping with patience
                break  # Not nice but it keeps the code clean

            if type(optimizer) == NTAvSGD:
                if (
                    (
                        "t0" not in optimizer.param_groups[0]
                        and len(losses_dev) > experiment_config.non_monotonic_interval
                        and loss_dev
                        > min(losses_dev[: -experiment_config.non_monotonic_interval])
                    )
                    # or (epoch > 10)
                ) and (optimizer.is_triggered == False):
                    optimizer.trigger()
                    logging.info(f"TRIGGERED!")
                    print(f"TRIGGERED at epoch {epoch} with loss {loss_dev}")

    if best_model_state is not None:
        best_model = experiment_config.model_type(
            emb_size=experiment_config.emb_size,
            hidden_size=experiment_config.hid_size,
            output_size=vocal_len,
            pad_index=lang.word2id["<pad>"],
            out_dropout=experiment_config.dropout_output,
            emb_dropout=experiment_config.dropout_embedding,
            n_layers=1,
            weight_tying=experiment_config.weight_tying,
        )
        best_model.load_state_dict(best_model_state)
        best_model.to(device)

        final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)

        writer.add_scalar("ppl/test", final_ppl)
        writer.add_scalar("ppl/dev_final", best_ppl)

        # Save the model
        model_path = os.path.join(file_name, "model.pt")
        torch.save(best_model_state, model_path)  # type: ignore

    if best_model_average is not None:
        best_model_avg = experiment_config.model_type(
            emb_size=experiment_config.emb_size,
            hidden_size=experiment_config.hid_size,
            output_size=vocal_len,
            pad_index=lang.word2id["<pad>"],
            out_dropout=experiment_config.dropout_output,
            emb_dropout=experiment_config.dropout_embedding,
            n_layers=1,
            weight_tying=experiment_config.weight_tying,
        )
        best_model_avg.load_state_dict(best_model_average)
        best_model_avg.to(device)

        final_ppl_avg, _ = eval_loop(test_loader, criterion_eval, best_model_avg)

        writer.add_scalar("ppl/test_avg", final_ppl_avg)

        # Save the averaged model
        model_path_avg = os.path.join(file_name, "model_avg.pt")
        torch.save(best_model_average, model_path_avg)


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
