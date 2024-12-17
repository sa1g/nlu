import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer, required
from torch.utils.data import DataLoader
from functools import partial
import logging

from tqdm import tqdm

from utils import read_file, Lang, PennTreeBank, collate_fn
from model import LM_RNN, LM_LSTM


def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        nn.init.xavier_uniform_(param[idx * mul : (idx + 1) * mul])
                elif "weight_hh" in name:
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        nn.init.orthogonal_(param[idx * mul : (idx + 1) * mul])
                elif "bias" in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)


def train_loop(data, optimizer, criterion, model: torch.nn.Module, clip=5):
    model.train()
    loss_array = []
    number_of_tokens = []

    for sample in data:
        optimizer.zero_grad()
        output = model(sample["source"])
        loss = criterion(output, sample["target"])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

    return sum(loss_array) / sum(number_of_tokens)


def eval_loop(data, eval_criterion, model):
    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []

    with torch.no_grad():
        for sample in data:
            output = model(sample["source"])
            loss = eval_criterion(output, sample["target"])
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])

    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return


def get_loaders_lang(
    path: str,
    train_batch_size: int = 128,
    dev_batch_size: int = 128,
    test_batch_size: int = 128,
) -> list[DataLoader | Lang]:
    logging.debug("Dataloading init")

    train_raw = read_file(f"{path}/ptb.train.txt")
    dev_raw = read_file(f"{path}/ptb.valid.txt")
    test_raw = read_file(f"{path}/ptb.test.txt")

    lang = Lang(train_raw, ["<pad>", "<eos>"])

    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),
        shuffle=True,
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=dev_batch_size,
        collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),
    )

    logging.debug("Dataloading done")
    return train_loader, dev_loader, test_loader, lang


def get_model(config: dict, device) -> nn.Module:
    if config["model_type"] == "LM_RNN":
        logging.debug("LM_RNN")
        model = LM_RNN(config).to(device)
    elif config["model_type"] == "LM_LSTM":
        logging.debug("LM_LSTM")
        model = LM_LSTM(config).to(device)
    # elif config["model_type"] == "LM_LSTM_WS":
    #     model = LM_LSTM_WS(config).to(device)
    # elif config["model_type"] == "LM_LSTM_VD":
    #     model = LM_LSTM_VD(config).to(device)

    # if init_weights:
    model.apply(init_weights)

    return model


def get_optimizer(model: nn.Module, config: dict = {}):
    if config["optim_name"] == "SGD":
        return optim.SGD(model.parameters(), lr=config["lr"])
    elif config["optim_name"] == "AdamW":
        return optim.AdamW(
            model.parameters(),
            lr=config["lr"],
            betas=config["betas"],
            eps=config["eps"],
            weight_decay=config["weight_decay"],
        )
    elif config["optim_name"] == "NTAvSGD":
        return NTAvSGD(
            model.parameters(),
            lr=config["lr"],
            momentum=0,
            dampening=0,
            weight_decay=config["weight_decay"],
            nesterov=False,
        )

    SystemError()


def train(
    model: nn.Module,
    optimizer_config: dict,
    lang: Lang,
    writer,
    n_epochs,
    clip: int,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    test_loader: DataLoader,
    device: str = "cpu",
    patience: int = 5
):
    """
    TODO: add docs
    Talk about early stopping, PPL and evaluation
    """
    criterion_train = torch.nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = torch.nn.CrossEntropyLoss(
        ignore_index=lang.word2id["<pad>"], reduction="sum"
    )

    optimizer = get_optimizer(model=model, config=optimizer_config)

    logging.debug(f"Got Optimizer: {optimizer.__class__.__name__}")

    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None

    logging.debug("Training")
    pbar = tqdm(range(1, n_epochs))

    # Training loop :)
    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_train, model, clip)
        if epoch % 1 == 0:
            sampled_epochs.append(epoch)

            losses_train.append(np.asarray(loss).mean())
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            losses_dev.append(np.asarray(loss_dev).mean())

            pbar.set_description("PPL: %f" % ppl_dev)

            writer.add_scalar("Loss/Train", losses_train[-1], epoch)
            writer.add_scalar("Loss/Test", losses_dev[-1], epoch)
            writer.add_scalar("PPL/Test", ppl_dev, epoch)

            # early stopping deactivated
            if patience != -1:
                if ppl_dev < best_ppl:
                    best_ppl = ppl_dev
                    best_model = copy.deepcopy(model).to("cpu")
                    patience = patience
                else:
                    patience -= 1

                if patience <= 0:
                    logging.info("My patience is done!")
                    break

        if optimizer.__class__.__name__ == "NTAvSGD":
            if (
                (
                    "t0" not in optimizer.param_groups[0]
                    and len(losses_dev) > optimizer_config["non_monotonic_interval"]
                    and loss_dev
                    > min(losses_dev[: -optimizer_config["non_monotonic_interval"]])
                    and optimizer_config["optim_name"] == "nmASGD"
                )
                or (epoch > 10)
            ) and (optimizer.is_triggered == False):
                optimizer.trigger()
                logging.debug(f"TRIGGERED!")

    logging.debug("Done")

    best_model.to(device)

    final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
    writer.add_scalar("PPL/Eval", final_ppl, len(sampled_epochs))
    logging.info("Test ppl: %f", final_ppl)

    path = f"bin/{best_model.name}.pt"
    torch.save(best_model.state_dict(), path)


class NTAvSGD(optim.SGD):
    def __init__(
        self, params, lr=1e-3, momentum=0, dampening=0, weight_decay=0, nesterov=False
    ):
        super(NTAvSGD, self).__init__(
            params, lr, momentum, dampening, weight_decay, nesterov
        )
        self.avg_params = None
        self.is_triggered = False

    def initialize_avg_params(self):
        self.avg_params = []
        for group in self.param_groups:
            avg_group = {}
            for param in group["params"]:
                if param.requires_grad:
                    avg_group[param] = torch.clone(param.data).detach()
            self.avg_params.append(avg_group)

    def update_avg_params(self):
        for avg_group, group in zip(self.avg_params, self.param_groups):
            for param in group["params"]:
                if param.requires_grad:
                    avg_group[param].data.mul_(0.5).add_(param.data, alpha=0.5)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        # Standard SGD step
        super().step()

        # Initialize avg_params on first step
        if self.avg_params is None:
            self.initialize_avg_params()

        # Update avg_params if triggered
        if self.is_triggered:
            self.update_avg_params()
            self.swap_params_with_avg()
            self.flatten_rnn_parameters()

    def trigger(self, state=True):
        self.is_triggered = state

    def swap_params_with_avg(self):
        if self.avg_params is not None:
            for avg_group, group in zip(self.avg_params, self.param_groups):
                for param in group["params"]:
                    if param.requires_grad:
                        param.data, avg_group[param].data = (
                            avg_group[param].data,
                            param.data,
                        )

    def flatten_rnn_parameters(self):
        for group in self.param_groups:
            for param in group["params"]:
                if isinstance(param, torch.nn.RNNBase):
                    param.flatten_parameters()

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
        super(NTAvSGD, self).load_state_dict(state_dict)
