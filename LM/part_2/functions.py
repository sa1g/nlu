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

    # vocab = get_vocab(train_raw, ["<pad>", "<eos>"])

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
    elif config["model_type"] == "LM_LSTM_WS":
        model = LM_LSTM_WS(config).to(device)
    elif config["model_type"] == "LM_LSTM_VD":
        model = LM_LSTM_VD(config).to(device)

    if init_weights:
        model.apply(config["init_weights"])

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
    elif config["optim_name"] == "nmASGD":
        return optim.ASGD(
            model.parameters(),
            lr=1e-2,
            t0=config.get("t0", 1e6),
            weight_decay=0
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
):
    """
    TODO: add docs
    Talk about early stopping, PPL and evaluation
    """
    criterion_train = torch.nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = torch.nn.CrossEntropyLoss(
        ignore_index=lang.word2id["<pad>"], reduction="sum"
    )

    if optimizer_config["optim_name"] == "nmASGD":
        config = copy.deepcopy(optimizer_config)
        config["optim_name"] = "SGD"
        optimizer = get_optimizer(model=model, config=config)

        del config
    else:
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
            if ppl_dev < best_ppl:
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to("cpu")
                patience = 10
            else:
                patience -= 1

            if patience <= 0:
                logging.info("My patience is done!")
                break

        logging.debug(f"Running optimizer: {optimizer.__class__.__name__}")
        logging.debug(f"t0 in stuff: {'t0' not in optimizer.param_groups[0]}")
        logging.debug(f"val to switch {optimizer_config['non_monotonic_interval']}")
        logging.debug(f"losses_dev len: {len(losses_dev)}")
        logging.debug(f"optim target: {optimizer_config['optim_name']}")

        if len(losses_dev) > 6:
            logging.debug(
                f"last stuff: {loss_dev > min(losses_dev[: -optimizer_config['non_monotonic_interval']])}"
            )
            logging.debug(
                min(losses_dev[: -optimizer_config["non_monotonic_interval"]])
            )

        if (
            optimizer.__class__.__name__ == "SGD"
            and "t0" not in optimizer.param_groups[0]
            and len(losses_dev) > optimizer_config["non_monotonic_interval"]
            and loss_dev
            > min(losses_dev[: -optimizer_config["non_monotonic_interval"]])
            and optimizer_config["optim_name"] == "nmASGD"
        ):
            optimizer = get_optimizer(model=model, config=optimizer_config)
            logging.debug(f"Got Optimizer: {optimizer.__class__.__name__}")

    logging.debug("Done")

    best_model.to(device)

    final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
    writer.add_scalar("PPL/Eval", final_ppl, len(sampled_epochs))
    logging.info("Test ppl: %f", final_ppl)

    path = f"bin/{best_model.name}.pt"
    torch.save(best_model.state_dict(), path)


# class NonMonotonicAvSGD(Optimizer):
#     def __init__(
#         self,
#         params,
#         lr=required,
#         momentum=0,
#         weight_decay=0,
#         logging_interval=10,
#         non_monotonic_interval=5,
#     ):
#         if lr is not required and lr < 0.0:
#             raise ValueError("Invalid learning rate: {}".format(lr))
#         if momentum < 0.0 or momentum >= 1.0:
#             raise ValueError("Invalid momentum value: {}".format(momentum))
#         if weight_decay < 0.0:
#             raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
#         if logging_interval <= 0:
#             raise ValueError(
#                 "Invalid logging_interval value: {}".format(logging_interval)
#             )
#         if non_monotonic_interval <= 0:
#             raise ValueError(
#                 "Invalid non_monotonic_interval value: {}".format(
#                     non_monotonic_interval
#                 )
#             )

#         defaults = dict(
#             lr=lr,
#             momentum=momentum,
#             weight_decay=weight_decay,
#             logging_interval=logging_interval,
#             non_monotonic_interval=non_monotonic_interval,
#         )
#         super(NonMonotonicAvSGD, self).__init__(params, defaults)

#         self.state["k"] = 0
#         self.state["t"] = 0
#         self.state["T"] = 0
#         self.state["logs"] = []
#         self.state["averaging"] = False
#         self.state["num_averaged"] = 0

#     def step(self, closure=None, current_loss=None):
#         """Performs a single optimization step.

#         Arguments:
#             closure (callable, optional): A closure that reevaluates the model and returns the loss.
#             current_loss (float, optional): The current validation loss to check for non-monotonic trigger.
#         """
#         loss = None
#         if closure is not None:
#             loss = closure()

#         for group in self.param_groups:
#             weight_decay = group["weight_decay"]
#             momentum = group["momentum"]

#             for p in group["params"]:
#                 if p.grad is None:
#                     continue
#                 d_p = p.grad.data
#                 if weight_decay != 0:
#                     # d_p.add_(weight_decay, p.data)
#                     d_p.add_(p.data, alpha=weight_decay)
#                 if momentum != 0:
#                     param_state = self.state[p]
#                     if "momentum_buffer" not in param_state:
#                         buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
#                     else:
#                         buf = param_state["momentum_buffer"]
#                         buf.mul_(momentum).add_(d_p)
#                     d_p = buf

#                 # Update parameters according to: θ_t = θ_{t-1} - η_t * g_t
#                 # p.data.add_(-group["lr"], d_p)
#                 p.data.add_(d_p, alpha=-group["lr"])

#         # Increment the step counter
#         self.state["k"] += 1

#         # Check if we should log the validation loss
#         if (
#             self.state["k"] % self.defaults["logging_interval"] == 0
#             and self.state["T"] == 0
#         ):
#             if current_loss is not None:
#                 self.state["logs"].append(current_loss)
#                 if len(self.state["logs"]) > self.defaults["non_monotonic_interval"]:
#                     min_past_loss = min(
#                         self.state["logs"][: -self.defaults["non_monotonic_interval"]]
#                     )
#                     if current_loss > min_past_loss:
#                         self.state["T"] = self.state["k"]
#                 self.state["t"] += 1

#         # If averaging is triggered, update the running average of parameters
#         if self.state["T"] != 0:
#             self.state["averaging"] = True
#             for group in self.param_groups:
#                 for p in group["params"]:
#                     if "average_params" not in self.state:
#                         self.state["average_params"] = {
#                             id(p): torch.clone(p.data).detach() for p in group["params"]
#                         }
#                     else:
#                         avg_p = self.state["average_params"][id(p)]
#                         avg_p.mul_(self.state["num_averaged"]).add_(p.data).div_(
#                             self.state["num_averaged"] + 1
#                         )
#                         self.state["average_params"][id(p)] = avg_p.clone()
#             self.state["num_averaged"] += 1

#         if self.state["averaging"]:
#             logging.info("TRIGGERED")
#             self.apply_averaging()
#         return loss

#     def apply_averaging(self):
#         """Applies the averaged parameters."""
#         if not self.state["averaging"]:
#             raise RuntimeError("Averaging has not been triggered yet.")
#         for group in self.param_groups:
#             for p in group["params"]:
#                 if (
#                     "average_params" in self.state
#                     and id(p) in self.state["average_params"]
#                 ):
#                     p.data.copy_(self.state["average_params"][id(p)])


# class NonMonotonicAvSGD(Optimizer):
#     def __init__(self, params, lr=0.01, lambd=0.0001, alpha=0.75, t0=1e6, weight_decay=0, logging_interval=10, non_monotonic_interval=5):
#         if lr < 0.0:
#             raise ValueError("Invalid learning rate: {}".format(lr))
#         if lambd < 0.0:
#             raise ValueError("Invalid lambd value: {}".format(lambd))
#         if alpha < 0.0:
#             raise ValueError("Invalid alpha value: {}".format(alpha))
#         if t0 < 0.0:
#             raise ValueError("Invalid t0 value: {}".format(t0))
#         if weight_decay < 0.0:
#             raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
#         if logging_interval <= 0:
#             raise ValueError("Invalid logging_interval value: {}".format(logging_interval))
#         if non_monotonic_interval <= 0:
#             raise ValueError("Invalid non_monotonic_interval value: {}".format(non_monotonic_interval))

#         defaults = dict(lr=lr, lambd=lambd, alpha=alpha, t0=t0, weight_decay=weight_decay, logging_interval=logging_interval, non_monotonic_interval=non_monotonic_interval)
#         super(NonMonotonicAvSGD, self).__init__(params, defaults)

#         self.state['k'] = 0
#         self.state['t'] = 0
#         self.state['T'] = 0
#         self.state['logs'] = []
#         self.state['averaging'] = False
#         self.state['num_averaged'] = 0

#     def step(self, closure=None, current_loss=None):
#         """Performs a single optimization step.

#         Arguments:
#             closure (callable, optional): A closure that reevaluates the model and returns the loss.
#             current_loss (float, optional): The current validation loss to check for non-monotonic trigger.
#         """
#         loss = None
#         if closure is not None:
#             loss = closure()

#         for group in self.param_groups:
#             weight_decay = group['weight_decay']

#             for p in group['params']:
#                 if p.grad is None:
#                     continue
#                 d_p = p.grad.data
#                 if weight_decay != 0:
#                     d_p.add_(weight_decay, p.data)
#                 param_state = self.state[p]

#                 if 'mu' not in param_state:
#                     param_state['mu'] = torch.clone(p.data).detach()
#                 if 'eta' not in param_state:
#                     param_state['eta'] = group['lr']

#                 mu = param_state['mu']
#                 eta = param_state['eta']

#                 p.data.add_(-eta, d_p)
#                 mu.add_(-eta * (p.data - mu))

#                 if self.state['T'] != 0:
#                     if 'average_params' not in self.state:
#                         self.state['average_params'] = {id(p): torch.clone(mu).detach() for p in group['params']}
#                     else:
#                         avg_p = self.state['average_params'][id(p)]
#                         avg_p.mul_(self.state['num_averaged']).add_(mu).div_(self.state['num_averaged'] + 1)
#                         self.state['average_params'][id(p)] = avg_p.clone()
#                     self.state['num_averaged'] += 1

#         self.state['k'] += 1

#         if self.state['k'] % self.defaults['logging_interval'] == 0 and self.state['T'] == 0:
#             if current_loss is not None:
#                 self.state['logs'].append(current_loss)
#                 if len(self.state['logs']) > self.defaults['non_monotonic_interval']:
#                     min_past_loss = min(self.state['logs'][:-self.defaults['non_monotonic_interval']])
#                     if current_loss > min_past_loss:
#                         self.state['T'] = self.state['k']
#                 self.state['t'] += 1

#         if self.state["ageraging"]:
#             self.apply_averaging()

#         return loss

#     def apply_averaging(self):
#         """Applies the averaged parameters."""
#         if not self.state['averaging']:
#             raise RuntimeError("Averaging has not been triggered yet.")
#         for group in self.param_groups:
#             for p in group['params']:
#                 if 'average_params' in self.state and id(p) in self.state['average_params']:
#                     p.data.copy_(self.state['average_params'][id(p)])
