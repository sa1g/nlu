from functions import (
    get_model,
    get_optimizer,
    init_weights,
    train,
    get_loaders_lang,
)
from torch.utils.tensorboard import SummaryWriter
import logging
import json

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)


import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-c", default="config.json", help="Config file json")


def main(train_config: dict, model_config: dict, optimizer_config: dict):
    train_loader, dev_loader, test_loader, lang = get_loaders_lang(
        train_config["dataset_path"],
        train_config["train_batch_size"],
        train_config["dev_batch_size"],
        train_config["test_batch_size"],
    )

    vocab_len = len(lang.word2id)
    pad_index = lang.word2id["<pad>"]
    output_size = vocab_len

    model_config["output_size"] = output_size
    model_config["pad_index"] = pad_index
    model_config["device"] = DEVICE
    model_config["init_weights"] = init_weights

    # MODEL SETUP
    model = get_model(model_config, DEVICE)

    logging.debug("Model done")

    # TENSORBOARD
    writer: SummaryWriter = SummaryWriter(log_dir=f"log/{model.name}")

    # TRAINING
    train(
        model=model,
        optimizer_config=optimizer_config,
        lang=lang,
        writer=writer,
        n_epochs=train_config["n_epochs"],
        clip=train_config["clip"],
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        device=DEVICE,
        patience=train_config["patience"]
    )


def load_config(config_file):
    with open(config_file, "r") as file:
        configs = json.load(file)
    return configs


if __name__ == "__main__":
    DEVICE = "cuda:0"

    args = parser.parse_args()
    config: dict = load_config(args.c)

    for key, config in config.items():
        logging.info(" !! Running %s !! ", key)
        assert config.get("model_type", "LM_RNN") in ["LM_RNN", "LM_LSTM"]
        assert config.get("optim_name", "SGD") in ["SGD", "AdamW", "NTAvSGD"]

        train_config = {
            "dataset_path": config.get("dataset_path", "../dataset"),
            "train_batch_size": config.get("train_batch_size", 128),
            "dev_batch_size": config.get("dev_batch_size", 128),
            "test_batch_size": config.get("test_batch_size", 128),
            "n_epochs": config.get("n_epochs", 1),
            "clip": config.get("clip", 5),
            "patience": config.get("patience", 5)
        }

        model_config = {
            "emb_size": config.get("emb_size", 300),
            "hid_size": config.get("hid_size", 300),
            "emb_dropout": config.get("emb_dropout", 0),
            "out_dropout": config.get("out_dropout", 0),
            "n_layers": config.get("n_layers", 1),
            "model_type": config.get("model_type", "LM_RNN"),
            "variational_dropout": config.get("variational_dropout", 0),
            "weight_tying": config.get("weight_tying", False),
            "optim_name": config.get("optim_name", "SGD"),
        }

        optimizer_config = {
            "optim_name": config.get("optim_name", "SGD"),
            "lr": config.get("lr", 0.0001),
            "betas": config.get("betas", (0.9, 0.999)),
            "eps": config.get("eps", 1e-08),
            "weight_decay": config.get("weight_decay", 0.01),
            "momentum": config.get("momentum", 0),
            "non_monotonic_interval": config.get("non_monotonic_interval", 5),
            "logging_interval": config.get("train_batch_size", 128)
        }

        main(train_config, model_config, optimizer_config)
