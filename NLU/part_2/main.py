# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
import json
from utils import get_loaders_lang
from functions import *
import os
from torch.utils.tensorboard import SummaryWriter
import logging
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c", default="config.json", help="Config file json")
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.WARNING)
# logging.basicConfig(
#     format="%(levelname)s:%(filename)s:%(funcName)s:%(lineno)d:%(message)s",
#     level=logging.DEBUG
# )


def main(
    train_config: dict, model_config: dict, optimizer_config: dict, device, PAD_TOKEN
):

    train_loader, dev_loader, test_loader, lang, w2id, slot2id, intent2id, tokenizer = (
        get_loaders_lang(
            train_config["dataset_path"],
            train_config["train_batch_size"],
            train_config["dev_batch_size"],
            train_config["test_batch_size"],
        )
    )

    name = f"BERT_emb_{model_config['emb_size']}_hid_{model_config['hid_size']}_edo_{model_config['emb_dropout']}_odo_{model_config['out_dropout']}_ido_{model_config['in_dropout']}_lay_{model_config['n_layers']}_bid_{model_config['bidirectional']}_{train_config['train_batch_size']}_{train_config['dev_batch_size']}_{train_config['test_batch_size']}"

    # TENSORBOARD
    writer: SummaryWriter = SummaryWriter(
        log_dir=f"log/{name}"
    )

    # Training
    train(
        model_config=model_config,
        optimizer_config=optimizer_config,
        train_config=train_config,
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        lang=lang,
        w2id=w2id,
        slot2id=slot2id,
        intent2id=intent2id,
        writer=writer,
        PAD_TOKEN=PAD_TOKEN,
        name=name,
        device=device,
        tokenizer=tokenizer
    )


def load_config(config_file):
    with open(config_file, "r") as file:
        configs = json.load(file)
    return configs


if __name__ == "__main__":
    # Global variables
    device = "cuda:0"
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Used to report errors on CUDA side
    PAD_TOKEN = 0

    args = parser.parse_args()
    config: dict = load_config(args.c)

    for key, config in config.items():
        logging.info(" !! Running %s !!", key)
        # add assert here if needed

        train_config = {
            "name": key,
            "dataset_path": config.get(
                "dataset_path", os.path.join("..", "dataset", "ATIS")
            ),
            "train_batch_size": config.get("train_batch_size", 128),
            "dev_batch_size": config.get("dev_batch_size", 128),
            "test_batch_size": config.get("test_batch_size", 128),
            "n_epochs": config.get("n_epochs", 1),
            "clip": config.get("clip", 5),
            "patience": config.get("patience", 5),
            "runs": config.get("runs", 5),
        }

        model_config = {
            "emb_size": config.get("emb_size", 300),
            "hid_size": config.get("hid_size", 300),
            "emb_dropout": config.get("emb_dropout", 0),
            "out_dropout": config.get("out_dropout", 0),
            "in_dropout": config.get("in_dropout", 0),
            "n_layers": config.get("n_layers", 1),
            "bidirectional": config.get("bidirectional", False),
        }

        optimizer_config = {"lr": config.get("lr", 0.001)}

        main(train_config, model_config, optimizer_config, device, PAD_TOKEN)
