import copy
import json
import logging
import os
import random
from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter


from utils import (
    get_data_and_mapping,
    preprocess_data,
    ATISDataset,
    collate_fn,
    get_device,
)
from model import IntentSlotModel
from functions import train_loop, eval_loop

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c", default="config.json", help="Config file json")
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)


def load_config(config_file):
    with open(config_file, "r") as file:
        configs = json.load(file)
    return configs


def main(config: dict):
    """
    Main function to train and evaluate an Intent and Slot model for Natural Language Understanding (NLU).
    Args:
        config (dict): Configuration dictionary containing hyperparameters and settings.
    The function performs the following steps:
    1. Loads and preprocesses the training, development, and test datasets.
    2. Initializes the tokenizer, datasets, and dataloaders.
    3. Logs the shapes of the input tensors for debugging.
    4. Sets up the TensorBoard writer for logging.
    5. Initializes variables for tracking performance metrics and early stopping.
    6. Runs multiple training iterations (runs) with the specified number of epochs.
    7. Trains the model using the training loop and evaluates it on the development set.
    8. Implements early stopping based on development loss.
    9. Evaluates the best model on the test set and logs the results.
    10. Saves the best model and optimizer state to a file.
    Returns:
        None
    """

    train_raw, dev_raw, test_raw, slots2id, id2slots, intent2id, id2intent = (
        get_data_and_mapping()
    )

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    processed_train = preprocess_data(train_raw, tokenizer, slots2id, intent2id)
    processed_test = preprocess_data(test_raw, tokenizer, slots2id, intent2id)
    processed_dev = preprocess_data(dev_raw, tokenizer, slots2id, intent2id)

    train_dataset = ATISDataset(processed_train)
    test_dataset = ATISDataset(processed_test)
    dev_dataset = ATISDataset(processed_dev)

    dev_dataloader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=config["dev_batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config["test_batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    for dataloader in dev_dataloader:
        logging.debug(dataloader["input_ids"].shape)
        logging.debug(dataloader["attention_mask"].shape)
        logging.debug(dataloader["token_type_ids"].shape)
        logging.debug(dataloader["slots"].shape)
        logging.debug(dataloader["intent"].shape)

        break

    name = config["name"]
    writer: SummaryWriter = SummaryWriter(log_dir=f"log/{name}")

    accuracy, f1, loss = 0, 0, [float("inf")]
    accuracies, f1s, dev_losses = [], [], []

    best_model = None
    best_f1 = 0

    early_stopping_counter = 0

    for r in tqdm(range(config["runs"]), desc="Runs"):
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config["train_batch_size"],
            shuffle=True,
            collate_fn=collate_fn,
        )

        total_steps = len(train_dataloader) * config["epochs"]
        warmup_steps = int(0.1 * total_steps)

        model = IntentSlotModel(len(slots2id), len(intent2id))
        model.to(get_device())

        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
        intent_loss_fn = torch.nn.CrossEntropyLoss()
        slot_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=slots2id["pad"])

        # scheduler = None
        if config["scheduler"]:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=len(train_dataloader) * config["epochs"],
            )
        else:
            scheduler = None

        train_losses = []
        dev_losses = []

        epochs_tqdm = tqdm(
            range(config["epochs"]),
            desc=f"Epochs | Acc: {accuracy:.4f} - F1: {f1:.4f} - Loss: {sum(loss)/len(loss):.4f}",
            leave=False,
        )

        for epoch in epochs_tqdm:
            t_loss = train_loop(
                model,
                train_dataloader,
                optimizer,
                intent_loss_fn,
                slot_loss_fn,
                slots2id,
                scheduler,
                config["grad_clip"],
            )

            train_losses.append(np.asarray(t_loss).mean())

            accuracy, f1, loss = eval_loop(
                model,
                dev_dataloader,
                intent_loss_fn,
                slot_loss_fn,
                tokenizer,
                id2slots,
                slots2id,
            )

            dev_losses.append(np.asarray(loss).mean())

            epochs_tqdm.set_description(
                f"Epochs | Acc: {accuracy:.4f} - F1: {f1:.4f} - Loss: {sum(loss)/len(loss):.4f}"
            )

            if f1 > best_f1:
                best_f1 = f1
                best_model = copy.deepcopy(model).to("cpu")

            writer.add_scalar("Loss/train", train_losses[-1], epoch)
            writer.add_scalar("Loss/dev", dev_losses[-1], epoch)
            writer.add_scalar("F1/dev", f1, epoch)
            writer.add_scalar("Accuracy/dev", accuracy, epoch)

            # Early stopping with a patience of 5, using the counter
            if len(dev_losses) > 5 and dev_losses[-1] > dev_losses[-2]:
                early_stopping_counter += 1
            else:
                early_stopping_counter = 0

            if early_stopping_counter >= 5:
                break

        accuracy, f1, loss = eval_loop(
            best_model.to(get_device()),
            test_dataloader,
            intent_loss_fn,
            slot_loss_fn,
            tokenizer,
            id2slots,
            slots2id,
        )

        writer.add_scalar("F1/test", f1, r)
        writer.add_scalar("Accuracy/test", accuracy, r)
        writer.add_scalar("Loss/test", sum(loss) / len(loss), r)

        accuracies.append(accuracy)
        f1s.append(f1)
        dev_losses.append(sum(loss) / len(loss))

        print(
            f"Accuracy: {accuracy:.4f} - F1: {f1:.4f} - Loss: {sum(loss)/len(loss):.4f}"
        )

    # writer.add_scalar("F1/test", np.array(f1s).mean(), r)
    # writer.add_scalar("Accuracy/test", np.array(accuracies).mean(), r)
    # writer.add_scalar("Loss/test", np.array(dev_losses).mean(), r)

    PATH = f"bin/{name}.pt"
    saving_object = {
        "model": best_model.to("cpu").state_dict(),
        "optimizer": optimizer.state_dict(),
        "slot2id": slots2id,
        "intent2id": intent2id,
    }
    torch.save(saving_object, PATH)
    writer.close()


if __name__ == "__main__":
    # set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    args = parser.parse_args()
    config: dict = load_config(args.c)

    for key, config in config.items():
        logging.info(" !! Running %s !!", key)

        config = {
            "name": key,
            "dataset_path": config.get(
                "dataset_path", os.path.join("..", "dataset", "ATIS")
            ),
            "train_batch_size": config.get("train_batch_size", 64),
            "dev_batch_size": config.get("dev_batch_size", 64),
            "test_batch_size": config.get("test_batch_size", 64),
            "runs": config.get("runs", 1),
            "epochs": config.get("epochs", 6),
            "grad_clip": config.get("grad_clip", True),
            "scheduler": config.get("scheduler", True),
            "lr": config.get("lr", 5e-5),
        }

        main(config)
