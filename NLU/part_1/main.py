# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
import json
from model import ModelIAS
from utils import get_loaders_lang
from functions import *
import os
from torch.utils.tensorboard import SummaryWriter
import logging
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c", default="config.json", help="Config file json")
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)


def main(train_config: dict, model_config: dict, optimizer_config: dict, device, PAD_TOKEN):
    
    train_loader, dev_loader, test_loader, lang = get_loaders_lang(
        train_config["dataset_path"],
        train_config["train_batch_size"],
        train_config["dev_batch_size"],
        train_config["test_batch_size"],
    )

    # Setup model config
    ## model_config is not fully used at the moment
    hid_size = 200
    emb_size = 300

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)

    # Get model
    model = ModelIAS(
        hid_size, out_slot, out_int, emb_size, vocab_len, pad_index=PAD_TOKEN
    ).to(device)
    model.apply(init_weights)

    # TENSORBOARD
    writer: SummaryWriter = SummaryWriter(
        log_dir=f"log/{model.name}-{train_config['train_batch_size']}-{train_config['dev_batch_size']}"
    )

    # Training
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
        PAD_TOKEN=PAD_TOKEN,
        device=device,
        patience=train_config["patience"],
    )

def load_config(config_file):
    with open(config_file, "r") as file:
        configs = json.load(file)
    return configs

if __name__ == "__main__":
    # Global variables
    device = "cuda:0"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Used to report errors on CUDA side
    PAD_TOKEN = 0

    args = parser.parse_args()
    config: dict = load_config(args.c)

    for key, config in config.items():
        logging.info(" !! Running %s !!", key)
        # add assert here if needed

        train_config = {
            "dataset_path": config.get("dataset_path", os.path.join("..", "dataset", "ATIS")),
            "train_batch_size": config.get("train_batch_size", 128),
            "dev_batch_size": config.get("dev_batch_size", 128),
            "test_batch_size": config.get("test_batch_size", 128),
            "n_epochs": config.get("n_epochs", 1),
            "clip": config.get("clip", 5),
            "patience": config.get("patience", 5),
        }

        model_config = {
            "emb_size": config.get("emb_size", 300),
            "hid_size": config.get("hid_size", 300),
            "emb_dropout": config.get("emb_dropout", 0),
            "out_dropout": config.get("out_dropout", 0),
            "n_layers": config.get("n_layers", 1),
            "bidirectional": config.get("bidirectional", False),
        }

        optimizer_config = {
            "lr": config.get("lr", 0.001)
        }

        main(train_config, model_config, optimizer_config, device, PAD_TOKEN)


    # # # train_loader, dev_loader, test_loader, lang = get_loaders_lang(
    # # #     "../dataset/ATIS", 128,64,64
    # # # )

    # !wget -P dataset/ATIS https://raw.githubusercontent.com/BrownFortress/IntentSlotDatasets/main/ATIS/test.json
    # !wget -P dataset/ATIS https://raw.githubusercontent.com/BrownFortress/IntentSlotDatasets/main/ATIS/train.json
    # !wget https://raw.githubusercontent.com/BrownFortress/NLU-2024-Labs/main/labs/conll.py

    # # # # hid_size = 200
    # # # # emb_size = 300

    # # # # lr = 0.0001 # learning rate
    # # # # clip = 5 # Clip the gradient

    # # # # out_slot = len(lang.slot2id)
    # # # # out_int = len(lang.intent2id)
    # # # # vocab_len = len(lang.word2id)

    # # # # model = ModelIAS(
    # # # #     hid_size, out_slot, out_int, emb_size, vocab_len, pad_index=PAD_TOKEN
    # # # # ).to(device)
    # # # # model.apply(init_weights)


    # # # # # """FROM HERE"""
    # # # # optimizer = optim.Adam(model.parameters(), lr=lr)
    # # # # criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    # # # # criterion_intents = nn.CrossEntropyLoss()  # Because we do not have the pad token

    # # # # # Simple Training loop
    # # # # n_epochs = 200
    # # # # patience = 3
    # # # # losses_train = []
    # # # # losses_dev = []
    # # # # sampled_epochs = []

    

    # # # # best_f1 = 0
    # # # # for x in tqdm(range(1, n_epochs)):
    # # # #     loss = train_loop(
    # # # #         train_loader,
    # # # #         optimizer,
    # # # #         criterion_slots,
    # # # #         criterion_intents,
    # # # #         model,
    # # # #         clip=clip,
    # # # #     )
    # # # #     if x % 5 == 0:  # We check the performance every 5 epochs
    # # # #         sampled_epochs.append(x)
    # # # #         losses_train.append(np.asarray(loss).mean())
    # # # #         results_dev, intent_res, loss_dev = eval_loop(
    # # # #             dev_loader, criterion_slots, criterion_intents, model, lang
    # # # #         )
    # # # #         losses_dev.append(np.asarray(loss_dev).mean())

    # # # #         f1 = results_dev["total"]["f"]
    # # # #         # For decreasing the patience you can also use the average between slot f1 and intent accuracy
    # # # #         if f1 > best_f1:
    # # # #             best_f1 = f1
    # # # #             # Here you should save the model
    # # # #             patience = 3
    # # # #         else:
    # # # #             patience -= 1
    # # # #         if patience <= 0:  # Early stopping with patience
    # # # #             break  # Not nice but it keeps the code clean

    # # # # results_test, intent_test, _ = eval_loop(
    # # # #     test_loader, criterion_slots, criterion_intents, model, lang
    # # # # )
    # # # # print("Slot F1: ", results_test["total"]["f"])
    # # # # print("Intent Accuracy:", intent_test["accuracy"])

    # # # # # # Saving the model:
    # # # # # PATH = os.path.join("bin", model_name)
    # # # # # saving_object = {"epoch": x,
    # # # # #                  "model": model.state_dict(),
    # # # # #                  "optimizer": optimizer.state_dict(),
    # # # # #                  "w2id": w2id,
    # # # # #                  "slot2id": slot2id,
    # # # # #                  "intent2id": intent2id}
    # # # # # torch.save(saving_object, PATH)

    # # # # plt.figure(num=3, figsize=(8, 5)).patch.set_facecolor("white")
    # # # # plt.title("Train and Dev Losses")
    # # # # plt.ylabel("Loss")
    # # # # plt.xlabel("Epochs")
    # # # # plt.plot(sampled_epochs, losses_train, label="Train loss")
    # # # # plt.plot(sampled_epochs, losses_dev, label="Dev loss")
    # # # # plt.legend()
    # # # # plt.show()

    # Multiple runs
    # To have reliable results on small corpora we have to train and test the model
    # from scratch for several times. At the end, we average the results and we
    # compute the standard deviation.

    # hid_size = 200
    # emb_size = 300

    # lr = 0.0001 # learning rate
    # clip = 5 # Clip the gradient

    # out_slot = len(lang.slot2id)
    # out_int = len(lang.intent2id)
    # vocab_len = len(lang.word2id)

    # n_epochs = 200
    # runs = 5

    # slot_f1s, intent_acc = [], []
    # for x in tqdm(range(0, runs)):
    #     model = ModelIAS(hid_size, out_slot, out_int, emb_size,
    #                     vocab_len, pad_index=PAD_TOKEN).to(device)
    #     model.apply(init_weights)

    #     optimizer = optim.Adam(model.parameters(), lr=lr)
    #     criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    #     criterion_intents = nn.CrossEntropyLoss()

    #     patience = 3
    #     losses_train = []
    #     losses_dev = []
    #     sampled_epochs = []
    #     best_f1 = 0
    #     for x in range(1,n_epochs):
    #         loss = train_loop(train_loader, optimizer, criterion_slots,
    #                         criterion_intents, model)
    #         if x % 5 == 0:
    #             sampled_epochs.append(x)
    #             losses_train.append(np.asarray(loss).mean())
    #             results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots,
    #                                                         criterion_intents, model, lang)
    #             losses_dev.append(np.asarray(loss_dev).mean())
    #             f1 = results_dev['total']['f']

    #             if f1 > best_f1:
    #                 best_f1 = f1
    #             else:
    #                 patience -= 1
    #             if patience <= 0: # Early stopping with patient
    #                 break # Not nice but it keeps the code clean

    #     results_test, intent_test, _ = eval_loop(test_loader, criterion_slots,
    #                                             criterion_intents, model, lang)
    #     intent_acc.append(intent_test['accuracy'])
    #     slot_f1s.append(results_test['total']['f'])
    # slot_f1s = np.asarray(slot_f1s)
    # intent_acc = np.asarray(intent_acc)
    # print('Slot F1', round(slot_f1s.mean(),3), '+-', round(slot_f1s.std(),3))
    # print('Intent Acc', round(intent_acc.mean(), 3), '+-', round(slot_f1s.std(), 3))
