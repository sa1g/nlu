# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
import copy
import logging
from matplotlib import pyplot as plt
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from model import ModelBert
from conll import evaluate
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from torch import optim


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


def train_loop(
    # data, optimizer, criterion_slots, criterion_intents, model, scaler, clip=5
    data, optimizer, criterion_slots, criterion_intents, model, clip=5

):
    model.train()
    loss_array = []

    for sample in data:
        optimizer.zero_grad()  # Zeroing the gradient

        slots, intent = model(
            sample["input_ids"], sample["attention_mask"], sample["token_type_ids"]
        )

        loss_intent = criterion_intents(intent, sample["intents"])
        loss_slot = criterion_slots(slots, sample["slots"])
        # In joint training we sum the losses.
        loss = loss_intent + loss_slot

        loss_array.append(loss.item())

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
    return loss_array


# def eval_loop(data, criterion_slots, criterion_intents, model, tokenizer, lang):
#     model.eval()
#     loss_array = []

#     ref_intents = []
#     hyp_intents = []

#     ref_slots = []
#     hyp_slots = []

#     with torch.no_grad():  # It used to avoid the creation of computational graph
#         for sample in data:
#             slots, intents = model(sample['input_ids'], sample['attention_mask'], sample['token_type_ids'])
#             loss_intent = criterion_intents(intents, sample["intents"])
#             loss_slot = criterion_slots(slots, sample["slots"])
#             loss = loss_intent + loss_slot
#             loss_array.append(loss.item())

#             # Intent inference
#             out_intents = [lang.id2intent[x] for x in torch.argmax(intents, dim=1).tolist()]
#             gt_intents = [lang.id2intent[x] for x in sample["intents"].tolist()]
#             ref_intents.extend(gt_intents)
#             hyp_intents.extend(out_intents)

#             # Slot inference
#             output_slots = torch.argmax(slots, dim=1)
#             for id_seq, seq in enumerate(output_slots):
#                 # Decode the tokens using BERT tokenizer
#                 tokens = tokenizer.convert_ids_to_tokens(sample['input_ids'][id_seq])
#                 gt_slots = [lang.id2slot[elem] for elem in sample["slots"][id_seq].tolist()]

#                 # Filter out special tokens
#                 filtered_tokens = []
#                 filtered_gt_slots = []
#                 filtered_pred_slots = []
#                 for i, token in enumerate(tokens):
#                     if token not in tokenizer.all_special_tokens:
#                         filtered_tokens.append(token)
#                         filtered_gt_slots.append(gt_slots[i])
#                         filtered_pred_slots.append(seq[i].item())

#                 ref_slots.append([(filtered_tokens[i], filtered_gt_slots[i]) for i in range(len(filtered_tokens))])
#                 hyp_slots.append([(filtered_tokens[i], lang.id2slot[filtered_pred_slots[i]]) for i in range(len(filtered_tokens))])

#     logging.debug(f"REFERENCE: {ref_slots[0]}")
#     logging.debug(f"HYPOTHEIS: {hyp_slots[0]}")
#     # exit()
#     # TODO: improve the evaluation as it is not working properly
#     try:
#         results = evaluate(ref_slots, hyp_slots)
#     except Exception as ex:
#         # Sometimes the model predicts a class that is not in REF
#         logging.warning("Warning: %s", ex)
#         ref_s = set([x[1] for x in ref_slots])
#         hyp_s = set([x[1] for x in hyp_slots])
#         logging.warning(hyp_s.difference(ref_s))
#         results = {"total": {"f": 0}}

#     report_intent = classification_report(ref_intents, hyp_intents, zero_division=False, output_dict=True)

#     # Log the reuslts of the evaluation
#     logging.debug("Intent Report: %s", report_intent)
#     logging.debug("Slot Report: %s", results)

#     return results, report_intent, loss_array

from torch.cuda.amp import autocast, GradScaler


def eval_loop(data, criterion_slots, criterion_intents, model, tokenizer, lang):
    model.eval()
    loss_array = []

    ref_intents = []
    hyp_intents = []

    ref_slots = []
    hyp_slots = []

    with torch.no_grad():  # It used to avoid the creation of computational graph
        for sample in data:
            slots, intents = model(
                sample["input_ids"],
                sample["attention_mask"],
                sample["token_type_ids"],
            )
            loss_intent = criterion_intents(intents, sample["intents"])
            loss_slot = criterion_slots(slots, sample["slots"])
            loss = loss_intent + loss_slot
            loss_array.append(loss.item())

            # Intent inference
            out_intents = [
                lang.id2intent[x] for x in torch.argmax(intents, dim=1).tolist()
            ]

            gt_intents = [lang.id2intent[x] for x in sample["intents"].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)

            # Slot inference
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                # Decode the tokens using BERT tokenizer
                tokens = tokenizer.convert_ids_to_tokens(
                    sample["input_ids"][id_seq]
                )
                
                gt_slots = [
                    lang.id2slot[elem] for elem in sample["slots"][id_seq].tolist()
                ]

                # Filter out special tokens
                filtered_tokens = []
                filtered_gt_slots = []
                filtered_pred_slots = []
                for i, token in enumerate(tokens):
                    if token not in tokenizer.all_special_tokens:
                        filtered_tokens.append(token)
                        filtered_gt_slots.append(gt_slots[i])
                        filtered_pred_slots.append(seq[i].item())

                ref_slots.append(
                    [
                        (filtered_tokens[i], filtered_gt_slots[i])
                        for i in range(len(filtered_tokens))
                    ]
                )
                hyp_slots.append(
                    [
                        (filtered_tokens[i], lang.id2slot[filtered_pred_slots[i]])
                        for i in range(len(filtered_tokens))
                    ]
                )

    logging.debug(f"REFERENCE: {ref_slots[0]}")
    logging.debug(f"HYPOTHEIS: {hyp_slots[0]}")
    # exit()
    # TODO: improve the evaluation as it is not working properly
    try:
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        logging.warning("Warning: %s", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        logging.warning(hyp_s.difference(ref_s))
        results = {"total": {"f": 0}}

    report_intent = classification_report(
        ref_intents, hyp_intents, zero_division=False, output_dict=True
    )

    # Log the reuslts of the evaluation
    logging.debug("Intent Report: %s", report_intent)
    logging.debug("Slot Report: %s", results)

    return results, report_intent, loss_array


def pad_list_of_lists(lists, pad_value=np.nan):
    max_length = max(len(lst) for lst in lists)
    return [lst + [pad_value] * (max_length - len(lst)) for lst in lists]

def train(
    model_config: dict,
    optimizer_config: dict,
    train_config: dict,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    test_loader: DataLoader,
    lang,
    w2id,
    slot2id,
    intent2id,
    writer,
    PAD_TOKEN,
    name: str,
    device: str,
    tokenizer,
):
    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)

    model_config["out_slot"] = out_slot
    model_config["out_int"] = out_int

    slot_f1s, intent_acc = [], []
    all_losses_train = []
    all_losses_dev = []

    for run in tqdm(range(train_config["runs"]), desc="Runs"):

        # Get model
        # model = ModelBert(model_config, vocab_len, name, pad_index=PAD_TOKEN).to(device)
        model = ModelBert(
            bert_model_name="bert-base-uncased", out_slot=out_slot, out_int=out_int, name=name
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=optimizer_config["lr"])
        criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
        criterion_intents = nn.CrossEntropyLoss()

        # scaler = torch.cuda.amp.GradScaler()

        run_patience = train_config["patience"]

        best_model = None
        best_f1 = 0

        losses_train = []
        losses_dev = []
        sampled_epochs = []

        for x in tqdm(
            range(train_config["n_epochs"]), desc=f"Run {run+1}", leave=False
        ):

            loss = train_loop(
                data=train_loader,
                optimizer=optimizer,
                criterion_slots=criterion_slots,
                criterion_intents=criterion_intents,
                model=model,
                clip=train_config["clip"],
                # scaler=scaler,
            )

            if x % 5 == 0:  # We check the performance every 5 epochs
                sampled_epochs.append(x)
                losses_train.append(np.asarray(loss).mean())
                results_dev, intent_res, loss_dev = eval_loop(
                    dev_loader,
                    criterion_slots,
                    criterion_intents,
                    model,
                    tokenizer,
                    lang,
                )
                losses_dev.append(np.asarray(loss_dev).mean())

                f1 = results_dev["total"]["f"]

                if f1 > best_f1:
                    best_f1 = f1
                    best_model = copy.deepcopy(model).to("cpu")

                    run_patience = train_config["patience"]
                else:
                    run_patience -= 1
                if run_patience <= 0:  # Early stopping with patience
                    break

        results_test, intent_test, _ = eval_loop(
            test_loader, criterion_slots, criterion_intents, model, tokenizer, lang
        )

        # # Evaluate the model
        # predictions, ground_truths = evaluate(model, test_loader, device)

        # # Print some example predictions and ground truths
        # for i in range(min(10, len(predictions))):  # Print first 10 examples
        #     print(f"Example {i+1}:")
        #     print("Predicted:", predictions[i])
        #     print("Ground Truth:", ground_truths[i])
        #     print()  # Blank line for separation

        intent_acc.append(intent_test["accuracy"])
        slot_f1s.append(results_test["total"]["f"])

        all_losses_train.append(losses_train)
        all_losses_dev.append(losses_dev)

    # Pad lists to the same length
    all_losses_train = pad_list_of_lists(all_losses_train)
    all_losses_dev = pad_list_of_lists(all_losses_dev)

    # Convert to numpy arrays and compute statistics, ignoring nan values
    all_losses_train = np.array(all_losses_train)
    all_losses_dev = np.array(all_losses_dev)

    avg_losses_train = np.nanmean(all_losses_train, axis=0)
    std_losses_train = np.nanstd(all_losses_train, axis=0)
    avg_losses_dev = np.nanmean(all_losses_dev, axis=0)
    std_losses_dev = np.nanstd(all_losses_dev, axis=0)

    slot_f1s = np.array(slot_f1s)
    intent_acc = np.array(intent_acc)

    for epoch, (
        avg_loss_train,
        std_loss_train,
        avg_loss_dev,
        std_loss_dev,
    ) in enumerate(
        zip(avg_losses_train, std_losses_train, avg_losses_dev, std_losses_dev)
    ):
        writer.add_scalar("Loss/Train_avg", avg_loss_train, epoch * 5)
        writer.add_scalar("Loss/Train_std", std_loss_train, epoch * 5)
        writer.add_scalar("Loss/Dev_avg", avg_loss_dev, epoch * 5)
        writer.add_scalar("Loss/Dev_std", std_loss_dev, epoch * 5)

    writer.add_scalar("Metrics/Slot_F1_avg", slot_f1s.mean())
    writer.add_scalar("Metrics/Slot_F1_std", slot_f1s.std())
    writer.add_scalar("Metrics/Intent_Acc_avg", intent_acc.mean())
    writer.add_scalar("Metrics/Intent_Acc_std", intent_acc.std())

    print("Slot F1", round(slot_f1s.mean(), 3), "+-", round(slot_f1s.std(), 3))
    print("Intent Acc", round(intent_acc.mean(), 3), "+-", round(intent_acc.std(), 3))

    # Saving the model:
    if best_model is not None:
        PATH = f"bin/{best_model.name}.pt"
        saving_object = {
            "epoch": x,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "w2id": w2id,
            "slot2id": slot2id,
            "intent2id": intent2id,
        }
        torch.save(saving_object, PATH)

    plt.figure(num=3, figsize=(8, 5)).patch.set_facecolor("white")
    plt.title("Train and Dev Losses")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.plot(sampled_epochs, losses_train, label="Train loss")
    plt.plot(sampled_epochs, losses_dev, label="Dev loss")
    plt.legend()
    plt.show()
