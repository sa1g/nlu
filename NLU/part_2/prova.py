from collections import Counter
import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from transformers import BertModel, BertTokenizer
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
from conll import evaluate


def load_data(path):
    dataset = []
    with open(path, "r") as file:
        dataset = json.loads(file.read())

    return dataset


class IntentSlotModel(nn.Module):
    def __init__(self, bert_model, num_intent_labels, num_slot_labels):
        super().__init__()
        self.bert = bert_model
        self.intent_classifier = nn.Linear(
            bert_model.config.hidden_size, num_intent_labels
        )
        self.slot_classifier = nn.Linear(bert_model.config.hidden_size, num_slot_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output

        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)

        return intent_logits, slot_logits


def create_dataloader(raw):
    inputs = [t["utterance"] for t in raw]
    intent_labels = [intent_label_map[t["intent"]] for t in raw]
    slot_labels = [
        [slot_label_map[label] for label in t["slots"].split(" ")] for t in raw
    ]

    # TRAIN size: 4480
    # DEV size: 498
    # TEST size: 893
    # # Words: 864
    # # Slots: 129
    # # Intent: 26

    # print(f"intent_labels: {intent_labels}")
    # print("intent_labels map: ", intent_label_map)

    # for i, _ in enumerate(raw):
    #     # print(f"raw: {raw[i]['intent']}, intent: {intent_labels[i]}")
    #     print(f"raw: {raw[i]['slots']}, slots: {slot_labels[i]}")

    #     # exit()

    # # print(f"slot labels: {slot_labels}")
    # # print("slot labels map: ", slot_label_map)
    # exit()

    # Tokenize inputs
    encoded_inputs = tokenizer(
        inputs,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=64,
    )
    input_ids = encoded_inputs["input_ids"]
    attention_mask = encoded_inputs["attention_mask"]

    # Pad slot labels to match the length of input_ids
    max_len = input_ids.shape[1]
    padded_slot_labels = []
    for label in slot_labels:
        padded_label = label + [slot_label_map["<PAD>"]] * (
            max_len - len(label)
        )  # Padding with "O"
        padded_slot_labels.append(padded_label)
    slot_labels = torch.tensor(padded_slot_labels)

    # Convert intent labels to tensors
    intent_labels = torch.tensor(intent_labels)

    # print(f"INTENTION LABELS: {intent_labels[0]}")
    # print(f"SLOT LABELS: {slot_labels[0]}")
    # exit()
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    intent_labels = intent_labels.to(device)
    slot_labels = slot_labels.to(device)

    dataset = TensorDataset(input_ids, attention_mask, intent_labels, slot_labels)
    dataloader = DataLoader(dataset, batch_size=64)
    return dataloader


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #####
    # LOAD DATA
    #####
    tmp_train_raw = load_data(os.path.join("../dataset", "ATIS", "train.json"))
    test_raw = load_data(os.path.join("../dataset", "ATIS", "test.json"))

    #####
    # TRAIN TEST DEV SPLIT
    #####
    portion = 0.10

    intents = [x["intent"] for x in tmp_train_raw]  # We stratify on intents
    count_y = Counter(intents)

    labels = []
    inputs = []
    mini_train = []

    for id_y, y in enumerate(intents):
        if count_y[y] > 1:  # If some intents occurs only once, we put them in training
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(tmp_train_raw[id_y])
    # Random Stratify
    X_train, X_dev, y_train, y_dev = train_test_split(
        inputs,
        labels,
        test_size=portion,
        random_state=42,
        shuffle=True,
        stratify=labels,
    )
    X_train.extend(mini_train)
    train_raw = X_train
    dev_raw = X_dev

    y_test = [x["intent"] for x in test_raw]

    print("TRAIN size:", len(train_raw))
    print("DEV size:", len(dev_raw))
    print("TEST size:", len(test_raw))

    #####
    # COUNT WORDS
    #####
    PAD_TOKEN = 0

    # Count the number of unique words, slot and intent tags
    words_set = set()
    slot_set = set()
    slot_set.add("<PAD>")
    intent_set = set()

    for example in train_raw:
        for word in example["utterance"].split():
            if word not in words_set:
                words_set.add(word)
        for slot in example["slots"].split():
            if slot not in slot_set:
                slot_set.add(slot)
        if example["intent"] not in intent_set:
            intent_set.add(example["intent"])

    for example in dev_raw:
        # for word in example['utterance'].split():
        #     if word not in words_set:
        #         words_set.add(word)
        for slot in example["slots"].split():
            if slot not in slot_set:
                slot_set.add(slot)
        if example["intent"] not in intent_set:
            intent_set.add(example["intent"])

    for example in test_raw:
        # for word in example['utterance'].split():
        #     if word not in words_set:
        #         words_set.add(word)
        for slot in example["slots"].split():
            if slot not in slot_set:
                slot_set.add(slot)
        if example["intent"] not in intent_set:
            intent_set.add(example["intent"])

    num_words = len(words_set)
    num_intent_labels = len(intent_set)
    num_slot_labels = len(slot_set)

    print("# Words:", num_words)
    print("# Slots:", num_slot_labels)
    print("# Intent:", num_intent_labels)

    #####
    # TOKENIZE DATA
    #####
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    slot_labels_list = list(slot_set)
    intent_labels_list = list(intent_set)

    slot_label_map = {label: idx for idx, label in enumerate(slot_labels_list)}

    tmp = list(slot_label_map.keys())[list(slot_label_map.values()).index(0)]
    tmp_value = slot_label_map["<PAD>"]
    slot_label_map["<PAD>"] = 0
    slot_label_map[tmp] = tmp_value

    intent_label_map = {label: idx for idx, label in enumerate(intent_labels_list)}

    train_dataloader = create_dataloader(train_raw)
    dev_dataloader = create_dataloader(dev_raw)

    #####
    # MODEL
    #####
    bert_model = BertModel.from_pretrained("bert-base-uncased")

    model = IntentSlotModel(bert_model, num_intent_labels, num_slot_labels)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

    # Loss functions
    # TODO: in intention maybe ignore padding, start, end sequence
    intent_loss_fn = nn.CrossEntropyLoss()
    slot_loss_fn = nn.CrossEntropyLoss()

    #####
    # Train
    #####
    def calculate_loss(
        intent_loss_fn,
        slot_loss_fn,
        intent_logits,
        slot_logits,
        intent_labels,
        slot_labels,
    ):
        intent_loss = intent_loss_fn(intent_logits, intent_labels)
        slot_loss = slot_loss_fn(
            slot_logits.view(-1, num_slot_labels), slot_labels.view(-1)
        )
        return intent_loss + slot_loss

    def train_loop(model, data, optimizer, intent_loss_fn, slot_loss_fn, clip=5):
        model.train()

        input_ids, attention_mask, intent_labels, slot_labels = data

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        intent_labels = intent_labels.to(device)
        slot_labels = slot_labels.to(device)

        # Forward
        intent_logits, slot_logits = model(input_ids, attention_mask)

        # intent_logits = torch.argmax(intent_logits, dim=1)
        # slot_logits = torch.argmax(slot_logits, dim=2)

        # print(f"intent_logits: {intent_logits}")
        # print(f"intent_labels: {intent_labels}")
        # print(f"slot_logits: {slot_logits}")
        # print(f"slot_labels: {slot_labels}")

        # print(f"intent_logits.shape: {intent_logits.shape}")
        # print(f"intent_labels.shape: {intent_labels.shape}")
        # print(f"slot_logits.shape: {slot_logits.shape}")
        # print(f"slot_labels.shape: {slot_labels.shape}")

        # Losses
        loss = calculate_loss(
            intent_loss_fn,
            slot_loss_fn,
            intent_logits,
            slot_logits,
            intent_labels,
            slot_labels,
        )
        # exit()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def eval_loop(model, data, intent_loss_fn, slot_loss_fn):
        model.eval()
        total_loss = []

        with torch.no_grad():
            input_ids, attention_mask, intent_labels, slot_labels = data

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            intent_labels = intent_labels.to(device)
            slot_labels = slot_labels.to(device)

            # Forward
            intent_logits, slot_logits = model(input_ids, attention_mask)
            total_loss.append(
                calculate_loss(
                    intent_loss_fn,
                    slot_loss_fn,
                    intent_logits,
                    slot_logits,
                    intent_labels,
                    slot_labels,
                ).item()
            )

            intent_p = torch.argmax(intent_logits, dim=1)
            slot_p = torch.argmax(slot_logits, dim=2)

            print("========== EVAL ===========")
            print(f"intent_p: {intent_p[0]}")
            print(f"intent_labels: {intent_labels[0]}")
            print(f"slot_p: {slot_p[0]}")
            print(f"slot_labels: {slot_labels[0]}")

            # Intention reference
            accuracy_intention = classification_report(
                intent_labels.to("cpu"),
                intent_p.to("cpu"),
                output_dict=True,
                zero_division=False,
            )
            exit()
            # Slot reference
            # TODO: understand hoow to make it work AAAAAaaaAAAAaaAAAAaaaAA 
            print(slot_labels.to("cpu").tolist())
            print(slot_p.to("cpu").tolist())

            # try:
            f1_slot = evaluate(slot_labels, slot_p)
            # except Exception as ex:
            # print("Error: ", ex)
            # exit()
            exit()

        # accuracy_intention = 0 # classification report
        print("====== ACCURACY INTENTION: ", accuracy_intention["accuracy"])

        return np.mean(total_loss), f1_slot, accuracy_intention

    model.train()
    for epoch in range(3):  # Example: 3 epochs
        for index, batch in enumerate(train_dataloader):
            loss = train_loop(model, batch, optimizer, intent_loss_fn, slot_loss_fn)

            if index % 5 == 0:
                eval_loop(model, batch, intent_loss_fn, slot_loss_fn)

            print(f"Epoch: {epoch}, Loss: {loss :.4f}")
