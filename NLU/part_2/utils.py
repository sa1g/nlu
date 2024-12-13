import json
import logging
import os
from collections import Counter
from sklearn.model_selection import train_test_split
import torch


def load_data(path):
    dataset = []
    with open(path, "r") as file:
        dataset = json.loads(file.read())

    return dataset


def split_sets(tmp_train_raw):
    portion = 0.10

    # We stratify on intents
    intents = [x["intent"] for x in tmp_train_raw]
    count_y = Counter(intents)

    labels = []
    inputs = []
    mini_train = []

    # If some intents occurs only once, we put them in training
    for id_y, y in enumerate(intents):
        if count_y[y] > 1:
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(tmp_train_raw[id_y])

    # Random Stratify
    X_train, X_dev, _, _ = train_test_split(
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

    return train_raw, dev_raw


def get_data_and_mapping(
    train=os.path.join("../dataset", "ATIS", "train.json"),
    test=os.path.join("../dataset", "ATIS", "test.json"),
):

    tmp_train_raw = load_data(train)
    test_raw = load_data(test)

    train_raw, dev_raw = split_sets(tmp_train_raw)

    logging.info("Train size: %d", len(train_raw))
    logging.info("Dev size: %d", len(dev_raw))
    logging.info("Test size: %d", len(test_raw))

    slots_set = set()
    intents_set = set()

    for phrases in [train_raw, dev_raw, test_raw]:
        for phrase in phrases:
            for slot in phrase["slots"].split():
                slots_set.add(slot)
            intents_set.add(phrase["intent"])

    slots2id = {"pad": 0}
    id2slots = {0: "O"}
    for slot in slots_set:
        slots2id[slot] = len(slots2id)
        id2slots[len(id2slots)] = slot

    intent2id = {}
    id2intent = {}
    for intent in intents_set:
        intent2id[intent] = len(intent2id)
        id2intent[len(id2intent)] = intent

    return train_raw, dev_raw, test_raw, slots2id, id2slots, intent2id, id2intent


def tokenize_and_preserve_labels(sentence, text_labels, tokenizer):
    text_labels = text_labels.split()
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence.split(), text_labels):
        tokenized_word = tokenizer.tokenize(word)
        # n_subwords = len(tokenized_word)

        # tokenized_sentence.extend(tokenized_word)

        # labels.extend([label] * n_subwords)

        # Add the tokenized word to the tokenized sentence
        tokenized_sentence.extend(tokenized_word)

        # Assign the original label to the first subtoken and padding (-100) to the rest
        labels.extend([label])
        labels.extend(["pad"] * (len(tokenized_word) - 1))

    return tokenized_sentence, labels


def tokenize_data(raw_data, tokenizer):
    processed_data = []
    for dset in raw_data:
        tokenized_set = {}
        tokenized_set["raw_intent"] = dset["intent"]
        tokenized_set["raw_slots"] = dset["slots"]
        tokenized_set["raw_utterance"] = dset["utterance"]

        tokenized_sentence, adapted_labels = tokenize_and_preserve_labels(
            dset["utterance"], dset["slots"], tokenizer
        )

        tokenized_set["tokenized_utterance"] = tokenized_sentence
        tokenized_set["tokenized_slots"] = adapted_labels

        processed_data.append(tokenized_set)
    return processed_data


def encode_data(tokenized_data, tokenizer, slots2id, intent2id):
    encoded_data = []
    for dset in tokenized_data:
        encoded_set = {}

        encoded_set["raw_intent"] = dset["raw_intent"]
        encoded_set["raw_slots"] = dset["raw_slots"]
        encoded_set["raw_utterance"] = dset["raw_utterance"]
        encoded_set["tokenized_utterance"] = dset["tokenized_utterance"]
        encoded_set["tokenized_slots"] = dset["tokenized_slots"]

        # Encode the tokenized utterance
        encoded_set["encoded_utterance"] = tokenizer.encode_plus(
            dset["tokenized_utterance"], add_special_tokens=False
        )

        # Encode the tokenized slots
        # encoded_set["encoded_slots"] = [
            # slots2id[slot] if slot != -100 else -100 for slot in dset["tokenized_slots"]
        # ]

        encoded_set["encoded_slots"] = [
            slots2id[slot] for slot in dset["tokenized_slots"] 
        ]

        # Encode the intent
        encoded_set["encoded_intent"] = intent2id[dset["raw_intent"]]

        encoded_data.append(encoded_set)
    return encoded_data


def check_preprocessing(encoded_data):
    err = 0
    for tokenized_data in encoded_data:
        if (
            len(tokenized_data["encoded_utterance"]["input_ids"])
            - len(tokenized_data["encoded_slots"])
        ) != 0:
            err += 1

            logging.debug(tokenized_data["tokenized_utterance"])
            logging.debug(tokenized_data["tokenized_slots"])
            logging.debug(tokenized_data["encoded_utterance"])
            logging.debug(tokenized_data["encoded_slots"])
            logging.debug(tokenized_data["encoded_intent"])

            logging.debug("\n\n")

    if err != 0:
        logging.error("There are %d errors in the preprocessing", err)
        raise ValueError("There are errors in the preprocessing")


def preprocess_data(raw_data, tokenizer, slots2id, intent2id):
    """
    Preprocess the raw data by tokenizing and encoding it.

    Args:
    - raw_data: list of dictionaries
    - tokenizer: BertTokenizer
    - slots2id: dictionary mapping slots to numerical ids
    - intent2id: dictionary mapping intents to numerical ids

    Note: the correctness of slots2id and intent2id is assumed.

    Each element in processed_train is a dictionary with the following keys:
    - raw_intent
    - raw_slots
    - raw_utterance
    - tokenized_utterance
    - tokenized_slots
    - encoded_utterance
    - encoded_slots
    - encoded_intent
    """

    # Tokenize `utterance` and `slots` with sub-token labelling. The subtoken is labelled with the same label
    processed_data = tokenize_data(raw_data, tokenizer)

    # Encode the tokenized data
    encoded_data = encode_data(processed_data, tokenizer, slots2id, intent2id)

    # Check if the preprocessing is correct
    check_preprocessing(encoded_data)

    return encoded_data


class ATISDataset(torch.utils.data.Dataset):
    def __init__(self, processed_data):
        self.input = [data["encoded_utterance"]["input_ids"] for data in processed_data]
        self.attention_mask = [
            data["encoded_utterance"]["attention_mask"] for data in processed_data
        ]
        self.token_type_ids = [
            data["encoded_utterance"]["token_type_ids"] for data in processed_data
        ]
        self.slots = [data["encoded_slots"] for data in processed_data]
        self.intent = [data["encoded_intent"] for data in processed_data]

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return (
            self.input[idx],
            self.attention_mask[idx],
            self.token_type_ids[idx],
            self.slots[idx],
            self.intent[idx],
        )


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_fn(data):
    device = get_device()

    # Get the max length of the input sequence
    max_len = max([len(sentence) for sentence, _, _, _, _ in data])

    # PAD all the input sequences to the max length
    slots_len = torch.tensor([len(slots) for _, _, _, slots, _ in data]).to(device)
    input_ids = torch.tensor(
        [sentence + [0] * (max_len - len(sentence)) for sentence, _, _, _, _ in data]
    ).to(device)
    attention_mask = torch.tensor(
        [[1] * len(mask) + [0] * (max_len - len(mask)) for _, mask, _, _, _ in data]
    ).to(device)
    token_type_ids = torch.tensor(
        [
            token_type_ids + [0] * (max_len - len(token_type_ids))
            for _, _, token_type_ids, _, _ in data
        ]
    ).to(device)
    slots = torch.tensor(
        [slots + [0] * (max_len - len(slots)) for _, _, _, slots, _ in data]
    ).to(device)
    intent = torch.tensor([intent for _, _, _, _, intent in data]).to(device)

    return {
        "slots_len": slots_len,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "slots": slots,
        "intent": intent,
    }
