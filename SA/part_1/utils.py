# Add functions or classes used for data loading and preprocessing
import os
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import torch
from transformers import BertTokenizer, get_linear_schedule_with_warmup


def load_data(path):
    def convert_tags(tags: list):
        """
        Extraction of the aspect terms only
        """
        new_tags = []

        n_tags = len(tags)
        i = 0
        while i < n_tags:
            if tags[i] == "T-POS" or tags[i] == "T-NEG" or tags[i] == "T-NEU":
                new_tags.append("T")
                i += 1
            else:
                new_tags.append("O")
                i += 1

        return new_tags

    raw_data = []

    with open(path, encoding="utf-8", mode="r") as f:
        lines = f.readlines()

        error = 0

        for line in lines:
            try:
                _, tags = line.split("####")

                tags = tags.split(" ")

                all_body, all_tags = [], []
                for tag in tags:
                    a = tag.split("=")

                    # if `\n` is present in the tag, remove it
                    if "\n" in a[1]:
                        a[1] = a[1].replace("\n", "")

                    all_body.append(a[0])
                    all_tags.append(a[1])

                if len(all_body) != len(all_tags):
                    error += 1
                    print(all_body)
                    print(all_tags)

                all_body = " ".join(all_body)
                all_tags = convert_tags(all_tags)

                raw_data.append({"utterance": all_body, "slot": " ".join(all_tags)})
            except Exception as e:
                error += 1
                print(f"Error processing line: {line}")
                print(e)

        if error != 0:
            print("Error: ", error)

        return raw_data


def split_sets(tmp_train_raw):
    portion = 0.15

    intents = [x["slot"] for x in tmp_train_raw]  # We stratify on intents
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


def tokenize_and_preserve_labels(sentence, text_labels, tokenizer):
    text_labels = text_labels.split()
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence.split(), text_labels):
        tokenized_word = tokenizer.tokenize(word)

        tokenized_sentence.extend(tokenized_word)

        labels.extend([label])
        labels.extend(["pad"] * (len(tokenized_word) - 1))

    return tokenized_sentence, labels


def get_data_and_mapping(
    train_path=os.path.join("..", "dataset", "laptop14_train.txt"),
    test_path=os.path.join("..", "dataset", "laptop14_test.txt"),
):
    tmp_train_raw = load_data(train_path)
    # print(tmp_train_raw)
    # exit()
    test_raw = load_data(test_path)

    train_raw, dev_raw = split_sets(tmp_train_raw)

    logging.info("Train size: %d", len(train_raw))
    logging.info("Dev size: %d", len(dev_raw))
    logging.info("Test size: %d", len(test_raw))

    slots_set = set()
    for phrases in [train_raw, dev_raw, test_raw]:
        for phrase in phrases:
            for slot in phrase["slot"].split(" "):
                slots_set.add(slot)

    slots2id = {"pad": 0}
    id2slots = {0: "pad"}

    for slot in slots_set:
        slots2id[slot] = len(slots2id)
        id2slots[len(id2slots)] = slot

    return train_raw, dev_raw, test_raw, slots2id, id2slots


def tokenize_data(raw_data, tokenizer):
    processed_data = []
    for dset in raw_data:
        tokenized_set = {}
        tokenized_set["raw_slots"] = dset["slot"]
        tokenized_set["raw_utterance"] = dset["utterance"]

        tokenized_sentence, adapted_labels = tokenize_and_preserve_labels(
            dset["utterance"], dset["slot"], tokenizer
        )

        tokenized_set["tokenized_utterance"] = tokenized_sentence
        tokenized_set["tokenized_slots"] = adapted_labels

        processed_data.append(tokenized_set)
    return processed_data


def encode_data(tokenized_data, tokenizer, slots2id):
    encoded_data = []
    for dset in tokenized_data:
        encoded_set = {}

        encoded_set["raw_slots"] = dset["raw_slots"]
        encoded_set["raw_utterance"] = dset["raw_utterance"]
        encoded_set["tokenized_utterance"] = dset["tokenized_utterance"]
        encoded_set["tokenized_slots"] = dset["tokenized_slots"]

        # Encode the tokenized utterance
        encoded_set["encoded_utterance"] = tokenizer.encode_plus(
            dset["tokenized_utterance"], add_special_tokens=False
        )

        # Encode the tokenized slots
        encoded_set["encoded_slots"] = [
            # slots2id[slot] if slot != 0 else 0 for slot in dset["tokenized_slots"]
            slots2id[slot]
            for slot in dset["tokenized_slots"]
        ]

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
            logging.debug("\n\n")

    if err != 0:
        logging.error("There are %d errors in the preprocessing", err)
        raise ValueError("There are errors in the preprocessing")


def preprocess_data(raw_data, tokenizer, slots2id):
    """
    Preprocess the raw data by tokenizing and encoding it.

    Args:
    - raw_data: list of dictionaries
    - tokenizer: BertTokenizer
    - slots2id: dictionary mapping slots to numerical ids
    - intent2id: dictionary mapping intents to numerical ids

    Note: the correctness of slots2id and intent2id is assumed.

    Each element in processed_train is a dictionary with the following keys:
    - raw_slots
    - raw_utterance
    - tokenized_utterance
    - tokenized_slots
    - encoded_utterance
    - encoded_slots
    """

    # Tokenize `utterance` and `slots` with sub-token labelling. The subtoken is labelled with the same label
    processed_data = tokenize_data(raw_data, tokenizer)

    # Encode the tokenized data
    encoded_data = encode_data(processed_data, tokenizer, slots2id)

    # Check if the preprocessing is correct
    check_preprocessing(encoded_data)

    return encoded_data


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_fn(data):
    device = get_device()

    # Get the max length of the input sequence
    max_len = max([len(sentence) for sentence, _, _, _ in data])

    # PAD all the input sequences to the max length
    slots_len = torch.tensor([len(slots) for _, _, _, slots in data]).to(device)
    input_ids = torch.tensor(
        [sentence + [0] * (max_len - len(sentence)) for sentence, _, _, _ in data]
    ).to(device)
    attention_mask = torch.tensor(
        [[1] * len(mask) + [0] * (max_len - len(mask)) for _, mask, _, _ in data]
    ).to(device)
    token_type_ids = torch.tensor(
        [
            token_type_ids + [0] * (max_len - len(token_type_ids))
            for _, _, token_type_ids, _ in data
        ]
    ).to(device)
    slots = torch.tensor(
        [slots + [0] * (max_len - len(slots)) for _, _, _, slots in data]
    ).to(device)

    return {
        "slots_len": slots_len,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "slots": slots,
    }


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

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return (
            self.input[idx],
            self.attention_mask[idx],
            self.token_type_ids[idx],
            self.slots[idx],
        )


SMALL_POSITIVE_CONST = 1e-4


def tag2ts(ts_tag_sequence):
    """
    Transform ts tag sequence to target spans
    :param ts_tag_sequence: tag sequence with 'T' and 'O'
    :return: List of (start, end) tuples for target spans
    """
    n_tags = len(ts_tag_sequence)
    ts_sequence = []
    beg, end = -1, -1
    for i in range(n_tags):
        ts_tag = ts_tag_sequence[i]
        if ts_tag == "T":
            if beg == -1:
                beg = i
            end = i
        elif ts_tag == "O" and beg != -1:
            ts_sequence.append((beg, end))
            beg, end = -1, -1
    if beg != -1:
        ts_sequence.append((beg, end))
    return ts_sequence


def match_ts(gold_ts_sequence, pred_ts_sequence):
    """
    Calculate the number of correctly predicted target spans
    :param gold_ts_sequence: gold standard target spans
    :param pred_ts_sequence: predicted target spans
    :return: hit_count, gold_count, pred_count
    """
    hit_count = 0
    gold_count = len(gold_ts_sequence)
    pred_count = len(pred_ts_sequence)

    for t in pred_ts_sequence:
        if t in gold_ts_sequence:
            hit_count += 1

    return hit_count, gold_count, pred_count


def evaluate_ts(gold_ts, pred_ts):
    """
    Evaluate the model performance for the binary tagging task
    :param gold_ts: gold standard ts tags
    :param pred_ts: predicted ts tags
    :return: Precision, Recall, F1 scores

    Adapted from:
    https://github.com/lixin4ever/E2E-TBSA/blob/master/evals.py#L51
    """
    assert len(gold_ts) == len(pred_ts)
    n_samples = len(gold_ts)

    n_tp_ts, n_gold_ts, n_pred_ts = 0, 0, 0

    for i in range(n_samples):
        g_ts_sequence = tag2ts(ts_tag_sequence=gold_ts[i])
        p_ts_sequence = tag2ts(ts_tag_sequence=pred_ts[i])

        hit_ts_count, gold_ts_count, pred_ts_count = match_ts(
            gold_ts_sequence=g_ts_sequence, pred_ts_sequence=p_ts_sequence
        )

        n_tp_ts += hit_ts_count
        n_gold_ts += gold_ts_count
        n_pred_ts += pred_ts_count

    precision = float(n_tp_ts) / (n_pred_ts + SMALL_POSITIVE_CONST)
    recall = float(n_tp_ts) / (n_gold_ts + SMALL_POSITIVE_CONST)
    f1_score = 2 * precision * recall / (precision + recall + SMALL_POSITIVE_CONST)

    return precision, recall, f1_score
