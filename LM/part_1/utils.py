# Add functions or classes used for data loading and preprocessing
from typing import List
import torch
from torch.utils import data
from torch.utils.data import DataLoader
from functools import partial

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class Lang:
    """
    This class computes and stores the vocabulary.
    Words are mapped to ids and vice versa.
    """

    def __init__(self, corpus, special_tokens=None):
        self.word2id = self.__get_vocab(corpus, special_tokens)
        self.id2word = {v: k for k, v in self.word2id.items()}

    def __get_vocab(self, corpus, special_tokens=None) -> dict:
        """
        Creates a dictionary that maps unique words to an unique id.

        Args:
            corpus (List(str)): list of phrases. Each phrase may end with `special_tokens[@]`.
            special_tokens (List(str), optional): List of special tokens. Defaults to None.

        Returns:
            dict: mapping between words to ids.
        """
        output = {}
        i = 0

        for st in special_tokens:
            output[st] = i
            i += 1

        for sentence in corpus:
            for w in sentence.split():
                if w not in output:
                    output[w] = i
                    i += 1

        return output


class PennTreeBank(data.Dataset):
    """
    Words-ids dataset manager.
    """

    def __init__(self, corpus: List[str], lang: Lang):
        self.source = []
        self.target = []

        for sentence in corpus:
            self.source.append(
                sentence.split()[0:-1]
            )  # We get from the first token till the second-last token

            self.target.append(
                sentence.split()[1:]
            )  # We get from the second token till the last token
            # See example in section 6.2

        self.source_ids = self.mapping_seq(self.source, lang)
        self.target_ids = self.mapping_seq(self.target, lang)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        src = torch.LongTensor(self.source_ids[idx])
        trg = torch.LongTensor(self.target_ids[idx])
        sample = {"source": src, "target": trg}
        return sample

    # Auxiliary methods

    def mapping_seq(self, data: List[List[str]], lang: Lang) -> List[List[id]]:
        """
        Map sequences of tokens to the corresponding class in Lang

        Args:
            data (List[List[str]]): corpus data splitted word by word. Phrase separation is still present.
            lang (Lang): active dictionary

        Returns:
            List[List[id]]: Mapped tokens
        """

        res = []

        for seq in data:
            tmp_seq = []
            for x in seq:
                if x in lang.word2id:
                    tmp_seq.append(lang.word2id[x])
                else:
                    print("OOV found!")
                    print(
                        "You have to deal with that"
                    )  # PennTreeBank doesn't have OOV but "Trust is good, control is better!"
                    break
            res.append(tmp_seq)

        return res


def read_file(path: str, eos_token: str = "<eos>") -> List[str]:
    """Load the corpus from a file.

    Args:
        path (str): file path and filename. E.g. "/dataset/data.txt"
        eos_token (str, optional): End of sentence token. Defaults to "<eos>".

    Returns:
        List[str]: list of phrases. Each phrase ends with `eos_token`.
    """
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line.strip() + " " + eos_token)
    return output


def get_vocab(corpus, special_tokens=None):
    """Vocab with tokens to ids

    Args:
        corpus (_type_): _description_
        special_tokens (list, optional): _description_. Defaults to [].

    Returns:
        _type_: _description_
    """
    output = {}
    i = 0
    for st in special_tokens:
        output[st] = i
        i += 1
    for sentence in corpus:
        for w in sentence.split():
            if w not in output:
                output[w] = i
                i += 1
    return output


def collate_fn(data: PennTreeBank, pad_token: dict) -> dict:
    """
    Collate function for batching variable-length sequences for training using PyTorch's DataLoader.

    This function sorts the input data by the length of the source sequences in descending order, pads the sequences to make them of equal length within a batch, and moves the processed data to the specified computational device.

    Args:
        data (PennTreeBank): A PennTreeBank object containing the input data.
        pad_token (dict): A dictionary mapping pad tokens to their corresponding IDs.

    Returns:
        dict: A dictionary containing the processed batched data, including the padded source and target sequences, the total number of tokens in the batch, and other relevant information.
    """

    def merge(sequences):
        """
        merge from batch * sent_len to batch * max_len
        """

        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths) == 0 else max(lengths)

        # Pad token is zero in this case
        # Create a matrix full of PAD_TOKEN (i.e. 0) with shape
        # [batch_size, maximum length of a sequence]
        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(pad_token)

        for i, seq in enumerate(sequences):
            end = lengths[i]
            # Copy each sequence into the matrix
            padded_seqs[i, :end] = seq

        # Remove these tensors from the computational graph
        padded_seqs = padded_seqs.detach()

        return padded_seqs, lengths

    # Sort data by seq lengths
    data.sort(key=lambda x: len(x["source"]), reverse=True)
    new_item = {}

    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    source, _ = merge(new_item["source"])
    target, lengths = merge(new_item["target"])

    new_item["source"] = source.to(DEVICE)
    new_item["target"] = target.to(DEVICE)
    new_item["number_tokens"] = sum(lengths)
    return new_item


def get_data_loaders(
    train_batch_size: int = 128,
    eval_batch_size: int = 128,
    test_batch_size=128,
    num_workers=0,
):
    """
    Retrieves the data loaders for training, evaluation, and testing.

    Args:
        train_batch_size (int, optional): The batch size for the training data loader. Defaults to 128.
        eval_batch_size (int, optional): The batch size for the evaluation data loader. Defaults to 128.
        test_batch_size (int, optional): The batch size for the testing data loader. Defaults to 128.

    Returns:
        tuple: A tuple containing the following elements:
            - lang (Lang): The language object containing the vocabulary.
            - train_loader (DataLoader): The data loader for the training dataset.
            - eval_loader (DataLoader): The data loader for the evaluation dataset.
            - test_loader (DataLoader): The data loader for the testing dataset.
    """

    train_raw = read_file("dataset/ptb.train.txt")
    dev_raw = read_file("dataset/ptb.valid.txt")
    test_raw = read_file("dataset/ptb.test.txt")

    # Define the dictionary.
    lang = Lang(train_raw, ["<pad>", "<eos>"])

    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    # Dataloader instantiation

    # collate_fn (Callable, optional): merges a list of samples to form a
    # mini-batch of Tensor(s). Used when using batched loading from a map-style dataset.

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),
        shuffle=True,
        num_workers=num_workers,
    )
    eval_loader = DataLoader(
        dev_dataset,
        batch_size=eval_batch_size,
        collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),
        num_workers=num_workers,
    )

    return lang, train_loader, eval_loader, test_loader
