from torch.utils import data
import torch

DEVICE = "cuda:0"

def read_file(path: str, eos_token: str = "<eos>") -> list[str]:
    """
    Read a file and strip each letter inside of it.
    """
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line.strip() + " " + eos_token)
    return output


def get_vocab(corpus: list[str], special_tokens: list[str] = []) -> dict[int]:
    """"
    Analize a corpus (in this case a list of letters with 
    <eos> flag) and assign in an unique way an id to each other.

    This transforms the corpus from a literal "dictionary" 
    to a numeric, trainable one.
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


class Lang:
    """
    Class to manage the embedding from word to number and vice versa.
    """

    def __init__(self, corpus: list[str], special_tokens: list[str] = []):
        self.word2id = self.get_vocab(corpus, special_tokens)
        self.id2word = {v: k for k, v in self.word2id.items()}

    def get_vocab(self, corpus: list[str], special_tokens: list[str] = []) -> dict[int]:
        """"
        Analize a corpus (in this case a list of letters with 
        <eos> flag) and assign in an unique way an id to each other.

        This transforms the corpus from a literal "dictionary" 
        to a numeric, trainable one.
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
    def __init__(self, corpus:list[str], lang: Lang):
        self.source = []
        self.target = []

        for sentence in corpus:
            # Get from the first token till the second-last 
            self.source.append(sentence.split()[0:-1])
            # Get from the second token till the last token
            self.target.append(sentence.split()[1:])

        self.source_ids = self.mapping_seq(self.source, lang)
        self.target_ids = self.mapping_seq(self.target, lang)

    def __len__(self):
        return len(self.source)
    
    def __getitem__(self, index) -> dict[torch.LongTensor]:
        src = torch.LongTensor(self.source_ids[index])
        trg = torch.LongTensor(self.target_ids[index])

        sample = {"source": src, "target": trg}
        return sample

    def mapping_seq(self, data: list[str, int] , lang: Lang) -> list[int]:
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq:
                if x in lang.word2id:
                    tmp_seq.append(lang.word2id[x])
                else:
                    print("OOV found!")
                    print("Manage it yourself GLHF")
                    break
            res.append(tmp_seq)
        return res
    

def collate_fn(data, pad_token):
    def merge(sequences):
        """
        merge from batch * sent_len to batch * max_len
        """
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths) == 0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(
            len(sequences), max_len).fill_(pad_token)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq  # We copy each sequence into the matrix
        padded_seqs = (
            padded_seqs.detach()
        )  # We remove these tensors from the computational graph
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
