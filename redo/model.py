import torch.nn as nn


class LM_RNN(nn.Module):
    def __init__(
        self,
        emb_size: int,
        hid_size: int,
        output_size: int,
        pad_index: int = 0,
        emb_dropout: int = 0,
        out_dropout: int = 0,
        n_layers: int = 1,
    ):
        super().__init__()

        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.rnn = nn.RNN(emb_size, hid_size, n_layers)
        self.out_dropout = nn.Dropout(out_dropout)
        self.output = nn.Linear(hid_size, output_size)

        self.name = f"{self.__class__.__name__}_emb_{emb_size}_hid_{hid_size}_edr_{emb_dropout}_odr_{out_dropout}_lay_{n_layers}"

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        emb = self.emb_dropout(emb)
        rnn, _ = self.rnn(emb)
        out = self.out_dropout(rnn)
        out = self.output(out)
        out = out.permute(0, 2, 1)
        return out


class LM_LSTM(nn.Module):
    def __init__(
        self,
        emb_size: int,
        hid_size: int,
        output_size: int,
        pad_index: int = 0,
        emb_dropout: int = 0,
        out_dropout: int = 0,
        n_layers: int = 1,
    ):
        super().__init__()

        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.rnn = nn.LSTM(emb_size, hid_size, n_layers, bidirectional=False)
        self.out_dropout = nn.Dropout(out_dropout)
        self.output = nn.Linear(hid_size, output_size)

        self.name = f"{self.__class__.__name__}_emb_{emb_size}_hid_{hid_size}_edr_{emb_dropout}_odr_{out_dropout}_lay_{n_layers}"

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        emb = self.emb_dropout(emb)
        rnn, _ = self.rnn(emb)
        out = self.out_dropout(rnn)
        out = self.output(out)
        out = out.permute(0, 2, 1)
        return out
