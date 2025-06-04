from abc import ABC, abstractmethod

import torch.nn as nn


class ModelApi(nn.Module, ABC):
    @abstractmethod
    def __init__(
        self,
        emb_size: int,
        hidden_size: int,
        output_size: int,
        pad_index: int = 0,
        out_dropout: float = 0.0,
        emb_dropout: float = 0.0,
        n_layers: int = 1,
    ):
        super().__init__()


class LM_RNN(ModelApi):
    def __init__(
        self,
        emb_size: int,
        hidden_size: int,
        output_size: int,
        pad_index: int = 0,
        out_dropout: float = 0.0,
        emb_dropout: float = 0.0,
        n_layers: int = 1,
    ):
        super().__init__(
            emb_size,
            hidden_size,
            output_size,
            pad_index,
            out_dropout,
            emb_dropout,
            n_layers,
        )

        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.rnn = nn.RNN(
            emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True
        )
        self.output = nn.Linear(hidden_size, output_size)

        self.pad_token = pad_index

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        rnn_out, _ = self.rnn(emb)
        output = self.output(rnn_out).permute(0, 2, 1)
        return output


class LM_LSTM(ModelApi):
    def __init__(
        self,
        emb_size: int,
        hidden_size: int,
        output_size: int,
        pad_index: int = 0,
        out_dropout: float = 0.0,
        emb_dropout: float = 0.0,
        n_layers: int = 1,
    ):
        super().__init__(
            emb_size,
            hidden_size,
            output_size,
            pad_index,
            out_dropout,
            emb_dropout,
            n_layers,
        )
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.lstm = nn.LSTM(
            emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True
        )
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        lstm_out, _ = self.lstm(emb)
        output = self.output(lstm_out).permute(0, 2, 1)
        return output
