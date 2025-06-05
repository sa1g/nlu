from abc import ABC, abstractmethod
from typing import Optional

import torch
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


class VariationalDropout(nn.Module):
    """
    Apply the same dropout mask across the time dimension.

    [Paper](https://arxiv.org/abs/1512.05287) -
    [Source](https://github.com/keitakurita/Better_LSTM_PyTorch/blob/master/better_lstm/model.py)
    """

    def __init__(self, dropout: float = 0.0, batch_first: Optional[bool] = False):
        super().__init__()
        self.dropout = dropout
        self.batch_first = batch_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply dropout only during training and if >0
        if not self.training or self.dropout <= 0.0:
            return x

        # Generate the mask
        if self.batch_first:
            mask = x.new_empty(x.size(0), 1, x.size(2), requires_grad=False).bernoulli(
                1 - self.dropout
            )
        else:
            mask = x.new_empty(1, x.size(0), x.size(2), requires_grad=False).bernoulli(
                1 - self.dropout
            )
        # Apply the mask
        x = x.masked_fill(mask == 0, 0) / (1 - self.dropout)
        # The scaling factor (1 - self.dropout) is applied to maintain the expected value of
        # the activations.

        return x


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
        weight_tying: bool = False,
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

        # Batch first as our data is batch_first
        # we apply variational dropout on the "token" dimension
        self.emb_dropout = VariationalDropout(emb_dropout, batch_first=True)
        self.out_dropout = VariationalDropout(out_dropout, batch_first=True)

        if weight_tying:
            assert (
                hidden_size == emb_size
            ), "Output size must be equal to embedding size for weight tying"
            self.output.weight = self.embedding.weight

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        emb = self.emb_dropout(emb)
        lstm_out, _ = self.lstm(emb)
        lstm_out = self.out_dropout(lstm_out)
        output = self.output(lstm_out).permute(0, 2, 1)
        return output
