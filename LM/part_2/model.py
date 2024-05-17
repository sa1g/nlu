import torch.nn as nn


class LM_RNN(nn.Module):
    """
    Base RNN model for language modeling
    """

    def __init__(
        self,
        emb_size,
        hidden_size,
        output_size,
        pad_index=0,
        out_dropout=None,
        emb_dropout=None,
        n_layers=1,
        lstm=False,
    ):
        super().__init__()

        self.switch = 0

        self.embedding = nn.Embedding(
            output_size, emb_size, padding_idx=pad_index)

        if emb_dropout is not None:
            self.switch = 1
            self.emb_dropout = nn.Dropout(emb_dropout)

        if lstm is True:
            self.__name__ = "LSTM"
            self.rnn = nn.LSTM(emb_size, hidden_size,
                               n_layers, bidirectional=False)
        else:
            self.__name__ = "RNN"
            self.rnn = nn.RNN(emb_size, hidden_size,
                              n_layers, bidirectional=False)

        if out_dropout is not None:
            self.switch = 2
            self.out_dropout = nn.Dropout(out_dropout)

        if emb_dropout is not None and out_dropout is not None:
            self.switch = 3

        self.pad_token = pad_index

        self.output = nn.Linear(hidden_size, output_size)

        self.output.weight = self.embedding.weight

    def forward(self, input_sequence):
        # Horrible implementation but it works
        if self.switch == 0:
            emb = self.embedding(input_sequence)
            rnn_out, _ = self.rnn(emb)
            output = self.output(rnn_out).permute(0, 2, 1)
        elif self.switch == 1:
            emb = self.embedding(input_sequence)
            emb = self.emb_dropout(emb)
            rnn_out, _ = self.rnn(emb)
            output = self.output(rnn_out).permute(0, 2, 1)
        elif self.switch == 2:
            emb = self.embedding(input_sequence)
            rnn_out, _ = self.rnn(emb)
            rnn_out = self.out_dropout(rnn_out)
            output = self.output(rnn_out).permute(0, 2, 1)
        else:
            emb = self.embedding(input_sequence)
            emb = self.emb_dropout(emb)
            rnn_out, _ = self.rnn(emb)
            rnn_out = self.out_dropout(rnn_out)
            output = self.output(rnn_out).permute(0, 2, 1)

        return output
