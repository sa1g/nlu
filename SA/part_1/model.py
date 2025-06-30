from transformers import AutoModel, BertModel
import torch


class SlotModel(torch.nn.Module):
    def __init__(self, slot_len: int):
        super().__init__()
        self.bert: BertModel = AutoModel.from_pretrained("bert-base-uncased")
        self.slot_classifier = torch.nn.Linear(self.bert.config.hidden_size, slot_len)

    def forward(self, x, attention_mask):
        outputs = self.bert(x, attention_mask=attention_mask)

        sequence_output = outputs.last_hidden_state
        slot_logits = self.slot_classifier(sequence_output)

        return slot_logits
