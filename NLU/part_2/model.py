import torch
from transformers import AutoModel, BertModel


class IntentSlotModel(torch.nn.Module):
    def __init__(self, slot_len: int, intent_len: int):
        super().__init__()
        self.bert: BertModel = AutoModel.from_pretrained("bert-base-uncased")

        self.intent_classifier = torch.nn.Linear(
            self.bert.config.hidden_size, intent_len
        )
        self.slot_classifier = torch.nn.Linear(self.bert.config.hidden_size, slot_len)

    def forward(self, x, attention_mask):
        outputs = self.bert(x, attention_mask=attention_mask)

        sequence_output = outputs.last_hidden_state
        slot_logits = self.slot_classifier(sequence_output)

        # Intent classification as standard in HuggingFace
        pooled_output = outputs.pooler_output
        intent_logits = self.intent_classifier(pooled_output)

        return intent_logits, slot_logits
