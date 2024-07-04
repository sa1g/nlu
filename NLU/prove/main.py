import logging
from pprint import pprint
import torch
from model import IntentSlotModel
from utils import get_data, Tokenizer, create_dataset
from torch.utils.data import DataLoader
from transformers import BertModel
from functions import train_loop

def custom_collate_fn(batch):
    def pad_sequence(sequences, padding_value=0):
        max_len = max([len(seq) for seq in sequences])
        padded_seqs = torch.full((len(sequences), max_len), padding_value)
        for i, seq in enumerate(sequences):
            padded_seqs[i, :len(seq)] = seq
        return padded_seqs

    input_ids = pad_sequence([item['input_ids'] for item in batch])
    attention_mask = pad_sequence([item['attention_mask'] for item in batch])
    token_type_ids = pad_sequence([item['token_type_ids'] for item in batch])
    slots = pad_sequence([item['slots'] for item in batch])
    intent = torch.stack([item['intent'] for item in batch])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids,
        'slots': slots,
        'intent': intent
    }

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    """
    LOAD DATA
    """

    train_raw, dev_raw, test_raw = get_data()

    tokenizer = Tokenizer(train_raw, dev_raw, test_raw)
    
    dataset = create_dataset(train_raw, tokenizer, device)

    bert_model = BertModel.from_pretrained("bert-base-uncased")

    model = IntentSlotModel(bert_model, tokenizer.slot_len, tokenizer.intent_len)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

    # Loss functions
    # TODO: in intention maybe ignore padding, start, end sequence
    intent_loss_fn = torch.nn.CrossEntropyLoss()
    slot_loss_fn = torch.nn.CrossEntropyLoss()

    for batch in dataset:
        
        # input_ids, attention_mask, intent_labels, slot_labels = batch
        # intent_labels = intent_labels.squeeze(1)

        loss = train_loop(model, batch, optimizer, intent_loss_fn, slot_loss_fn, tokenizer)
        print(f"Loss: {loss}")

        # intent_logits, slot_logits = model(input_ids, attention_mask)
        # intent_loss = intent_loss_fn(intent_logits, intent_labels)
        # slot_loss = slot_loss_fn(
        #     slot_logits.view(-1, tokenizer.slot_len), slot_labels.view(-1)
        # )

        # loss = intent_loss + slot_loss
        # print(f"Loss: {loss}")
        # exit()


        # # waht it is:
        # torch.Size([64, 26])
        # torch.Size([64, 1])
        # torch.Size([64, 64, 130])
        # torch.Size([64, 64])

        # # what it should be
        # torch.Size([64, 26])
        # torch.Size([64])
        # torch.Size([64, 64, 132])
        # torch.Size([64, 64])



        

    exit()

    train_dataset = MyDataset(train_raw, tokenizer, device)
    dev_dataset = MyDataset(dev_raw, tokenizer, device)
    test_dataset = MyDataset(test_raw, tokenizer, device)

    train_dataloader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
    dev_dataloader = DataLoader(
        dev_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)
    test_dataloader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)




