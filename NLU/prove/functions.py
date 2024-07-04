import torch
from model import IntentSlotModel
from utils import Tokenizer
from sklearn.metrics import classification_report
from conll import evaluate




def calculate_loss(
    intent_loss_fn,
    slot_loss_fn,
    intent_logits,
    slot_logits,
    intent_labels,
    slot_labels,
    tokenizer: Tokenizer,
):
    intent_loss = intent_loss_fn(intent_logits, intent_labels)
    slot_loss = slot_loss_fn(
        slot_logits.view(-1, tokenizer.slot_len), slot_labels.view(-1)
    )

    return intent_loss + slot_loss


def train_loop(
    model: IntentSlotModel,
    data,
    optimizer,
    intent_loss_fn,
    slot_loss_fn,
    tokenizer: Tokenizer,
    clip=5,
    device: str = "cuda:0",
):
    model.train()

    input_ids, attention_mask, intent_labels, slot_labels = data
    intent_labels = intent_labels.squeeze(1)

    intent_logits, slot_logits = model(input_ids, attention_mask)

    loss = calculate_loss(
        intent_loss_fn=intent_loss_fn,
        intent_logits=intent_logits,
        intent_labels=intent_labels,
        slot_loss_fn=slot_loss_fn,
        slot_logits=slot_logits,
        slot_labels=slot_labels,
        tokenizer=tokenizer,
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def eval_loop(
    model: IntentSlotModel,
    data,
    intent_loss_fn,
    slot_loss_fn,
    tokenizer: Tokenizer,
):

    model.eval()
    total_loss = []

    with torch.no_grad():
        input_ids, attention_mask, intent_labels, slot_labels = data
        intent_labels = intent_labels.squeeze(1)

        # Forward
        intent_logits, slot_logits = model(input_ids, attention_mask)
        total_loss.append(
            calculate_loss(
                intent_loss_fn=intent_loss_fn,
                intent_logits=intent_logits,
                intent_labels=intent_labels,
                slot_loss_fn=slot_loss_fn,
                slot_logits=slot_logits,
                slot_labels=slot_labels,
                tokenizer=tokenizer,
            )
        )

        intent_hyp = torch.argmax(intent_logits, dim=1)
        slot_hpy = torch.argmax(slot_logits, dim=2)

        print("========== EVAL ===========")
        # Intent: accuracy
        accuracy_intention = classification_report(
                intent_labels.to("cpu"),
                intent_hyp.to("cpu"),
                output_dict=True,
                zero_division=False,
            )['accuracy']
        
        # for input in input_ids:
        #     print(input.shape.tolist()[0])
        #     input_size = input.shape
        #     break

        ref, hyp = [], []

        if slot_labels.shape != slot_hpy.shape:
            print("Shape mismatch")
            print(slot_labels.shape)
            print(slot_hpy.shape)
            exit()

        for i, (s_ref, s_hyp) in enumerate(zip(slot_labels, slot_hpy)):
            tmp_ref = []
            tmp_hyp = []

            for j, (r,h) in enumerate(zip(s_ref, s_hyp)):
                tmp_ref.append((j, r))
                tmp_hyp.append((j, h))

            ref.append(tmp_ref)
            hyp.append(tmp_hyp)

        f1_slot = evaluate(ref, hyp)
        print(f1_slot)

        exit()
