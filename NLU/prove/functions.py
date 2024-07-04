from model import IntentSlotModel
from utils import Tokenizer


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

    # loss = train_loop(model, batch, optimizer, intent_loss_fn, slot_loss_fn, tokenizer)
    # print(f"Loss: {loss}")

    intent_logits, slot_logits = model(input_ids, attention_mask)
    # intent_loss = intent_loss_fn(intent_logits, intent_labels)
    # slot_loss = slot_loss_fn(
        # slot_logits.view(-1, tokenizer.slot_len), slot_labels.view(-1)
    # )

    # loss = intent_loss + slot_loss
    # print(f"Loss: {loss}")
    # exit()


    # Losses
    loss = calculate_loss(
        intent_loss_fn=intent_loss_fn,
        intent_logits=intent_logits,
        intent_labels=intent_labels,
        slot_loss_fn=slot_loss_fn,
        slot_logits=slot_logits,
        slot_labels=slot_labels,
        tokenizer=tokenizer,
    )
    # exit()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
