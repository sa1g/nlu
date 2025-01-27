from collections import Counter
from sklearn.metrics import classification_report
from tqdm import tqdm
from conll import evaluate
import torch
from model import IntentSlotModel


def calculate_loss(
    data,
    intent_loss_fn,
    slot_loss_fn,
    intent_logits,
    slot_logits,
    intent_labels,
    slot_labels,
    slots2id,
    id2slots,
    id2intent,
):
    """
    Calculate the combined loss for intent classification and slot filling.
    Args:
        intent_loss_fn (callable): Loss function for intent classification.
        slot_loss_fn (callable): Loss function for slot filling.
        intent_logits (torch.Tensor): Logits output from the model for intent classification.
        slot_logits (torch.Tensor): Logits output from the model for slot filling.
        intent_labels (torch.Tensor): Ground truth labels for intent classification.
        slot_labels (torch.Tensor): Ground truth labels for slot filling.
        slots2id (dict): Dictionary mapping slot names to their corresponding IDs.
    Returns:
        torch.Tensor: Combined loss for intent classification and slot filling.
    """

    intent_count = Counter(data["intent"].tolist())
    intent_weights = (
        torch.tensor([1 / (intent_count[x] + 1) for x in id2intent.keys()])
        .float()
        .to("cuda")
    )
    criterion_intents = torch.nn.CrossEntropyLoss(weight=intent_weights)

    slot_count = Counter(data["slots"].flatten().tolist())
    slot_weights = (
        torch.tensor([1 / (slot_count[x] + 1) for x in id2slots.keys()])
        .float()
        .to("cuda")
    )
    criterion_slots = torch.nn.CrossEntropyLoss(
        weight=slot_weights, ignore_index=slots2id["pad"]
    )

    loss_intent = criterion_intents(intent_logits, intent_labels)
    loss_slot = criterion_slots(
        slot_logits.view(-1, len(slots2id)), slot_labels.view(-1)
    )

    loss = loss_intent + loss_slot

    return loss


def eval_loop(
    model: IntentSlotModel,
    dataloader,
    intent_loss_fn,
    slot_loss_fn,
    tokenizer,
    id2slots,
    slots2id,
    id2intent,
):
    """
    Evaluate the performance of an intent and slot filling model on a given dataset.
    Args:
        model (IntentSlotModel): The model to evaluate.
        dataloader (DataLoader): DataLoader providing the evaluation data.
        intent_loss_fn (callable): Loss function for intent classification.
        slot_loss_fn (callable): Loss function for slot filling.
        tokenizer (Tokenizer): Tokenizer used to decode input_ids.
        id2slots (dict): Mapping from slot IDs to slot labels.
        slots2id (dict): Mapping from slot labels to slot IDs.
    Returns:
        tuple: A tuple containing:
            - accuracy_intention (float): Accuracy of the intent classification.
            - f1_slot (float): F1 score of the slot filling.
            - total_loss (list): List of loss values for each batch.
    """

    model.eval()
    total_loss = []

    ref_intents = []
    hyp_intents = []

    ref_slots = []
    hyp_slots = []

    with torch.no_grad():
        for data in dataloader:
            input_ids = data["input_ids"]
            attention_mask = data["attention_mask"]
            token_type_ids = data["token_type_ids"]
            intent_labels = data["intent"]
            slot_labels = data["slots"]
            slots_len = data["slots_len"]

            # Forward pass
            intent_logits, slot_logits = model(
                input_ids, attention_mask, token_type_ids
            )
            loss = calculate_loss(
                data=data,
                intent_loss_fn=intent_loss_fn,
                intent_logits=intent_logits,
                intent_labels=intent_labels,
                slot_loss_fn=slot_loss_fn,
                slot_logits=slot_logits,
                slot_labels=slot_labels,
                slots2id=slots2id,
                id2slots=id2slots,
                id2intent=id2intent,
            ).item()
            total_loss.append(loss)

            # Intent inference
            intent_hyp = torch.argmax(intent_logits, dim=1)
            ref_intents.extend(intent_labels.cpu().tolist())
            hyp_intents.extend(intent_hyp.cpu().tolist())

            # Slot filling inference
            slot_hyp = torch.argmax(slot_logits, dim=2)

            for i in range(input_ids.size(0)):
                seq_length = slots_len[i]
                utterance = tokenizer.tokenize(
                    tokenizer.decode(
                        input_ids[i].cpu().tolist(), include_special_tokens=False
                    )
                )[:seq_length]

                tmp_ref = [
                    (utterance[j], id2slots[slot_labels[i][j].item()])
                    for j in range(seq_length)
                ]
                tmp_hyp = [
                    (utterance[j], id2slots[slot_hyp[i][j].item()])
                    for j in range(seq_length)
                ]

                ref_slots.append(tmp_ref)
                hyp_slots.append(tmp_hyp)

        f1_slot = evaluate(ref_slots, hyp_slots)

        accuracy_intention = classification_report(
            ref_intents,
            hyp_intents,
            output_dict=True,
            zero_division=False,
        )["accuracy"]

        return accuracy_intention, f1_slot["total"]["f"], total_loss


def train_loop(
    model: IntentSlotModel,
    train_dataloader,
    optimizer,
    intent_loss_fn,
    slot_loss_fn,
    slots2id,
    id2slots,
    id2intent,
    tokenizer,
    scheduler=None,
    grad_clip=False,
):
    """
    Trains the given model for one epoch using the provided dataloader, optimizer, and loss functions.
    Args:
        model (IntentSlotModel): The model to be trained.
        train_dataloader (DataLoader): DataLoader for the training data.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        intent_loss_fn (callable): Loss function for intent classification.
        slot_loss_fn (callable): Loss function for slot filling.
        slots2id (dict): Dictionary mapping slot labels to their corresponding IDs.
        scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler. Defaults to None.
        grad_clip (bool, optional): Whether to apply gradient clipping. Defaults to False.
    Returns:
        list: List of loss values for each batch.
    """

    model.train()

    loss_array = []
    batch_tqdm = tqdm(
        enumerate(train_dataloader), desc=f"Batch | Loss: {0:.4f}", leave=False
    )

    for _, data in batch_tqdm:
        optimizer.zero_grad()
        intent_logits, slot_logits = model(
            data["input_ids"], data["attention_mask"], data["token_type_ids"]
        )

        loss = calculate_loss(
            data=data,
            intent_loss_fn=intent_loss_fn,
            intent_logits=intent_logits,
            intent_labels=data["intent"],
            slot_loss_fn=slot_loss_fn,
            slot_logits=slot_logits,
            slot_labels=data["slots"],
            slots2id=slots2id,
            id2slots=id2slots,
            id2intent=id2intent,
        )

        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        batch_tqdm.set_description(f"Batch | Loss: {loss.item():.4f}")
        loss_array.append(loss.item())

    return loss_array
