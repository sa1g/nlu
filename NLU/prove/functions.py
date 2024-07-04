def calculate_loss(
    intent_loss_fn,
    slot_loss_fn,
    intent_logits,
    slot_logits,
    intent_labels,
    slot_labels,
):
    intent_loss = intent_loss_fn(intent_logits, intent_labels)
    slot_loss = slot_loss_fn(slot_logits, slot_labels)

    return intent_loss + slot_loss


def train_loop(
    model, data, optimizer, intent_loss_fn, slot_loss_fn, clip=5, device: str = "cuda:0"
):
    model.train()

    # input_ids, attention_mask, intent_labels, slot_labels = data

    # input_ids = input_ids.to(device)
    # attention_mask = attention_mask.to(device)
    # intent_labels = intent_labels.to(device)
    # slot_labels = slot_labels.to(device)

    # Forward
    intent_logits, slot_logits = model(data['input_ids'], data['attention_mask'])

    # intent_logits = torch.argmax(intent_logits, dim=1)
    # slot_logits = torch.argmax(slot_logits, dim=2)

    # print(f"intent_logits: {intent_logits}")
    # print(f"intent_labels: {intent_labels}")
    # print(f"slot_logits: {slot_logits}")
    # print(f"slot_labels: {slot_labels}")

    # print(f"intent_logits.shape: {intent_logits.shape}")
    # print(f"intent_labels.shape: {intent_labels.shape}")
    # print(f"slot_logits.shape: {slot_logits.shape}")
    # print(f"slot_labels.shape: {slot_labels.shape}")

    # Losses
    loss = calculate_loss(
        intent_loss_fn,
        slot_loss_fn,
        intent_logits,
        slot_logits,
        data['intent'],
        data['slots'],
    )
    # exit()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()
