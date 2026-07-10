import torch

def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for x, lengths, y, _ in loader:
        x = x.to(device)
        lengths = lengths.to(device)
        y = y.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(x, lengths)
            loss = criterion(logits, y)

            if is_train:
                loss.backward()
                optimizer.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_count += bs

    return total_loss / max(total_count, 1), total_correct / max(total_count, 1)