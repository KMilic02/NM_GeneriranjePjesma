import torch
import math

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits, _ = model(x)

        loss = criterion(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits, _ = model(x)
        loss = criterion(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity