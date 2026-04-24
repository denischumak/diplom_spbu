import torch.nn as nn
import torch


import torch
import torch.nn as nn


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        kernel_size = 3
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, dilation=dilation
        )
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, dilation=dilation
        )
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = x + residual
        x = self.act(x)
        return x


class MaskedGlobalAvgPool1d(nn.Module):
    def forward(self, x, lengths):
        """
        x: [B, C, T]
        lengths: [B]
        """
        b, c, t = x.shape
        device = x.device
        idx = torch.arange(t, device=device).unsqueeze(0).expand(b, t)
        mask = (
            (idx < lengths.unsqueeze(1)).float().unsqueeze(1)
        )  # [B,1,T], put 1 if value wasn't padded

        x = x * mask
        summed = x.sum(dim=-1)
        denom = mask.sum(dim=-1).clamp(min=1.0)
        return summed / denom


class GestureTCN(nn.Module):
    def __init__(self, tcn_cfg: dict, num_classes):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(
                tcn_cfg["in_channels"], tcn_cfg["hidden"], kernel_size=5, padding=2
            ),
            nn.GroupNorm(num_groups=8, num_channels=tcn_cfg["hidden"]),
            nn.ReLU(inplace=True),
        )
        # receptive field = 1 + s * (k-1) * (2^n - 1), where n - number of residual blocks, s - number of convs in each block
        # receptive field = 29 in this case
        dilations = [
            1,
            2,
            4,
        ]
        self.blocks = nn.Sequential(
            *[
                ResidualTCNBlock(tcn_cfg["hidden"], d, tcn_cfg["dropout"])
                for d in dilations
            ]
        )

        self.pool = MaskedGlobalAvgPool1d()

        self.head = nn.Sequential(
            nn.Linear(tcn_cfg["hidden"], 32),
            nn.ReLU(inplace=True),
            nn.Dropout(tcn_cfg["dropout"]),
            nn.Linear(32, num_classes),
        )

    def forward(self, x, lengths):
        """
        x: [B, C, T]
        """
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x, lengths)
        logits = self.head(x)
        return logits


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
