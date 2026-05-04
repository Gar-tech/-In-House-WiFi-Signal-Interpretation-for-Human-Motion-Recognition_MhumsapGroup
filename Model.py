import torch
import torch.nn as nn
import torch.nn.functional as F

class CSI_Attention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv1d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv1d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv1d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch, ch, time = x.size()
        q = self.query(x).view(batch, -1, time).permute(0, 2, 1)
        k = self.key(x).view(batch, -1, time)
        attention = torch.bmm(q, k)
        attention = F.softmax(attention, dim=-1)
        v = self.value(x).view(batch, -1, time)
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(batch, ch, time)
        return self.gamma * out + x

class AmplitudeOnlyModel(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(30, 64, 7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4)
        )
        self.attn = CSI_Attention(64)
        self.features = nn.Sequential(
            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(128, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.attn(x)
        x = self.features(x)
        return self.classifier(torch.flatten(x, 1))