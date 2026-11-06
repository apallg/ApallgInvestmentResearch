import torch
from torch import nn


class HorizonFusionNet(nn.Module):
    def __init__(self, daily_features, arm):
        super().__init__()
        self.arm = arm
        self.daily = nn.LSTM(daily_features, 32, num_layers=1, batch_first=True)
        if arm == "NT" or arm == "NR":
            self.monthly = nn.LSTM(12, 16, num_layers=1, batch_first=True)
            extra = 16
        elif arm == "NF":
            self.monthly = nn.Sequential(nn.Linear(144, 12), nn.ReLU(), nn.Linear(12, 16), nn.ReLU())
            extra = 16
        else:
            self.monthly = None
            extra = 0
        self.output = nn.Linear(32 + extra, 1)

    def forward(self, daily, monthly=None):
        daily_state = self.daily(daily)[0][:, -1, :]
        if self.monthly is None:
            state = daily_state
        elif self.arm in ("NT", "NR"):
            state = torch.cat([daily_state, self.monthly(monthly)[0][:, -1, :]], dim=1)
        else:
            state = torch.cat([daily_state, self.monthly(monthly.reshape(len(monthly), 144))], dim=1)
        return self.output(state).squeeze(-1)
