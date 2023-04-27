import torch


class QuantileLoss(torch.nn.Module):
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.huber_loss = torch.nn.HuberLoss(reduction="none", delta=delta)

    def forward(self, q: torch.Tensor, g: torch.Tensor):
        n = q.shape[-1]
        tau = torch.arange(0, n).unsqueeze(0).to(q.device) / n
        h = self.huber_loss(g, q)
        k = torch.abs(tau - (g - q > 0).float())
        return torch.mean(h * k)


class ValueRiskManagement(torch.nn.Module):
    def __init__(self, u_q: float = 0.75):
        super(ValueRiskManagement, self).__init__()
        self.u_q = u_q

    def forward(self, q: torch.Tensor):
        # q shape = (N, n)
        j = int(self.u_q * q.shape[-1])
        return torch.mean(q[:, j:], dim=-1)
