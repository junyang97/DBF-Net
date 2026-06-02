import torch
import torch.nn as nn
import torch.nn.functional as F

from fusion.models.WACV_2024_EGA import EGA

class EG_Gate(nn.Module):

    def __init__(self, in_channels, gate_channels=None):
        super(EG_Gate, self).__init__()

        self.in_channels = in_channels
        self.ega = EGA(in_channels)

        if gate_channels is None:
            gate_channels = max(in_channels // 2, 16)

        self.gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, gate_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(gate_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        self.gamma = nn.Parameter(torch.tensor(0.3))

    def forward(self, x, edge_inp, pred):

        assert x is not None, "EG_Gate: x cannot be None"
        assert edge_inp is not None, "EG_Gate: edge_inp cannot be None"
        assert pred is not None, "EG_Gate: pred cannot be None"

        if edge_inp.shape[2:] != x.shape[2:]:
            edge_inp = F.interpolate(edge_inp, size=x.shape[2:], mode='bilinear', align_corners=True)

        if pred.shape[2:] != x.shape[2:]:
            pred = F.interpolate(pred, size=x.shape[2:], mode='bilinear', align_corners=True)

        if pred.shape[1] > 1:
            pred_for_ega = torch.mean(pred, dim=1, keepdim=True)
        else:
            pred_for_ega = pred

        x_ega = self.ega(edge_inp, x, pred_for_ega)

        delta = x_ega - x

        alpha = self.gate(torch.cat([x, x_ega], dim=1))

        out = x + self.gamma * alpha * delta

        return out, alpha, x_ega, delta


if __name__ == "__main__":
    x = torch.randn(1, 128, 56, 56).cuda()
    edge = torch.randn(1, 1, 56, 56).cuda()
    pred = torch.randn(1, 1, 56, 56).cuda()

    block = EG_Gate(128).cuda()
    out, alpha, x_ega, delta = block(x, edge, pred)

    print("x      :", x.shape)
    print("edge   :", edge.shape)
    print("pred   :", pred.shape)
    print("x_ega  :", x_ega.shape)
    print("alpha  :", alpha.shape)
    print("delta  :", delta.shape)
    print("out    :", out.shape)
    print("gamma  :", block.gamma.item())