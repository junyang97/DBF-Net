import torch
import torch.nn as nn
import torch.nn.functional as F

from fusion.models.WACV_2024_EGA import EGA


class EG_Gate(nn.Module):
    """
    简化后的 Edge-Guided Gate
    设计原则：
    1. 当前 decoder 特征 x 作为主语义输入
    2. coarse prediction pred 作为 EGA 的预测引导
    3. edge_inp 作为边界引导
    4. 输出采用残差增强，而不是强行完全替换原特征

    输入:
        x        : 当前 decoder block 输出特征, [B, C, H, W]
        edge_inp : 边缘图, [B, 1, H, W] 或 [B, C_e, H, W]
        pred     : coarse segmentation logits / prob map, 建议单通道或多类logits
    输出:
        out      : 融合后的特征
        alpha    : 空间门控图 [B,1,H,W]
        x_ega    : EGA增强后的特征
        delta    : 增强残差 x_ega - x
    """

    def __init__(self, in_channels, gate_channels=None):
        super(EG_Gate, self).__init__()

        self.in_channels = in_channels
        self.ega = EGA(in_channels)

        if gate_channels is None:
            gate_channels = max(in_channels // 2, 16)

        # 用当前特征 x 和 EGA 输出 x_ega 生成空间门控
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, gate_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(gate_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        # 可学习残差缩放
        self.gamma = nn.Parameter(torch.tensor(0.3))

    def forward(self, x, edge_inp, pred):
        """
        x       : 当前 decoder 特征
        edge_inp: 边界引导
        pred    : coarse prediction logits
        """
        assert x is not None, "EG_Gate: x cannot be None"
        assert edge_inp is not None, "EG_Gate: edge_inp cannot be None"
        assert pred is not None, "EG_Gate: pred cannot be None"

        # 对齐空间尺寸
        if edge_inp.shape[2:] != x.shape[2:]:
            edge_inp = F.interpolate(edge_inp, size=x.shape[2:], mode='bilinear', align_corners=True)

        if pred.shape[2:] != x.shape[2:]:
            pred = F.interpolate(pred, size=x.shape[2:], mode='bilinear', align_corners=True)

        # 如果 pred 是多类输出，压成单通道边界引导更稳
        # 这里保留 logits 形式送进 EGA，让 EGA 内部自己 sigmoid
        if pred.shape[1] > 1:
            pred_for_ega = torch.mean(pred, dim=1, keepdim=True)
        else:
            pred_for_ega = pred

        # EGA增强
        x_ega = self.ega(edge_inp, x, pred_for_ega)

        # 残差增强
        delta = x_ega - x

        # 由原特征和增强特征联合生成门控
        alpha = self.gate(torch.cat([x, x_ega], dim=1))

        # 最终输出：原特征 + 门控残差
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