import torch
import torch.nn as nn
import torch.nn.functional as F

def gauss_kernel(channels=3, cuda=True):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    if cuda:
        kernel = kernel.cuda()
    return kernel


def downsample(x):
    return x[:, :, ::2, ::2]


def conv_gauss(img, kernel):
    img = F.pad(img, (2, 2, 2, 2), mode='reflect')
    out = F.conv2d(img, kernel, groups=img.shape[1])
    return out


def upsample(x, channels):
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
    cc = cc.permute(0, 1, 3, 2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
    x_up = cc.permute(0, 1, 3, 2)
    return conv_gauss(x_up, 4 * gauss_kernel(channels))


def make_laplace(img, channels):
    filtered = conv_gauss(img, gauss_kernel(channels))
    down = downsample(filtered)
    up = upsample(down, channels)
    if up.shape[2] != img.shape[2] or up.shape[3] != img.shape[3]:
        up = nn.functional.interpolate(up, size=(img.shape[2], img.shape[3]))
    diff = img - up
    return diff


def make_laplace_pyramid(img, level, channels):
    current = img
    pyr = []
    for _ in range(level):
        filtered = conv_gauss(current, gauss_kernel(channels))
        down = downsample(filtered)
        up = upsample(down, channels)
        if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
            up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
        diff = current - up
        pyr.append(diff)
        current = down
    pyr.append(current)
    return pyr


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )

    def forward(self, x):
        avg_out = self.mlp(F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))))
        max_out = self.mlp(F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))))
        channel_att_sum = avg_out + max_out

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.spatial = nn.Conv2d(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        x_compress = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio)
        self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        x_out = self.SpatialGate(x_out)
        return x_out

class EGA(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.res_align = nn.Conv2d(in_channels, in_channels, 1)
        self.in_channels = in_channels

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.cbam = CBAM(in_channels)

    def forward(self, edge_feature, x, pred):
        residual = self.res_align(x)
        xsize = x.size()[2:]
        pred = torch.sigmoid(pred)

        background_att = 1 - pred
        background_x = x * background_att

        edge_pred = make_laplace(pred, 1)
        pred_feature = x * edge_pred

        edge_input = F.interpolate(edge_feature, size=xsize, mode='bilinear', align_corners=True)
        input_feature = x * edge_input

        fusion_feature = torch.cat([background_x, pred_feature, input_feature], dim=1)

        fusion_feature = self.fusion_conv(fusion_feature)

        attention_map = self.attention(fusion_feature)
        fusion_feature = fusion_feature * attention_map

        out = fusion_feature + residual
        out = self.cbam(out)
        return out

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
