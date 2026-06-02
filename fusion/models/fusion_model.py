import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/media/zjt/A2BA6F20E45A4F62/sjy1/VmTu1/fusion/VisionMamba")
import fusion.VisionMamba.Vim as Vim

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from VisionMamba.Vim import VisionMamba
except ImportError:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "Vim",
        os.path.join(project_root, "VisionMamba", "Vim.py")
    )
    vim_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vim_module)
    VisionMamba = vim_module.VisionMamba

from fusion.models.vit_segvm_modeling import VisionTransformer, CONFIGS
from fusion.configs.config_setting_polyp import setting_config
from fusion.models.ega_gatenocsca import EG_Gate


class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=True):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_batchnorm
        )
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels)
        super().__init__(conv, bn, relu)


class AdaptiveGuideGate(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        alpha = self.pool(x)
        alpha = self.fc1(alpha)
        alpha = self.relu(alpha)
        alpha = self.fc2(alpha)
        alpha = self.sigmoid(alpha)
        return alpha


class SpatialGuideGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 4, 8)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)


class DecoderBlockEG(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        skip_channels=0,
        use_batchnorm=True,
        use_ega=False,
        block_id=None,
        ega_input_mode="pred",
        pred_channels=1
    ):
        super().__init__()
        self.use_ega = use_ega
        self.block_id = block_id
        self.ega_input_mode = ega_input_mode

        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        if use_ega:
            self.ega_gate = EG_Gate(in_channels=out_channels)
            self.guide_gate = AdaptiveGuideGate(out_channels)
            self.pred_proj = nn.Conv2d(pred_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x, skip=None, cs_feat=None, edge_feat=None, pred_feat=None, debug_prefix=""):
        ega_triggered = False

        x = self.up(x)
        if skip is not None:
            if x.size(2) != skip.size(2) or x.size(3) != skip.size(3):
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)

        if self.use_ega and edge_feat is not None and (pred_feat is not None or cs_feat is not None):
            if self.ega_input_mode == "pred":
                guide_feat = pred_feat
                if guide_feat is None:
                    raise ValueError("pred mode requires pred_feat.")
                if guide_feat.shape[2:] != x.shape[2:]:
                    guide_feat = F.interpolate(guide_feat, size=x.shape[2:], mode='bilinear', align_corners=True)
                guide_feat = self.pred_proj(guide_feat)

            elif self.ega_input_mode == "cs":
                guide_feat = cs_feat
                if guide_feat is None:
                    raise ValueError("cs mode requires cs_feat.")
                if guide_feat.shape[2:] != x.shape[2:]:
                    guide_feat = F.interpolate(guide_feat, size=x.shape[2:], mode='bilinear', align_corners=True)
                if guide_feat.shape[1] != x.shape[1]:
                    raise ValueError(
                        f"cs mode expects cs_feat channels == x channels, "
                        f"but got cs_feat={guide_feat.shape[1]}, x={x.shape[1]}"
                    )

            elif self.ega_input_mode == "adaptive":
                if pred_feat is None or cs_feat is None:
                    raise ValueError("adaptive mode requires both pred_feat and cs_feat.")

                if pred_feat.shape[2:] != x.shape[2:]:
                    pred_feat = F.interpolate(pred_feat, size=x.shape[2:], mode='bilinear', align_corners=True)
                if cs_feat.shape[2:] != x.shape[2:]:
                    cs_feat = F.interpolate(cs_feat, size=x.shape[2:], mode='bilinear', align_corners=True)

                pred_feat = self.pred_proj(pred_feat)

                if cs_feat.shape[1] != x.shape[1]:
                    raise ValueError(
                        f"adaptive mode expects cs_feat channels == x channels, "
                        f"but got cs_feat={cs_feat.shape[1]}, x={x.shape[1]}"
                    )

                alpha = self.guide_gate(x)
                guide_feat = alpha * pred_feat + (1.0 - alpha) * cs_feat

                with torch.no_grad():
                    print(
                        f"{debug_prefix}[AdaptiveGuide] block {self.block_id} | "
                        f"alpha_mean={alpha.mean().item():.4f}, "
                        f"alpha_min={alpha.min().item():.4f}, "
                        f"alpha_max={alpha.max().item():.4f}, "
                        f"pred_proj_shape={tuple(pred_feat.shape)}, "
                        f"cs_shape={tuple(cs_feat.shape)}"
                    )

            else:
                raise ValueError(f"Unknown ega_input_mode: {self.ega_input_mode}")

            if edge_feat.shape[2:] != x.shape[2:]:
                edge_feat = F.interpolate(edge_feat, size=x.shape[2:], mode='bilinear', align_corners=True)

            x, _, _, _ = self.ega_gate(x, edge_feat, guide_feat)
            ega_triggered = True

        return x, ega_triggered


class DecoderCupEG(nn.Module):
    def __init__(self, config, use_ega_layers=(2, 3), ega_input_mode="pred"):
        super().__init__()
        self.config = config
        self.use_ega_layers = tuple(use_ega_layers)
        self.ega_input_mode = ega_input_mode

        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True
        )

        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = list(self.config.skip_channels)
            for i in range(4 - self.config.n_skip):
                skip_channels[3 - i] = 0
        else:
            skip_channels = [0, 0, 0, 0]

        blocks = []
        for i, (in_ch, out_ch, sk_ch) in enumerate(zip(in_channels, out_channels, skip_channels)):
            use_ega = i in self.use_ega_layers
            blocks.append(
                DecoderBlockEG(
                    in_ch,
                    out_ch,
                    sk_ch,
                    use_ega=use_ega,
                    block_id=i,
                    ega_input_mode=ega_input_mode,
                    pred_channels=config.n_classes
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.last_debug_info = {}

    def forward(self, hidden_states, features=None, cs_feats=None, edge_feats=None, pred_feats=None,
                return_all=False, debug_prefix=""):
        B, n_patch, hidden = hidden_states.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1).contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)

        stage_feats = []
        ega_trigger_count = 0
        ega_triggered_blocks = []

        for i, decoder_block in enumerate(self.blocks):
            skip = features[i] if (features is not None and i < self.config.n_skip) else None

            curr_cs = None
            if cs_feats is not None and i < len(cs_feats):
                curr_cs = cs_feats[i]

            curr_edge = None
            if edge_feats is not None and i < len(edge_feats):
                curr_edge = edge_feats[i]

            curr_pred = None
            if pred_feats is not None and i < len(pred_feats):
                curr_pred = pred_feats[i]

            x, ega_triggered = decoder_block(
                x,
                skip=skip,
                cs_feat=curr_cs,
                edge_feat=curr_edge,
                pred_feat=curr_pred,
                debug_prefix=debug_prefix
            )

            stage_feats.append(x)

            if ega_triggered:
                ega_trigger_count += 1
                ega_triggered_blocks.append(i)

        self.last_debug_info = {
            "num_stage_feats": len(stage_feats),
            "stage_feat_shapes": [tuple(t.shape) for t in stage_feats],
            "ega_trigger_count": ega_trigger_count,
            "ega_triggered_blocks": ega_triggered_blocks,
            "use_ega_layers": self.use_ega_layers,
        }

        if return_all:
            return x, stage_feats
        return x


class AdaptiveCrossGatingFusion(nn.Module):
    def __init__(self, trans_channels: int, mamba_channels: int, fusion_channels: int):
        super().__init__()
        self.trans_proj = nn.Conv2d(trans_channels, fusion_channels, kernel_size=1)
        self.mamba_proj = nn.Conv2d(mamba_channels, fusion_channels, kernel_size=1)

        self.gate_generator = nn.Sequential(
            nn.Conv2d(fusion_channels * 2, fusion_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, feat_trans, feat_mamba):
        f_trans_proj = self.trans_proj(feat_trans)
        f_mamba_proj = self.mamba_proj(feat_mamba)

        concat_features = torch.cat([f_trans_proj, f_mamba_proj], dim=1)
        gate_map = self.gate_generator(concat_features)

        fused = gate_map * f_trans_proj + (1 - gate_map) * f_mamba_proj
        return fused


class CGAFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_fc = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x, guide):
        ch_att = self.channel_fc(self.avg_pool(guide))
        x_ch = x * ch_att

        cat_feat = torch.cat([x, guide], dim=1)
        max_feat, _ = torch.max(cat_feat, dim=1, keepdim=True)
        avg_feat = torch.mean(cat_feat, dim=1, keepdim=True)
        sp_att = self.spatial_conv(torch.cat([max_feat, avg_feat], dim=1))
        x_sp = x_ch * sp_att

        return x_sp


class A2B_Fusion(nn.Module):
    def __init__(self, trans_channels, mamba_channels, fusion_channels):
        super().__init__()
        self.A = AdaptiveCrossGatingFusion(trans_channels, mamba_channels, fusion_channels)
        self.B = CGAFusion(fusion_channels)

    def forward(self, feat_trans, feat_mamba):
        F_A = self.A(feat_trans, feat_mamba)
        F_final = self.B(F_A, F_A)
        return F_final


class LateFusionSegEG(nn.Module):
    def __init__(
        self,
        transunet: VisionTransformer,
        visionmamba: VisionMamba,
        fused_channels: int = 768,
        num_classes: int = 9,
        use_ega_layers=(2, 3),
        ega_input_mode="pred",
        use_cross_guided_fusion=True,
        use_two_stage_decoder=True,
        use_ega_refine=True
    ):
        super().__init__()
        self.use_detach = True
        self.ega_input_mode = ega_input_mode

        self.use_cross_guided_fusion = use_cross_guided_fusion
        self.use_two_stage_decoder = use_two_stage_decoder
        self.use_ega_refine = use_ega_refine

        if use_ega_refine:
            self.use_ega_layers = tuple(use_ega_layers)
        else:
            self.use_ega_layers = tuple()

        self.transunet_encoder = transunet.transformer
        self.visionmamba = visionmamba

        self.decoder = DecoderCupEG(
            transunet.config,
            use_ega_layers=self.use_ega_layers,
            ega_input_mode=ega_input_mode
        )

        self.segmentation_head = transunet.segmentation_head

        mamba_channels = visionmamba.embed_dim
        transunet_bottle_channels = transunet.transformer.encoder.encoder_norm.normalized_shape[0]

        self.fusion = A2B_Fusion(
            trans_channels=transunet_bottle_channels,
            mamba_channels=mamba_channels,
            fusion_channels=fused_channels
        )

        self.simple_fusion = nn.Conv2d(
            transunet_bottle_channels + mamba_channels,
            fused_channels,
            kernel_size=1,
            bias=False
        )

        self.last_debug_info = {}

    def generate_edge(self, pred):
        prob = torch.sigmoid(pred)

        if not hasattr(self, "_debug_printed"):
            self._debug_printed = 0

        if self._debug_printed < 5:
            with torch.no_grad():
                prob_min = prob.min().item()
                prob_max = prob.max().item()
                prob_mean = prob.mean().item()
                prob_std = prob.std().item()
                near_zero = (prob < 0.01).float().mean().item()
                near_one = (prob > 0.99).float().mean().item()

            self._debug_printed += 1

        sobel_x = torch.tensor(
            [[[-1, 0, 1],
              [-2, 0, 2],
              [-1, 0, 1]]],
            dtype=prob.dtype,
            device=prob.device
        ).unsqueeze(0)

        sobel_y = torch.tensor(
            [[[-1, -2, -1],
              [0, 0, 0],
              [1, 2, 1]]],
            dtype=prob.dtype,
            device=prob.device
        ).unsqueeze(0)

        if prob.shape[1] == 1:
            grad_x = F.conv2d(prob, sobel_x, padding=1)
            grad_y = F.conv2d(prob, sobel_y, padding=1)
        else:
            grad_x = F.conv2d(
                prob,
                sobel_x.repeat(prob.shape[1], 1, 1, 1),
                padding=1,
                groups=prob.shape[1]
            )
            grad_y = F.conv2d(
                prob,
                sobel_y.repeat(prob.shape[1], 1, 1, 1),
                padding=1,
                groups=prob.shape[1]
            )
            grad_x = grad_x.mean(dim=1, keepdim=True)
            grad_y = grad_y.mean(dim=1, keepdim=True)

        edge = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        return edge

    def forward(self, x, validate_ega=True):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        x_tu, _, features = self.transunet_encoder(x)

        if not hasattr(self, "_debug_feature_shape_printed"):
            print("[Debug] x_tu shape:", x_tu.shape)
            print("[Debug] TransUNet features:")
            for i, f in enumerate(features):
                print(f"  features[{i}] shape:", f.shape)
            self._debug_feature_shape_printed = True

        feat_m = self.visionmamba(x, return_features=True)

        B, n_patch, hidden = x_tu.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x_tu_reshaped = x_tu.permute(0, 2, 1).contiguous().view(B, hidden, h, w)

        if feat_m.shape[2:] != x_tu_reshaped.shape[2:]:
            feat_m = F.interpolate(
                feat_m,
                size=x_tu_reshaped.shape[2:],
                mode='bilinear',
                align_corners=True
            )

        if self.use_cross_guided_fusion:
            fused = self.fusion(x_tu_reshaped, feat_m)
        else:
            fused = self.simple_fusion(torch.cat([x_tu_reshaped, feat_m], dim=1))

        fused_reshaped_back = fused.flatten(2).transpose(-1, -2)

        decoded_features, first_stage_feats = self.decoder(
            fused_reshaped_back,
            features,
            cs_feats=None,
            edge_feats=None,
            pred_feats=None,
            return_all=True,
            debug_prefix="[PASS1] "
        )

        pred = self.segmentation_head(decoded_features)

        if not self.use_two_stage_decoder:
            self.last_debug_info = {
                "mode": "one_stage",
                "use_cross_guided_fusion": self.use_cross_guided_fusion,
                "use_two_stage_decoder": self.use_two_stage_decoder,
                "use_ega_refine": self.use_ega_refine,
                "first_stage_feats_len": len(first_stage_feats),
                "first_stage_feat_shapes": [tuple(t.shape) for t in first_stage_feats],
            }
            return pred

        edge_input = pred.detach() if self.use_detach else pred
        edge_map = self.generate_edge(edge_input)

        cs_feats = first_stage_feats
        edge_feats = [
            F.interpolate(edge_map, size=cs.shape[2:], mode='bilinear', align_corners=True)
            for cs in cs_feats
        ]
        pred_feats = [
            F.interpolate(edge_input, size=cs.shape[2:], mode='bilinear', align_corners=True)
            for cs in cs_feats
        ]

        decoded_refined = self.decoder(
            fused_reshaped_back,
            features,
            cs_feats=cs_feats,
            edge_feats=edge_feats,
            pred_feats=pred_feats,
            return_all=False,
            debug_prefix="[PASS2] "
        )

        final_pred = self.segmentation_head(decoded_refined)

        self.last_debug_info = {
            "mode": "two_stage",
            "use_cross_guided_fusion": self.use_cross_guided_fusion,
            "use_two_stage_decoder": self.use_two_stage_decoder,
            "use_ega_refine": self.use_ega_refine,
            "first_stage_feats_len": len(first_stage_feats),
            "first_stage_feat_shapes": [tuple(t.shape) for t in first_stage_feats],
            "edge_feats_len": len(edge_feats),
            "edge_feat_shapes": [tuple(t.shape) for t in edge_feats],
            "decoder_last_debug": self.decoder.last_debug_info,
        }

        if validate_ega and self.use_two_stage_decoder and self.use_ega_refine and len(self.use_ega_layers) > 0:
            triggered = self.decoder.last_debug_info.get("ega_trigger_count", 0)
            if triggered == 0:
                raise RuntimeError(
                    f"EGA validation failed: use_ega_layers={self.use_ega_layers}, "
                    f"but no EGA block was triggered in PASS2."
                )

        return final_pred


def build_fusion_model(
    load_pretrained=False,
    npz_path=None,
    ega_input_mode="pred",
    use_cross_guided_fusion=True,
    use_two_stage_decoder=True,
    use_ega_refine=True
):
    num_classes = setting_config.model_config['num_classes']
    img_size = setting_config.input_size_h

    config_vit = CONFIGS['R50-ViT-B_16']
    config_vit.n_classes = num_classes
    config_vit.n_skip = 3

    transunet_part = VisionTransformer(config_vit, img_size=img_size, num_classes=num_classes)

    visionmamba_part = VisionMamba(
        img_size=img_size,
        patch_size=16,
        embed_dim=192,
        depth=24,
        num_classes=num_classes,
        drop_rate=0.1,
        drop_path_rate=0.2,
        if_cls_token=False,
        rms_norm=False,
        residual_in_fp32=True,
        fused_add_norm=False
    )

    final_model = LateFusionSegEG(
        transunet_part,
        visionmamba_part,
        num_classes=num_classes,
        use_ega_layers=(2,3),
        ega_input_mode=ega_input_mode,
        use_cross_guided_fusion=use_cross_guided_fusion,
        use_two_stage_decoder=use_two_stage_decoder,
        use_ega_refine=use_ega_refine
    )
    return final_model