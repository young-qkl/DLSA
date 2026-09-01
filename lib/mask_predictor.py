import torch
from torch import nn
from torch.nn import functional as F
from arc import AdaptiveRotatedConv2d, RountingFunction
from alti import (
    AdaptiveGLFLiteDecoderBlock,
    GLFLiteDecoderBlock,
    TextGuidedDecoderBlock,
)
from sgamc import SgAMC

class SimpleDecoding(nn.Module):
    def __init__(self, c4_dims, factor=2, args=None):
        super(SimpleDecoding, self).__init__()
        self.args = args
        self.use_text_decoder = bool(getattr(args, "use_text_decoder", False))
        self.text_decoder_stages = getattr(args, "text_decoder_stages", "all")
        self.text_decoder_variant = getattr(args, "text_decoder_variant", "attn")
        self.td_res_scale_init = float(getattr(args, "td_res_scale_init", 0.0))
        self.td_gate_bias_init = float(getattr(args, "td_gate_bias_init", -4.0))
        self.td_alpha_bias_init = float(getattr(args, "td_alpha_bias_init", -2.0))
        self.td_alpha_weight_std = float(getattr(args, "td_alpha_weight_std", 0.0))
        self.td_uncertainty_gate = bool(getattr(args, "td_uncertainty_gate", False))
        self.alti_ablation = getattr(args, "alti_ablation", "none")
        # Existing decoder-side SgAMC modules are kept for state_dict compatibility,
        # but are disabled by default because the original decoding path did not pass
        # language into the classifier.
        self.use_decoder_sgamc = bool(getattr(args, "use_decoder_sgamc", False))

        hidden_size = c4_dims // factor
        c4_size = c4_dims
        c3_size = c4_dims // (factor ** 1)
        c2_size = c4_dims // (factor ** 2)
        c1_size = c4_dims // (factor ** 3)

        self.conv1_4 = nn.Conv2d(c4_size + c3_size, hidden_size, 3, padding=1, bias=False)
        routing_function1 = RountingFunction(in_channels=hidden_size, kernel_number=1)
        self.conv2_4 = AdaptiveRotatedConv2d(
            in_channels=hidden_size, out_channels=hidden_size,
            kernel_size=3, padding=1, rounting_func=routing_function1,
            bias=False, kernel_number=1,
        )
        self.bn1_4 = nn.BatchNorm2d(hidden_size)
        self.relu1_4 = nn.ReLU()
        self.bn2_4 = nn.BatchNorm2d(hidden_size)
        self.relu2_4 = nn.ReLU()

        self.conv1_3 = nn.Conv2d(hidden_size + c2_size, hidden_size, 3, padding=1, bias=False)
        routing_function2 = RountingFunction(in_channels=hidden_size, kernel_number=1)
        self.conv2_3 = AdaptiveRotatedConv2d(
            in_channels=hidden_size, out_channels=hidden_size,
            kernel_size=3, padding=1, rounting_func=routing_function2,
            bias=False, kernel_number=1,
        )
        self.bn1_3 = nn.BatchNorm2d(hidden_size)
        self.relu1_3 = nn.ReLU()
        self.bn2_3 = nn.BatchNorm2d(hidden_size)
        self.relu2_3 = nn.ReLU()

        self.conv1_2 = nn.Conv2d(hidden_size + c1_size, hidden_size, 3, padding=1, bias=False)
        routing_function3 = RountingFunction(in_channels=hidden_size, kernel_number=1)
        self.conv2_2 = AdaptiveRotatedConv2d(
            in_channels=hidden_size, out_channels=hidden_size,
            kernel_size=3, padding=1, rounting_func=routing_function3,
            bias=False, kernel_number=1,
        )
        self.bn1_2 = nn.BatchNorm2d(hidden_size)
        self.relu1_2 = nn.ReLU()
        self.bn2_2 = nn.BatchNorm2d(hidden_size)
        self.relu2_2 = nn.ReLU()

        sg_mode = getattr(args, "sg_mode", "language")
        self.sg_amc = SgAMC(in_channels=1024, out_channels=1024, mode=sg_mode)
        self.sg_amc4 = SgAMC(in_channels=c4_size, out_channels=c4_size, mode=sg_mode)
        self.sg_amc3 = SgAMC(in_channels=hidden_size, out_channels=hidden_size, mode=sg_mode)
        self.sg_amc2 = SgAMC(in_channels=hidden_size, out_channels=hidden_size, mode=sg_mode)
        self.sg_amc1 = SgAMC(in_channels=hidden_size, out_channels=hidden_size, mode=sg_mode)

        self.tg4 = self._make_text_decoder_block(c4_size)
        self.tg3 = self._make_text_decoder_block(hidden_size)
        self.tg2 = self._make_text_decoder_block(hidden_size)
        self.tg1 = self._make_text_decoder_block(hidden_size)

        self.conv1_1 = nn.Conv2d(hidden_size, 2, 1)

    def _make_text_decoder_block(self, channels):
        if self.text_decoder_variant == "attn":
            return TextGuidedDecoderBlock(
                channels, 768,
                res_scale_init=self.td_res_scale_init,
                gate_bias_init=self.td_gate_bias_init,
            )
        if self.text_decoder_variant == "glf":
            return GLFLiteDecoderBlock(
                channels, 768,
                res_scale_init=self.td_res_scale_init,
                gate_bias_init=self.td_gate_bias_init,
            )
        if self.text_decoder_variant == "glf_adapt":
            return AdaptiveGLFLiteDecoderBlock(
                channels, 768,
                res_scale_init=self.td_res_scale_init,
                gate_bias_init=self.td_gate_bias_init,
                alpha_bias_init=self.td_alpha_bias_init,
                alpha_weight_std=self.td_alpha_weight_std,
                uncertainty_gate=self.td_uncertainty_gate,
                alti_ablation=self.alti_ablation,
            )
        raise ValueError(f"Unknown text_decoder_variant: {self.text_decoder_variant}")

    def _enable_tg(self, name):
        if not self.use_text_decoder:
            return False
        mode = self.text_decoder_stages
        if mode == "none":
            return False
        if mode == "tg1":
            return name == "tg1"
        if mode == "tg2_tg1":
            return name in ["tg2", "tg1"]
        if mode == "all":
            return name in ["tg4", "tg3", "tg2", "tg1"]
        if mode == "tg4_tg3":
            return name in ["tg4", "tg3"]
        raise ValueError(f"Unknown text_decoder_stages: {mode}")

    def _call_sgamc_compat(self, module, x, l_feats=None, l_mask=None, lang_feat=None, stage_id=0):
        # Kept only for optional decoder-SgAMC experiments. The default path never calls this.
        try:
            return module(x, l_feats=l_feats, l_mask=l_mask, lang_feat=lang_feat, stage_id=stage_id)
        except TypeError:
            return module(x, lang_feat, stage_id=stage_id)

    def forward(self, x_c4, x_c3, x_c2, x_c1, lang_feat=None, l_feats=None, l_mask=None, return_feat=False):
        # Stage 4: top feature before upsampling.
        if self._enable_tg("tg4") and l_feats is not None:
            x_c4 = self.tg4(x_c4, l_feats, l_mask)
        if self.use_decoder_sgamc and lang_feat is not None:
            x_c4 = self._call_sgamc_compat(self.sg_amc4, x_c4, l_feats, l_mask, lang_feat, stage_id=3)

        if x_c4.size(-2) < x_c3.size(-2) or x_c4.size(-1) < x_c3.size(-1):
            x_c4 = F.interpolate(input=x_c4, scale_factor=2, mode="bilinear", align_corners=True)

        x = torch.cat([x_c4, x_c3], dim=1)
        x = self.conv1_4(x)
        x = self.bn1_4(x)
        x = self.relu1_4(x)

        if self._enable_tg("tg3") and l_feats is not None:
            x = self.tg3(x, l_feats, l_mask)
        if self.use_decoder_sgamc and lang_feat is not None:
            x = self._call_sgamc_compat(self.sg_amc3, x, l_feats, l_mask, lang_feat, stage_id=2)

        x = self.conv2_4(x)
        x = self.bn2_4(x)
        x = self.relu2_4(x)

        # Stage 3.
        if x.size(-2) < x_c2.size(-2) or x.size(-1) < x_c2.size(-1):
            x = F.interpolate(input=x, scale_factor=2, mode="bilinear", align_corners=True)
        x = torch.cat([x, x_c2], dim=1)
        x = self.conv1_3(x)
        x = self.bn1_3(x)
        x = self.relu1_3(x)

        if self._enable_tg("tg2") and l_feats is not None:
            x = self.tg2(x, l_feats, l_mask)
        if self.use_decoder_sgamc and lang_feat is not None:
            x = self._call_sgamc_compat(self.sg_amc2, x, l_feats, l_mask, lang_feat, stage_id=1)

        x = self.conv2_3(x)
        x = self.bn2_3(x)
        x = self.relu2_3(x)

        # Stage 2 / final decoder feature.
        if x.size(-2) < x_c1.size(-2) or x.size(-1) < x_c1.size(-1):
            x = F.interpolate(input=x, scale_factor=2, mode="bilinear", align_corners=True)
        x = torch.cat([x, x_c1], dim=1)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu1_2(x)

        if self._enable_tg("tg1") and l_feats is not None:
            x = self.tg1(x, l_feats, l_mask)
        if self.use_decoder_sgamc and lang_feat is not None:
            x = self._call_sgamc_compat(self.sg_amc1, x, l_feats, l_mask, lang_feat, stage_id=0)

        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.relu2_2(x)

        dec_feat = x
        logits = self.conv1_1(dec_feat)
        if return_feat:
            return logits, dec_feat
        return logits
