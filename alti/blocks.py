import torch
from torch import nn
from torch.nn import functional as F


class TextGuidedDecoderBlock(nn.Module):
    """
    Lightweight token-aware text-guided decoder block.

    Args:
        v_dim: channel dimension of decoder feature x.
        l_dim: channel dimension of BERT token features, normally 768.
        hidden_dim: attention hidden dimension. Defaults to v_dim.
        res_scale_init: initial residual scale. 0.0 makes the block identity-safe.
        gate_bias_init: negative value keeps the initial gate small.

    Input:
        x:       [B, C, H, W]
        l_feats: [B, 768, L] or [B, L, 768]
        l_mask:  [B, L, 1] or [B, 1, L] or [B, L]

    Output:
        x_refined: [B, C, H, W]
    """
    total_forward_calls = 0

    def __init__(self, v_dim, l_dim=768, hidden_dim=None, res_scale_init=0.0, gate_bias_init=-4.0):
        super().__init__()
        hidden_dim = hidden_dim or v_dim
        self.v_dim = v_dim
        self.l_dim = l_dim
        self.hidden_dim = hidden_dim

        self.q_proj = nn.Conv2d(v_dim, hidden_dim, kernel_size=1)
        self.k_proj = nn.Conv1d(l_dim, hidden_dim, kernel_size=1)
        self.v_proj = nn.Conv1d(l_dim, hidden_dim, kernel_size=1)

        self.out_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, v_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(v_dim),
            nn.GELU(),
        )

        self.gate = nn.Conv2d(v_dim * 2, v_dim, kernel_size=1)
        if self.gate.bias is not None:
            nn.init.constant_(self.gate.bias, float(gate_bias_init))

        # Identity-safe residual: tanh(0) = 0, so a newly inserted block does not
        # perturb a pretrained checkpoint before fine-tuning.
        self.res_scale = nn.Parameter(torch.tensor(float(res_scale_init)))
        self.scale = hidden_dim ** -0.5

    def _normalize_l_feats(self, l_feats):
        if l_feats is None:
            return None
        if l_feats.dim() != 3:
            raise ValueError(f"TextGuidedDecoderBlock expects 3D l_feats, got {tuple(l_feats.shape)}")
        if l_feats.size(1) == self.l_dim:
            return l_feats
        if l_feats.size(-1) == self.l_dim:
            return l_feats.transpose(1, 2).contiguous()
        raise ValueError(f"Unexpected l_feats shape: {tuple(l_feats.shape)}")

    @staticmethod
    def _mask_to_valid(l_mask, expected_len=None):
        if l_mask is None:
            return None
        if l_mask.dim() == 3:
            if l_mask.size(-1) == 1:
                valid = l_mask.squeeze(-1)
            elif l_mask.size(1) == 1:
                valid = l_mask.squeeze(1)
            else:
                raise ValueError(f"Unexpected l_mask shape: {tuple(l_mask.shape)}")
        elif l_mask.dim() == 2:
            valid = l_mask
        else:
            raise ValueError(f"Unexpected l_mask shape: {tuple(l_mask.shape)}")
        valid = valid.bool()
        if expected_len is not None and valid.size(1) != expected_len:
            raise ValueError(f"l_mask length {valid.size(1)} does not match text length {expected_len}")
        return valid

    def forward(self, x, l_feats, l_mask=None):
        if l_feats is None:
            return x

        l_feats = self._normalize_l_feats(l_feats)
        TextGuidedDecoderBlock.total_forward_calls += 1

        B, C, H, W = x.shape
        _, _, L = l_feats.shape

        q = self.q_proj(x).flatten(2).transpose(1, 2)  # [B, HW, D]
        k = self.k_proj(l_feats)                       # [B, D, L]
        v = self.v_proj(l_feats).transpose(1, 2)       # [B, L, D]

        attn = torch.matmul(q, k) * self.scale          # [B, HW, L]
        valid = self._mask_to_valid(l_mask, expected_len=L)
        if valid is not None:
            attn = attn.masked_fill(~valid.unsqueeze(1), -1e4)
        attn = F.softmax(attn, dim=-1)

        txt = torch.matmul(attn, v)                     # [B, HW, D]
        txt = txt.transpose(1, 2).contiguous().view(B, self.hidden_dim, H, W)
        txt_res = self.out_proj(txt)                    # [B, C, H, W]

        gate = torch.sigmoid(self.gate(torch.cat([x, txt_res], dim=1)))
        scale = torch.tanh(self.res_scale)
        return x + scale * gate * txt_res


class GLFLiteDecoderBlock(nn.Module):
    """
    TD-v3: identity-safe GLF-lite decoder block.

    This block follows the stable global-language modulation idea used by GLF-style
    decoders, but keeps the TD-v2 identity-safe residual design. It avoids dense
    pixel-to-token attention and instead pools valid text tokens into a global
    language vector, then uses the vector to modulate decoder features.

    Input:
        x:       [B, C, H, W]
        l_feats: [B, 768, L] or [B, L, 768]
        l_mask:  [B, L, 1] or [B, 1, L] or [B, L]

    Output:
        x_refined: [B, C, H, W]
    """
    total_forward_calls = 0

    def __init__(self, v_dim, l_dim=768, hidden_dim=None, res_scale_init=0.0, gate_bias_init=-4.0):
        super().__init__()
        hidden_dim = hidden_dim or v_dim
        self.v_dim = v_dim
        self.l_dim = l_dim
        self.hidden_dim = hidden_dim

        # Language projection. Linear is used after masked average pooling.
        self.lang_proj = nn.Sequential(
            nn.Linear(l_dim, v_dim),
            nn.Tanh(),
        )

        # Vision projection resembles a lightweight GLF branch.
        self.vis_proj = nn.Sequential(
            nn.Conv2d(v_dim, v_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(v_dim),
            nn.Tanh(),
        )

        self.out_proj = nn.Sequential(
            nn.Conv2d(v_dim, v_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(v_dim),
            nn.GELU(),
        )

        self.gate = nn.Conv2d(v_dim * 2, v_dim, kernel_size=1)
        if self.gate.bias is not None:
            nn.init.constant_(self.gate.bias, float(gate_bias_init))

        # Identity-safe residual scale. tanh(0)=0, so the newly inserted block
        # does not perturb the pretrained checkpoint at the beginning of fine-tuning.
        self.res_scale = nn.Parameter(torch.tensor(float(res_scale_init)))

    def _normalize_l_feats(self, l_feats):
        if l_feats is None:
            return None
        if l_feats.dim() != 3:
            raise ValueError(f"GLFLiteDecoderBlock expects 3D l_feats, got {tuple(l_feats.shape)}")
        if l_feats.size(1) == self.l_dim:
            return l_feats
        if l_feats.size(-1) == self.l_dim:
            return l_feats.transpose(1, 2).contiguous()
        raise ValueError(f"Unexpected l_feats shape: {tuple(l_feats.shape)}")

    @staticmethod
    def _mask_to_valid(l_mask, expected_len=None):
        if l_mask is None:
            return None
        if l_mask.dim() == 3:
            if l_mask.size(-1) == 1:
                valid = l_mask.squeeze(-1)
            elif l_mask.size(1) == 1:
                valid = l_mask.squeeze(1)
            else:
                raise ValueError(f"Unexpected l_mask shape: {tuple(l_mask.shape)}")
        elif l_mask.dim() == 2:
            valid = l_mask
        else:
            raise ValueError(f"Unexpected l_mask shape: {tuple(l_mask.shape)}")
        valid = valid.bool()
        if expected_len is not None and valid.size(1) != expected_len:
            raise ValueError(f"l_mask length {valid.size(1)} does not match text length {expected_len}")
        return valid

    def _masked_avg_text(self, l_feats, l_mask=None):
        # l_feats: [B, 768, L]
        B, _, L = l_feats.shape
        valid = self._mask_to_valid(l_mask, expected_len=L)
        if valid is None:
            return l_feats.mean(dim=-1)
        weight = valid.to(dtype=l_feats.dtype, device=l_feats.device).unsqueeze(1)  # [B, 1, L]
        denom = weight.sum(dim=-1).clamp_min(1.0)  # [B, 1]
        return (l_feats * weight).sum(dim=-1) / denom

    def forward(self, x, l_feats, l_mask=None):
        if l_feats is None:
            return x

        l_feats = self._normalize_l_feats(l_feats)
        GLFLiteDecoderBlock.total_forward_calls += 1

        lang_vec = self._masked_avg_text(l_feats, l_mask)          # [B, 768]
        lang_gate = self.lang_proj(lang_vec).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]

        vis = self.vis_proj(x)                                     # [B, C, H, W]
        txt_res = self.out_proj(vis * lang_gate)                   # [B, C, H, W]

        gate = torch.sigmoid(self.gate(torch.cat([x, txt_res], dim=1)))
        scale = torch.tanh(self.res_scale)
        return x + scale * gate * txt_res


class AdaptiveGLFLiteDecoderBlock(GLFLiteDecoderBlock):
    """
    TD-v4A / ALTI-GLF and TD-v4A-U / UA-ALTI decoder block.

    It keeps the TD-v3 GLF-lite residual branch and adds a sample-wise
    adaptive alpha gate.  When uncertainty_gate=True, the alpha gate uses
    richer visual statistics: GAP, GMP, and spatial STD.
    """
    total_forward_calls = 0

    def __init__(
        self,
        v_dim,
        l_dim=768,
        hidden_dim=None,
        res_scale_init=0.0,
        gate_bias_init=-4.0,
        alpha_bias_init=-2.0,
            alti_ablation='none',
        alpha_weight_std=0.0,
        uncertainty_gate=False,
    ):
        super().__init__(
            v_dim=v_dim,
            l_dim=l_dim,
            hidden_dim=hidden_dim,
            res_scale_init=res_scale_init,
            gate_bias_init=gate_bias_init,
        )
        self.uncertainty_gate = bool(uncertainty_gate)
        self.alpha_bias_init = float(alpha_bias_init)
        self.alpha_weight_std = float(alpha_weight_std)
        self.alti_ablation = alti_ablation
        self.lang_gate_proj = nn.Linear(l_dim, v_dim)
        gate_in_dim = v_dim * (5 if self.uncertainty_gate else 3)
        gate_hidden = max(v_dim // 4, 32)
        self.inject_gate = nn.Sequential(
            nn.Linear(gate_in_dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
        )

        if self.alpha_weight_std > 0:
            nn.init.normal_(self.inject_gate[-1].weight, std=self.alpha_weight_std)
        else:
            nn.init.zeros_(self.inject_gate[-1].weight)
        nn.init.constant_(self.inject_gate[-1].bias, self.alpha_bias_init)

        self.reset_alpha_stats()

    def reset_alpha_stats(self):
        self.alpha_sum = 0.0
        self.alpha_count = 0
        self.last_alpha = None

    def _compute_alpha(self, x, lang_vec):
        # x: [B, C, H, W], lang_vec: [B, 768]
        v_avg = x.mean(dim=(2, 3))
        l_pool_proj = self.lang_gate_proj(lang_vec)

        if self.uncertainty_gate:
            v_max = x.amax(dim=(2, 3))
            v_std = x.flatten(2).std(dim=-1)
            gate_input = torch.cat([
                v_avg,
                v_max,
                v_std,
                l_pool_proj,
                v_avg * l_pool_proj,
            ], dim=-1)
        else:
            gate_input = torch.cat([
                v_avg,
                l_pool_proj,
                v_avg * l_pool_proj,
            ], dim=-1)

        alpha = torch.sigmoid(self.inject_gate(gate_input)).view(x.size(0), 1, 1, 1)
        return alpha

    def forward(self, x, l_feats, l_mask=None):
        if l_feats is None:
            return x

        l_feats = self._normalize_l_feats(l_feats)
        AdaptiveGLFLiteDecoderBlock.total_forward_calls += 1

        lang_vec = self._masked_avg_text(l_feats, l_mask)          # [B, 768]
        lang_gate = self.lang_proj(lang_vec).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]

        vis = self.vis_proj(x)                                     # [B, C, H, W]
        glf_res = self.out_proj(vis * lang_gate)                   # [B, C, H, W]

        local_gate = torch.sigmoid(self.gate(torch.cat([x, glf_res], dim=1)))
        alpha = self._compute_alpha(x, lang_vec)
        scale = torch.tanh(self.res_scale)
        if self.alti_ablation == 'no_alpha':
            alpha = torch.ones_like(alpha)

        if self.alti_ablation == 'no_local':
            local_gate = torch.ones_like(local_gate)

        if self.alti_ablation == 'no_scale':
            scale = torch.ones_like(scale)

        self.last_alpha = alpha.detach()
        with torch.no_grad():
            self.alpha_sum += float(alpha.detach().sum().item())
            self.alpha_count += int(alpha.numel())

        return x + scale * alpha * local_gate * glf_res

