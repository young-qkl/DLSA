import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


class SgAMC(nn.Module):
    """
    Token-aware Semantic-guided Adaptive Multi-scale Convolution.

    输入支持：
    1) x: (B, N, C)，backbone 中的 Swin token
    2) x: (B, C, H, W)，decoder 中的 feature map

    文本输入：
    l_feats: (B, 768, L) 或 (B, L, 768)
    l_mask:  (B, L, 1) 或 (B, 1, L) 或 (B, L)

    输出：
    与 x 输入形状一致。
    """

    def __init__(
        self,
        in_channels,
        out_channels=None,
        mode='language',
        lang_channels=768,
        hidden_dim=128,
        kernels=(1, 3, 5, 7),
        attn_drop=0.0,
        kernel_ablation='none',
    ):
        super().__init__()
        self.mode = mode
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.lang_channels = lang_channels
        self.hidden_dim = hidden_dim

        # Kernel branch ablation switch.
        # SgAMC is created inside the backbone, so we support both an init argument
        # and an environment variable for direct experiment control.
        # Valid values:
        #   none  : use full kernel set {1,3,5,7}
        #   wo_k1 : remove 1x1 branch
        #   wo_k3 : remove 3x3 branch
        #   wo_k5 : remove 5x5 branch
        #   wo_k7 : remove 7x7 branch
        kernel_ablation = os.environ.get('SGAMC_KERNEL_ABLATION', kernel_ablation)
        alias = {
            'full': 'none',
            'drop1': 'wo_k1',
            'drop3': 'wo_k3',
            'drop5': 'wo_k5',
            'drop7': 'wo_k7',
            'without1': 'wo_k1',
            'without3': 'wo_k3',
            'without5': 'wo_k5',
            'without7': 'wo_k7',
        }
        kernel_ablation = alias.get(str(kernel_ablation), str(kernel_ablation))
        valid_ablation = {'none', 'wo_k1', 'wo_k3', 'wo_k5', 'wo_k7'}
        if kernel_ablation not in valid_ablation:
            raise ValueError(
                f"Unsupported SGAMC_KERNEL_ABLATION={kernel_ablation}. "
                f"Expected one of {sorted(valid_ablation)}."
            )

        remove_kernel = {
            'wo_k1': 1,
            'wo_k3': 3,
            'wo_k5': 5,
            'wo_k7': 7,
        }.get(kernel_ablation, None)
        if remove_kernel is not None:
            kernels = tuple(k for k in kernels if k != remove_kernel)
            if len(kernels) == 0:
                raise ValueError("SgAMC kernel ablation removed all branches.")

        self.kernel_ablation = kernel_ablation
        self.kernels = kernels
        self.num_branches = len(kernels)

        # 多尺度卷积分支：仍然对 channel-mean map 做卷积，保持你原始设计的轻量性
        self.convs = nn.ModuleList([
            nn.Conv2d(1, 1, kernel_size=k, padding=k // 2)
            for k in kernels
        ])

        # visual token projection
        self.vis_proj = nn.Linear(in_channels, hidden_dim)

        # language token projection
        self.lang_key = nn.Conv1d(lang_channels, hidden_dim, kernel_size=1)
        self.lang_value = nn.Conv1d(lang_channels, hidden_dim, kernel_size=1)

        # 每个 kernel 分支一个 learnable query，用来从文本 token 中提取 branch-specific semantic context
        self.branch_queries = nn.Parameter(
            torch.randn(self.num_branches, hidden_dim) * 0.02
        )

        self.scale = hidden_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)

        # 用 visual token 和 branch text context 共同决定每个位置的 kernel 权重
        self.route_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

        # 把四个 kernel response 融合成一个 modulation map
        self.fusion_proj = nn.Sequential(
            nn.Conv2d(self.num_branches, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 方便后续可视化和写论文
        self.last_token_attn = None       # (B, 4, L)
        self.last_kernel_weights = None   # (B, K, H, W)

    def _to_feature_map(self, x):
        """
        把输入统一成：
        input_map:   (B, C, H, W)
        token_feats: (B, H*W, C)
        input_is_4d: bool
        """
        if x.dim() == 4:
            B, C, H, W = x.shape
            input_map = x
            token_feats = x.flatten(2).transpose(1, 2)  # (B, HW, C)
            return input_map, token_feats, H, W, True

        if x.dim() == 3:
            B, N, C = x.shape
            H = W = int(np.sqrt(N))
            assert H * W == N, f"SgAMC expects square token map, but got N={N}"
            input_map = x.transpose(1, 2).contiguous().view(B, C, H, W)
            token_feats = x
            return input_map, token_feats, H, W, False

        raise ValueError(f"Unsupported x shape: {x.shape}")

    def _normalize_text_inputs(self, l_feats, l_mask, lang_feat, device):
        """
        输出：
        lang_seq: (B, 768, L)
        valid:    (B, L), bool
        """
        # 兼容旧调用：self.sg_amc(x, lang_feat, stage_id=...)
        if l_feats is not None and l_feats.dim() == 2 and lang_feat is None:
            lang_feat = l_feats
            l_feats = None

        if l_feats is None:
            assert lang_feat is not None, "Either l_feats or lang_feat must be provided."
            lang_seq = lang_feat.unsqueeze(-1)  # (B, 768, 1)
            valid = torch.ones(
                lang_seq.size(0), 1,
                dtype=torch.bool,
                device=device
            )
            return lang_seq, valid

        # l_feats can be (B, 768, L) or (B, L, 768)
        if l_feats.size(1) == self.lang_channels:
            lang_seq = l_feats
        elif l_feats.size(-1) == self.lang_channels:
            lang_seq = l_feats.transpose(1, 2).contiguous()
        else:
            raise ValueError(f"Unexpected l_feats shape: {l_feats.shape}")

        B, _, L = lang_seq.shape

        if l_mask is None:
            valid = torch.ones(B, L, dtype=torch.bool, device=device)
        else:
            if l_mask.dim() == 3:
                if l_mask.size(-1) == 1:
                    valid = l_mask.squeeze(-1)
                elif l_mask.size(1) == 1:
                    valid = l_mask.squeeze(1)
                else:
                    raise ValueError(f"Unexpected l_mask shape: {l_mask.shape}")
            elif l_mask.dim() == 2:
                valid = l_mask
            else:
                raise ValueError(f"Unexpected l_mask shape: {l_mask.shape}")

            valid = valid.to(device=device).bool()

        return lang_seq, valid

    def _fixed_or_avg_weights(self, B, H, W, device):
        weights = torch.zeros(B, self.num_branches, H, W, device=device)

        if self.mode == 'avg':
            weights.fill_(1.0 / self.num_branches)
            return weights

        # Map fixed-kernel modes to the current branch index.
        # When a branch is removed, its fixed mode is no longer valid.
        fixed_map = {f'fixed{k}': i for i, k in enumerate(self.kernels)}
        if self.mode in fixed_map:
            weights[:, fixed_map[self.mode], :, :] = 1.0
            return weights

        return None

    def forward(
        self,
        x,
        l_feats=None,
        l_mask=None,
        lang_feat=None,
        stage_id=0,
        return_attn=False
    ):
        input_map, token_feats, H, W, input_is_4d = self._to_feature_map(x)
        B, C, _, _ = input_map.shape
        device = input_map.device
        N = H * W

        # 1. multi-kernel branch responses
        mean_input = input_map.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        branch_maps = [conv(mean_input) for conv in self.convs]
        x_stack = torch.cat(branch_maps, dim=1)  # (B, K, H, W)

        # 2. non-language routing modes
        fixed_weights = self._fixed_or_avg_weights(B, H, W, device)
        if self.mode != 'language' and fixed_weights is not None:
            weights_grid = fixed_weights

        else:
            # 3. normalize text tokens
            lang_seq, valid = self._normalize_text_inputs(
                l_feats=l_feats,
                l_mask=l_mask,
                lang_feat=lang_feat,
                device=device
            )  # lang_seq: (B, 768, L), valid: (B, L)

            # 4. branch-specific text attention
            # K, V: (B, L, D)
            K_txt = self.lang_key(lang_seq).transpose(1, 2)
            V_txt = self.lang_value(lang_seq).transpose(1, 2)

            # Q_branch: (K, D)
            Q_branch = F.normalize(self.branch_queries, dim=-1)

            # attn_logits: (B, K, L)
            attn_logits = torch.einsum('kd,bld->bkl', Q_branch, K_txt) * self.scale
            attn_logits = attn_logits.masked_fill(~valid.unsqueeze(1), -1e4)

            token_attn = F.softmax(attn_logits, dim=-1)
            token_attn = self.attn_drop(token_attn)

            # branch_text: (B, K, D)
            branch_text = torch.einsum('bkl,bld->bkd', token_attn, V_txt)

            # 5. visual-token and branch-text compatibility
            visual_ctx = self.vis_proj(token_feats)  # (B, N, D)

            visual_expand = visual_ctx.unsqueeze(2).expand(
                B, N, self.num_branches, self.hidden_dim
            )
            text_expand = branch_text.unsqueeze(1).expand(
                B, N, self.num_branches, self.hidden_dim
            )

            route_input = torch.cat(
                [visual_expand, text_expand, visual_expand * text_expand],
                dim=-1
            )  # (B, N, K, 3D)

            scores = self.route_mlp(route_input).squeeze(-1)  # (B, N, K)
            weights = F.softmax(scores, dim=-1)  # (B, N, K)

            weights_grid = weights.view(B, H, W, self.num_branches).permute(
                0, 3, 1, 2
            ).contiguous()  # (B, K, H, W)

            self.last_token_attn = token_attn.detach()
            self.last_kernel_weights = weights_grid.detach()

        # 6. weighted multi-scale fusion
        fusion_feat = weights_grid * x_stack  # (B, K, H, W)
        weight_map = self.fusion_proj(fusion_feat)  # (B, 1, H, W)

        # 7. semantic modulation
        output = input_map * (1.0 + weight_map)

        if return_attn:
            aux = {
                "token_attn": self.last_token_attn,
                "kernel_weights": self.last_kernel_weights,
                "weight_map": weight_map.detach()
            }
        else:
            aux = None

        if input_is_4d:
            return (output, aux) if return_attn else output

        out = output.flatten(2).transpose(1, 2).contiguous()  # (B, N, C)
        return (out, aux) if return_attn else out