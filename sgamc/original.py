import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SgAMC(nn.Module):
    def __init__(self, in_channels, out_channels, mode='language', num_stages=4, stage_dim=16, use_stage_embedding=False):
        super(SgAMC, self).__init__()
        self.mode = mode
        self.num_stages = num_stages
        self.use_stage_embedding = use_stage_embedding

        # 多分支卷积
        self.conv1 = nn.Conv2d(1, 1, kernel_size=1)
        self.conv3 = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(1, 1, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(1, 1, kernel_size=7, padding=3)

        # Token-wise gating FC: input is [token_feature; lang_feature]
        self.token_fc = nn.Linear(in_channels + 768, 4)

        # Attention-based fusion projection
        self.fusion_proj = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, lang_feat, stage_id):
        bs, n, dim = x.shape
        h, w = int(np.sqrt(n)), int(np.sqrt(n))

        # reshape x to feature map
        input = x.view(bs, h, w, dim).permute(0, 3, 1, 2)  # (B, dim, h, w)
        mean_input = torch.mean(input, dim=1, keepdim=True)  # (B,1,h,w)

        # 多分支卷积
        x1 = self.conv1(mean_input)
        x3 = self.conv3(mean_input)
        x5 = self.conv5(mean_input)
        x7 = self.conv7(mean_input)
        x_stack = torch.cat([x1, x3, x5, x7], dim=1)  # (B,4,h,w)

        # Token-wise conditioning
        # Prepare token-wise features
        token_feats = input.permute(0,2,3,1).reshape(bs, h*w, dim)  # (B, n, dim)
        lang_feat_expanded = lang_feat.unsqueeze(1).expand(-1, n, -1)  # (B, n, 768)
        token_condition = torch.cat([token_feats, lang_feat_expanded], dim=-1)  # (B, n, dim+768)

        # Predict per-token per-branch weights
        weights = self.token_fc(token_condition)  # (B, n, 4)
        weights = F.softmax(weights, dim=-1).view(bs, h, w, 4).permute(0, 3, 1, 2)  # (B,4,h,w)

        # Attention-based fusion
        fusion_feat = weights * x_stack  # (B,4,h,w), element-wise multiplication
        weight_map = self.fusion_proj(fusion_feat)  # (B,1,h,w)

        # modulation
        output = input * (1 + weight_map)
        # reshape back
        out = output.reshape(bs, dim, -1).permute(0, 2, 1)  # (B, n, dim)

        return out