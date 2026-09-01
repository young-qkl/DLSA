import torch
import torch.nn as nn
import torch.nn.functional as F


class LUPPoolingPrototype(nn.Module):
    def __init__(self, in_dim, embed_dim=256):
        """
        in_dim: 输入特征维度 (backbone输出的channel数)
        embed_dim: embedding后的维度，默认256
        """
        super(LUPPoolingPrototype, self).__init__()

        # 1. 使用1x1卷积将 deep feature maps 投影到 embedding space
        self.projector = nn.Conv2d(in_dim, embed_dim, kernel_size=1)

        # 2. 定义 MLP，用于匹配 pooled instance embedding 与 global prototype
        # 输入：concat(pooled_feat, global_proto) -> embed_dim*2
        # 输出：匹配分数 (1 scalar per instance)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, feat, masks):
        """
        feat: (B,C,H,W) deep feature maps
        masks: (B,1,H,W) binary mask after connected component filtering
        Returns:
            scores: (B,) matching scores for each instance
        """
        B, C, H, W = feat.shape

        # === Step 1. 特征投影到 embedding space ===
        embed = self.projector(feat)  # (B,embed_dim,H,W)

        pooled_feats = []
        for b in range(B):
            mask = masks[b, 0]  # (H,W) binary mask
            embed_b = embed[b]  # (embed_dim,H,W)

            # === Step 2. Masked Average Pooling ===
            # 将 embedding 乘以 mask，仅保留 instance 内的特征
            masked_embed = embed_b * mask.unsqueeze(0)  # (embed_dim,H,W)

            area = mask.sum() + 1e-6  # 避免除零
            pooled = masked_embed.view(embed_b.shape[0], -1).sum(dim=1) / area  # (embed_dim,)
            pooled_feats.append(pooled)

        # === Step 3. 堆叠 pooled embedding (B,embed_dim) ===
        pooled_feats = torch.stack(pooled_feats, dim=0)  # (B,embed_dim)

        # === Step 4. 计算 Global Prototype ===
        # 对 embedding feature maps 做 global average pooling
        proto = embed.mean(dim=[2, 3])  # (B,embed_dim)

        # === Step 5. 将 pooled_feat 与 global_proto concat 做匹配 ===
        concat = torch.cat([pooled_feats, proto], dim=1)  # (B,embed_dim*2)

        # === Step 6. MLP 输出匹配分数 ===
        scores = self.mlp(concat).squeeze(1)  # (B,)

        return scores  # 每个 instance 的匹配 score
