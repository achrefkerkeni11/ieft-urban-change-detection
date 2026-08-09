import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvGN(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        groups = 1
        for cand in [8, 4, 2, 1]:
            if out_ch % cand == 0:
                groups = cand
                break
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class _ResBlock(nn.Module):
    def __init__(self, ch: int, dropout: float = 0.0):
        super().__init__()
        groups = 1
        for cand in [8, 4, 2, 1]:
            if ch % cand == 0:
                groups = cand
                break
        self.body = nn.Sequential(
            _ConvGN(ch, ch),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
            nn.GroupNorm(groups, ch),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.body(x))


class LevelChangeHead(nn.Module):
    def __init__(self, hidden_size: int = 384, out_dim: int = 160, dropout: float = 0.08):
        super().__init__()
        in_dim = 4 * hidden_size + 1
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2),
            nn.LayerNorm(out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )
        self.head = nn.Linear(out_dim, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, f_t1: torch.Tensor, f_t2: torch.Tensor):
        diff = torch.abs(f_t2 - f_t1)
        prod = f_t1 * f_t2
        f1_n = F.normalize(f_t1, p=2, dim=-1, eps=1e-6)
        f2_n = F.normalize(f_t2, p=2, dim=-1, eps=1e-6)
        cos_sim = (f1_n * f2_n).sum(dim=-1, keepdim=True)
        feat = torch.cat([f_t1, f_t2, diff, prod, cos_sim], dim=-1)
        embed = self.proj(feat)
        logit = self.head(embed).squeeze(-1)
        return logit, embed


class RGBFusionEncoder(nn.Module):
    def __init__(self, in_ch: int = 12, dropout: float = 0.08):
        super().__init__()
        self.e1 = nn.Sequential(_ConvGN(in_ch, 32), _ResBlock(32, dropout))
        self.e2 = nn.Sequential(_ConvGN(32, 64, k=3, s=2, p=1), _ResBlock(64, dropout))
        self.e3 = nn.Sequential(_ConvGN(64, 96, k=3, s=2, p=1), _ResBlock(96, dropout))
        self.e4 = nn.Sequential(_ConvGN(96, 128, k=3, s=2, p=1), _ResBlock(128, dropout))

    def forward(self, x: torch.Tensor):
        x1 = self.e1(x)
        x2 = self.e2(x1)
        x3 = self.e3(x2)
        x4 = self.e4(x3)
        return x1, x2, x3, x4


class _UpFuse(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.08):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1)
        self.fuse = nn.Sequential(
            _ConvGN(out_ch + skip_ch, out_ch),
            _ResBlock(out_ch, dropout),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


class PixelObjectDecoder(nn.Module):
    def __init__(self, token_dim: int = 160, dropout: float = 0.08):
        super().__init__()
        self.token_stem = nn.Sequential(_ConvGN(token_dim, 160), _ResBlock(160, dropout))
        self.up0 = _UpFuse(160, 128, 128, dropout=dropout)
        self.up1 = _UpFuse(128, 96, 96, dropout=dropout)
        self.up2 = _UpFuse(96, 64, 64, dropout=dropout)
        self.up3 = _UpFuse(64, 32, 32, dropout=dropout)
        self.refine = nn.Sequential(_ConvGN(32, 32), _ResBlock(32, dropout))
        self.out_head = nn.Conv2d(32, 1, 1)
        self.boundary_head = nn.Conv2d(32, 1, 1)

    def forward(self, token_map: torch.Tensor, rgb_feats):
        x1, x2, x3, x4 = rgb_feats
        x = self.token_stem(token_map)
        x = self.up0(x, x4)
        x = self.up1(x, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.refine(x)
        return self.out_head(x), self.boundary_head(x)


class MultiScalePixelObjectChangeDecoder(nn.Module):
    def __init__(self, hidden_size: int = 384, grid_size: int = 16, num_levels: int = 4, decoder_dim: int = 160, dropout: float = 0.08):
        super().__init__()
        self.grid_size = grid_size
        self.num_levels = num_levels
        self.decoder_dim = decoder_dim

        self.level_heads = nn.ModuleList([
            LevelChangeHead(hidden_size=hidden_size, out_dim=decoder_dim, dropout=dropout)
            for _ in range(num_levels)
        ])
        self.level_logit_weights = nn.Parameter(torch.zeros(num_levels))
        self.level_feat_weights = nn.Parameter(torch.zeros(num_levels))

        self.rgb_encoder = RGBFusionEncoder(in_ch=12, dropout=dropout)
        self.pixel_decoder = PixelObjectDecoder(token_dim=decoder_dim, dropout=dropout)

    def forward(self, level_t1_feats: list, level_t2_feats: list, t1: torch.Tensor, t2: torch.Tensor):
        assert len(level_t1_feats) == self.num_levels
        assert len(level_t2_feats) == self.num_levels
        b = level_t1_feats[0].size(0)
        g = self.grid_size

        level_logits = []
        level_embeds = []
        for i in range(self.num_levels):
            lg, emb = self.level_heads[i](level_t1_feats[i], level_t2_feats[i])
            level_logits.append(lg)
            level_embeds.append(emb)

        lw = torch.softmax(self.level_logit_weights, dim=0)
        fw = torch.softmax(self.level_feat_weights, dim=0)

        coarse_logits = 0.0
        fused_feat = 0.0
        for i in range(self.num_levels):
            coarse_logits = coarse_logits + lw[i] * level_logits[i]
            fused_feat = fused_feat + fw[i] * level_embeds[i]

        token_map = fused_feat.view(b, g, g, self.decoder_dim).permute(0, 3, 1, 2).contiguous()

        diff = torch.abs(t2 - t1)
        prod = t1 * t2
        rgb_in = torch.cat([t1, t2, diff, prod], dim=1)
        rgb_feats = self.rgb_encoder(rgb_in)
        dense_logits, boundary_logits = self.pixel_decoder(token_map, rgb_feats)

        return coarse_logits, dense_logits, boundary_logits, level_logits
