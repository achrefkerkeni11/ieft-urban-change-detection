import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _pick_num_groups(channels: int, max_groups: int = 8) -> int:
    for g in [8, 4, 2, 1]:
        if g <= max_groups and channels % g == 0:
            return g
    return 1


class ConvGNAct(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        k=3,
        s=1,
        p=1,
        d=1,
        groups=8,
        act=True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=k,
            stride=s,
            padding=p,
            dilation=d,
            bias=False,
        )
        self.norm = nn.GroupNorm(
            num_groups=_pick_num_groups(out_ch, groups),
            num_channels=out_ch,
        )
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1, act=True):
        super().__init__()
        self.dw = nn.Conv2d(
            in_ch,
            in_ch,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            groups=in_ch,
            bias=False,
        )
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(
            num_groups=_pick_num_groups(out_ch, 8),
            num_channels=out_ch,
        )
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ResidualDSBlock(nn.Module):
    def __init__(self, ch, dilation=1, dropout=0.0):
        super().__init__()
        self.block1 = DepthwiseSeparableConv(ch, ch, dilation=dilation, act=True)
        self.block2 = DepthwiseSeparableConv(ch, ch, dilation=dilation, act=False)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.out_act = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.block1(x)
        x = self.drop(x)
        x = self.block2(x)
        x = x + residual
        x = self.out_act(x)
        return x


class MultiDilatedContext(nn.Module):
    def __init__(self, ch, dropout=0.0):
        super().__init__()
        mid = max(ch // 4, 32)

        self.b0 = ConvGNAct(ch, mid, k=1, s=1, p=0)
        self.b1 = ConvGNAct(ch, mid, k=3, s=1, p=1, d=1)
        self.b2 = ConvGNAct(ch, mid, k=3, s=1, p=2, d=2)
        self.b3 = ConvGNAct(ch, mid, k=3, s=1, p=3, d=3)

        self.pool_proj = ConvGNAct(ch, mid, k=1, s=1, p=0)
        self.fuse = nn.Sequential(
            ConvGNAct(mid * 5, ch, k=1, s=1, p=0),
            ResidualDSBlock(ch, dilation=1, dropout=dropout),
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        p = F.adaptive_avg_pool2d(x, output_size=1)
        p = self.pool_proj(p)
        p = F.interpolate(p, size=(h, w), mode="bilinear", align_corners=False)

        y = torch.cat([self.b0(x), self.b1(x), self.b2(x), self.b3(x), p], dim=1)
        y = self.fuse(y)
        return y


class UpFuseBlock(nn.Module):
    def __init__(self, ch, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            ConvGNAct(ch * 2, ch, k=1, s=1, p=0),
            ResidualDSBlock(ch, dilation=1, dropout=dropout),
            ResidualDSBlock(ch, dilation=2, dropout=dropout),
        )

    def forward(self, x_low, x_high_up):
        x = torch.cat([x_low, x_high_up], dim=1)
        return self.block(x)


class DetailRefineBlock(nn.Module):
    def __init__(self, ch, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            ResidualDSBlock(ch, dilation=1, dropout=dropout),
            ResidualDSBlock(ch, dilation=1, dropout=dropout),
            ResidualDSBlock(ch, dilation=2, dropout=dropout),
        )

    def forward(self, x):
        return self.block(x)


class BoundaryHead(nn.Module):
    def __init__(self, in_ch, mid_ch=128, dropout=0.0):
        super().__init__()
        self.stem = ConvGNAct(in_ch, mid_ch, k=1, s=1, p=0)
        self.refine = nn.Sequential(
            ResidualDSBlock(mid_ch, dilation=1, dropout=dropout),
            ResidualDSBlock(mid_ch, dilation=2, dropout=dropout),
        )
        self.head = nn.Conv2d(mid_ch, 1, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        x = self.refine(x)
        return self.head(x)


class LightMultiScaleChangeDecoder(nn.Module):
    def __init__(self, in_dim=768, decoder_dim=256, dropout=0.1):
        super().__init__()

        if in_dim % 4 != 0:
            raise ValueError(
                f"LightMultiScaleChangeDecoder attend in_dim divisible par 4, reçu: {in_dim}"
            )

        self.in_dim = int(in_dim)
        self.base_dim = int(in_dim // 4)
        self.decoder_dim = int(decoder_dim)

        branch_dim = max(decoder_dim // 2, 64)
        boundary_dim = max(decoder_dim // 2, 64)

        self.proj_t1 = ConvGNAct(self.base_dim, branch_dim, k=1, s=1, p=0)
        self.proj_t2 = ConvGNAct(self.base_dim, branch_dim, k=1, s=1, p=0)
        self.proj_abs = ConvGNAct(self.base_dim, branch_dim, k=1, s=1, p=0)
        self.proj_temp = ConvGNAct(self.base_dim, branch_dim, k=1, s=1, p=0)
        self.proj_signed = ConvGNAct(self.base_dim, branch_dim, k=1, s=1, p=0)

        self.main_stem = nn.Sequential(
            ConvGNAct(branch_dim * 3, decoder_dim, k=1, s=1, p=0),
            ResidualDSBlock(decoder_dim, dilation=1, dropout=dropout),
        )

        self.down_s2 = ConvGNAct(decoder_dim, decoder_dim, k=3, s=2, p=1)
        self.down_s3 = ConvGNAct(decoder_dim, decoder_dim, k=3, s=2, p=1)

        self.ctx_s1 = MultiDilatedContext(decoder_dim, dropout=dropout)
        self.ctx_s2 = MultiDilatedContext(decoder_dim, dropout=dropout)
        self.ctx_s3 = MultiDilatedContext(decoder_dim, dropout=dropout)

        self.fuse_s2 = UpFuseBlock(decoder_dim, dropout=dropout)
        self.fuse_s1 = UpFuseBlock(decoder_dim, dropout=dropout)

        self.detail_stem = nn.Sequential(
            ConvGNAct(branch_dim * 4, decoder_dim, k=1, s=1, p=0),
            ResidualDSBlock(decoder_dim, dilation=1, dropout=dropout),
        )
        self.local_contrast_proj = ConvGNAct(branch_dim, decoder_dim, k=1, s=1, p=0)
        self.detail_refine = DetailRefineBlock(decoder_dim, dropout=dropout)

        self.main_detail_gate = nn.Sequential(
            nn.Conv2d(decoder_dim * 2, decoder_dim, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(decoder_dim, decoder_dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.final_refine = nn.Sequential(
            ResidualDSBlock(decoder_dim, dilation=1, dropout=dropout),
            ResidualDSBlock(decoder_dim, dilation=2, dropout=dropout),
            nn.Dropout2d(dropout),
        )

        self.main_head = nn.Conv2d(decoder_dim, 1, kernel_size=1)
        self.detail_head = nn.Conv2d(decoder_dim, 1, kernel_size=1)
        self.boundary_head = BoundaryHead(
            in_ch=decoder_dim * 2,
            mid_ch=boundary_dim,
            dropout=dropout,
        )

        self.detail_weight = 0.35
        self.boundary_weight = 0.20

    def _tokens_to_map(self, patch_tokens):
        b, n, c = patch_tokens.shape
        g = int(math.sqrt(n))
        if g * g != n:
            raise ValueError(f"Le nombre de patch tokens n'est pas un carré parfait: N={n}")
        x = patch_tokens.transpose(1, 2).contiguous().view(b, c, g, g)
        return x

    def _high_pass(self, x):
        low = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        return x - low

    def _split_modal_parts(self, x):
        if x.shape[1] != self.in_dim:
            raise ValueError(
                f"Canaux inattendus: attendu {self.in_dim}, reçu {x.shape[1]}"
            )
        t1, t2, dabs, temp = torch.chunk(x, chunks=4, dim=1)
        dsigned = t2 - t1
        return t1, t2, dabs, temp, dsigned

    def forward(self, patch_tokens):
        x = self._tokens_to_map(patch_tokens)
        t1, t2, dabs, temp, dsigned = self._split_modal_parts(x)

        t1_p = self.proj_t1(t1)
        t2_p = self.proj_t2(t2)
        abs_p = self.proj_abs(dabs)
        temp_p = self.proj_temp(temp)
        signed_p = self.proj_signed(dsigned)

        main_in = torch.cat([abs_p, signed_p, temp_p], dim=1)
        main_s1 = self.main_stem(main_in)
        main_s1 = self.ctx_s1(main_s1)

        main_s2 = self.down_s2(main_s1)
        main_s2 = self.ctx_s2(main_s2)

        main_s3 = self.down_s3(main_s2)
        main_s3 = self.ctx_s3(main_s3)

        main_s3_up = F.interpolate(
            main_s3,
            size=main_s2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        main_f2 = self.fuse_s2(main_s2, main_s3_up)

        main_f2_up = F.interpolate(
            main_f2,
            size=main_s1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        main_f1 = self.fuse_s1(main_s1, main_f2_up)

        hp_abs = self._high_pass(abs_p)
        hp_signed = self._high_pass(signed_p)
        local_contrast = torch.abs(t2_p - t1_p)
        local_contrast = self.local_contrast_proj(local_contrast)

        detail_in = torch.cat([abs_p, signed_p, hp_abs, hp_signed], dim=1)
        detail = self.detail_stem(detail_in)
        detail = detail + local_contrast
        detail = self.detail_refine(detail)

        gate = self.main_detail_gate(torch.cat([main_f1, detail], dim=1))
        fused = main_f1 + gate * detail
        fused = self.final_refine(fused)

        detail_hp = self._high_pass(detail)
        boundary_in = torch.cat([detail, detail_hp], dim=1)

        main_logits = self.main_head(fused)
        detail_logits = self.detail_head(detail)
        boundary_logits = self.boundary_head(boundary_in)

        logits = (
            main_logits
            + self.detail_weight * detail_logits
            + self.boundary_weight * boundary_logits
        )

        logits = logits.squeeze(1)
        return logits