# Implementation from the paper: SegDiff: Unsupervised Video Object Segmentation with Differentiable Object Representations
# implement G, F, E_dot_D, LUT which is mentioned in the paper (SegDiff, https://arxiv.org/pdf/2112.00390)
# this outputs x_tminus_1 conditioned on x_t (previous result), I (image), t(time step)
# x_tminus_1 = E_dot_D(F(x_t) + G(I), LUT(t))
# G: RRDBs (Residual Dense Blocks) for image I : Multi-level image features without batchnorm
# F : 2D-convolutional layer with single-channel input and output oc C channels
# E_dot_D : Encoder and Decoder with skip connections (e_theta) : Each level residual blocks, attention layer.
# Bottle-neck layer : two residual blocks with an attention layer in between


import autorootcwd
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.archs.rrdb import RRDB, make_layer
from einops import rearrange, reduce
from functools import partial
import math
from src.utils.registry import ARCH_REGISTRY


class G_model(nn.Module):
    def __init__(self, in_channels=1, out_channels=128, nf=64, nb=1, gc=32):
        # in_channels: number of input channels
        # out_channels: number of output channels
        # nf: number of filters
        # nb: number of RRDB blocks
        # gc: group norm

        super(G_model, self).__init__()
        RRDB_block_f = partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_channels, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_channels, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk  # residual connection around RRDB blocks
        out = self.conv_last(self.lrelu(self.HRconv(fea)))
        return out


# Define F
class F_model(nn.Module):
    """ F is a 2D-convolutional layer with single-channel input and output oc C channels """

    def __init__(self, in_channels=1, out_channels=128):
        super(F_model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)

# Define LUT layer


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings following the implementation in Ho et al. "Denoising Diffusion Probabilistic
    Models" https://arxiv.org/abs/2006.11239.

    Args:
        timesteps: a 1-D Tensor of N indices, one per batch element.
        embedding_dim: the dimension of the output.
        max_period: controls the minimum frequency of the embeddings.
    """
    if timesteps.ndim != 1:
        raise ValueError("Timesteps should be a 1d-array")

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(start=0,
                                                    end=half_dim, dtype=torch.float32, device=timesteps.device)
    freqs = torch.exp(exponent / half_dim)

    args = timesteps[:, None].float() * freqs[None, :]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        embedding = torch.nn.functional.pad(embedding, (0, 1, 0, 0))

    return embedding


class LUT(nn.Module):
    def __init__(self, embedding_dim: int = 64, out_channels: int = 64):
        super(LUT, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(out_channels, embedding_dim), nn.SiLU(
            ), nn.Linear(embedding_dim, embedding_dim)
        )
        self.embedding_dim = embedding_dim

    def forward(self, t: torch.Tensor):
        t = get_timestep_embedding(t, embedding_dim=self.embedding_dim)
        t = self.mlp(t)  # MLP layer
        return t

# Define E_dot_D
# First define residual block


def exists(x):
    return x is not None


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):  # use scale_shift for time embedding
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResidualBlock(nn.Module):
    """ ResidualBlock inputs x and t and outputs x_hat
    This acts as a building block for E_dot_D """

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)

        return h + self.res_conv(x)


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    """

    def __init__(self, channels, use_conv):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    """

    def __init__(self, channels, stride=2, use_conv=True):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv

        if use_conv:
            self.op = nn.Conv2d(channels, channels,
                                3, stride, padding=1)
        else:
            self.op = nn.AvgPool2d(stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.attention = QKVAttention()
        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        h = self.attention(qkv)
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (C * 3) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x C x T] tensor after attention.
        """
        ch = qkv.shape[1] // 3
        q, k, v = torch.split(qkv, ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        return torch.einsum("bts,bcs->bct", weight, v)


class E_dot_D(nn.Module):
    """ input: G(I) + F(x_t), LUT(t)
    size: G+F = (1,64,128,128)
    size: LUT + (1,64)
    """

    def __init__(self, in_channels=64, out_channels=1, num_res_blocks=4, channel_mul=(1, 2, 4, 8), num_heads=1, conv_resample=False):
        # in_channels: number of input channels
        # out_channels: number of output channels
        # num_res_blocks: number of residual blocks for each downsampling
        # channel_mul: channel multipliers
        # num_heads: number of attention heads
        # conv_resample: use convolutional downsampling and ups

        super().__init__()

        self.num_resolutions = len(channel_mul)
        self.num_res_blocks = num_res_blocks

        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        in_ch = in_channels

        for i in range(self.num_resolutions):
            out_ch = channel_mul[i] * in_channels
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResidualBlock(in_ch, out_ch, time_emb_dim=in_channels))
                in_ch = out_ch
            if i != self.num_resolutions - 1:
                self.down_blocks.append(Downsample(in_ch, use_conv=conv_resample))

        # Bottleneck
        self.bottleneck = nn.ModuleList([
            ResidualBlock(in_ch, in_ch, time_emb_dim=in_channels),
            AttentionBlock(in_ch, num_heads=num_heads)
        ])

        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(self.num_resolutions)):
            out_ch = channel_mul[max(i - 1, 0)] * in_channels
            for j in range(num_res_blocks + 1):
                self.up_blocks.append(ResidualBlock(in_ch, out_ch, time_emb_dim=in_channels))
                in_ch = out_ch
            if i != 0:
                self.up_blocks.append(Upsample(in_ch, use_conv=conv_resample))

        # Final convolution
        self.conv_out = nn.Conv2d(in_ch, out_channels, 3, padding=1)

    def forward(self, x, t):
        # Downsampling
        skips = []
        for block in self.down_blocks:
            if isinstance(block, Downsample):
                skips.append(x)
                x = block(x)
            else:
                x = block(x, t)

        # Bottleneck
        for block in self.bottleneck:
            x = block(x) if isinstance(block, AttentionBlock) else block(x, t)

        # Upsampling
        for block in self.up_blocks:
            if isinstance(block, Upsample):
                x = block(x)
                skip = skips.pop()
                x = x + skip
            else:
                x = block(x, t)

        # Final convolution
        x = self.conv_out(x)
        return x

@ARCH_REGISTRY.register()
class SegDiffUnet(nn.Module):
    def __init__(self, in_channels, out_channels, nf=64):
        super(SegDiffUnet, self).__init__()
        self.G = G_model(in_channels=in_channels, out_channels=nf)
        self.F = F_model(in_channels=in_channels, out_channels=nf)
        self.LUT = LUT(embedding_dim=nf, out_channels=nf)
        self.E_dot_D = E_dot_D(in_channels=nf, out_channels=out_channels)
    def forward(self, x_t, I, t): # x, c, t
        G_output = self.G(I)
        F_output = self.F(x_t)
        LUT_output = self.LUT(t)
        E_dot_D_output = self.E_dot_D(G_output + F_output, LUT_output)
        return E_dot_D_output

if __name__ == "__main__":
    # Define input dimensions
    batch_size = 1
    img_channels = 1
    img_height = 128
    img_width = 128
    time_steps = 10
    embedding_dim = 64

    # Create dummy inputs
    x_t = torch.randn(batch_size, img_channels, img_height, img_width)
    I = torch.randn(batch_size, img_channels, img_height, img_width)
    t = torch.randint(0, time_steps, (batch_size,))

    # Initialize models
    G_func = G_model(in_channels=img_channels, out_channels=embedding_dim)
    F_func = F_model(in_channels=img_channels, out_channels=embedding_dim)
    LUT_func = LUT(embedding_dim=embedding_dim,
                    out_channels=embedding_dim)
    E_dot_D_func = E_dot_D(in_channels=embedding_dim,
                            out_channels=img_channels)

    # Forward pass through G
    G_output = G_func(I)
    print("G output shape:", G_output.shape)

    # Forward pass through F
    F_output = F_func(x_t)
    print("F output shape:", F_output.shape)

    # Forward pass through LUT
    LUT_output = LUT_func(t)
    print("LUT output shape:", LUT_output.shape)

    # Forward pass through E_dot_D
    E_dot_D_output = E_dot_D_func(G_output + F_output, LUT_output)
    print("E_dot_D output shape:", E_dot_D_output.shape)
    
    # Add SegDiffUnet
    SegDiffUnet_model = SegDiffUnet(in_channels=img_channels, out_channels=img_channels)
    SegDiffUnet_output = SegDiffUnet_model(x_t, I, t)
    print("SegDiffUnet output shape:", SegDiffUnet_output.shape)

