import autorootcwd
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from src.utils.registry import ARCH_REGISTRY

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, k_size=3, dilation=1):
        super(ConvBlock, self).__init__()
        self.init(in_channels, out_channels, stride, k_size, dilation)
    
    def init(self, in_channels, out_channels, stride=1, k_size=3, dilation=1):
        raise NotImplementedError("Subclass must implement abstract method")

class ResidualBlock(ConvBlock):
    expansion = 1

    def init(self, in_channels, out_channels, stride, k_size, dilation):
        p = k_size // 2 * dilation
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=k_size,
                               stride=stride, padding=p, bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=k_size,
                               stride=1, padding=p, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class RecurrentBlock(nn.Module):
    def __init__(self, out_ch, k_size, t=2, groups=1):
        super(RecurrentBlock, self).__init__()
        self.t = t
        self.conv = nn.Conv2d(out_ch, out_ch, kernel_size=k_size, stride=1, 
                              padding=k_size//2, bias=False, groups=groups)

    def forward(self, x):
        for _ in range(self.t):
            if _ == 0:
                x1 = self.conv(x)
            x1 = self.conv(x + x1)
        return x1

class RRCNNBlock(ConvBlock):
    def init(self, in_channels, out_channels, stride, k_size, dilation):
        assert dilation == 1
        t = 2
        self.RCNN = nn.Sequential(
            RecurrentBlock(out_channels, k_size=k_size, t=t),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            RecurrentBlock(out_channels, k_size=k_size, t=t),
            nn.BatchNorm2d(out_channels),
        )
        self.Conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x1 = self.Conv(x)
        x2 = self.RCNN(x1)
        out = x1 + x2
        return F.relu(out)

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class RecurrentConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = RecurrentBlock(dim, k_size=7, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = RecurrentBlock(dim, k_size=3)
        self.act = nn.GELU()
        self.pwconv2 = RecurrentBlock(dim, k_size=3)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = x.permute(0, 2, 3, 1)
            x = self.gamma * x
            x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x

@ARCH_REGISTRY.register()
class FRNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, ls_mid_ch=([32]*6), out_k_size=11, k_size=3,
                 cls_init_block=ResidualBlock, cls_conv_block=ResidualBlock):
        super().__init__()
        self.dict_module = nn.ModuleDict()
        self.ls_mid_ch = ls_mid_ch

        ch1 = in_channels
        for i, ch2 in enumerate(ls_mid_ch):
            if ch1 != ch2:
                self.dict_module.add_module(f"conv{i}", cls_init_block(ch1, ch2, k_size=k_size))
            else:
                if cls_conv_block == RecurrentConvNeXtBlock:
                    module = RecurrentConvNeXtBlock(dim=ch1, layer_scale_init_value=1)
                else:
                    module = cls_conv_block(ch1, ch2, k_size=k_size)
                self.dict_module.add_module(f"conv{i}", module)
            ch1 = ch2

        self.dict_module.add_module("final", nn.Sequential(
            nn.Conv2d(ch1, out_channels*4, out_k_size, padding=out_k_size//2, bias=False),
            nn.Sigmoid()
        ))

    def forward(self, x):
        for i in range(len(self.ls_mid_ch)):
            x = self.dict_module[f'conv{i}'](x)
        x = self.dict_module['final'](x)
        return torch.max(x, dim=1, keepdim=True)[0]
    
def test_frnet():
    ch_in = 1 
    ch_out = 1 

    model = FRNet(ch_in, ch_out, 
                   cls_init_block=ResidualBlock, cls_conv_block=ResidualBlock)

    x = torch.randn(1, 1, 256, 256)
    model.eval()  # 평가 모드로 설정
    with torch.no_grad():
        output = model(x)    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    
    # 모델 파라미터 수 계산
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Number of parameters: {num_params:,}")
    print()

if __name__ == "__main__":
    test_frnet()