import autorootcwd
from .base_model import BaseModel
import torch
import numpy as np
import torch.nn.functional as F


def exists(x):
    return x is not None


def default(val, d):
    # val이 존재하는지 확인, 존재하지 않으면 d 반환
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(
        ((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class SegDiffModel(BaseModel):

    def __init__(self, arch='FRNet', criterion='MonaiDiceCELoss',
                 device='cuda:0', mode='train', beta_schedule='linear',
                 timesteps=100):
        super().__init__(arch=arch, criterion=criterion, device=device, mode=mode)

        self.arch = self.arch.to(self.device)
        self.criterion = self.criterion.to(self.device)
        self.optimizer = torch.optim.Adam(self.arch.parameters(), lr=2e-3)
        self.timesteps = timesteps

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        # beta 값만 있지만 미리 계산해놓으면 좋은 값들
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)  # 누적곱, alpha_t_bar
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        def register_buffer(name, val): return self.register_buffer(
            name, val.to(torch.float32))  # register_buffer 함수로 쉽게 모델에 넣을 수 있게 함

        register_buffer('betas', betas)  # SegDiffModel.betas = betas 와 유사
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # Diffusion q(x_t | x_{t-1}) and others 을 위하여
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod',
                        torch.sqrt(1.0 - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod',
                        torch.log(1.0 - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod',
                        torch.sqrt(1.0 / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod',
                        torch.sqrt(1.0 / alphas_cumprod - 1))

        # Caculations for posterior q(x_{t-1} | x_t, x_0), 즉 sampling을 위해
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))

        register_buffer('posterior_variance', posterior_variance)

        # log calculation clipped because the posterior variance is 0 at the beginning
        register_buffer('posterior_log_variance_clipped',
                        torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas *
                        torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev)
                        * torch.sqrt(alphas) / (1. - alphas_cumprod))

    @torch.no_grad()
    def q_sample(self, x_0, t, noise=None):
        """
        Sample from the Gaussian distribution 
        NOTE x_0 is the true segmentation map
        q(x_t | x_0) = N(x_t; sqrt(alpha_t_bar) * x_0, (1 - alpha_t_bar) * I_nxn)
        """
        if noise is None:
            noise = torch.randn_like(x_0)  # Generate noise
        assert noise.shape == x_0.shape, 'Noise shape must be equal to x_start shape'

        sqrt_alpha_t_bar = extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        alpha_t_bar = extract(self.alphas_cumprod, t, x_0.shape)

        _mean = x_0 * sqrt_alpha_t_bar
        _variance = (1.0 - alpha_t_bar)

        x_t = _mean + noise * torch.sqrt(_variance)

        return x_t

    def sample_and_estimate(self, img, seg):
        times = torch.randint(
            0, self.timesteps, (img.shape[0],), device=img.device).long()
        seg = normalize_to_neg_one_to_one(seg)
        x_0 = seg

        x_t = self.q_sample(x_0, times)
        model_out = self.arch(img, x_t, cond=img)
        return model_out
    
    