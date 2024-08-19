import autorootcwd
from .supervised_model import SupervisedModel
import torch
import torch.nn.functional as F
import math
from functools import partial
from collections import namedtuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
import numpy as np
from src.metrics.vessel_2d import dice_metric
from .segdiff_model import SegDiffModel

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])
# ModelPrediction: 모델 예측 결과를 저장하는 namedtuple (pred_noise: 예측된 noise, pred_x_start: 예측된 x_start)
# prediction = ModelPrediction(pred_noise, pred_x_start)
# prediction.pred_noise, prediction.pred_x_start


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


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) /
                      tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def get_best_worst_slices(pred_seg, label, n=2, smooth=1e-5):
    # 각 슬라이스별 Dice score 계산
    dice_scores = []
    for i in range(pred_seg.shape[0]):
        dice_score = dice_metric(pred_seg[i], label[i], smooth)
        dice_scores.append(dice_score.item() if isinstance(dice_score, torch.Tensor) else dice_score)
    
    # Dice score를 기준으로 정렬된 인덱스 얻기
    sorted_indices = np.argsort(dice_scores)
    
    # 최악의 n개와 최고의 n개 슬라이스 인덱스 선택
    worst_indices = sorted_indices[:n]
    best_indices = sorted_indices[-n:][::-1]  # 역순으로 정렬
    
    return best_indices, worst_indices

class BerDiffModel(SegDiffModel):
    "BerDiff: Conditional Bernoulli Diffusion Model for Medical Image Segmentation MICCAI 2023" 
    
    
