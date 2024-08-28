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
from src.utils.registry import ARCH_REGISTRY, LOSS_REGISTRY
from monai.losses import DiceLoss

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])
# ModelPrediction: 모델 예측 결과를 저장하는 namedtuple (pred_noise: 예측된 noise, pred_x_start: 예측된 x_start)
# prediction = ModelPrediction(pred_noise, pred_x_start)
# prediction.pred_noise, prediction.pred_x_start

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    # Convert np.sqrt(2.0 / np.pi) to a tensor
    return 0.5 * (1.0 + torch.tanh(torch.tensor(np.sqrt(2.0 / np.pi)) * (x + 0.044715 * torch.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """

    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs

def exists(x):
    return x is not None


def default(val, d):
    # val이 존재하는지 확인, 존재하지 않으�� d 반환
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

class SegDiffModel(SupervisedModel):
    """
    주의할 점: 논문 구현할 때는 논문 확인하면서 notation을 그대로 유지하면 좋음 (다른데에서 코드 가져오더라도)
    그래야 논문을 이해하는데 도움이 되고, 다른 사람들이 코드를 이해하는데 도움이 됨
    """

    def __init__(self, arch='SegDiffUnet', criterion='pred_x0',
                 mode='train', beta_schedule='sigmoid', min_snr_loss_weight=True, min_snr_gamma=5, timesteps=100, name=None,
                 mse_weight=100.0, dice_weight=10, vb_weight=0.1):

        super(SegDiffModel, self).__init__(arch=arch, criterion=criterion, mode=mode, name=name)
        # 위에서 criterion은 사용 안함, 논문대로 MSE loss 사용
        # arch 모델은 eta_theta = D(E(F(x_t) + G(I), t), t) 이��. 여기에서 I 는 이미지, x_t는 t시점의 segmentation map
        # x_t 와 I 를 혼동하면 안됨

        self.name = name if name else f"{arch}__{beta_schedule}__{criterion}__{timesteps}"
        self.optimizer = torch.optim.AdamW(self.arch.parameters(), lr=1e-3)
        self.timesteps = timesteps
        self.criterion = criterion

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == 'sigmoid':
            betas = sigmoid_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        # beta 값만 있지만 미리 계산해놓으면 좋은 값들
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)  # 누적곱, alpha_t_bar
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        # helper function to register buffer from float64 to float32
        # register_buffer를 써야하는 이유: https://www.ai-bio.info/pytorch-register-buffer
        # register_buffer에 등록된 애들은 자동으로 cuda로 옮기기 편하다.
        # register_buffer에 등록된 애들은 학습 되지 않는다! (아주 중요!!)

        def register_buffer(name, val): return self.register_buffer(
            name, val.to(torch.float32))

        # q_sample 때 필요한 값들 (미리 준비)
        register_buffer('betas', betas)  # SegDiffModel.betas = betas 와 유사
        register_buffer('alphas_cumprod', alphas_cumprod)  # alpha_t_bar
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))

        # posterior_mean_variance 계산할 때 필요한 값들 (미리 준비)
        register_buffer('sqrt_recip_alphas_cumprod',
                        torch.sqrt(1.0 / alphas_cumprod))

        register_buffer('sqrt_recipm1_alphas_cumprod',
                        torch.sqrt(1.0 / alphas_cumprod - 1))

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped',
                        torch.log(posterior_variance.clamp(min=1e-20)))
        # Posterior variance 대신에 log variance 사용 주로 함 (numerical stability)
        # log variance를 사용하면 variance가 0에 가까울 때 numerical stability를 높일 수 있다.

        # posterior mean 계산에 필요한 계수들
        # posterior_mean = posterior_mean_coef1 * x_start + posterior_mean_coef2 * x_t
        # posterior_mean_coef1 = beta_t * sqrt(alpha_t_bar_prev) / (1 - alpha_t_bar)
        # posterior_mean_coef2 = (1 - alpha_t_bar_prev) * sqrt(alpha_t) / (1 - alpha_t_bar)
        register_buffer('posterior_mean_coef1', betas *
                        torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev)
                        * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # loss weight
        # snr
        snr = alphas_cumprod / (1-alphas_cumprod)
        maybe_clipped_snr = snr.clone()

        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max=min_snr_gamma)  # min_snr_gamma

        # default: pred_v
        if criterion == 'pred_noise':  # 기존의 DDPM: noise 예측
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif criterion == 'pred_x0':  # x0를 예측하는 것
            register_buffer('loss_weight', maybe_clipped_snr)
        else:
            raise ValueError(f'unknown criterion {criterion}')
        self.dice_loss = DiceLoss(sigmoid=False, reduction="none")

        self.mse_weight = mse_weight
        self.dice_weight = dice_weight
        self.vb_weight = vb_weight

    ############### Training stage ###############
    def predict_start_from_noise(self, x_t, t, noise):
        """
        현재 시점 t에서의 샘플 x_t 와 노이즈 (예측된)를 제거하여 초기 시점 0의 샘플 x_0을 예측하는 함수
        x_t: 현재 시점 t의 샘플
        t: 현재 시점
        noise: 노이즈 (예측한)
        """
        sqrt_recip_alphas_cumprod = extract(
            self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod = extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

        # 수식 참고 (아래)
        # x_0 = sqrt(1 / alpha_t_bar) * x_t - sqrt(1 / alpha_t_bar - 1) * noise
        # (alpha_t_bar = alphas_cumprod (t시점까지의 누적곱))

        x_start = sqrt_recip_alphas_cumprod * x_t - noise * sqrt_recipm1_alphas_cumprod
        return x_start

    def predict_noise_from_start(self, x_t, t, x_start):
        sqrt_recip_alphas_cumprod = extract(
            self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod = extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

        noise = (sqrt_recip_alphas_cumprod * x_t - x_start) / \
            (sqrt_recipm1_alphas_cumprod)
        return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, c, clip_x_start=False):
        # here, condition (c) is same as I below
        # eta_theta = D(E(F(x_t) + G(I), t), t)

        model_output = self.arch(x, c, t)
        maybe_clip = partial(torch.clamp, min=-1.,
                             max=1.) if clip_x_start else identity

        if self.criterion == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.criterion == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        # namedtuple 로 pred_noise, x_start 반환
        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, c, clip_denoised=True):
        preds = self.model_predictions(x, t, c)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance_clipped = self.q_posterior(
            x_start, x, t)
        return model_mean, posterior_variance, posterior_log_variance_clipped, x_start

    @torch.no_grad()
    def p_sample(self, x, t, c, clip_denoised=True):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full(
            (x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, t=batched_times, c=c, clip_denoised=clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, cond, verbose=True):
        shape = cond.shape
        batch = shape[0]

        img = torch.randn(shape, device=self.device)

        x_start = None

        time_range = reversed(range(0, self.timesteps))
        if verbose:
            time_range = tqdm(time_range, desc='sampling loop time step', total=self.timesteps)

        for t in time_range:
            # 주의, 여기에서 cond 가 I, img 는 x 이다.
            img, x_start = self.p_sample(img, t, cond)

        img = unnormalize_to_zero_to_one(img)
        return img

    def q_sample(self, x_start, t, noise=None):
        """
        Sample from the Gaussian distribution 
        NOTE x_start(x_0) is the true segmentation map
        q(x_t | x_0) = N(x_t; sqrt(alpha_t_bar) * x_0, (1 - alpha_t_bar) * I_nxn)
        """
        if noise is None:
            noise = torch.randn_like(x_start)  # Generate noise
        assert noise.shape == x_start.shape, 'Noise shape must be equal to x_start shape'

        sqrt_alpha_t_bar = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        alpha_t_bar = extract(self.alphas_cumprod, t, x_start.shape)

        _mean = x_start * sqrt_alpha_t_bar
        _variance = (1.0 - alpha_t_bar)

        x_t = _mean + noise * torch.sqrt(_variance)

        return x_t

    def _vb_terms_bpd(self, x_start, x_t, t, c, clip_denoised=True):
        """
        Get a term for the variational lower-bound.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior(x_start, x_t, t) # x_{t-1} mean variance
        out_mean, _, out_log_variance, out_x_start = self.p_mean_variance(x_t, t, c, clip_denoised=clip_denoised)
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out_mean, out_log_variance
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out_mean, log_scales=0.5 * out_log_variance
        )
        
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0), decoder_nll, kl)
        return output

    def p_losses(self, x_start, t, cond, noise=None):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        model_output = self.arch(x, cond, t)

        if self.criterion == 'pred_noise':
            target = noise
        elif self.criterion == 'pred_x0':
            target = x_start
        else:
            raise ValueError(f'unknown criterion {self.criterion}')
        
        # Calculate individual losses
        mse_loss = F.mse_loss(model_output, target, reduction='none')
        mse_loss = reduce(mse_loss, 'b ... -> b', 'mean')

        seg_out = unnormalize_to_zero_to_one(model_output)
        seg_out.clamp_(0., 1.)

        x_start_seg = unnormalize_to_zero_to_one(x_start)
        dice_loss = self.dice_loss(seg_out, x_start_seg)
        dice_loss = reduce(dice_loss, 'b ... -> b', 'mean')

        vb_loss = self._vb_terms_bpd(
            x_start=x_start,
            x_t=x,
            t=t,
            c=cond,
            clip_denoised=False,
        )
        
        # Apply weights and combine losses
        weighted_mse_loss = self.mse_weight * mse_loss
        weighted_dice_loss = self.dice_weight * dice_loss
        weighted_vb_loss = self.vb_weight * vb_loss
        
        combined_loss = weighted_mse_loss + weighted_dice_loss + weighted_vb_loss

        # Apply loss weight
        combined_loss = combined_loss * extract(self.loss_weight, t, combined_loss.shape)
        
        loss_dict = {
            'mse': mse_loss.mean().item(),
            'dice': dice_loss.mean().item(),
            'vb': vb_loss.mean().item(),
            'combined': combined_loss.mean().item()
        }
        
        return combined_loss.mean(), loss_dict

    def forward(self, batch):
        img, label = self.feed_data(batch)
        label = normalize_to_neg_one_to_one(label)
        times = torch.randint(0, self.timesteps, (1,), device=self.device).long()
        loss, loss_dict = self.p_losses(x_start=label, t=times, cond=img)
        return loss, loss_dict
    
    def train_step(self, batch):
        self.optimizer.zero_grad()
        img, label = self.feed_data(batch)

        if img.ndim == 3:
            img = rearrange(img, 'c h w -> 1 c h w')

        if label.ndim == 3:
            label = rearrange(label, 'c h w -> 1 c h w')

        times = torch.randint(0, self.timesteps, (1,), device=self.device).long()
        label = normalize_to_neg_one_to_one(label)

        loss, loss_dict = self.p_losses(x_start=label, t=times, cond=img)
        loss.backward()
        self.optimizer.step()
        return loss.item(), loss_dict

    def valid_step(self, batch, verbose=False):
        img, label = self.feed_data(batch)
        output = self.p_sample_loop(cond=img, verbose=verbose)
        return output, label
    
    def infer_step(self, batch, verbose=True):
        img, label = self.feed_data(batch)
        output = self.p_sample_loop(cond=img, verbose=verbose)
        
        output = (output > 0.5) # thresholding
        output = output.float()
        return output, label
    
    def val_one_epoch(self, dataloader, current_epoch, total_epoch=0, verbose=True, base_folder='results/seg_diff'):
        # First run the validation step
        pred_seg = None
        label = None
        im = None
        with tqdm(total=len(dataloader), desc=f"Validation Epoch {current_epoch}/{total_epoch}") as pbar:
            for batch_idx, batch in enumerate(dataloader):
                
                _pred_seg, _label = self.valid_step(batch, verbose=verbose)
                _pred_seg = _pred_seg.detach().cpu().numpy()
                _label = _label.detach().cpu().numpy()
                _im = batch['image'].detach().cpu().numpy()
                
                if pred_seg is None:
                    pred_seg = _pred_seg
                    label = _label
                    im = _im
                    
                else:
                    pred_seg = np.vstack((pred_seg, _pred_seg))
                    label = np.vstack((label, _label))
                    im = np.vstack((im, _im))
                pbar.update(1)
        
        pred_seg = (pred_seg > 0.5)
        total_metric = self.compute_metrics(pred_seg, label)
        avg_metrics = {key: value / len(dataloader) for key, value in total_metric.items()}
     
        print(
            f"Dice: {avg_metrics['Dice']:.4f}, IoU: {avg_metrics['IoU']:.4f}, "
            f"Accuracy: {avg_metrics['Accuracy']:.4f}, Sensitivity: {avg_metrics['Sensitivity']:.4f}, "
            f"Specificity: {avg_metrics['Specificity']:.4f}, clDice: {avg_metrics['clDice']:.4f}"
        )
        
        if verbose:
            best_indices, worst_indices = get_best_worst_slices(pred_seg, label)
            print(f"Best slice indices: {best_indices}")
            print(f"Worst slice indices: {worst_indices}")
            
            # Ensure the base folder exists
            os.makedirs(base_folder, exist_ok=True)

            fig, axes = plt.subplots(2, 6, figsize=(36, 10))
            
            for i, idx in enumerate(list(best_indices[:2]) + list(worst_indices[:2])):
                row = i // 2
                col = (i % 2) * 3

                # Plot input image
                axes[row, col].imshow(im[idx].squeeze(), cmap='gray')
                axes[row, col].axis('off')
                axes[row, col].set_title(f'{"Best" if i < 2 else "Worst"} Input\n(Index: {idx})')

                # Plot prediction
                axes[row, col+1].imshow(im[idx].squeeze(), cmap='gray')
                axes[row, col+1].imshow(pred_seg[idx].squeeze(), cmap='gray')
                axes[row, col+1].axis('off')
                axes[row, col+1].set_title(f'{"Best" if i < 2 else "Worst"} Prediction')
                
                # Plot ground truth
                axes[row, col+2].imshow(label[idx].squeeze(), cmap='gray')
                axes[row, col+2].axis('off')
                axes[row, col+2].set_title(f'{"Best" if i < 2 else "Worst"} Ground Truth')
            
            plt.tight_layout()
            plt.savefig(os.path.join(base_folder, f'validation_epoch_{current_epoch}.png'))
            plt.close()
                
    