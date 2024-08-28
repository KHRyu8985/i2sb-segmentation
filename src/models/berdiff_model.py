import autorootcwd
from src.models.segdiff_model import SegDiffModel
from src.models.segdiff_model import *
from torch.distributions.binomial import Binomial
import torch

# Usage of binomial
# out = Binomial(total_count=1, probs=torch.tensor([0.1, 0.5])).sample()


class BerDiffModel(SegDiffModel):
    "BerDiff: Conditional Bernoulli Diffusion Model for Medical Image Segmentation MICCAI 2023"

    def __init__(self, arch='SegDiffUnet', criterion='pred_x0',
                 mode='train', beta_schedule='sigmoid', min_snr_loss_weight=True, min_snr_gamma=5, timesteps=100, name=None):
        super(BerDiffModel, self).__init__(arch=arch, criterion=criterion, mode=mode, beta_schedule=beta_schedule,
                                           min_snr_loss_weight=min_snr_loss_weight, min_snr_gamma=min_snr_gamma,
                                           timesteps=timesteps, name=name)

        self.name = name if name else f"BERDIFF__{arch}__{beta_schedule}__{criterion}__{timesteps}"

    def q_mean(self, x_start, t):
        """
        Get the distribution q(x_t | x_0) for q_sample function
        """

        mean = extract(self.alphas_cumprod, t, x_start.shape) * x_start + \
            (1 - extract(self.alphas_cumprod, t, x_start.shape)) / 2

        return mean

    def q_sample(self, x_start, t):
        """
        Sample from q(x_t | x_0)
        """

        mean = self.q_mean(x_start, t)
        return Binomial(total_count=1, probs=mean).sample()

    def q_posterior_mean(self, x_start, x_t, t):
        """
        Get distribution q(x_{t-1} | x_t, x_0)
        q(y_{t-1} | y_t, y_0) = B(y_{t-1}; θ_post (y_t, y_0)),
        """
        assert x_start.shape == x_t.shape

        theta_1 = (extract(self.alphas, t, x_start.shape) * (1-x_t) + (1 - extract(self.alphas, t, x_start.shape)) / 2) * \
            (extract(self.alphas_cumprod, t-1, x_start.shape) * (1-x_start) +
             (1 - extract(self.alphas_cumprod, t-1, x_start.shape)) / 2)

        theta_2 = (extract(self.alphas, t, x_start.shape) * x_t + (1 - extract(self.alphas, t, x_start.shape)) / 2) * \
            (extract(self.alphas_cumprod, t-1, x_start.shape) * x_start +
             (1 - extract(self.alphas_cumprod, t-1, x_start.shape)) / 2)

        posterior_mean = theta_2 / (theta_1 + theta_2)

        return posterior_mean

    def predict_start_from_noise(self, x_t, t, noise):
        return torch.abs(x_t - noise)

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        A = (extract(self.alphas, t, x_t.shape) * (1-x_t) +
             (1 - extract(self.alphas, t, x_t.shape)) / 2)
        B = (extract(self.alphas, t, x_t.shape) * x_t +
             (1 - extract(self.alphas, t, x_t.shape)) / 2)
        C = (1 - extract(self.alphas_cumprod, t-1, x_t.shape)) / 2
        numerator = A * C * xprev + B * C * \
            (xprev - 1) + A * xprev * extract(self.alphas_cumprod, t-1, x_t.shape)
        denominator = (B + A * xprev - B * xprev) * \
            extract(self.alphas_cumprod, t-1, x_t.shape)
        return (numerator / denominator)


    def p_mean(self, x, t, c):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        """
        model_output = self.arch(x, t, c)

        if self.criterion == 'pred_x0':
            pred_xstart = model_output
        elif self.criterion == 'pred_noise':
            pred_xstart = self.predict_start_from_noise(model_output, t, c)
        else:
            raise ValueError(f"Criterion {self.criterion} not recognized")
        
        return ModelPrediction(pred_xstart, model_output)