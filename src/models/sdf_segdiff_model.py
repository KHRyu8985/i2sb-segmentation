import autorootcwd
from .segdiff_model import SegDiffModel
import torch
import torch.nn.functional as F
from src.distance.distance import compute_sdf 
from .segdiff_model import *

class SDFSegDiffModel(SegDiffModel):

    @torch.no_grad()
    def feed_data(self, batch):
        img, label = batch['image'], batch['label']
        label = label.to(self.device)
        sdf_label = compute_sdf(label, delta=3.0) # signed distance transform
        sdf_label = sdf_label.float()

        img = img.to(self.device)

        return img, label, sdf_label
    
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

        vb_loss = self._vb_terms_bpd(
            x_start=x_start,
            x_t=x,
            t=t,
            c=cond,
            clip_denoised=False,
        )
                        
        # Apply weights and combine losses
        weighted_mse_loss = self.mse_weight * mse_loss
        weighted_vb_loss = self.vb_weight * vb_loss
        
        combined_loss = weighted_mse_loss + weighted_vb_loss

        # Apply loss weight
        combined_loss = combined_loss * extract(self.loss_weight, t, combined_loss.shape)
        
        loss_dict = {
            'mse': mse_loss.mean().item(),
            'vb': vb_loss.mean().item(),
            'combined': combined_loss.mean().item()
        }
        
        return combined_loss.mean(), loss_dict

    def forward(self, batch):
        img, label, sdf_label = self.feed_data(batch)
        times = torch.randint(0, self.timesteps, (1,), device=self.device).long()
        loss, loss_dict = self.p_losses(x_start=sdf_label, t=times, cond=img)        
        return loss, loss_dict

    def valid_step(self, batch, verbose=False):
        img, label, sdf_label = self.feed_data(batch)
        output = self.p_sample_loop(cond=img, verbose=verbose)
        return output, sdf_label
    
    def infer_step(self, batch, verbose=True):
        img, label, sdf_label = self.feed_data(batch)
        output = self.p_sample_loop(cond=img, verbose=verbose)
        
        output = (output >= 0) # thresholding
        output = output.float()
        return output, label      
    
