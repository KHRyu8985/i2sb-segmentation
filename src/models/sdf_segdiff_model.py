import autorootcwd
from .segdiff_model import SegDiffModel
import torch
import torch.nn.functional as F
import cupy as cp
import cupyx.scipy.ndimage as ndi

def torch_to_cupy(x):
    return cp.asarray(x.detach().cpu().numpy())

def cupy_to_torch(x):
    return torch.from_numpy(cp.asnumpy(x)).cuda()

class SDFSegDiffModel(SegDiffModel):
    
    def __init__(self, arch='SegDiffUnet', criterion='pred_x0',
                 mode='train', beta_schedule='sigmoid', min_snr_loss_weight=True, min_snr_gamma=5, timesteps=100, name=None):

        super(SDFSegDiffModel, self).__init__(arch=arch, criterion=criterion, mode=mode, beta_schedule=beta_schedule,
                                              min_snr_loss_weight=min_snr_loss_weight, min_snr_gamma=min_snr_gamma, 
                                              timesteps=timesteps, name=name)

        self.name = name if name else f"SDFSEGDIFF__{arch}__{beta_schedule}__{criterion}__{timesteps}"
        

    def _transform_sdf(self, x, delta=5.0):  # delta is the truncation threshold
        if isinstance(x, torch.Tensor):
            x = torch_to_cupy(x)
        else:
            x = cp.asarray(x)
        
        # Compute distance transforms
        dist_outside = ndi.distance_transform_edt(1 - x)
        dist_inside = ndi.distance_transform_edt(x)
        
        # Apply truncation
        sdf = cp.where(x == 1, -cp.minimum(dist_inside, delta),
                       cp.where(x == 0, cp.minimum(dist_outside, delta), 0))
        
        # Normalize
        max_dist = cp.maximum(cp.max(cp.abs(sdf)), delta)
        sdf_normalized = sdf / max_dist
        
        return cupy_to_torch(sdf_normalized) if isinstance(x, torch.Tensor) else sdf_normalized
    
    def _sdf_to_binary(self, sdf):
        if isinstance(sdf, torch.Tensor):
            sdf = torch_to_cupy(sdf)
        else:
            sdf = cp.asarray(sdf)
        
        binary_mask = (sdf <= 0).astype(cp.float32)
        
        return cupy_to_torch(binary_mask) if isinstance(sdf, torch.Tensor) else binary_mask

    def forward(self, batch):
        img, label = self.feed_data(batch)
        label = self._transform_sdf(label)   
        times = torch.randint(0, self.timesteps, (1,),
                              device=self.device).long()
        loss = self.p_losses(x_start=label, t=times, cond=img)
        return loss
    
    def valid_step(self, batch, verbose=False):
        img, label = self.feed_data(batch)
        output = self.p_sample_loop(cond=img, verbose=verbose)
        output = self._sdf_to_binary(output)
        return output, label
        
    