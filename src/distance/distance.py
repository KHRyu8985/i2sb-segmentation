from monai.transforms import distance_transform_edt
import torch

def compute_sdf(x, delta=3.0, invert_sign=True):
    dist_outside = distance_transform_edt(1 - x)
    dist_inside = distance_transform_edt(x)
    
    sdf = torch.where(x == 1, -torch.minimum(dist_inside, torch.tensor(delta)),
                      torch.where(x == 0, torch.minimum(dist_outside, torch.tensor(delta)), torch.tensor(0.0)))
    
    max_dist = torch.maximum(torch.max(torch.abs(sdf)), torch.tensor(delta))
    normalized_sdf = sdf / max_dist
    
    return -normalized_sdf if invert_sign else normalized_sdf