import imageio
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.distance.distance import compute_sdf
from src.distance.skeletonize import Skeletonize

def load_and_preprocess_image(path):
    img = imageio.imread(path) / 255.
    return torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

def main():
    # Load and preprocess image
    x = load_and_preprocess_image('/home/kanghyun/i2sb-segmentation/data/ROSSA/train_sam/label/315.png')
    
    # Compute SDF
    sdf_normalized = compute_sdf(x, delta=3.0)
    
    # Compute binary segmentation and skeleton
    sdf_normalized_np = sdf_normalized.cpu().numpy().squeeze()
    
    binary_seg = torch.sigmoid(sdf_normalized * 1500)
    binary_seg = binary_seg.cpu().numpy().squeeze()
    #binary_seg = (sdf_normalized_np >= 0).astype(np.float32)
    
    skeletonization_module = Skeletonize(probabilistic=False, simple_point_detection='Boolean')
    skeleton = skeletonization_module(x).numpy().squeeze()
    
    # Prepare ground truth
    gt = x.cpu().numpy().squeeze()
    
    # Plot results
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    images = [gt, binary_seg, sdf_normalized_np, skeleton]
    titles = ["GT", "DISTANCE_TF_TO_BINARY", "NORM_SIGNED_DT", "SKELETON"]
    
    for ax, img, title in zip(axs.ravel(), images, titles):
        cmap = 'hot' if title == "NORM_SIGNED_DT" else 'gray'
        im = ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('results/unit_test/distance_transform_results.png')
    plt.close()

    # Add CUDA testing
    if torch.cuda.is_available():
        device = torch.device("cuda")
        x_cuda = x.to(device)
        
        # Compute SDF on CUDA
        sdf_normalized_cuda = compute_sdf(x_cuda, delta=3.0)
        
        # Compare CPU and CUDA results
        sdf_normalized_cpu = sdf_normalized.cpu().numpy()
        sdf_normalized_cuda_cpu = sdf_normalized_cuda.cpu().numpy()
        
        is_close = np.allclose(sdf_normalized_cpu, sdf_normalized_cuda_cpu, atol=1e-6)
        print(f"CPU and CUDA EDT results are close: {is_close}")
        
        # Compute skeleton on CUDA
        skeleton_cuda = skeletonization_module(x_cuda).cpu().numpy().squeeze()
        is_close = np.allclose(skeleton, skeleton_cuda, atol=1e-6)
        print(f"CPU and CUDA skeleton results are close: {is_close}")   
        
        
if __name__ == "__main__":
    main()