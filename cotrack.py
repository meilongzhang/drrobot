import torch
from scene import Scene, RobotScene
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from gaussian_renderer import GaussianModel
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from cotracker.utils.visualizer import Visualizer
from transformers import Sam3Processor, Sam3Model
import numpy as np

def sample_points_from_mask(mask: np.ndarray, n: int = 200, seed: int = 0) -> np.ndarray:
    """
    mask: (H,W) bool or {0,1}
    returns: (n,2) float32 points in (x,y) pixel coords
    """
    rng = np.random.default_rng(seed)
    ys, xs = np.where(mask.astype(bool))
    if len(xs) == 0:
        raise ValueError("Mask is empty—no pixels to sample.")

    n = min(n, len(xs))
    idx = rng.choice(len(xs), size=n, replace=False)
    pts = np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)  # (n,2) (x,y)
    return pts

def make_cotracker_queries(points_xy: np.ndarray, device: str = "cuda", t0: int = 0) -> torch.Tensor:
    """
    points_xy: (N,2) float32 (x,y)
    returns: (1,N,3) float tensor (t0,x,y)
    """
    pts = torch.from_numpy(points_xy).to(device=device, dtype=torch.float32)  # (N,2)
    t = torch.full((pts.shape[0], 1), float(t0), device=device)               # (N,1)
    queries = torch.cat([t, pts], dim=1)[None, ...]                           # (1,N,3)
    return queries


if __name__ == "__main__":
    import moviepy as mpy
    from moviepy.editor import ImageSequenceClip
    import copy
    import numpy as np
    from PIL import Image
    import imageio.v3 as iio

    device = 'cuda'
    grid_size = 10

    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    # Download the video or use video from specified path
    #video_path = "./out_video_v3.mp4"
    video_path = "/nethome/chuang475/datasets/egodex/part2/insert_remove_furniture_bench_round_table/999.mp4"
    #url = 'https://github.com/facebookresearch/co-tracker/raw/refs/heads/main/assets/apple.mp4' 
    frames = iio.imread(video_path, plugin="pyav")  # plugin="pyav"

    video = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)  # B T C H W

    
    image = Image.fromarray(frames[0])
    inputs = processor(images=image, text="left hand", return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=inputs["original_sizes"].tolist()
    )[0]

    masks = results["masks"]  # list/stack of masks depending on HF version
    print(f"Found {len(masks)} objects")

    #mask0 = np.any(masks, axis=0)

    # Handle HF variants: (N,H,W), (N,1,H,W), list[Tensor], numpy, etc.
    if isinstance(masks, (list, tuple)):
        masks = torch.stack([m if isinstance(m, torch.Tensor) else torch.from_numpy(m) for m in masks], dim=0)

    if isinstance(masks, torch.Tensor):
        if masks.ndim == 4 and masks.shape[1] == 1:      # (N,1,H,W) -> (N,H,W)
            masks = masks[:, 0]
        elif masks.ndim == 2:                             # (H,W) -> (1,H,W)
            masks = masks[None, ...]

        mask0 = masks.any(dim=0)                          # (H,W), bool tensor
        mask0 = mask0.cpu().numpy()
    else:
        # pure numpy fallback (rare)
        mask0 = np.any(np.asarray(masks), axis=0)


    points_xy = sample_points_from_mask(mask0, n=256, seed=0)
    queries = make_cotracker_queries(points_xy, device=device, t0=0)  # (1,N,3)


    # Run Offline CoTracker:
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
    pred_tracks, pred_visibility = cotracker(video, queries=queries) # B T N 2,  B T N 1

    vis = Visualizer(save_dir="./saved_videos", linewidth=1, mode='cool', tracks_leave_trace=-1)
    vis.visualize(video, pred_tracks, pred_visibility, filename='test_egodex_onehand')
