import torch
import numpy as np
import json
from scene import Scene, RobotScene
from tqdm import tqdm
from os import makedirs
from PIL import Image
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from gaussian_renderer import GaussianModel
from scene.cameras import Camera, Camera_Pose
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render, render_3d
from cotrack import sample_points_from_mask, make_cotracker_queries
from transformers import Sam3Processor, Sam3Model
from cotracker.utils.visualizer import Visualizer
from utils.mujoco_utils import simulate_mujoco_scene, compute_camera_extrinsic_matrix

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def skew_symmetric(v: torch.Tensor) -> torch.Tensor:
    """
    v: (3,)
    returns: (3,3)
    """
    O = torch.zeros((), device=v.device, dtype=v.dtype)
    x, y, z = v[0], v[1], v[2]
    return torch.stack([
        torch.stack([O, -z,  y]),
        torch.stack([z,  O, -x]),
        torch.stack([-y, x,  O]),
    ])


def so3_exp_map(w: torch.Tensor) -> torch.Tensor:
    """
    Axis-angle exponential map.
    w: (3,)
    returns R: (3,3)
    """
    theta = torch.linalg.norm(w).clamp_min(1e-8)
    K = skew_symmetric(w / theta)
    I = torch.eye(3, device=w.device, dtype=w.dtype)
    return I + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * (K @ K)


class GlobalAlign(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rotvec = torch.nn.Parameter(torch.zeros(3, dtype=torch.float32))
        self.trans = torch.nn.Parameter(torch.zeros(3, dtype=torch.float32))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        R = so3_exp_map(self.rotvec)
        return X @ R.T + self.trans


def load_manual_world_transform(json_path: str):
    """
    Load a manually configured source-world -> MuJoCo-world transform.

    Supported JSON layouts:
      - [{"camera_base_ori": [[...],[...],[...]], "camera_base_pos": [...]}]
      - {"camera_base_ori": [[...],[...],[...]], "camera_base_pos": [...]}
      - {"R": [[...],[...],[...]], "t": [...]}
      - {"transform": [[...],[...],[...],[...]]}
    Returns:
      R: (3, 3) float32 torch tensor
      t: (3,)    float32 torch tensor
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        if len(data) == 0:
            raise ValueError(f"No transforms found in {json_path}.")
        data = data[0]

    if "transform" in data:
        T = np.asarray(data["transform"], dtype=np.float32)
        if T.shape != (4, 4):
            raise ValueError(f"Expected 4x4 transform in {json_path}, got {T.shape}.")
        R = torch.from_numpy(T[:3, :3])
        t = torch.from_numpy(T[:3, 3])
        return R, t

    if "camera_base_ori" in data and "camera_base_pos" in data:
        R = torch.tensor(data["camera_base_ori"], dtype=torch.float32)
        t = torch.tensor(data["camera_base_pos"], dtype=torch.float32)
        return R, t

    if "R" in data and "t" in data:
        R = torch.tensor(data["R"], dtype=torch.float32)
        t = torch.tensor(data["t"], dtype=torch.float32)
        return R, t

    raise ValueError(
        f"Unrecognized transform format in {json_path}. "
        "Expected camera_base_ori/camera_base_pos, R/t, or transform."
    )


def apply_rigid_transform(X: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    X: (..., 3)
    R: (3, 3)
    t: (3,)
    """
    return X @ R.T + t

def normalize_track_sequence(X: torch.Tensor, vis: torch.Tensor = None, eps: float = 1e-8):
    """
    X:   (T, N, 3)
    vis: (T, N) bool
    Returns:
      X_norm : (T, N, 3)
      center : (3,)
      scale  : scalar
    """
    finite = torch.isfinite(X).all(dim=-1)
    valid = finite if vis is None else (finite & vis)

    X_safe = torch.where(valid[..., None], X, torch.zeros_like(X))
    denom = valid.sum().clamp_min(1).float()

    center = X_safe.sum(dim=(0, 1)) / denom
    Xc = X - center

    sq = torch.where(valid[..., None], Xc.square(), torch.zeros_like(Xc)).sum()
    scale = torch.sqrt(sq / (denom * 3.0) + eps)

    Xn = Xc / scale
    return Xn, center, scale


def masked_centroids(X: torch.Tensor, V: torch.Tensor):
    """
    X: (T, N, 3)
    V: (T, N) bool
    Returns:
      C: (T, 3) centroid per frame
      valid_frame: (T,) bool
    """
    finite = torch.isfinite(X).all(dim=-1)
    V = V & finite

    X_safe = torch.where(V[..., None], X, torch.zeros_like(X))
    counts = V.sum(dim=1)  # (T,)
    valid_frame = counts > 0

    counts_f = counts.clamp_min(1).float().unsqueeze(-1)  # (T,1)
    C = X_safe.sum(dim=1) / counts_f
    return C, valid_frame


def velocity_loss(X_src: torch.Tensor, V_src: torch.Tensor,
                  X_tgt: torch.Tensor, V_tgt: torch.Tensor) -> torch.Tensor:
    """
    Compare centroid velocities over time.
    Works even when source/target have different numbers of points.
    """
    if X_src.shape[0] < 2 or X_tgt.shape[0] < 2:
        return torch.tensor(0.0, device=X_src.device)

    C_src, valid_src = masked_centroids(X_src, V_src)  # (T,3), (T,)
    C_tgt, valid_tgt = masked_centroids(X_tgt, V_tgt)  # (T,3), (T,)

    d_src = C_src[1:] - C_src[:-1]   # (T-1, 3)
    d_tgt = C_tgt[1:] - C_tgt[:-1]   # (T-1, 3)

    valid = valid_src[1:] & valid_src[:-1] & valid_tgt[1:] & valid_tgt[:-1]
    if valid.sum() == 0:
        return torch.tensor(0.0, device=X_src.device)

    diff2 = (d_src - d_tgt).square().sum(dim=-1)  # (T-1,)
    return diff2[valid].mean()


def initialize_alignment_from_means(aligner, G_tracks_3d, G_vis_3d, cam_list, gaussians, dof, stride=80):
    """
    Crude translation initialization:
    align source centroid to robot centroid from a zero-pose render_3d snapshot.
    """
    with torch.no_grad():
        valid = G_vis_3d & torch.isfinite(G_tracks_3d).all(dim=-1)
        if valid.sum() > 0:
            src_mean = G_tracks_3d[valid].mean(dim=0)
        else:
            src_mean = torch.zeros(3, device=G_tracks_3d.device)

        q0 = torch.zeros(1, dof, device=G_tracks_3d.device)
        out_tracks, out_vis = render_3d_tracks_perframe([cam_list[0]], gaussians, q0, stride=stride)
        robot_valid = out_vis[0] > 0.5
        if robot_valid.sum() > 0:
            robot_mean = out_tracks[0][robot_valid].mean(dim=0)
        else:
            robot_mean = torch.zeros(3, device=G_tracks_3d.device)

        aligner.trans.copy_(robot_mean - src_mean)

class DummyCam:
    def __init__(self, azimuth=0, elevation=-45, distance=3):
        self.azimuth = azimuth
        self.elevation = elevation
        self.distance = distance
        self.lookat = [0, 0, 0]  # Force lookat to be [0, 0, 0]

def make_open_uniform_knots(K: int, degree: int):
    """
    Open-uniform (clamped) knot vector.
    Returns a 1D tensor of length K + degree + 1.
    """
    p = degree
    n = K - 1
    m = n + p + 1  # last knot index
    # Open uniform: first p+1 knots = 0, last p+1 knots = 1, interior uniformly spaced
    knots = torch.zeros(m + 1, dtype=torch.float64)
    knots[m - p : m + 1] = 1.0
    num_interior = (m + 1) - 2 * (p + 1)
    if num_interior > 0:
        interior = torch.linspace(0, 1, num_interior + 2, dtype=torch.float64)[1:-1]
        knots[p + 1 : p + 1 + num_interior] = interior
    return knots

def bspline_basis_matrix(T: int, K: int, degree: int = 3):
    """
    Build B in R^{T x K} where B[t,k] = B_k(u_t) for u_t in [0,1].
    Uses Cox-de Boor recursion.
    """
    p = degree
    knots = make_open_uniform_knots(K, p)  # length K+p+1

    # Parameter values u_t: sample in [0, 1], include endpoints
    u = torch.linspace(0, 1, T, dtype=torch.float64)

    # Base case p=0
    # N_{i,0}(u) = 1 if knots[i] <= u < knots[i+1], except at u=1 we put it in the last span
    N = torch.zeros((T, K), dtype=torch.float64)
    for i in range(K):
        left = knots[i]
        right = knots[i + 1]
        # u in [left, right)
        mask = (u >= left) & (u < right)
        N[mask, i] = 1.0
    # Special-case u=1 to land on the last basis
    N[u == 1.0, :] = 0.0
    N[u == 1.0, K - 1] = 1.0

    # Recursion for p=1..degree
    for d in range(1, p + 1):
        N_new = torch.zeros((T, K), dtype=torch.float64)
        for i in range(K):
            # term1: (u - t_i) / (t_{i+d} - t_i) * N_{i,d-1}
            denom1 = knots[i + d] - knots[i]
            if denom1 > 0:
                term1 = (u - knots[i]) / denom1 * N[:, i]
            else:
                term1 = torch.zeros_like(u)

            # term2: (t_{i+d+1} - u) / (t_{i+d+1} - t_{i+1}) * N_{i+1,d-1}
            if i + 1 < K:
                denom2 = knots[i + d + 1] - knots[i + 1]
                if denom2 > 0:
                    term2 = (knots[i + d + 1] - u) / denom2 * N[:, i + 1]
                else:
                    term2 = torch.zeros_like(u)
            else:
                term2 = torch.zeros_like(u)

            N_new[:, i] = term1 + term2
        N = N_new

    # Cast to float32 for typical torch use
    return N.to(dtype=torch.float32)  # [T, K]

def initialize_gaussians(model_path=None, from_ckpt=False):
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    opt = OptimizationParams(parser)
    args = get_combined_args(parser)

    if model_path is not None:
        args.model_path = model_path


    gaussians = GaussianModel(model.sh_degree, opt)
    
    scene = RobotScene(args, gaussians, opt_params=opt, from_ckpt=True, load_iteration=-1)

    gaussians.model_path = scene.model_path #todo, find a cleaner way to do this

    bg_color = [1, 1, 1]
    background_color = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    return gaussians, background_color, scene.getSampleCameras(stage='pose_conditioned'), gaussians.chain

def render_robot_video(camera, gaussians, bg, q_traj):
    frames = []
    for t in range(q_traj.shape[0]):
        camera.joint_pose = q_traj[t]
        frame = render(camera, gaussians, bg)["render"]          # (3,H,W), torch
        frames.append(frame)
    video = torch.stack(frames, dim=0)                           # (T,3,H,W)
    video = video.unsqueeze(0)                                   # (1,T,3,H,W)
    return video

def render_tracks(camera, gaussians, bg, q_traj, stride=100):
    track = []
    vis = []
    for t in range(q_traj.shape[0]):
        camera.joint_pose = q_traj[t]
        rendering = render(camera, gaussians, bg)#["viewspace_points"][0] 
        points = rendering['viewspace_points'][0][::stride,:]
        visibility_filter = rendering['visibility_filter'][::stride] # (S, )
        track.append(points)
        vis.append(visibility_filter)
        del rendering
    tracks = torch.stack(track, dim=0)   # (T, S, 2)
    visibilities = torch.stack(vis, dim=0)    # (T, S)                          
    return tracks, visibilities


def render_3d_tracks_perframe(cam_list, gaussians, q_traj, stride=50):
    """
    For each frame t, set cam_list[t].joint_pose = q_traj[t] and call render_3d.
    Returns:
      tracks      : (T, S, 3)  world-space 3-D means (differentiable w.r.t. q_traj)
      visibilities: (T, S)     bool/float visibility filter
    """
    from gaussian_renderer import render_3d
    track = []
    vis   = []
    for t in range(q_traj.shape[0]):
        cam_list[t].joint_pose = q_traj[t]          # keep on computation graph
        out = render_3d(cam_list[t], gaussians)
        track.append(out['viewspace_points'][0][::stride])    # (S, 3)
        vis.append(out['visibility_filter'][::stride].float())# (S,)
    tracks       = torch.stack(track, dim=0)   # (T, S, 3)
    visibilities = torch.stack(vis,   dim=0)   # (T, S)
    return tracks, visibilities


def soft_chamfer(P, S, tau=50.0):
    # P: (M,2), S: (N,2)
    d2 = ((P[:, None, :] - S[None, :, :])**2).sum(dim=-1)  # (M,N)
    p2s = -tau * torch.logsumexp(-d2 / tau, dim=1)
    s2p = -tau * torch.logsumexp(-d2 / tau, dim=0)
    return p2s.mean() + s2p.mean()

def chamfer_distance(P, S):
    # P: (M,3), S: (N,3)
    d2 = ((P[:, None, :] - S[None, :, :]) ** 2).sum(dim=-1)
    return d2.min(dim=1).values.mean() + d2.min(dim=0).values.mean()

def sinkhorn_trajectory_loss(
    G_vis, G_tracks,
    rendered_visibilities, rendered_tracks,
    q_traj, T,
    eps_ot=0.05,
    iters=60,
    unmatched_cost=1e3,
    smooth_w=0.1,
    velocity_w=0.1,
    min_shared_visible=3,
):
    """
    Trajectory-space Sinkhorn OT loss.
    """

    S = G_tracks[:T]                 # (T, K, 2)
    P = rendered_tracks[:T]          # (T, M, 2)
    S_vis = G_vis[:T] > 0.5          # (T, K)
    P_vis = rendered_visibilities[:T] > 0.5  # (T, M)

    T_, M, _ = P.shape
    _, K, _ = S.shape

    # ---- Trajectory cost matrix C (M x K) ----
    diff = P[:, :, None, :] - S[:, None, :, :]      # (T, M, K, 2)
    diff2 = (diff ** 2).sum(-1)                     # (T, M, K)

    valid = P_vis[:, :, None] & S_vis[:, None, :]   # (T, M, K)
    valid_f = valid.float()

    denom = valid_f.sum(0)                           # (M, K)
    safe_denom = denom.clamp_min(1.0)
    C = (diff2 * valid_f).sum(0) / safe_denom       # (M, K)

    # Add a trajectory velocity term so correspondences prefer smooth temporal behavior.
    if T_ > 1:
        vel_diff = (P[1:, :, None, :] - P[:-1, :, None, :]) - (S[1:, None, :, :] - S[:-1, None, :, :])
        vel_diff2 = (vel_diff ** 2).sum(-1)         # (T-1, M, K)
        vel_valid = valid[1:] & valid[:-1]
        vel_valid_f = vel_valid.float()
        vel_denom = vel_valid_f.sum(0).clamp_min(1.0)
        C = C + velocity_w * (vel_diff2 * vel_valid_f).sum(0) / vel_denom

    # If never visible together → make very expensive
    C = torch.where(denom >= min_shared_visible, C, torch.full_like(C, unmatched_cost))

    # ---- Sinkhorn (log-domain, stable) ----
    logK = -C / eps_ot

    log_a = torch.full((M,), -torch.log(torch.tensor(float(M), device=C.device)), device=C.device)
    log_b = torch.full((K,), -torch.log(torch.tensor(float(K), device=C.device)), device=C.device)

    u = torch.zeros_like(log_a)
    v = torch.zeros_like(log_b)

    for _ in range(iters):
        u = log_a - torch.logsumexp(logK + v[None, :], dim=1)
        v = log_b - torch.logsumexp(logK.t() + u[None, :], dim=1)

    logPi = logK + u[:, None] + v[None, :]
    Pi = torch.exp(logPi)

    ot_loss = (Pi * C).sum()

    # ---- Smoothness (unchanged from your version) ----
    smooth = ((q_traj[1:] - q_traj[:-1]) ** 2).mean()

    total = ot_loss + smooth_w * smooth

    return ot_loss, smooth, total

def filter_cotracker_queries(G_tracks, G_vis, min_visibility_ratio=0.3):
    """
    Keep only trajectories that are visible for at least a minimum ratio of frames.
    This removes unstable tracks and reduces discontinuous correspondences.
    """
    vis = G_vis.float()
    keep = vis.mean(dim=0) >= min_visibility_ratio
    if keep.sum() == 0:
        return G_tracks, G_vis
    return G_tracks[:, keep], G_vis[:, keep]

def tracks_from_video_with_sam(video_frames_uint8, text="robot arm", n_points=32, seed=0, indices=[0]):
    # video_frames_uint8: (T,H,W,3) uint8
    if isinstance(indices, set):
        indices = sorted(indices)
    else:
        indices = list(indices)

    if not indices:
        return {}

    num_frames = len(video_frames_uint8)
    for idx in indices:
        if idx < 0 or idx >= num_frames:
            raise IndexError(f"Frame index {idx} is out of bounds for {num_frames} frames.")

    sam_model = Sam3Model.from_pretrained("facebook/sam3").to(device).eval()
    sam_proc = Sam3Processor.from_pretrained("facebook/sam3")

    images = [Image.fromarray(video_frames_uint8[i]) for i in indices]
    texts = [text] * len(images)
    inputs = sam_proc(images=images, text=texts, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = sam_model(**inputs)

    results = sam_proc.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=inputs["original_sizes"].tolist()
    )

    points_by_index = {}
    for frame_idx, result in zip(indices, results):
        masks = result["masks"]

        # Normalize HF output variants to a single (H,W) boolean mask per frame.
        if isinstance(masks, (list, tuple)):
            if len(masks) == 0:
                combined_mask = None
            else:
                masks = torch.stack(
                    [m if isinstance(m, torch.Tensor) else torch.from_numpy(m) for m in masks],
                    dim=0,
                )

        if isinstance(masks, torch.Tensor):
            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks[:, 0]
            elif masks.ndim == 2:
                masks = masks[None, ...]
            combined_mask = masks.any(dim=0).cpu().numpy()
        elif masks is None:
            combined_mask = None
        else:
            masks = np.asarray(masks)
            combined_mask = None if masks.size == 0 else np.any(masks, axis=0)

        if combined_mask is None or not np.any(combined_mask):
            points_by_index[frame_idx] = np.zeros((0, 2), dtype=np.float32)
            continue

        points_by_index[frame_idx] = sample_points_from_mask(
            combined_mask,
            n=n_points,
            seed=seed + frame_idx,
        )

    # FREE VRAM
    del outputs, inputs, sam_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if len(indices) == 1:
        return points_by_index[indices[0]]
    return points_by_index

def spline_trajectory(B, C_raw):
    Q = B @ C_raw  # [T, D]
    # If pose_normalized is expected in [-1,1], this is a safe guard:
    Q = torch.tanh(Q)
    return Q  # [20, 7]

def accel_penalty(Q):
    return ((Q[2:] - 2*Q[1:-1] + Q[:-2])**2).mean()

def chamfer_loss(G_vis, G_tracks, rendered_visibilities, rendered_tracks, q_traj, T):
    # soft chamfer per timestep, optionally mask by visibility
    loss = 0.0
    for t in range(T):
        # optionally keep only visible source points
        vis = (G_vis[t] > 0.5)
        S_t = G_tracks[t][vis]
        
        p_vis = (rendered_visibilities[t] > 0.5)
        P_t = rendered_tracks[t][p_vis]
        loss = loss + soft_chamfer(P_t, S_t, tau=300.0)

    loss = loss / T
    smooth = ((q_traj[1:] - q_traj[:-1])**2).mean()
    total = loss + 0.1 * smooth
    return loss, smooth, total

def set_camera_resolution(cam, H, W):
    # Common field names across 3DGS forks
    if hasattr(cam, "image_height"): cam.image_height = int(H)
    if hasattr(cam, "image_width"):  cam.image_width  = int(W)
    if hasattr(cam, "H"): cam.H = int(H)
    if hasattr(cam, "W"): cam.W = int(W)

    # If intrinsics exist, scale them to match new resolution
    # We need old size to compute scale; try to read it from common fields.
    oldH = getattr(cam, "image_height", getattr(cam, "H", None))
    oldW = getattr(cam, "image_width",  getattr(cam, "W", None))

    # If oldH/oldW were overwritten above, you may want to pass them in explicitly instead.
    # So: only scale if the camera exposes fx/fy/cx/cy and we can infer old sizes another way.
    # If your Camera class has original size fields, use those.

def scale_camera_intrinsics(cam, sx, sy):
    # fx/fy/cx/cy case
    for name in ["fx", "cx"]:
        if hasattr(cam, name): setattr(cam, name, getattr(cam, name) * sx)
    for name in ["fy", "cy"]:
        if hasattr(cam, name): setattr(cam, name, getattr(cam, name) * sy)

    # K matrix case
    if hasattr(cam, "K") and cam.K is not None:
        K = cam.K
        K = K.clone()
        K[0, 0] *= sx; K[0, 2] *= sx
        K[1, 1] *= sy; K[1, 2] *= sy
        cam.K = K

if __name__ == "__main__":
    import moviepy as mpy
    from moviepy.editor import ImageSequenceClip
    import copy
    import numpy as np
    from PIL import Image, ImageDraw
    import imageio.v3 as iio
    import h5py
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 – registers 3d projection

    video_path = "./999.mp4"
    frames = iio.imread(video_path, plugin="pyav")  # plugin="pyav"
    Ts, Hs, Ws, _ = frames.shape

    # Pick source queries via SAM3 on the first, middle, and last frames.
    query_indices = sorted({0, Ts // 2, Ts - 1})
    src_pts_by_idx = tracks_from_video_with_sam(
        frames,
        text="left hand",
        n_points=256,
        seed=0,
        indices=query_indices,
    )

    query_chunks = []
    for t in query_indices:
        pts_xy = src_pts_by_idx[t]
        if pts_xy.shape[0] == 0:
            continue
        query_chunks.append(make_cotracker_queries(pts_xy, device=device, t0=t))

    if not query_chunks:
        raise ValueError("SAM3 did not produce any valid masks on the selected query frames.")

    src_queries = torch.cat(query_chunks, dim=1)  # (1,N_total,3)

    source_video = torch.from_numpy(frames).to(device).permute(0,3,1,2).float().unsqueeze(0)  # (1,Ts,3,H,W)

    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device).eval()
    for p in cotracker.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        G_tracks, G_vis = cotracker(
            source_video,
            queries=src_queries,
            backward_tracking=True,
        )  # (1,Ts,N,2), (1,Ts,N,1)
    G_tracks = G_tracks[0]                # (Ts,N,2) (21, 64, 2)
    G_vis = G_vis[0,:,:]                # (Ts,N)
    G_tracks, G_vis = filter_cotracker_queries(G_tracks, G_vis, min_visibility_ratio=0.35)
    #print("ground truth tracks shape=", G_tracks.shape)

    del source_video
    del cotracker




    # ── Load depth maps and camera parameters ────────────────────────────────
    depth_data  = np.load("999_depths.npz")
    depths_np   = depth_data["depths"]          # (T_d, H_d, W_d)  float32, metric metres
    D_T, D_H, D_W = depths_np.shape

    with h5py.File("999.hdf5", "r") as f:
        K_np          = f["camera/intrinsic"][:]   # (3, 3)  – intrinsics for the depth-map resolution
        T_cam_world   = f["transforms/camera"][:]  # (T_d, 4, 4) – camera-to-world pose per frame

    # Camera intrinsics (depth-map space: D_H × D_W)
    fx = float(K_np[0, 0])
    fy = float(K_np[1, 1])
    cx = float(K_np[0, 2])
    cy = float(K_np[1, 2])

    # ── Lift 2-D tracks → 3-D world coordinates ──────────────────────────────
    # G_tracks: (Ts, N, 2)  pixel coords (x=col, y=row) in the *source video* space (Hs × Ws)
    # Depth map may have a different resolution, so we scale pixel coords accordingly.
    sx = D_W / Ws   # column scale  video → depth-map
    sy = D_H / Hs   # row    scale  video → depth-map

    G_tracks_np = G_tracks.cpu().numpy()   # (Ts, N, 2)
    G_vis_np    = G_vis.cpu().numpy()      # (Ts, N)

    T_use = min(Ts, D_T)
    N_pts = G_tracks_np.shape[1]

    tracks_3d = np.full((T_use, N_pts, 3), np.nan, dtype=np.float32)  # world-space XYZ

    for t in range(T_use):
        for n in range(N_pts):
            if G_vis_np[t, n] < 0.5:
                continue

            # Track pixel coordinates in source-video space
            u_vid = G_tracks_np[t, n, 0]   # x (column)
            v_vid = G_tracks_np[t, n, 1]   # y (row)

            # Map to depth-map pixel space
            u_d = u_vid * sx
            v_d = v_vid * sy
            u_i = int(round(u_d))
            v_i = int(round(v_d))

            if not (0 <= u_i < D_W and 0 <= v_i < D_H):
                continue

            z = float(depths_np[t, v_i, u_i])
            if z <= 0.0 or not np.isfinite(z):
                continue

            # Unproject to camera space
            xc = (u_d - cx) * z / fx
            yc = (v_d - cy) * z / fy
            zc = z
            p_cam = np.array([xc, yc, zc, 1.0], dtype=np.float64)

            # Transform to world space.
            # transforms/camera is assumed to be the camera-to-world (c2w) matrix
            # so  p_world = T_c2w @ p_cam
            p_world = T_cam_world[t].astype(np.float64) @ p_cam
            #p_world = np.linalg.inv(T_cam_world[t].astype(np.float64)) @ p_cam
            tracks_3d[t, n] = p_world[:3].astype(np.float32)

    n_valid = int(np.isfinite(tracks_3d[:, :, 0]).sum())
    print(f"3D tracks: shape={tracks_3d.shape}, valid points={n_valid}")

    TRAIL   = 15   # number of past frames to draw as a trailing line
    FPS_OUT = 20

    # ── Animate 3D tracks and save as MP4 ────────────────────────────────────
    # Pre-compute global axis limits (ignore NaN)
    x_all = tracks_3d[:, :, 0]
    y_all = tracks_3d[:, :, 1]
    z_all = tracks_3d[:, :, 2]
    xlim = (float(np.nanmin(x_all)), float(np.nanmax(x_all)))
    ylim = (float(np.nanmin(y_all)), float(np.nanmax(y_all)))
    zlim = (float(np.nanmin(z_all)), float(np.nanmax(z_all)))

    # One fixed colour per track for easy visual identification
    cmap    = plt.cm.tab20
    colours = [cmap(i % 20) for i in range(N_pts)]

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")

    def update(t: int):
        ax.cla()
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(f"3-D hand point tracks — frame {t:03d} / {T_use - 1}")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)

        for n in range(N_pts):
            t0  = max(0, t - TRAIL)
            seg = tracks_3d[t0 : t + 1, n, :]        # (trail, 3)
            valid_mask = np.all(np.isfinite(seg), axis=1)

            # Draw trailing line
            if valid_mask.sum() >= 2:
                s = seg[valid_mask]
                ax.plot(s[:, 0], s[:, 1], s[:, 2],
                        color=colours[n], lw=0.8, alpha=0.6)

            # Draw current point
            cur = tracks_3d[t, n]
            if np.all(np.isfinite(cur)):
                ax.scatter(*cur, color=colours[n], s=18, depthshade=False, zorder=5)

    ani     = FuncAnimation(fig, update, frames=T_use, interval=int(1000 / FPS_OUT))
    out_3d  = "3d_hand_tracks.mp4"
    writer  = FFMpegWriter(fps=FPS_OUT, metadata={"title": "3-D hand tracks"})
    ani.save(out_3d, writer=writer, dpi=120)
    plt.close(fig)
    print(f"Saved 3-D track video → {out_3d}")

    # ── Also save a static overview (all tracks projected onto world XY plane) ─
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    for n in range(N_pts):
        seg = tracks_3d[:, n, :]          # (T, 3)
        valid = np.all(np.isfinite(seg), axis=1)
        if valid.sum() < 2:
            continue
        ax2.plot(seg[valid, 0], seg[valid, 1], color=colours[n], lw=0.6, alpha=0.7)
        ax2.scatter(seg[valid, 0][-1], seg[valid, 1][-1],
                    color=colours[n], s=20, zorder=5)
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_title("3-D hand point tracks — top view (XY plane)")
    ax2.set_aspect("equal", "box")
    fig2.tight_layout()
    out_static = "3d_hand_tracks_topview.png"
    fig2.savefig(out_static, dpi=150)
    plt.close(fig2)
    print(f"Saved static top-view overview → {out_static}")

    # ── Reproject 3-D tracks to original camera view and overlay on video ─────
    def project_world_to_video(p_world_xyz, t_idx):
        """
        Project a world-space 3D point into source-video pixels for frame t_idx.
        Returns (u_video, v_video) or None if point is behind camera / out of view.
        """
        p_world_h = np.array([p_world_xyz[0], p_world_xyz[1], p_world_xyz[2], 1.0], dtype=np.float64)

        # transforms/camera was used as camera-to-world in unprojection, so invert for world->camera.
        T_world_cam = np.linalg.inv(T_cam_world[t_idx].astype(np.float64))
        #T_world_cam = T_cam_world[t_idx].astype(np.float64)
        p_cam = T_world_cam @ p_world_h
        z_cam = p_cam[2]

        if z_cam <= 1e-8 or (not np.isfinite(z_cam)):
            return None

        # Camera projection in depth-map pixel space
        u_d = fx * (p_cam[0] / z_cam) + cx
        v_d = fy * (p_cam[1] / z_cam) + cy

        # Convert depth-map pixel space back to source-video pixel space
        u_v = u_d / sx
        v_v = v_d / sy

        if not (0.0 <= u_v < Ws and 0.0 <= v_v < Hs):
            return None

        return float(u_v), float(v_v)

    # Convert matplotlib RGBA colours to RGB uint8 for drawing
    colours_rgb = [tuple(int(255 * c) for c in rgba[:3]) for rgba in colours]

    overlay_frames = []
    for t in range(T_use):
        frame_pil = Image.fromarray(frames[t].astype(np.uint8))
        draw = ImageDraw.Draw(frame_pil)

        for n in range(N_pts):
            t0 = max(0, t - TRAIL)
            seg_world = tracks_3d[t0 : t + 1, n, :]  # (trail, 3)

            # Project valid 3D segment points into this frame
            projected = []
            for k in range(seg_world.shape[0]):
                p = seg_world[k]
                if not np.all(np.isfinite(p)):
                    continue
                uv = project_world_to_video(p, t)
                if uv is not None:
                    projected.append(uv)

            color = colours_rgb[n]

            # Draw trailing polyline
            if len(projected) >= 2:
                draw.line(projected, fill=color, width=2)

            # Draw current point if available
            p_cur = tracks_3d[t, n]
            if np.all(np.isfinite(p_cur)):
                uv_cur = project_world_to_video(p_cur, t)
                if uv_cur is not None:
                    r = 3
                    x, y = uv_cur
                    draw.ellipse((x - r, y - r, x + r, y + r), fill=color)

        overlay_frames.append(np.asarray(frame_pil))

    overlay_frames = np.stack(overlay_frames, axis=0)
    out_overlay = "3d_hand_tracks_on_video.mp4"

    # Use pyav if available (same plugin used for reading); fallback to imageio default writer.
    try:
        iio.imwrite(out_overlay, overlay_frames, plugin="pyav", fps=FPS_OUT)
    except Exception:
        iio.imwrite(out_overlay, overlay_frames, fps=FPS_OUT)

    print(f"Saved camera-view 3-D overlay video → {out_overlay}")
    torch.cuda.empty_cache()


















    #     # ── Start of rewritten retargeting code ───────────────────────────────────
    # # Goal:
    # #   - DO NOT use EgoDex camera poses as renderer cameras
    # #   - Apply a manual source-world -> MuJoCo transform
    # #   - Render robot with a valid renderer-native camera
    # #   - Match 3D motion directly in MuJoCo world

    # print("\n=== Rewritten retargeting: use manual source-world -> MuJoCo alignment ===")

    # # Keep sequence manageable for debugging
    # T_use = min(T_use, 20)

    # # Source 3D tracks as tensors
    # G_tracks_3d = torch.from_numpy(tracks_3d[:T_use]).to("cuda")          # (T, N_src, 3)
    # G_vis_3d = torch.from_numpy(G_vis_np[:T_use]).to("cuda") > 0.5        # (T, N_src) bool

    # # Also mask invalid depth / NaN points
    # depth_valid = torch.isfinite(G_tracks_3d).all(dim=-1)
    # G_vis_3d = G_vis_3d & depth_valid

    # # Apply the manually configured source-world -> MuJoCo transform.
    # manual_R, manual_t = load_manual_world_transform("camera_extrinsics_ego.json")
    # manual_R = manual_R.to(G_tracks_3d.device)
    # manual_t = manual_t.to(G_tracks_3d.device)
    # G_tracks_3d_mujoco = apply_rigid_transform(G_tracks_3d, manual_R, manual_t)

    # # Load pretrained differentiable robot renderer
    # gaussians, bg, sample_cams, chain = initialize_gaussians(
    #     model_path="results/shadow_hand/",
    #     from_ckpt=True,
    # )

    # # Try to infer DoF robustly
    # dof = None
    # for attr in ["dof", "n_joints"]:
    #     if hasattr(chain, attr):
    #         dof = int(getattr(chain, attr))
    #         break
    # if dof is None:
    #     dof = 24  # fallback
    # print(f"Using dof={dof}")

    # # Use one valid renderer-native camera from the robot dataset.
    # # This avoids mixing ARKit cameras into MuJoCo world.
    # render_cam = sample_cams[0]
    # cam_list = [render_cam for _ in range(T_use)]

    # # Freeze camera params if this camera object has parameters
    # if hasattr(render_cam, "parameters"):
    #     for p in render_cam.parameters():
    #         p.requires_grad_(False)

    # # B-spline trajectory parameterization
    # K_ctrl = max(4, T_use // 4)
    # B_mat = bspline_basis_matrix(T=T_use, K=K_ctrl, degree=3).to("cuda")   # (T, K_ctrl)
    # q_ctrl = torch.nn.Parameter(torch.zeros(K_ctrl, dof, device="cuda"))

    # # Optimize only the robot joint trajectory; the world-frame alignment is fixed.
    # optimizer_3d = torch.optim.Adam([
    #     {"params": [q_ctrl], "lr": 1e-2},
    # ])
    # scheduler_3d = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer_3d, T_max=300, eta_min=1e-3
    # )

    # print(
    #     f"Starting optimization — T={T_use}, dof={dof}, K_ctrl={K_ctrl}, "
    #     f"source pts={G_tracks_3d.shape[1]}"
    # )

    # STRIDE = 80
    # N_ITER = 300

    # for it in range(N_ITER):
    #     optimizer_3d.zero_grad()

    #     # Robot joint trajectory parameterized by spline control points
    #     Q = spline_trajectory(B_mat, q_ctrl)   # (T_use, dof)

    #     # Render robot 3D points in MuJoCo world
    #     R_tracks, R_vis = render_3d_tracks_perframe(
    #         cam_list, gaussians, Q, stride=STRIDE
    #     )  # R_tracks: (T,S,3), R_vis: (T,S)

    #     R_vis_bool = R_vis > 0.5

    #     # Per-frame Chamfer in MuJoCo-world 3D
    #     loss_3d = torch.tensor(0.0, device="cuda")
    #     n_frames_used = 0

    #     for t in range(T_use):
    #         S_mask = G_vis_3d[t]       # source valid points
    #         P_mask = R_vis_bool[t]     # rendered valid points

    #         if S_mask.sum() < 3 or P_mask.sum() < 3:
    #             continue

    #         S_t = G_tracks_3d_mujoco[t][S_mask]    # (Ns, 3)
    #         P_t = R_tracks[t][P_mask]              # (Np, 3)

    #         loss_3d = loss_3d + chamfer_distance(P_t, S_t)
    #         n_frames_used += 1

    #     if it == 0 or it % 10 == 0:
    #         s_counts = [int(G_vis_3d[t].sum().item()) for t in range(T_use)]
    #         p_counts = [int((R_vis[t] > 0.5).sum().item()) for t in range(T_use)]
    #         print("source valid counts per frame:", s_counts)
    #         print("render valid counts per frame:", p_counts)

    #     if n_frames_used == 0:
    #         print(f"iter {it}: no valid frames, skipping")
    #         continue

    #     loss_3d = loss_3d / n_frames_used

    #     # Motion regularization / temporal matching
    #     vel = velocity_loss(G_tracks_3d_mujoco, G_vis_3d, R_tracks, R_vis_bool)
    #     smooth = accel_penalty(Q)
    #     vel_reg = ((Q[1:] - Q[:-1]) ** 2).mean()

    #     total = (
    #         loss_3d
    #         + 0.2 * vel
    #         + 0.05 * smooth
    #         + 0.02 * vel_reg
    #     )

    #     total.backward()
    #     torch.nn.utils.clip_grad_norm_([q_ctrl], max_norm=1.0)
    #     optimizer_3d.step()
    #     scheduler_3d.step()

    #     if it % 20 == 0 or it == N_ITER - 1:
    #         with torch.no_grad():
    #             print(
    #                 f"iter {it:3d} | "
    #                 f"loss_3d={float(loss_3d):.4f} | "
    #                 f"vel={float(vel):.4f} | "
    #                 f"smooth={float(smooth):.4f} | "
    #                 f"frames={n_frames_used}/{T_use}"
    #             )

    # # ── Save optimized trajectory ─────────────────────────────────────────────
    # with torch.no_grad():
    #     Q_final = spline_trajectory(B_mat, q_ctrl)   # (T_use, dof)

    # np.save("optimized_q_traj.npy", Q_final.detach().cpu().numpy())
    # print("Saved optimized trajectory → optimized_q_traj.npy")

    # # ── Optional: save manually transformed source tracks in MuJoCo world ─────
    # with torch.no_grad():
    #     G_aligned_np = G_tracks_3d_mujoco.detach().cpu().numpy()

    # np.save("aligned_tracks_3d_mujoco.npy", G_aligned_np)
    # print("Saved aligned source tracks in MuJoCo frame → aligned_tracks_3d_mujoco.npy")

    # # ── Render final robot video from the renderer-native camera ──────────────
    # final_frames = []
    # with torch.no_grad():
    #     for t in range(T_use):
    #         render_cam.joint_pose = Q_final[t]
    #         img = render(render_cam, gaussians, bg)["render"]  # (3,H,W)
    #         final_frames.append(
    #             (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    #         )

    # out_robot = "retargeted_robot_hand_fixedcam.mp4"
    # try:
    #     iio.imwrite(out_robot, np.stack(final_frames), plugin="pyav", fps=FPS_OUT)
    # except Exception:
    #     iio.imwrite(out_robot, np.stack(final_frames), fps=FPS_OUT)

    # print(f"Saved retargeted robot render → {out_robot}")















    # # start of retargeting code
    # # ── Load pretrained differentiable robot-hand renderer ──────────────────
    # gaussians, bg, sample_cams, chain = initialize_gaussians(
    #     model_path="results/shadow_hand/", from_ckpt=True)
    # dof = 24 #chain.dof

    # # ── Build per-frame frozen cameras from HDF5 extrinsics ──────────────────
    # # K_np is defined at depth-map resolution (D_H × D_W), so use those as the
    # # image dimensions for the renderer camera.  FoV is derived from K_np.
    # import math as _math
    # FoVx = 2.0 * _math.atan(D_W / (2.0 * fx))
    # FoVy = 2.0 * _math.atan(D_H / (2.0 * fy))

    # # T_cam_world[t] is camera-to-world (c2w); Camera_Pose needs world-to-camera (w2c).
    # per_frame_cams = []
    # dummy_joint = torch.zeros(dof, device="cuda")
    # for t in range(T_use):
    #     w2c = torch.tensor(
    #         np.linalg.inv(T_cam_world[t].astype(np.float64)),
    #         dtype=torch.float32, device="cuda"
    #     )
    #     cam_t = Camera_Pose(
    #         w2c, FoVx, FoVy,
    #         image_width=D_W, image_height=D_H,
    #         joint_pose=dummy_joint,
    #         zero_init=True,
    #     ).cuda()
    #     # Freeze camera — only the trajectory is optimized, not the viewpoint.
    #     for p in cam_t.parameters():
    #         p.requires_grad_(False)
    #     per_frame_cams.append(cam_t)

    # # ── B-spline trajectory setup ─────────────────────────────────────────────
    # K_ctrl = max(4, T_use // 4)
    # B_mat  = bspline_basis_matrix(T=T_use, K=K_ctrl, degree=3).to("cuda")  # (T_use, K_ctrl)
    # q_ctrl = torch.nn.Parameter(torch.zeros(K_ctrl, dof, device="cuda"))   # learnable control pts

    # optimizer_3d = torch.optim.Adam([q_ctrl], lr=2e-2)
    # scheduler_3d = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer_3d, T_max=300, eta_min=1e-3)

    # # Source 3-D tracks as tensors (NaN where invisible)
    # G_tracks_3d = torch.from_numpy(tracks_3d).to("cuda")   # (T_use, N_src, 3)
    # G_vis_3d    = torch.from_numpy(G_vis_np[:T_use]).to("cuda")  # (T_use, N_src)
    # # Mask out frames/points with no valid depth (NaN in tracks_3d)
    # depth_valid = torch.isfinite(G_tracks_3d).all(dim=-1)   # (T_use, N_src)
    # G_vis_3d    = (G_vis_3d > 0.5) & depth_valid            # (T_use, N_src) bool

    # print(f"Starting 3-D flow matching optimisation — T={T_use}, dof={dof}, "
    #       f"K_ctrl={K_ctrl}, source pts={N_pts}")

    # # ── Optimisation loop ─────────────────────────────────────────────────────
    # STRIDE = 50   # subsample Gaussians; reduces cost while keeping signal
    # N_ITER = 300

    # for it in range(N_ITER):
    #     optimizer_3d.zero_grad()

    #     Q = spline_trajectory(B_mat, q_ctrl)   # (T_use, dof), tanh-bounded

    #     # Render 3-D Gaussian means per frame using HDF5 camera extrinsics
    #     R_tracks, R_vis = render_3d_tracks_perframe(
    #         per_frame_cams, gaussians, Q, stride=STRIDE)
    #     # R_tracks : (T_use, S, 3)   world-space, differentiable
    #     # R_vis    : (T_use, S)      float 0/1

    #     # ── Per-frame soft-Chamfer loss in 3-D ───────────────────────────────
    #     loss_3d = torch.tensor(0.0, device="cuda")
    #     n_frames_used = 0
    #     for t in range(T_use):
    #         S_mask = G_vis_3d[t]            # (N_src,) bool — valid source pts
    #         P_mask = R_vis[t] > 0.5        # (S,)    bool — visible rendered pts
    #         # if S_mask.sum() < 2 or P_mask.sum() < 2:
    #         #     continue
    #         S_t = G_tracks_3d[t][S_mask]   # (N_s, 3)
    #         #print("is there any NaN in S_t?", S_t.isnan().any())
    #         P_t = R_tracks[t]#[P_mask]      # (N_p, 3)
    #         #print(f"  iter {it}: S_t.shape={S_t.shape}, P_t.shape={P_t.shape}")
    #         loss_3d = loss_3d + soft_chamfer(P_t, S_t, tau=0.1)
    #         n_frames_used += 1

    #     if n_frames_used == 0:
    #         print(f"  iter {it}: no valid frames — skipping")
    #         continue

    #     loss_3d  = loss_3d / n_frames_used
    #     smooth   = accel_penalty(Q)
    #     vel_reg  = ((Q[1:] - Q[:-1]) ** 2).mean()
    #     total    = loss_3d + 0.05 * smooth + 0.02 * vel_reg

    #     total.backward()
    #     torch.nn.utils.clip_grad_norm_([q_ctrl], max_norm=1.0)
    #     optimizer_3d.step()
    #     scheduler_3d.step()

    #     if it % 20 == 0 or it == N_ITER - 1:
    #         print(f"  iter {it:3d}  loss_3d={float(loss_3d):.4f}  "
    #               f"smooth={float(smooth):.4f}  total={float(total):.4f}  "
    #               f"frames={n_frames_used}/{T_use}")

    # # ── Save optimised trajectory and final render ────────────────────────────
    # with torch.no_grad():
    #     Q_final = spline_trajectory(B_mat, q_ctrl)   # (T_use, dof)

    # np.save("optimized_q_traj.npy", Q_final.cpu().numpy())
    # print("Saved optimized trajectory → optimized_q_traj.npy")

    # # Render the final robot video using the first per-frame camera (frame 0)
    # # for a representative viewpoint — or use any fixed cam from the scene.
    # # `render()` calls into TorchScripted positional encoding in `lrs`, which
    # # can reject inference tensors even when gradients are disabled.
    # with torch.no_grad():
    #     sample_cam = next(iter(sample_cams))
    #     # Re-use cam0 from per_frame_cams to render at the HDF5 viewpoint
    #     render_cam = per_frame_cams[0]
    #     final_frames = []
    #     for t in range(T_use):
    #         render_cam = per_frame_cams[t]
    #         render_cam.joint_pose = Q_final[t]
    #         img = render(render_cam, gaussians, bg)["render"]  # (3, H, W)
    #         final_frames.append(
    #             (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    #         )

    # out_robot = "retargeted_robot_hand.mp4"
    # try:
    #     iio.imwrite(out_robot, np.stack(final_frames), plugin="pyav", fps=FPS_OUT)
    # except Exception:
    #     iio.imwrite(out_robot, np.stack(final_frames), fps=FPS_OUT)
    # print(f"Saved retargeted robot render → {out_robot}")
