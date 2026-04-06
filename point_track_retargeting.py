import torch
import numpy as np
from scene import Scene, RobotScene
from tqdm import tqdm
from os import makedirs
from PIL import Image
from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from gaussian_renderer import GaussianModel
from scene.cameras import Camera, Camera_Pose
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from cotrack import sample_points_from_mask, make_cotracker_queries
from transformers import Sam3Processor, Sam3Model
from cotracker.utils.visualizer import Visualizer
from utils.mujoco_utils import simulate_mujoco_scene, compute_camera_extrinsic_matrix

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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


def soft_chamfer(P, S, tau=50.0):
    # P: (M,2), S: (N,2)
    d2 = ((P[:, None, :] - S[None, :, :])**2).sum(dim=-1)  # (M,N)
    p2s = -tau * torch.logsumexp(-d2 / tau, dim=1)
    s2p = -tau * torch.logsumexp(-d2 / tau, dim=0)
    return p2s.mean() + s2p.mean()

def sinkhorn_trajectory_loss(
    G_vis, G_tracks,
    rendered_visibilities, rendered_tracks,
    q_traj, T,
    eps_ot=0.05,
    iters=60,
    unmatched_cost=1e3,
    smooth_w=0.1,
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

    denom = valid_f.sum(0).clamp_min(1.0)           # (M, K)
    C = (diff2 * valid_f).sum(0) / denom            # (M, K)

    # If never visible together → make very expensive
    C = torch.where(denom > 0, C, torch.full_like(C, unmatched_cost))

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
    #image0 = Image.fromarray(video_frames_uint8[0])
    #inputs = sam_proc(images=image0, text=text, return_tensors="pt").to(device)
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
    from PIL import Image
    import imageio.v3 as iio

    #video_path = "./saved_videos/test_egodex_onehand.mp4"
    #video_path = "/nethome/chuang475/datasets/egodex/part2/insert_remove_furniture_bench_round_table/999.mp4"
    video_path = "./999.mp4"
    #video_path = "./out_video.mp4"
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
    #print("ground truth tracks shape=", G_tracks.shape)

    del source_video
    del cotracker
    torch.cuda.empty_cache()

    # setup differentiable renderer
    gaussians, bg, cams, _ = initialize_gaussians(model_path="results/shadow_hand/", from_ckpt=True)
    #gaussians, bg, cams, _ = initialize_gaussians(model_path="results/universal_robots_ur5e_experiment/", from_ckpt=True)
    cam = next(iter(cams))

    dummy_cam = DummyCam()
    camera_extrinsic_matrix = compute_camera_extrinsic_matrix(dummy_cam)
    learnable_cam = Camera_Pose(
        torch.tensor(camera_extrinsic_matrix).clone().detach().float().cuda(),
        cam.FoVx, cam.FoVy,
        cam.image_width, cam.image_height,
        joint_pose=None,          # or pass your joint tensor if Camera_Pose expects it
        zero_init=True            # important: start from given pose then learn delta
    ).cuda()


    T = Ts #min(20, Ts)          # match length
    dof = 24 #6 #24
    K = T//4
    #B = None
    B = bspline_basis_matrix(T=T, K=K, degree=3).to("cuda")  # [20, 10]

    q_traj = torch.nn.Parameter(torch.ones(K, dof, device="cuda"))
    #q_traj = torch.nn.Parameter(torch.zeros(T, dof, device=device))
    # opt = torch.optim.Adam([q_traj], lr=2e-2)

    opt = torch.optim.Adam([
        {"params": [q_traj], "lr": 2e-2},
        {"params": learnable_cam.parameters(), "lr": 1e-2},
    ])
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    # best_total = float("inf")
    # best_iter = None
    # best_q_traj = None
    # best_cam_state = None
    for it in range(200):
        opt.zero_grad()
        with torch.cuda.amp.autocast(dtype=torch.float32):
            Q = spline_trajectory(B, q_traj)# if B else q_traj
            rendered_tracks, rendered_visibilities = render_tracks(learnable_cam, gaussians, bg, Q)

            # soft chamfer per timestep, optionally mask by visibility
            loss = 0.0
            # visis = 0.0
            # tots = 0.0

            # init_g_vis = (G_vis[0] > 0.5)
            # init_p_vis = (rendered_visibilities[0] > 0.5)
            for t in range(T):
                l = soft_chamfer(G_tracks[t], rendered_tracks[t]) / len(rendered_tracks[t])
                # cur_g_vis = (G_vis[t] > 0.5)
                # cur_p_vis = (rendered_visibilities[t] > 0.5)
               
                # init_S_t = G_tracks[t][init_g_vis]
                # init_P_t = rendered_tracks[t][init_p_vis]

                # cur_S_t = G_tracks[t][cur_g_vis]
                # cur_P_t = rendered_tracks[t][cur_p_vis]
                
                # # tots += len(rendered_visibilities[0])
                # # visis += sum(p_vis)
                # l = ((soft_chamfer(init_S_t, init_P_t, tau=300.0)/sum(init_p_vis)) + (soft_chamfer(cur_S_t, cur_P_t, tau=300.0)/sum(cur_p_vis)))/2
                loss = loss + l


            loss = loss / T
            smooth = ((q_traj[1:] - q_traj[:-1])**2).mean()
            total = loss + 0.1 * smooth
            # print(f"there are {visis/T} visible gaussians")
            # print(f"there are {tots/T} total gaussians")
            #loss, smooth, total = sinkhorn_trajectory_loss(G_vis, G_tracks, rendered_visibilities, rendered_tracks, q_traj, T)

        # if not torch.isfinite(total):
        #     print(f"Skipping iteration {it} because total loss is non-finite: {float(total.detach())}")
        #     continue

        # total_value = float(total.detach())
        # if total_value < best_total:
        #     best_total = total_value
        #     best_iter = it
        #     best_q_traj = q_traj.detach().clone()
        #     best_cam_state = {
        #         name: param.detach().clone()
        #         for name, param in learnable_cam.state_dict().items()
        #     }

        scaler.scale(total).backward()
        scaler.step(opt)
        scaler.update()
        print("q_traj.grad mean abs =", q_traj.grad.abs().mean().item(), "max =", q_traj.grad.abs().max().item())
        #opt.step()

        if it % 10 == 0:
            print(it, float(total), float(loss), float(smooth))

    # if best_q_traj is not None and best_cam_state is not None:
    #     with torch.no_grad():
    #         q_traj.copy_(best_q_traj)
    #     learnable_cam.load_state_dict(best_cam_state)
    #     print(f"Restored best parameters from iteration {best_iter} with total loss {best_total}")
    # else:
    #     print("No finite loss was found; rendering with the latest parameters.")

    with torch.no_grad():
        Q = spline_trajectory(B, q_traj)# if B else q_traj   # (T,dof)

    with torch.inference_mode():
        video = render_robot_video(learnable_cam, gaussians, bg, Q)
        pred_tracks, pred_visibility = render_tracks(learnable_cam, gaussians, bg, Q)

    pred_tracks = pred_tracks.unsqueeze(0) # (1, T, N, 2)
    pred_visibility = pred_visibility.unsqueeze(0).unsqueeze(-1) # (1, T, N, 1)
    video = torch.clamp(video, 0, 1)
    video = video.cpu().detach().numpy()[0]
    print(video.shape)
    video = np.transpose(video, (0, 2, 3, 1))    # (T, H, W, 3)
    video = (video * 255).astype(np.uint8)

    clip = ImageSequenceClip(list(video), fps=10)

    #test_name = "rendered_noinit_bspline_chamfer_ur5e"
    test_name = "rendered_shadow_noinit_bspline_chamfer_optcamera"
    clip.write_videofile(test_name + ".mp4") 

    frames = iio.imread(test_name + ".mp4", plugin="FFMPEG")  # plugin="pyav"
    video = torch.tensor(frames).permute(0, 3, 1, 2)[:-1,:,:,:][None].float().to(device)  # B T C H W

    vis = Visualizer(save_dir="./" + test_name, linewidth=1, mode='cool', tracks_leave_trace=-1)
    vis.visualize(video, pred_tracks, pred_visibility, filename=test_name)
