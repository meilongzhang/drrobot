import argparse
from dataclasses import dataclass

import h5py
import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image, ImageDraw

from cotrack import make_cotracker_queries
from gaussian_renderer import render, render_3d
from point_track_retargeting_2 import (
    filter_cotracker_queries,
    initialize_gaussians,
    tracks_from_video_with_sam,
)


device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class OrientedBox3D:
    center: np.ndarray
    axes: np.ndarray
    extents: np.ndarray
    corners: np.ndarray


def parse_args():
    parser = argparse.ArgumentParser(description="Estimate 3D bounding boxes from source-video tracks and rendered Gaussian means.")
    parser.add_argument("--run_source_box", action="store_true")
    parser.add_argument("--run_render_box", action="store_true")
    parser.add_argument("--video_path", default="./999.mp4")
    parser.add_argument("--depth_path", default="./999_depths.npz")
    parser.add_argument("--hdf5_path", default="./999.hdf5")
    parser.add_argument("--text", default="left hand")
    parser.add_argument("--n_points", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min_visibility_ratio", type=float, default=0.35)
    parser.add_argument("--model_path", default="results/shadow_hand/")
    parser.add_argument("--stage", choices=["canonical", "pose_conditioned"], default="pose_conditioned")
    parser.add_argument("--camera_index", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--quantile", type=float, default=0.02)
    parser.add_argument("--visible_only", action="store_true")
    parser.add_argument("--source_box_npz_path", default="./source_video_bbox_3d.npz")
    parser.add_argument("--source_overlay_path", default="./source_video_bbox_3d_overlay.mp4")
    parser.add_argument("--render_box_npz_path", default="./canonical_robot_bbox_3d.npz")
    parser.add_argument("--render_overlay_path", default="./canonical_robot_bbox_overlay.png")
    parser.add_argument("--fps", type=int, default=20)
    return parser.parse_args()


def ensure_right_handed(axes):
    axes = np.asarray(axes, dtype=np.float64).copy()
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1.0
    return axes


def fit_oriented_box(points_xyz, quantile=0.02):
    points_xyz = np.asarray(points_xyz, dtype=np.float64)
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError(f"Expected points shaped (N, 3), got {points_xyz.shape}.")
    if points_xyz.shape[0] < 3:
        raise ValueError("Need at least 3 points to estimate a 3D bounding box.")

    centroid = points_xyz.mean(axis=0)
    centered = points_xyz - centroid
    cov = centered.T @ centered / max(points_xyz.shape[0], 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    axes = ensure_right_handed(eigvecs[:, order])

    local_pts = centered @ axes
    lo = np.quantile(local_pts, quantile, axis=0)
    hi = np.quantile(local_pts, 1.0 - quantile, axis=0)
    extents = 0.5 * np.maximum(hi - lo, 1e-6)
    local_center = 0.5 * (lo + hi)
    center_world = centroid + axes @ local_center
    corners_world = make_box_corners(center_world, axes, extents)

    return OrientedBox3D(
        center=center_world.astype(np.float32),
        axes=axes.astype(np.float32),
        extents=extents.astype(np.float32),
        corners=corners_world.astype(np.float32),
    )


def make_box_corners(center, axes, extents):
    signs = np.array(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ],
        dtype=np.float32,
    )
    local_corners = signs * extents[None, :]
    return center[None, :] + local_corners @ axes.T


def infer_zero_joint_pose(camera):
    joint_pose = camera.joint_pose
    if joint_pose is None:
        return None
    if isinstance(joint_pose, torch.Tensor):
        return torch.zeros_like(joint_pose, device=joint_pose.device)
    joint_pose_np = np.asarray(joint_pose)
    return torch.zeros(joint_pose_np.shape, dtype=torch.float32, device=device)


def load_video_frames(video_path):
    frames = iio.imread(video_path, plugin="pyav")
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected video frames shaped (T, H, W, 3), got {frames.shape}.")
    return frames


def track_points_in_video(frames, text, n_points, seed, min_visibility_ratio):
    num_frames = frames.shape[0]
    query_indices = sorted({0, num_frames // 2, num_frames - 1})
    src_pts_by_idx = tracks_from_video_with_sam(
        frames,
        text=text,
        n_points=n_points,
        seed=seed,
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

    src_queries = torch.cat(query_chunks, dim=1)
    source_video = torch.from_numpy(frames).to(device).permute(0, 3, 1, 2).float().unsqueeze(0)

    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device).eval()
    for param in cotracker.parameters():
        param.requires_grad_(False)

    with torch.no_grad():
        tracks_2d, visibility = cotracker(
            source_video,
            queries=src_queries,
            backward_tracking=True,
        )

    tracks_2d = tracks_2d[0]
    visibility = visibility[0, :, :]
    tracks_2d, visibility = filter_cotracker_queries(
        tracks_2d,
        visibility,
        min_visibility_ratio=min_visibility_ratio,
    )

    del source_video
    del cotracker
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return tracks_2d.cpu().numpy(), visibility.cpu().numpy()


def load_depth_and_camera(depth_path, hdf5_path):
    depth_data = np.load(depth_path)
    depths_np = depth_data["depths"]
    with h5py.File(hdf5_path, "r") as handle:
        K_np = handle["camera/intrinsic"][:]
        T_cam_world = handle["transforms/camera"][:]
    return depths_np, K_np, T_cam_world


def lift_tracks_to_3d(tracks_2d, visibility, depths_np, K_np, T_cam_world, video_height, video_width):
    depth_frames, depth_height, depth_width = depths_np.shape
    T_use = min(tracks_2d.shape[0], depth_frames, T_cam_world.shape[0])
    num_points = tracks_2d.shape[1]

    fx = float(K_np[0, 0])
    fy = float(K_np[1, 1])
    cx = float(K_np[0, 2])
    cy = float(K_np[1, 2])
    scale_x = depth_width / float(video_width)
    scale_y = depth_height / float(video_height)

    tracks_3d = np.full((T_use, num_points, 3), np.nan, dtype=np.float32)
    vis_3d = visibility[:T_use] > 0.5

    for t in range(T_use):
        for n in range(num_points):
            if not vis_3d[t, n]:
                continue

            u_video = float(tracks_2d[t, n, 0])
            v_video = float(tracks_2d[t, n, 1])
            u_depth = u_video * scale_x
            v_depth = v_video * scale_y
            u_idx = int(round(u_depth))
            v_idx = int(round(v_depth))

            if not (0 <= u_idx < depth_width and 0 <= v_idx < depth_height):
                continue

            z = float(depths_np[t, v_idx, u_idx])
            if z <= 0.0 or not np.isfinite(z):
                continue

            x_cam = (u_depth - cx) * z / fx
            y_cam = (v_depth - cy) * z / fy
            p_cam = np.array([x_cam, y_cam, z, 1.0], dtype=np.float64)
            p_world = T_cam_world[t].astype(np.float64) @ p_cam
            tracks_3d[t, n] = p_world[:3].astype(np.float32)

    vis_3d &= np.isfinite(tracks_3d).all(axis=-1)
    return tracks_3d, vis_3d


def project_world_to_image(points_world, camera):
    points_world = np.asarray(points_world, dtype=np.float64)
    ones = np.ones((points_world.shape[0], 1), dtype=np.float64)
    points_world_h = np.concatenate([points_world, ones], axis=1)

    viewmat = camera.world_view_transform.transpose(0, 1).detach().cpu().numpy()
    points_cam = (viewmat @ points_world_h.T).T
    z = points_cam[:, 2]

    tanfovx = np.tan(float(camera.FoVx) * 0.5)
    tanfovy = np.tan(float(camera.FoVy) * 0.5)
    focal_x = float(camera.image_width) / (2.0 * tanfovx)
    focal_y = float(camera.image_height) / (2.0 * tanfovy)
    cx = float(camera.image_width) / 2.0
    cy = float(camera.image_height) / 2.0

    uv = np.full((points_world.shape[0], 2), np.nan, dtype=np.float32)
    valid = z > 1e-8
    if valid.any():
        uv[valid, 0] = focal_x * (points_cam[valid, 0] / z[valid]) + cx
        uv[valid, 1] = focal_y * (points_cam[valid, 1] / z[valid]) + cy
    return uv, valid


def project_world_to_video(points_world, t_idx, T_cam_world, K_np, scale_x, scale_y, video_height, video_width):
    points_world = np.asarray(points_world, dtype=np.float64)
    ones = np.ones((points_world.shape[0], 1), dtype=np.float64)
    points_world_h = np.concatenate([points_world, ones], axis=1)

    T_world_cam = np.linalg.inv(T_cam_world[t_idx].astype(np.float64))
    points_cam = (T_world_cam @ points_world_h.T).T
    z = points_cam[:, 2]

    fx = float(K_np[0, 0])
    fy = float(K_np[1, 1])
    cx = float(K_np[0, 2])
    cy = float(K_np[1, 2])

    uv = np.full((points_world.shape[0], 2), np.nan, dtype=np.float32)
    valid = z > 1e-8
    if valid.any():
        u_depth = fx * (points_cam[valid, 0] / z[valid]) + cx
        v_depth = fy * (points_cam[valid, 1] / z[valid]) + cy
        uv[valid, 0] = u_depth / scale_x
        uv[valid, 1] = v_depth / scale_y

    in_view = (
        np.isfinite(uv[:, 0])
        & np.isfinite(uv[:, 1])
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < video_width)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < video_height)
    )
    return uv, valid & in_view


def draw_projected_box(image_rgb, box, camera):
    corners_2d, valid = project_world_to_image(box.corners, camera)
    image_pil = Image.fromarray(image_rgb.astype(np.uint8))
    draw = ImageDraw.Draw(image_pil)

    edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7),
    ]
    for i, j in edges:
        if valid[i] and valid[j]:
            draw.line([tuple(corners_2d[i]), tuple(corners_2d[j])], fill=(0, 255, 0), width=3)

    center_2d, center_valid = project_world_to_image(box.center[None, :], camera)
    if bool(center_valid[0]):
        cx, cy = center_2d[0]
        draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=(255, 255, 255), outline=(0, 0, 0))

    return np.asarray(image_pil)


def draw_projected_box_on_frame(image_rgb, box, projected_points, valid_mask, color=(0, 255, 0)):
    image_pil = Image.fromarray(image_rgb.astype(np.uint8))
    draw = ImageDraw.Draw(image_pil)
    edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7),
    ]
    for i, j in edges:
        if valid_mask[i] and valid_mask[j]:
            draw.line([tuple(projected_points[i]), tuple(projected_points[j])], fill=color, width=3)
    return np.asarray(image_pil)


def estimate_robot_bbox(model_path, camera_index, stage, stride, quantile, visible_only):
    gaussians, background, cameras, _ = initialize_gaussians(model_path=model_path, from_ckpt=True)
    if len(cameras) == 0:
        raise ValueError(f"No sample cameras were found for model path {model_path}.")
    if camera_index < 0 or camera_index >= len(cameras):
        raise IndexError(f"camera_index {camera_index} is out of bounds for {len(cameras)} sample cameras.")

    camera = cameras[camera_index]
    zero_joint_pose = infer_zero_joint_pose(camera)
    if zero_joint_pose is not None:
        camera.joint_pose = zero_joint_pose

    with torch.no_grad():
        rendered_3d = render_3d(camera, gaussians, stage=stage)
        means3d = rendered_3d["viewspace_points"][0]
        visibility = rendered_3d["visibility_filter"]

        if stride > 1:
            means3d = means3d[::stride]
            visibility = visibility[::stride]

        if visible_only:
            means3d = means3d[visibility > 0.5]

        means3d_np = means3d.detach().cpu().numpy()
        bbox = fit_oriented_box(means3d_np, quantile=quantile)

        rendered = render(camera, gaussians, background, stage=stage)["render"]
        image_rgb = torch.clamp(rendered, 0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy()
        image_rgb = (255.0 * image_rgb).astype(np.uint8)
        overlay_rgb = draw_projected_box(image_rgb, bbox, camera)

    return means3d_np, bbox, image_rgb, overlay_rgb


def estimate_source_video_bbox(video_path, depth_path, hdf5_path, text, n_points, seed, min_visibility_ratio, quantile):
    frames = load_video_frames(video_path)
    tracks_2d, visibility = track_points_in_video(
        frames=frames,
        text=text,
        n_points=n_points,
        seed=seed,
        min_visibility_ratio=min_visibility_ratio,
    )
    depths_np, K_np, T_cam_world = load_depth_and_camera(depth_path, hdf5_path)
    tracks_3d, vis_3d = lift_tracks_to_3d(
        tracks_2d=tracks_2d,
        visibility=visibility,
        depths_np=depths_np,
        K_np=K_np,
        T_cam_world=T_cam_world,
        video_height=frames.shape[1],
        video_width=frames.shape[2],
    )

    valid_points = tracks_3d[vis_3d]
    if valid_points.shape[0] < 3:
        raise ValueError("Not enough valid 3D lifted source points to estimate a bounding box.")

    bbox = fit_oriented_box(valid_points, quantile=quantile)
    scale_x = depths_np.shape[2] / float(frames.shape[2])
    scale_y = depths_np.shape[1] / float(frames.shape[1])

    overlay_frames = []
    T_overlay = min(frames.shape[0], T_cam_world.shape[0])
    for t in range(T_overlay):
        projected, valid_mask = project_world_to_video(
            bbox.corners,
            t,
            T_cam_world,
            K_np,
            scale_x,
            scale_y,
            frames.shape[1],
            frames.shape[2],
        )
        overlay_frames.append(draw_projected_box_on_frame(frames[t], bbox, projected, valid_mask))

    overlay_frames = np.stack(overlay_frames, axis=0)
    return valid_points.astype(np.float32), bbox, overlay_frames


def save_bbox(path, bbox, points_xyz, key_prefix="bbox"):
    np.savez(
        path,
        gaussian_means_xyz=points_xyz,
        **{
            f"{key_prefix}_center": bbox.center,
            f"{key_prefix}_axes": bbox.axes,
            f"{key_prefix}_extents": bbox.extents,
            f"{key_prefix}_corners": bbox.corners,
        },
    )


def main():
    args = parse_args()
    run_source_box = args.run_source_box
    run_render_box = args.run_render_box
    if not run_source_box and not run_render_box:
        run_source_box = True
        run_render_box = True

    if run_source_box:
        source_points_xyz, source_bbox, source_overlay_frames = estimate_source_video_bbox(
            video_path=args.video_path,
            depth_path=args.depth_path,
            hdf5_path=args.hdf5_path,
            text=args.text,
            n_points=args.n_points,
            seed=args.seed,
            min_visibility_ratio=args.min_visibility_ratio,
            quantile=args.quantile,
        )
        save_bbox(args.source_box_npz_path, source_bbox, source_points_xyz, key_prefix="source_bbox")
        try:
            iio.imwrite(args.source_overlay_path, source_overlay_frames, plugin="pyav", fps=args.fps)
        except Exception:
            iio.imwrite(args.source_overlay_path, source_overlay_frames, fps=args.fps)
        print(f"Estimated source-video 3D bbox from {source_points_xyz.shape[0]} lifted points")
        print(f"Source center  : {source_bbox.center}")
        print(f"Source extents : {source_bbox.extents}")
        print(f"Saved source-video bbox data to {args.source_box_npz_path}")
        print(f"Saved source-video overlay to {args.source_overlay_path}")

    if run_render_box:
        means3d_np, render_bbox, _image_rgb, overlay_rgb = estimate_robot_bbox(
            model_path=args.model_path,
            camera_index=args.camera_index,
            stage=args.stage,
            stride=args.stride,
            quantile=args.quantile,
            visible_only=args.visible_only,
        )
        save_bbox(args.render_box_npz_path, render_bbox, means3d_np, key_prefix="render_bbox")
        iio.imwrite(args.render_overlay_path, overlay_rgb)
        print(f"Estimated renderer 3D bbox from {means3d_np.shape[0]} Gaussian means")
        print(f"Render center  : {render_bbox.center}")
        print(f"Render extents : {render_bbox.extents}")
        print(f"Saved renderer bbox data to {args.render_box_npz_path}")
        print(f"Saved renderer overlay to {args.render_overlay_path}")


if __name__ == "__main__":
    main()
