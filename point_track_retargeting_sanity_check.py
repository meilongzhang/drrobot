import json
import math
from argparse import ArgumentParser

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene import RobotScene
from scene.cameras import Camera_Pose
from scene.gaussian_model import GaussianModel
from scipy.spatial.transform import Rotation
import h5py


device = "cuda" if torch.cuda.is_available() else "cpu"


def initialize_gaussians(model_path: str):
    parser = ArgumentParser(description="Sanity-check renderer camera pose")
    model = ModelParams(parser, sentinel=True)
    PipelineParams(parser)
    opt = OptimizationParams(parser)
    args = get_combined_args(parser)
    args.model_path = model_path

    gaussians = GaussianModel(model.sh_degree, opt)
    scene = RobotScene(args, gaussians, opt_params=opt, from_ckpt=True, load_iteration=-1)
    gaussians.model_path = scene.model_path
    return gaussians, gaussians.chain

def load_intrinsics(json_path: str, camera_name: str = "left"):
    with open(json_path, "r") as f:
        data = json.load(f)

    if camera_name not in data:
        raise ValueError(f"Camera '{camera_name}' not found in {json_path}.")

    intr = data[camera_name]
    fx = float(intr["fx"])
    fy = float(intr["fy"])
    cx = float(intr["cx"])
    cy = float(intr["cy"])

    width = int(round(cx * 2.0))
    height = int(round(cy * 2.0))
    fov_scale = 1.0
    fovx = 2.0 * math.atan(width / (2.0 * fx))
    fovy = 2.0 * math.atan(height / (2.0 * fy))
    fovx = min(fovx * fov_scale, math.radians(175.0))
    fovy = min(fovy * fov_scale, math.radians(175.0))

    return fovx, fovy, width, height


def load_intrinsics_hdf5(hdf5_path: str):
    with h5py.File(hdf5_path, "r") as f:
        K = f["camera/intrinsic"][()]  # (3, 3)

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    # infer image size (same assumption as your original code)
    width = int(round(cx * 2.0))
    height = int(round(cy * 2.0))

    fov_scale = 1.0
    fovx = 2.0 * math.atan(width / (2.0 * fx))
    fovy = 2.0 * math.atan(height / (2.0 * fy))
    fovx = min(fovx * fov_scale, math.radians(175.0))
    fovy = min(fovy * fov_scale, math.radians(175.0))

    return fovx, fovy, width, height


def load_extrinsics(json_path: str):
    with open(json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        if len(data) == 0:
            raise ValueError(f"No camera entries found in {json_path}.")
        data = data[0]

    # I know this says c2w but it is actually cam to robot. in COLMAP
    R_c2w = np.asarray(data["camera_base_ori"], dtype=np.float32)
    t_c2w = np.asarray(data["camera_base_pos"], dtype=np.float32)

    # convert from c2w to w2c 
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ t_c2w

    # construct the 4x4 quaternion-based transformation matrix for MuJoCo
    T_w2c = np.eye(4, dtype=np.float32)
    T_w2c[:3, :3] = R_w2c
    T_w2c[:3, 3] = t_w2c
    return torch.tensor(T_w2c, dtype=torch.float32, device=device)

def load_extrinsics_hdf5(hdf5_path: str):

    # cam to world extrinsics
    with h5py.File(hdf5_path, "r") as f:
        # indexing to get the first extrinsics matrix
        T_c2w = f["transforms/camera"][()][0]  # (4, 4)
    
    # world to robot
    BASE_T_1 = np.array([[0.0, -1.0,  0.0,  0.0],
                    [ 0.5,  0.0,  0.866,  0.2],
                    [-0.866,  0.0,  0.5,  1.50],
                    [ 0.0,  0.0,  0.0,  1.0]])  
    
    T_c2r = BASE_T_1 @ T_c2w

    T_r2c = np.linalg.inv(T_c2r)

    return torch.tensor(T_r2c, dtype=torch.float32, device=device)


def infer_dof(chain) -> int:
    for attr in ["dof", "n_joints"]:
        if hasattr(chain, attr):
            return int(getattr(chain, attr))
    return 24


def save_scene_overview(gaussians, w2c: torch.Tensor, output_path: str):
    with torch.no_grad():
        points = gaussians.get_xyz.detach().cpu().numpy()

    step = max(1, points.shape[0] // 3000)
    points_plot = points[::step]
    center = points.mean(axis=0)

    w2c_np = w2c.detach().cpu().numpy()
    c2w = np.linalg.inv(w2c_np)
    camera_center = c2w[:3, 3]
    camera_rotation = c2w[:3, :3]
    camera_x_axis = camera_rotation @ np.array([1.0, 0.0, 0.0], dtype=np.float32)
    camera_y_axis = camera_rotation @ np.array([0.0, 1.0, 0.0], dtype=np.float32)
    camera_z_axis = camera_rotation @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
    camera_x_axis = camera_x_axis / max(np.linalg.norm(camera_x_axis), 1e-8)
    camera_y_axis = camera_y_axis / max(np.linalg.norm(camera_y_axis), 1e-8)
    camera_z_axis = camera_z_axis / max(np.linalg.norm(camera_z_axis), 1e-8)

    points_min = points_plot.min(axis=0)
    points_max = points_plot.max(axis=0)
    scene_extent = np.linalg.norm(points_max - points_min)
    arrow_len = max(0.1, 0.25 * scene_extent)
    axis_len = max(0.1, 0.2 * scene_extent)
    cube_len = 0.45 * arrow_len

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        points_plot[:, 0],
        points_plot[:, 1],
        points_plot[:, 2],
        s=1,
        alpha=0.15,
        c="tab:blue",
    )
    ax.scatter(center[0], center[1], center[2], s=80, c="tab:green", label="robot center")
    ax.scatter(
        camera_center[0],
        camera_center[1],
        camera_center[2],
        s=80,
        c="tab:red",
        label="camera",
    )

    cube_offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [cube_len, 0.0, 0.0],
            [0.0, cube_len, 0.0],
            [0.0, 0.0, cube_len],
            [cube_len, cube_len, 0.0],
            [cube_len, 0.0, cube_len],
            [0.0, cube_len, cube_len],
            [cube_len, cube_len, cube_len],
        ],
        dtype=np.float32,
    )
    cube_corners = camera_center[None, :] + cube_offsets @ camera_rotation.T
    cube_edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (1, 5),
        (2, 4), (2, 6),
        (3, 5), (3, 6),
        (4, 7), (5, 7), (6, 7),
    ]
    cube_faces = [
        [cube_corners[idx] for idx in [0, 1, 4, 2]],
        [cube_corners[idx] for idx in [0, 1, 5, 3]],
        [cube_corners[idx] for idx in [0, 2, 6, 3]],
        [cube_corners[idx] for idx in [7, 5, 1, 4]],
        [cube_corners[idx] for idx in [7, 6, 2, 4]],
        [cube_corners[idx] for idx in [7, 6, 3, 5]],
    ]
    cube_face_collection = Poly3DCollection(
        cube_faces,
        facecolors="0.82",
        edgecolors="none",
        alpha=1.0,
    )
    ax.add_collection3d(cube_face_collection)
    emphasized_edges = {
        (0, 1): "tab:red",
        (0, 2): "tab:cyan",
        (0, 3): "tab:pink",
    }
    for start_idx, end_idx in cube_edges:
        start = cube_corners[start_idx]
        end = cube_corners[end_idx]
        edge_key = (start_idx, end_idx)
        color = emphasized_edges.get(edge_key, "0.55")
        linewidth = 2.8 if edge_key in emphasized_edges else 1.1
        alpha = 1.0 if edge_key in emphasized_edges else 0.55
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
        )

    ax.quiver(
        camera_center[0],
        camera_center[1],
        camera_center[2],
        camera_x_axis[0],
        camera_x_axis[1],
        camera_x_axis[2],
        length=arrow_len,
        color="tab:red",
        linewidth=2.0,
    )
    ax.quiver(
        camera_center[0],
        camera_center[1],
        camera_center[2],
        camera_y_axis[0],
        camera_y_axis[1],
        camera_y_axis[2],
        length=arrow_len,
        color="tab:cyan",
        linewidth=2.0,
    )
    ax.quiver(
        camera_center[0],
        camera_center[1],
        camera_center[2],
        camera_z_axis[0],
        camera_z_axis[1],
        camera_z_axis[2],
        length=arrow_len,
        color="tab:pink",
        linewidth=2.0,
    )
    camera_x_tip = camera_center + arrow_len * camera_x_axis
    camera_y_tip = camera_center + arrow_len * camera_y_axis
    camera_z_tip = camera_center + arrow_len * camera_z_axis
    ax.text(camera_x_tip[0], camera_x_tip[1], camera_x_tip[2], "cam +X", color="tab:red")
    ax.text(camera_y_tip[0], camera_y_tip[1], camera_y_tip[2], "cam +Y", color="tab:cyan")
    ax.text(camera_z_tip[0], camera_z_tip[1], camera_z_tip[2], "cam +Z", color="tab:pink")
    ax.quiver(0.0, 0.0, 0.0, axis_len, 0.0, 0.0, color="tab:orange", linewidth=2.0)
    ax.quiver(0.0, 0.0, 0.0, 0.0, axis_len, 0.0, color="tab:purple", linewidth=2.0)
    ax.quiver(0.0, 0.0, 0.0, 0.0, 0.0, axis_len, color="tab:brown", linewidth=2.0)
    ax.text(axis_len, 0.0, 0.0, "+X", color="tab:orange")
    ax.text(0.0, axis_len, 0.0, "+Y", color="tab:purple")
    ax.text(0.0, 0.0, axis_len, "+Z", color="tab:brown")
    ax.text(
        center[0],
        center[1],
        center[2],
        f"center=({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})",
        color="tab:green",
    )

    all_points = np.vstack(
        [
            points_plot,
            np.zeros((1, 3), dtype=np.float32),
            np.array([[axis_len, 0.0, 0.0], [0.0, axis_len, 0.0], [0.0, 0.0, axis_len]], dtype=np.float32),
            camera_center[None, :],
            camera_x_tip[None, :],
            camera_y_tip[None, :],
            camera_z_tip[None, :],
            cube_corners,
        ]
    )
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    mid = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins) + 1e-3

    ax.set_xlim(mid[0] - radius, mid[0] + radius)
    ax.set_ylim(mid[1] - radius, mid[1] + radius)
    ax.set_zlim(mid[2] - radius, mid[2] + radius)
    ax.set_box_aspect([1.0, 1.0, 1.0])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Robot Gaussians and Camera Pose")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_scene_overview_camera_frame(gaussians, w2c: torch.Tensor, output_path: str):
    with torch.no_grad():
        points = gaussians.get_xyz.detach().cpu().numpy()

    step = max(1, points.shape[0] // 3000)
    points_plot = points[::step]

    w2c_np = w2c.detach().cpu().numpy()
    rotation_w2c = w2c_np[:3, :3]
    translation_w2c = w2c_np[:3, 3]

    points_cam = points_plot @ rotation_w2c.T + translation_w2c[None, :]
    center_cam = points_cam.mean(axis=0)
    robot_origin_cam = translation_w2c

    points_min = points_cam.min(axis=0)
    points_max = points_cam.max(axis=0)
    scene_extent = np.linalg.norm(points_max - points_min)
    arrow_len = max(0.1, 0.25 * scene_extent)
    axis_len = max(0.1, 0.2 * scene_extent)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        points_cam[:, 0],
        points_cam[:, 1],
        points_cam[:, 2],
        s=1,
        alpha=0.15,
        c="tab:blue",
    )
    ax.scatter(
        robot_origin_cam[0],
        robot_origin_cam[1],
        robot_origin_cam[2],
        s=80,
        c="tab:green",
        label="robot origin in camera frame",
    )
    ax.scatter(0.0, 0.0, 0.0, s=80, c="tab:red", label="camera origin")

    ax.quiver(0.0, 0.0, 0.0, axis_len, 0.0, 0.0, color="tab:red", linewidth=2.0)
    ax.quiver(0.0, 0.0, 0.0, 0.0, axis_len, 0.0, color="tab:cyan", linewidth=2.0)
    ax.quiver(0.0, 0.0, 0.0, 0.0, 0.0, axis_len, color="tab:pink", linewidth=2.0)
    ax.text(axis_len, 0.0, 0.0, "cam +X", color="tab:red")
    ax.text(0.0, axis_len, 0.0, "cam +Y", color="tab:cyan")
    ax.text(0.0, 0.0, axis_len, "cam +Z", color="tab:pink")

    robot_axis_x = rotation_w2c[:, 0]
    robot_axis_y = rotation_w2c[:, 1]
    robot_axis_z = rotation_w2c[:, 2]
    ax.quiver(
        robot_origin_cam[0],
        robot_origin_cam[1],
        robot_origin_cam[2],
        robot_axis_x[0],
        robot_axis_x[1],
        robot_axis_x[2],
        length=arrow_len,
        color="tab:orange",
        linewidth=2.0,
    )
    ax.quiver(
        robot_origin_cam[0],
        robot_origin_cam[1],
        robot_origin_cam[2],
        robot_axis_y[0],
        robot_axis_y[1],
        robot_axis_y[2],
        length=arrow_len,
        color="tab:purple",
        linewidth=2.0,
    )
    ax.quiver(
        robot_origin_cam[0],
        robot_origin_cam[1],
        robot_origin_cam[2],
        robot_axis_z[0],
        robot_axis_z[1],
        robot_axis_z[2],
        length=arrow_len,
        color="tab:brown",
        linewidth=2.0,
    )
    robot_x_tip = robot_origin_cam + arrow_len * robot_axis_x
    robot_y_tip = robot_origin_cam + arrow_len * robot_axis_y
    robot_z_tip = robot_origin_cam + arrow_len * robot_axis_z
    ax.text(robot_x_tip[0], robot_x_tip[1], robot_x_tip[2], "robot +X", color="tab:orange")
    ax.text(robot_y_tip[0], robot_y_tip[1], robot_y_tip[2], "robot +Y", color="tab:purple")
    ax.text(robot_z_tip[0], robot_z_tip[1], robot_z_tip[2], "robot +Z", color="tab:brown")
    ax.text(
        center_cam[0],
        center_cam[1],
        center_cam[2],
        f"center=({center_cam[0]:.3f}, {center_cam[1]:.3f}, {center_cam[2]:.3f})",
        color="tab:green",
    )

    all_points = np.vstack(
        [
            points_cam,
            np.zeros((1, 3), dtype=np.float32),
            np.array([[axis_len, 0.0, 0.0], [0.0, axis_len, 0.0], [0.0, 0.0, axis_len]], dtype=np.float32),
            robot_origin_cam[None, :],
            robot_x_tip[None, :],
            robot_y_tip[None, :],
            robot_z_tip[None, :],
        ]
    )
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    mid = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins) + 1e-3

    ax.set_xlim(mid[0] - radius, mid[0] + radius)
    ax.set_ylim(mid[1] - radius, mid[1] + radius)
    ax.set_zlim(mid[2] - radius, mid[2] + radius)
    ax.set_box_aspect([1.0, 1.0, 1.0])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Robot Gaussians in Camera Frame")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    # extrinsics_path = "camera_extrinsics_ego.json"
    # intrinsics_path = "camera_intrinsics_epic.json"
    extrinsics_path = "999.hdf5"
    intrinsics_path = "999.hdf5"
    model_path = "results/unitree_g1"
    output_path = "sanity_check_render.png"
    overview_path = "sanity_check_scene_overview.png"
    overview_camera_frame_path = "sanity_check_scene_overview_camera_frame.png"

    gaussians, chain = initialize_gaussians(model_path)
    dof = infer_dof(chain)

    # w2c = load_extrinsics(extrinsics_path)
    w2c = load_extrinsics_hdf5(extrinsics_path)
    # fovx, fovy, width, height = load_intrinsics(intrinsics_path, camera_name="left")
    fovx, fovy, width, height = load_intrinsics_hdf5(intrinsics_path)
    joint_pose = torch.zeros(dof, dtype=torch.float32, device=device)

    camera = Camera_Pose(
        start_pose_w2c=w2c,
        FoVx=fovx,
        FoVy=fovy,
        image_width=width,
        image_height=height,
        joint_pose=joint_pose,
        zero_init=True,
    ).to(device)

    with torch.no_grad():
        camera.w.zero_()
        camera.v.zero_()

    background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)

    with torch.no_grad():
        rendered = render(camera, gaussians, background)["render"]

    image = (rendered.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(image).save(output_path)
    save_scene_overview(gaussians, w2c, overview_path)
    save_scene_overview_camera_frame(gaussians, w2c, overview_camera_frame_path)

    print(f"Saved render to {output_path}")
    print(f"Saved 3D overview to {overview_path}")
    print(f"Saved camera-frame 3D overview to {overview_camera_frame_path}")
    print(f"Image size: {width}x{height}")
    print(f"DoF: {dof}")


if __name__ == "__main__":
    main()
