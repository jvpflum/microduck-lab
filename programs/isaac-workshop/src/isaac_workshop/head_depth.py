"""Head-attached Isaac RTX RGB/depth instrumentation for pinned MicroDuck.

Imports Isaac Lab only while configuring or collecting a native sensor. A new
image group is opt-in; the existing 61D proprioceptive group is never modified.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
import time
import zlib

import numpy as np


COMMIT = "b03345bbf2c9a54eb03f9bd823ae19c0ca3a014e"
SENSOR_NAME = "head_depth"
HEAD_PATH = "Geometry/trunk_base/neck/neck_pitch_1/yaw_roll_motion/jaw_soft"


@dataclass(frozen=True)
class CameraSpec:
    width: int = 160
    height: int = 120
    horizontal_fov_deg: float = 87.0
    focal_length_mm: float = 24.0
    near_m: float = 0.03
    far_m: float = 5.0
    update_period_s: float = 0.02
    # Authored lens translation, moved 2 mm along head-local -Z to avoid its surface.
    offset_pos_m: tuple = (0.0155, -0.0000913778, -0.05748)
    # Neutral head local -Z is robot forward; +X is up. Add 20 degrees downward pitch.
    pitch_about_head_y_deg: float = 110.0
    warmup_samples: int = 5
    minimum_valid_fraction: float = 0.01
    floor_tolerance_m: float = 0.02
    maximum_behind_floor_fraction: float = 0.05
    minimum_floor_hits: int = 10
    snapshot_interval: int = 25

    def validate(self):
        if self.width < 2 or self.height < 2 or not 1 < self.horizontal_fov_deg < 179:
            raise ValueError("Invalid camera resolution or field of view")
        if not 0 < self.near_m < self.far_m or self.update_period_s <= 0:
            raise ValueError("Invalid clipping interval or update period")
        if self.warmup_samples < 0 or self.snapshot_interval < 1:
            raise ValueError("Invalid recording schedule")
        if not all(math.isfinite(x) for x in (*self.offset_pos_m, self.pitch_about_head_y_deg)):
            raise ValueError("Nonfinite camera extrinsics")

    @property
    def quaternion_xyzw(self):
        half = math.radians(self.pitch_about_head_y_deg) / 2
        return (0.0, math.sin(half), 0.0, math.cos(half))

    @property
    def focal_pixels(self):
        return self.width / (2 * math.tan(math.radians(self.horizontal_fov_deg) / 2))

    def manifest(self):
        self.validate()
        return dict(asdict(self), source_commit=COMMIT, sensor_name=SENSOR_NAME,
                    relative_head_prim=HEAD_PATH, quaternion_xyzw=self.quaternion_xyzw,
                    offset_convention="world (+X forward, +Z up)",
                    requested_intrinsics=[[self.focal_pixels, 0, self.width / 2],
                                          [0, self.focal_pixels, self.height / 2], [0, 0, 1]],
                    depth_semantics="distance_to_image_plane: optical-axis metres, not radial range",
                    calibration_status="engineering assumption, not article or hardware calibration",
                    extrinsics_provenance="Pinned robot USD lens origin plus 2 mm forward; engineered 20 degree downward optical axis",
                    no_hit_semantics="Positive infinity retained in raw depth; validity mask excludes it",
                    policy_receives_depth=False)


def configure_head_camera(env_cfg, spec=None):
    """Add the actual sensor before launch; caller must set cli.enable_cameras=True."""
    spec = spec or CameraSpec()
    spec.validate()
    from isaaclab.sensors import CameraCfg
    from isaaclab.sim import PinholeCameraCfg
    from isaaclab_physx.renderers import IsaacRtxRendererCfg
    from isaaclab_newton.physics import NewtonCfg, MJWarpSolverCfg

    if not isinstance(env_cfg.sim.physics, NewtonCfg) or not isinstance(env_cfg.sim.physics.solver_cfg, MJWarpSolverCfg):
        raise ValueError("Head-depth probe requires the existing Newton MJWarp task")
    if hasattr(env_cfg.scene, SENSOR_NAME):
        raise ValueError("Refusing to replace an existing head-depth sensor")
    if not str(env_cfg.scene.robot.spawn.usd_path).endswith("/microduck_walk_backlash.usd"):
        raise ValueError("Head attachment is bound to the inspected walking-backlash USD")
    # One render opportunity per control interval; do not change physics dt or decimation.
    if not math.isclose(env_cfg.sim.dt * env_cfg.decimation, spec.update_period_s, abs_tol=1e-8):
        raise ValueError("Probe sampling interval must match the unchanged control interval")
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + HEAD_PATH + "/HeadDepth",
        height=spec.height, width=spec.width, update_period=spec.update_period_s,
        data_types=["rgb", "distance_to_image_plane"], update_latest_camera_pose=True,
        renderer_cfg=IsaacRtxRendererCfg(depth_clipping_behavior="none"),
        spawn=PinholeCameraCfg(focal_length=spec.focal_length_mm,
                              horizontal_aperture=spec.focal_length_mm * spec.width / spec.focal_pixels,
                              clipping_range=(spec.near_m, spec.far_m)),
        offset=CameraCfg.OffsetCfg(pos=spec.offset_pos_m, rot=spec.quaternion_xyzw, convention="world"),
    )
    setattr(env_cfg.scene, SENSOR_NAME, camera)
    return spec.manifest()


def configure_depth_observation_group(env_cfg, spec=None):
    """Opt in to a separate [N,2,H,W] image group for the new CNN policy.

    Invoke after configure_head_camera and before constructing the environment.
    Pinned ObservationManager returns a single term directly, preserving its
    spatial axes. No history, noise, scaling, or flattening is applied here.
    """
    from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
    from isaaclab.utils import configclass
    spec = spec or CameraSpec()
    spec.validate()
    if not hasattr(env_cfg.scene, SENSOR_NAME):
        raise ValueError("Configure the real head camera before adding its observation group")
    if hasattr(env_cfg.observations, SENSOR_NAME):
        raise ValueError("Refusing to replace an existing head-depth observation group")

    @configclass
    class HeadDepthObservationCfg(ObservationGroupCfg):
        normalized_depth_mask = ObservationTermCfg(func=head_depth_observation, params={"spec": asdict(spec)})
        concatenate_terms = True
        concatenate_dim = 0  # Channel axis, excluding the environment batch axis.
        enable_corruption = False
        history_length = 0
        flatten_history_dim = False

    setattr(env_cfg.observations, SENSOR_NAME, HeadDepthObservationCfg())
    return {"group": SENSOR_NAME, "shape_without_batch": [2, spec.height, spec.width],
            "channels": ["linear_optical_depth_near_to_far", "finite_in_range_validity"],
            "invalid_depth_value": 1.0, "invalid_mask_value": 0.0,
            "stock_proprioceptive_groups_unchanged": True}


def head_depth_observation(env, spec=None):
    """Real camera input [N,2,H,W]: normalized optical depth plus valid-pixel mask.

    This callable is opt-in for a NEW perception observation group. It neither
    mutates the sensor's metric buffer nor changes the stock proprioceptive group.
    """
    import torch
    if isinstance(spec, dict):
        spec = CameraSpec(**spec)
    spec = spec or CameraSpec()
    depth = env.scene.sensors[SENSOR_NAME].data.output["distance_to_image_plane"].torch
    if tuple(depth.shape[1:]) != (spec.height, spec.width, 1):
        raise RuntimeError(f"Unexpected camera depth shape: {tuple(depth.shape)}")
    valid = torch.isfinite(depth) & (depth >= spec.near_m) & (depth < spec.far_m)
    safe = torch.where(valid, depth, torch.full_like(depth, spec.far_m))
    normalized = (safe-spec.near_m)/(spec.far_m-spec.near_m)
    return torch.cat((normalized, valid.to(depth.dtype)), dim=-1).permute(0, 3, 1, 2).contiguous()


def rotation_xyzw(quaternion):
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or not np.isclose(np.linalg.norm(q), 1.0, atol=1e-3):
        raise ValueError("Invalid XYZW unit quaternion")
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def depth_geometry(depth, intrinsics, position, quaternion_ros, spec, floor_z=0.0):
    """Audit metric optical depth against the known floor, allowing nearer obstacles."""
    depth = np.asarray(depth)
    k = np.asarray(intrinsics, dtype=np.float64)
    position = np.asarray(position, dtype=np.float64)
    if depth.shape != (spec.height, spec.width) or k.shape != (3, 3) or position.shape != (3,):
        raise ValueError("Unexpected depth, intrinsics or position shape")
    if not np.isfinite(k).all() or not np.isfinite(position).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError("Invalid pose or intrinsics")
    rotation = rotation_xyzw(quaternion_ros)
    valid = np.isfinite(depth) & (depth >= spec.near_m) & (depth < spec.far_m)
    rows, cols = np.indices(depth.shape)
    rays_camera = np.stack(((cols-k[0, 2])/k[0, 0], (rows-k[1, 2])/k[1, 1], np.ones_like(cols)), axis=-1)
    rays_world = rays_camera @ rotation.T
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = (float(floor_z)-position[2]) / rays_world[..., 2]
    eligible = (rays_world[..., 2] < -1e-6) & (expected >= spec.near_m) & (expected < spec.far_m) & valid
    # Optical-axis depth scales a ray whose camera z component is 1; do not normalize it.
    with np.errstate(invalid="ignore"):
        world_z = position[2] + depth * rays_world[..., 2]
    residual = np.abs(world_z - floor_z)
    floor_hits = eligible & (residual <= spec.floor_tolerance_m)
    behind = eligible & (world_z < floor_z - spec.floor_tolerance_m)
    count = int(eligible.sum())
    result = {"valid_pixels": int(valid.sum()), "valid_fraction": float(valid.mean()),
              "raw_depth_finite_fraction": float(np.isfinite(depth).mean()),
              "nan_pixels": int(np.isnan(depth).sum()), "positive_infinity_pixels": int(np.isposinf(depth).sum()),
              "negative_infinity_pixels": int(np.isneginf(depth).sum()),
              "negative_finite_pixels": int((np.isfinite(depth) & (depth < 0)).sum()),
              "valid_min_m": float(depth[valid].min()) if valid.any() else None,
              "valid_max_m": float(depth[valid].max()) if valid.any() else None,
              "floor_test_pixels": count, "floor_consistent_pixels": int(floor_hits.sum()),
              "behind_floor_pixels": int(behind.sum()),
              "behind_floor_fraction": float(behind.sum()/count) if count else None,
              "floor_residual_median_m": float(np.median(residual[floor_hits])) if floor_hits.any() else None}
    return result, valid


def clock_progress(previous, episode_id, frame, timestamp, reset=False):
    """Sensor clocks reset per episode; a repeated capture is never a fresh frame."""
    if not math.isfinite(timestamp) or timestamp < 0 or frame < 1:
        return False
    if previous is None:
        return True
    episode, last_frame, last_time = previous
    if reset and episode_id <= episode:
        return False
    return episode_id > episode or (episode_id == episode and frame > last_frame and timestamp > last_time + 1e-7)


def _png(path, pixels):
    pixels = np.asarray(pixels, dtype=np.uint8)
    if pixels.ndim == 2:
        pixels = np.repeat(pixels[..., None], 3, axis=2)
    height, width, channels = pixels.shape
    if channels != 3:
        raise ValueError("PNG requires three RGB channels")
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    rows = b"".join(b"\x00" + np.ascontiguousarray(row).tobytes() for row in pixels)
    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
                           + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class HeadDepthRecorder:
    """Collect one live native camera. Call capture after step, finalize before Kit exits."""
    def __init__(self, env, output_dir, spec=None, floor_z=0.0):
        self.spec = spec or CameraSpec()
        self.spec.validate()
        self.env = env.unwrapped
        if self.env.num_envs != 1:
            raise ValueError("Component probe requires exactly one environment")
        self.camera = self.env.scene.sensors[SENSOR_NAME]
        self.robot = self.env.scene["robot"]
        matches = [i for i, name in enumerate(self.robot.body_names) if name == "jaw_soft"]
        if len(matches) != 1:
            raise RuntimeError("Expected one physical jaw_soft head body")
        self.head_index = matches[0]
        renderer = type(self.camera._renderer)
        if renderer.__name__ != "IsaacRtxRenderer" or not renderer.__module__.startswith("isaaclab_physx."):
            raise RuntimeError("Sensor is not using the native Isaac Sim RTX renderer")
        if not math.isfinite(floor_z):
            raise ValueError("Floor height must be finite")
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=False)
        (self.output / "frames").mkdir()
        self.trace = (self.output / "sensor-trace.jsonl").open("x", encoding="utf-8")
        self.floor_z = float(floor_z)
        self.rows = []
        self.previous = None
        self.final = None
        self.manifest = dict(self.spec.manifest(), actual_renderer=f"{renderer.__module__}.{renderer.__name__}",
                             actual_camera_prim=self.camera.cfg.prim_path, expected_floor_z_m=self.floor_z,
                             module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                             timestamp_source="Pinned SensorBase._timestamp_last_update; episode-relative seconds",
                             sampling_boundary="post_step_after_autoreset; reset rows start a new episode")
        _json(self.output / "camera-manifest.json", self.manifest)

    def capture(self, step, episode_id=0, reset=False):
        if self.final is not None or self.trace.closed:
            raise RuntimeError("Recorder has already been finalized or closed")
        import warp as wp
        data = self.camera.data  # Lazy capture; read matching pixels and capture-pose buffers together.
        rgb = data.output["rgb"].torch[0].detach().cpu().numpy().copy()
        depth = data.output["distance_to_image_plane"].torch[0, ..., 0].detach().cpu().numpy().copy()
        k = data.intrinsic_matrices.torch[0].detach().cpu().numpy().copy()
        pos = data.pos_w.torch[0].detach().cpu().numpy().copy()
        quat_ros = data.quat_w_ros.torch[0].detach().cpu().numpy().copy()
        quat_world = data.quat_w_world.torch[0].detach().cpu().numpy().copy()
        frame = int(self.camera.frame.torch[0])
        timestamp = float(wp.to_torch(self.camera._timestamp_last_update)[0])
        if rgb.shape != (self.spec.height, self.spec.width, 3) or rgb.dtype != np.uint8:
            raise RuntimeError(f"Unexpected RGB contract: {rgb.shape}, {rgb.dtype}")
        geometry, valid = depth_geometry(depth, k, pos, quat_ros, self.spec, self.floor_z)
        head_pos = self.robot.data.body_link_pos_w.torch[0, self.head_index].detach().cpu().numpy()
        head_quat = self.robot.data.body_link_quat_w.torch[0, self.head_index].detach().cpu().numpy()
        head_rotation = rotation_xyzw(head_quat)
        expected_pos = head_pos + head_rotation @ np.asarray(self.spec.offset_pos_m)
        expected_rotation = head_rotation @ rotation_xyzw(self.spec.quaternion_xyzw)
        actual_rotation = rotation_xyzw(quat_world)
        attachment_error = float(np.linalg.norm(pos - expected_pos))
        angular_error = float(math.acos(np.clip((np.trace(expected_rotation.T @ actual_rotation)-1)/2, -1, 1)))
        intrinsic_error = float(np.max(np.abs(k-np.asarray(self.spec.manifest()["requested_intrinsics"]))))
        previous = self.previous
        progressed = clock_progress(previous, episode_id, frame, timestamp, reset)
        fields_valid = (math.isfinite(timestamp) and timestamp >= 0 and np.isfinite(rgb).all()
                        and geometry["nan_pixels"] == 0 and geometry["negative_infinity_pixels"] == 0
                        and geometry["negative_finite_pixels"] == 0 and intrinsic_error < 0.05)
        row = dict(step=int(step), episode_id=int(episode_id), reset=bool(reset), sensor_frame=frame,
                   sensor_timestamp_s=timestamp, host_monotonic_ns=time.monotonic_ns(),
                   simulation_physics_steps=int(self.env._sim_step_counter),
                   simulation_time_s=float(self.env._sim_step_counter * self.env.physics_dt),
                   position_w_m=pos.tolist(), quaternion_ros_xyzw=quat_ros.tolist(),
                   quaternion_world_xyzw=quat_world.tolist(), intrinsic_matrix=k.tolist(),
                   rgb_shape=list(rgb.shape), depth_shape=list(depth.shape), rgb_dtype=str(rgb.dtype), depth_dtype=str(depth.dtype),
                   rgb_spatially_nonuniform=bool(np.any(rgb != rgb[0, 0])),
                   capture_fields_valid=bool(fields_valid), frame_and_timestamp_progress=bool(progressed),
                   head_attachment_position_error_m=attachment_error, head_attachment_angle_error_rad=angular_error,
                   rigid_head_attachment_verified=attachment_error < 0.002 and angular_error < 0.01,
                   intrinsic_max_error_px=intrinsic_error, geometry=geometry,
                   warmup=len(self.rows) < self.spec.warmup_samples)
        stem = f"{len(self.rows):06d}"
        path = self.output / "frames" / f"{stem}.npz"
        np.savez_compressed(path, rgb=rgb, depth_m=depth, valid_mask=valid, intrinsic_matrix=k,
                            position_w_m=pos, quaternion_ros_xyzw=quat_ros,
                            sensor_frame=np.int64(frame), sensor_timestamp_s=np.float64(timestamp),
                            episode_id=np.int64(episode_id), control_step=np.int64(step))
        row["frame_file"] = str(path.relative_to(self.output))
        row["frame_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        if len(self.rows) % self.spec.snapshot_interval == 0:
            _png(self.output / f"rgb-{stem}.png", rgb)
            preview = np.zeros(depth.shape, dtype=np.uint8)
            preview[valid] = np.clip((1-depth[valid]/self.spec.far_m)*255, 0, 255).astype(np.uint8)
            _png(self.output / f"depth-preview-{stem}.png", preview)
        self.trace.write(json.dumps(row, allow_nan=False) + "\n")
        self.trace.flush()
        self.rows.append(row)
        self.previous = (int(episode_id), frame, timestamp)
        return row

    def finalize(self):
        if self.final is not None:
            return self.final
        measured = [row for row in self.rows if not row["warmup"]]
        eligible = sum(row["geometry"]["floor_test_pixels"] for row in measured)
        floor_hits = sum(row["geometry"]["floor_consistent_pixels"] for row in measured)
        behind = sum(row["geometry"]["behind_floor_pixels"] for row in measured)
        checks = {
            "measured_stream_present": bool(measured),
            "all_capture_fields_valid": bool(measured) and all(row["capture_fields_valid"] for row in measured),
            "valid_depth_observed": bool(measured) and all(row["geometry"]["valid_fraction"] >= self.spec.minimum_valid_fraction for row in measured),
            "rgb_scene_observed": bool(measured) and any(row["rgb_spatially_nonuniform"] for row in measured),
            "frame_and_timestamp_progress": bool(measured) and all(row["frame_and_timestamp_progress"] for row in measured),
            "rigid_head_attachment_verified": bool(measured) and all(row["rigid_head_attachment_verified"] for row in measured),
            "floor_geometry_consistent": eligible > 0 and floor_hits >= self.spec.minimum_floor_hits and behind/eligible <= self.spec.maximum_behind_floor_fraction,
        }
        self.final = dict(collected_successfully=all(checks.values()), checks=checks, samples=len(self.rows),
                          measured_samples=len(measured), warmup_samples=len(self.rows)-len(measured),
                          raw_depth_all_finite=all(row["geometry"]["raw_depth_finite_fraction"] == 1 for row in measured) if measured else False,
                          positive_infinity_background_allowed=True, depth_conditioned_policy=False,
                          floor_test_pixels=eligible, floor_consistent_pixels=floor_hits, behind_floor_pixels=behind,
                          trace_sha256=hashlib.sha256((self.output / "sensor-trace.jsonl").read_bytes()).hexdigest(),
                          camera_manifest=self.manifest,
                          limits="Geometry check uses known horizontal floor; closer obstacles are accepted. No calibrated hardware or learned obstacle perception is claimed.")
        _json(self.output / "sensor-summary.json", self.final)
        return self.final

    def close(self):
        if not self.trace.closed:
            self.trace.close()
