"""Frozen V89 and fixed-target diagnostics in CPU MuJoCo and Isaac Lab MJWarp."""

from __future__ import annotations

import argparse
import json
import math
import hashlib
import copy
from contextlib import nullcontext
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation


parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
parser.add_argument("--physics", default="newton_mjwarp", choices=["newton_mjwarp"])
parser.add_argument("--control-mode", default="independent", choices=["independent", "shared_cpu"])
parser.add_argument("--sensor-source", default="legacy", choices=["native", "legacy"])
parser.add_argument("--audit-only", action="store_true")
parser.add_argument("--case", default="policy", choices=["policy", "stand", "pulse", "roll"])
parser.add_argument("--steps", type=int, default=500)
add_launcher_args(parser)
args = parser.parse_args()

import mujoco
import numpy as np
import onnxruntime as ort
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.physics import PhysicsCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.sim.schemas.schemas_cfg import NewtonArticulationCfg


HOME = Path.home()
PROJECT = HOME / "isaaclab-pilot"
USD = PROJECT / "duckwing-v89-official-r3/robot_allcollisions_rollers/robot_allcollisions_rollers.usda"
V91 = HOME / "home-ray-envs/v91-project"
SOURCE = V91 / "upstream/microduck_rl/src/mjlab_microduck/robot/microduck/scene_rollers.xml"
POLICY = V91 / "reports/v89-maturity-brake-20260912/checkpoint-450.onnx"
TERM_WIDTHS = (3, 3, 14, 14, 14, 3, 4, 6)
HISTORY_LENGTH = 256
DECIMATION = 4

STAND = {
    "left_hip_yaw": 0.0,
    "left_hip_roll": -0.0873,
    "left_hip_pitch": -0.4579,
    "left_knee": -0.0049,
    "left_ankle": 0.4530,
    "neck_pitch": 0.3491,
    "head_pitch": 0.3491,
    "head_yaw": 0.0,
    "head_roll": 0.0,
    "right_hip_yaw": 0.0,
    "right_hip_roll": 0.0873,
    "right_hip_pitch": 0.4579,
    "right_knee": 0.0049,
    "right_ankle": -0.4530,
}
MOTOR_JOINTS = list(STAND)
DEFAULT = np.asarray([STAND[name] for name in MOTOR_JOINTS], dtype=np.float32)
CONTACT_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    static_friction=1.0,
    dynamic_friction=1.0,
    restitution=0.0,
)


DUCKWING = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(USD),
        variants={"Physics": "mujoco"},
        make_uninstanceable=True,
        physics_material=CONTACT_MATERIAL,
        articulation_props=[NewtonArticulationCfg(self_collision_enabled=True)],
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.1385),
        joint_pos={**STAND, "passive_.*_wheel": 0.0},
    ),
    joint_ordering="mjwarp",
    body_ordering="mjwarp",
    actuators={
        "motors": ImplicitActuatorCfg(
            joint_names_expr=MOTOR_JOINTS,
            joint_effort_limit=0.9625,
            stiffness=0.55,
            damping=0.0,
            armature=0.0018,
            friction=0.0048,
        ),
        "passive_wheels": ImplicitActuatorCfg(
            joint_names_expr=["passive_.*_wheel"],
            joint_effort_limit=1.0e-9,
            stiffness=0.0,
            damping=0.0,
            armature=0.0001,
            friction=0.003,
        ),
    },
)


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(physics_material=CONTACT_MATERIAL),
    )
    robot: ArticulationCfg = DUCKWING.replace(prim_path="{ENV_REGEX_NS}/Robot")


class History:
    def __init__(self) -> None:
        self.frames: np.ndarray | None = None

    def pack(self, frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame, dtype=np.float32)
        assert frame.shape == (61,)
        if self.frames is None:
            self.frames = np.repeat(frame[None, :], HISTORY_LENGTH, axis=0)
        else:
            self.frames[:-1] = self.frames[1:]
            self.frames[-1] = frame
        pieces = []
        offset = 0
        for width in TERM_WIDTHS:
            pieces.append(self.frames[:, offset : offset + width].reshape(-1))
            offset += width
        packed = np.concatenate(pieces).astype(np.float32)
        assert packed.shape == (15616,)
        return packed


def command_for(step: int) -> np.ndarray:
    command = np.zeros(13, dtype=np.float32)
    if 100 <= step < 250:
        command[0] = 0.8
    return command


def inverse_rotate_one(quat_wxyz: np.ndarray, vector: np.ndarray) -> np.ndarray:
    cross = np.cross(quat_wxyz[1:4], vector) * 2
    return vector - quat_wxyz[0] * cross + np.cross(quat_wxyz[1:4], cross)


def cpu_setup() -> tuple[mujoco.MjModel, mujoco.MjData, list[int], list[int], int, int]:
    model = mujoco.MjModel.from_xml_path(str(SOURCE))
    model.opt.timestep = 0.005
    model.actuator_forcerange[:] = (-0.9625, 0.9625)
    model.actuator_forcelimited[:] = 1
    for joint_index in range(model.njnt):
        name = model.joint(joint_index).name
        if name.startswith("passive_") and "wheel" in name:
            model.dof_frictionloss[int(model.jnt_dofadr[joint_index])] = 0.003
    qpos_indices = []
    qvel_indices = []
    for name in MOTOR_JOINTS:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos_indices.append(int(model.jnt_qposadr[joint_id]))
        qvel_indices.append(int(model.jnt_dofadr[joint_id]))
    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    gyro = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
    gyro_start = int(model.sensor_adr[gyro])
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:7] = (0.0, 0.0, 0.1385, 1.0, 0.0, 0.0, 0.0)
    data.qpos[qpos_indices] = DEFAULT
    data.ctrl[:] = DEFAULT
    if args.case == "roll":
        data.qvel[0] = 0.3
    mujoco.mj_forward(model, data)
    return model, data, qpos_indices, qvel_indices, trunk, gyro_start


def cpu_observation(
    data: mujoco.MjData,
    qpos_indices: list[int],
    qvel_indices: list[int],
    trunk: int,
    gyro_start: int,
    previous_action: np.ndarray,
    command: np.ndarray,
) -> np.ndarray:
    angular_velocity = np.asarray(data.sensordata[gyro_start : gyro_start + 3], dtype=np.float32)
    quat = np.asarray(data.xquat[trunk], dtype=np.float32)
    projected_gravity = inverse_rotate_one(quat, np.asarray([0.0, 0.0, -1.0], dtype=np.float32))
    return np.concatenate(
        [
            angular_velocity,
            projected_gravity,
            np.asarray(data.qpos[qpos_indices], dtype=np.float32) - DEFAULT,
            np.asarray(data.qvel[qvel_indices], dtype=np.float32),
            previous_action,
            command,
        ]
    ).astype(np.float32)


def isaac_observation(robot, motor_indices: list[int], previous_action: np.ndarray, command: np.ndarray) -> np.ndarray:
    angular_velocity = robot.data.root_ang_vel_b.torch[0].detach().cpu().numpy().astype(np.float32)
    projected_gravity = robot.data.projected_gravity_b.torch[0].detach().cpu().numpy().astype(np.float32)
    if args.sensor_source == "legacy":
        # V89 reads MuJoCo's pre-integration IMU/kinematics but post-integration
        # joint state. Newton's public body state is refreshed after integration.
        from isaaclab_newton.physics.newton_manager import NewtonManager
        data = NewtonManager._solver.mjw_data
        quat = data.xquat.numpy()[0, 1].astype(np.float32)
        angular_velocity = inverse_rotate_one(quat, data.cvel.numpy()[0, 1, :3].astype(np.float32))
        projected_gravity = inverse_rotate_one(quat, np.asarray([0.0, 0.0, -1.0], dtype=np.float32))
    joint_pos = robot.data.joint_pos.torch[0, motor_indices].detach().cpu().numpy().astype(np.float32) - DEFAULT
    joint_vel = robot.data.joint_vel.torch[0, motor_indices].detach().cpu().numpy().astype(np.float32)
    return np.concatenate([angular_velocity, projected_gravity, joint_pos, joint_vel, previous_action, command]).astype(np.float32)


def tilt_from_quat(quat_wxyz: np.ndarray) -> float:
    return float(math.acos(np.clip(1.0 - 2.0 * (quat_wxyz[1] ** 2 + quat_wxyz[2] ** 2), -1.0, 1.0)))


def tilt_from_isaac_quat(quat_xyzw: np.ndarray) -> float:
    return tilt_from_quat(np.asarray(quat_xyzw)[[3, 0, 1, 2]])


def main() -> None:
    assert args.steps > 0
    assert hashlib.sha256(POLICY.read_bytes()).hexdigest() == "1693fd8608037a6b73a128ba737c57adc7bcb6228cac2cc8603719f4be740c3a"
    session = ort.InferenceSession(str(POLICY), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    cpu_model, cpu_data, cpu_qpos, cpu_qvel, cpu_trunk, cpu_gyro_start = cpu_setup()
    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args) as physics:
        assert isinstance(physics, NewtonCfg)
        physics.solver_cfg = MJWarpSolverCfg(
            njmax=2048,
            nconmax=256,
            cone="pyramidal",
            impratio=1,
            integrator="euler",
            tolerance=1e-8,
            enable_multiccd=True,
            disable_sensors=args.deterministic,
        )
        if args.deterministic:
            physics.deterministic_mode = "run_to_run"
        physics.num_substeps = 1
        physics.debug_mode = False
        physics.use_cuda_graph = True
        simulation = SimulationContext(
            SimulationCfg(
                dt=0.005,
                render_interval=4,
                device=args.device,
                physics=physics,
                physics_material=CONTACT_MATERIAL,
            )
        )
        scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=1.0, replicate_physics=True, clone_in_fabric=True))
        from pxr import UsdPhysics
        # The source power-support collision is the only member of mask group
        # 2 and cannot contact group 1 (floor/robot). Preserve that exclusion.
        masked = [g for g in range(cpu_model.ngeom) if cpu_model.geom_contype[g] == 2]
        assert len(masked) == 1 and cpu_model.geom_conaffinity[masked[0]] == 2
        disabled_shapes = []
        for prim in sim_utils.get_current_stage().Traverse():
            if "/power_support_1/" in str(prim.GetPath()) and prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
                disabled_shapes.append(str(prim.GetPath()))
        assert len(disabled_shapes) == 1, disabled_shapes
        simulation.reset()
        robot = scene["robot"]
        robot.write_root_pose_to_sim_index(root_pose=robot.data.default_root_pose.torch.clone())
        robot.write_root_velocity_to_sim_index(root_velocity=robot.data.default_root_vel.torch.clone())
        robot.write_joint_state_to_sim(
            robot.data.default_joint_pos.torch.clone(),
            robot.data.default_joint_vel.torch.clone(),
        )
        robot.reset()
        scene.update(0.0)
        if args.case == "roll":
            velocity = robot.data.default_root_vel.torch.clone()
            velocity[:, 0] = 0.3
            robot.write_root_velocity_to_sim_index(root_velocity=velocity)
        motor_indices = [robot.joint_names.index(name) for name in MOTOR_JOINTS]
        if args.audit_only:
            from isaaclab_newton.physics.newton_manager import NewtonManager
            solver = NewtonManager._solver
            converted = solver.mj_model
            converted.opt.timestep = 0.005
            mapped = mujoco.MjData(converted)
            mapped.qpos[:7] = cpu_data.qpos[:7]
            joint_map = {}
            for j in range(1, cpu_model.njnt):
                name = cpu_model.joint(j).name
                matches = [k for k in range(1, converted.njnt) if converted.joint(k).name.endswith("_" + name)]
                assert len(matches) == 1, (name, matches)
                k = matches[0]
                joint_map[j] = k
                mapped.qpos[converted.jnt_qposadr[k]] = cpu_data.qpos[cpu_model.jnt_qposadr[j]]
            for a in range(converted.nu):
                cj = int(converted.actuator_trnid[a, 0])
                originals = [j for j, k in joint_map.items() if k == cj]
                if originals:
                    sj = originals[0]
                    motors = np.flatnonzero(cpu_model.actuator_trnid[:, 0] == sj)
                    if len(motors):
                        mapped.ctrl[a] = cpu_data.ctrl[motors[0]]
            mujoco.mj_forward(converted, mapped)
            body_map = {}
            for b in range(1, cpu_model.nbody):
                source_name = cpu_model.body(b).name
                semantic = "neck_pitch_1" if source_name == "neck_pitch" else source_name
                matches = [c for c in range(1, converted.nbody) if converted.body(c).name.endswith("_" + semantic)]
                assert len(matches) == 1, (source_name, matches)
                body_map[b] = matches[0]
            aligned = {
                "bodies": [{"name": cpu_model.body(b).name,
                            "position_error_m": float(np.max(np.abs(cpu_data.xpos[b] - mapped.xpos[c]))),
                            "rotation_error": float(np.max(np.abs(cpu_data.xmat[b] - mapped.xmat[c]))),
                            "com_error_m": float(np.max(np.abs(cpu_data.xipos[b] - mapped.xipos[c]))),
                            "mass_error": float(cpu_model.body_mass[b] - converted.body_mass[c]),
                            "inertia_world_error": float(np.max(np.abs(cpu_data.ximat[b].reshape(3,3) @ np.diag(cpu_model.body_inertia[b]) @ cpu_data.ximat[b].reshape(3,3).T - mapped.ximat[c].reshape(3,3) @ np.diag(converted.body_inertia[c]) @ mapped.ximat[c].reshape(3,3).T)))} for b,c in body_map.items()],
                "joints": [{"name": cpu_model.joint(j).name,
                            "source_accel": float(cpu_data.qacc[cpu_model.jnt_dofadr[j]]),
                            "converted_accel": float(mapped.qacc[converted.jnt_dofadr[k]]),
                            "source_passive": float(cpu_data.qfrc_passive[cpu_model.jnt_dofadr[j]]),
                            "converted_passive": float(mapped.qfrc_passive[converted.jnt_dofadr[k]])} for j,k in joint_map.items()],
                "source_root_accel": cpu_data.qacc[:6].tolist(),
                "converted_root_accel": mapped.qacc[:6].tolist(),
                "source_contacts": [{"geoms": [int(c.geom1), int(c.geom2)], "dist": float(c.dist), "pos": c.pos.tolist()} for c in cpu_data.contact],
                "converted_contacts": [{"geoms": [int(c.geom1), int(c.geom2)], "dist": float(c.dist), "pos": c.pos.tolist()} for c in mapped.contact],
            }
            def describe(model):
                groups = {"body": (model.nbody, ("mass", "inertia", "ipos", "iquat", "pos", "quat")),
                          "jnt": (model.njnt, ("type", "qposadr", "dofadr", "pos", "axis", "range")),
                          "geom": (model.ngeom, ("type", "bodyid", "size", "pos", "quat", "friction", "condim", "solref", "solimp", "margin", "gap", "contype", "conaffinity")),
                          "actuator": (model.nu, ("trnid", "gear", "gainprm", "biasprm", "forcerange", "ctrlrange"))}
                result = {}
                for prefix, (count, fields) in groups.items():
                    kind = {"body": mujoco.mjtObj.mjOBJ_BODY, "jnt": mujoco.mjtObj.mjOBJ_JOINT, "geom": mujoco.mjtObj.mjOBJ_GEOM, "actuator": mujoco.mjtObj.mjOBJ_ACTUATOR}[prefix]
                    result[prefix] = [{"name": mujoco.mj_id2name(model, kind, i), **{f: np.asarray(getattr(model, prefix + "_" + f)[i]).tolist() for f in fields}} for i in range(count)]
                result["dof"] = {f: np.asarray(getattr(model, "dof_" + f)).tolist() for f in ("damping", "armature", "frictionloss")}
                result["opt"] = {f: np.asarray(getattr(model.opt, f)).tolist() for f in ("timestep", "integrator", "solver", "iterations", "ls_iterations", "tolerance", "ls_tolerance", "cone", "impratio", "gravity", "enableflags", "disableflags")}
                return result
            Path(args.output).write_text(json.dumps({"source": describe(cpu_model), "converted": describe(converted), "aligned": aligned, "robot_joint_names": list(robot.joint_names), "solver_type": type(solver).__name__}, indent=2))
            return
        import warp as wp
        import mujoco_warp as mjw
        # A direct source-MJCF CUDA control distinguishes importer errors from
        # the solver's CPU/GPU differences, without modifying the CPU reference.
        shadow_cpu_model = copy.copy(cpu_model)
        if args.deterministic:
            shadow_cpu_model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
        shadow_model = mjw.put_model(shadow_cpu_model)
        shadow_data = mjw.put_data(shadow_cpu_model, cpu_data, nworld=1, nconmax=256, njmax=2048)
        from isaaclab_newton.physics.newton_manager import NewtonManager
        solver = NewtonManager._solver
        # put_model silently floors CPU tolerance at 1e-6. This small robot's
        # frozen contract uses 1e-8; verify the active GPU field, not host metadata.
        runtime_tolerances = {}
        for label, gpu_model in (("isaac", solver.mjw_model), ("source_gpu", shadow_model)):
            field = gpu_model.opt.tolerance
            runtime_tolerances[label] = {"before": float(np.asarray(field.numpy()).reshape(-1)[0]) if hasattr(field, "numpy") else float(field)}
            if hasattr(field, "fill_"):
                field.fill_(cpu_model.opt.tolerance)
            else:
                gpu_model.opt.tolerance = float(cpu_model.opt.tolerance)
            runtime_tolerances[label]["after"] = float(cpu_model.opt.tolerance)
        # Capture the unchanged four-substep source GPU control to avoid
        # thousands of Python launches during this diagnostic.
        # Match Newton's pinned deterministic kernel options for the direct
        # source control. Internal sensors must be disabled in this mode; gyro
        # is reconstructed from the same pre-integration cvel/xquat below.
        if args.deterministic:
            from newton._src.solvers.mujoco.solver_mujoco import _mujoco_warp_deterministic_max_records
            assert solver._deterministic_max_records >= _mujoco_warp_deterministic_max_records(shadow_cpu_model, shadow_data)
        with (solver._scoped_mujoco_warp_execution() if args.deterministic else nullcontext()):
            with wp.ScopedCapture(device=simulation.device) as capture:
                for _ in range(DECIMATION):
                    mjw.step(shadow_model, shadow_data)
        shadow_graph = capture.graph
        shadow_previous = np.zeros(14, dtype=np.float32)
        shadow_history = History()
        shadow_fell = False
        shadow_max_separation = 0.0
        cpu_history = History()
        isaac_history = History()
        cpu_previous = np.zeros(14, dtype=np.float32)
        isaac_previous = np.zeros(14, dtype=np.float32)
        finite = True
        cpu_fell = False
        isaac_fell = False
        initial_observation_max_abs = None
        initial_action_max_abs = None
        observation_max_abs = []
        action_mean_abs = []
        snapshots = []
        initial_alignment = None
        for step in range(args.steps):
            command = command_for(step)
            cpu_frame = cpu_observation(
                cpu_data,
                cpu_qpos,
                cpu_qvel,
                cpu_trunk,
                cpu_gyro_start,
                cpu_previous,
                command,
            )
            isaac_frame = isaac_observation(robot, motor_indices, isaac_previous, command)
            cpu_input = cpu_history.pack(cpu_frame)
            isaac_input = isaac_history.pack(isaac_frame)
            cpu_action = session.run(None, {input_name: cpu_input[None]})[0][0].astype(np.float32)
            isaac_action = session.run(None, {input_name: isaac_input[None]})[0][0].astype(np.float32)
            sd_qp = shadow_data.qpos.numpy()[0]
            sd_qv = shadow_data.qvel.numpy()[0]
            sd_quat = shadow_data.xquat.numpy()[0, cpu_trunk]
            shadow_gyro = inverse_rotate_one(sd_quat, shadow_data.cvel.numpy()[0, cpu_trunk, :3]) if args.deterministic else shadow_data.sensordata.numpy()[0, cpu_gyro_start:cpu_gyro_start+3]
            shadow_frame = np.concatenate([shadow_gyro, inverse_rotate_one(sd_quat, np.array([0.,0.,-1.], np.float32)), sd_qp[cpu_qpos] - DEFAULT, sd_qv[cpu_qvel], shadow_previous, command]).astype(np.float32)
            shadow_action = session.run(None, {input_name: shadow_history.pack(shadow_frame)[None]})[0][0].astype(np.float32)
            if args.case != "policy":
                cpu_action = np.zeros(14, dtype=np.float32)
                if args.case == "pulse" and 25 <= step < 50:
                    cpu_action[2] = 0.1
                isaac_action = cpu_action.copy()
                shadow_action = cpu_action.copy()
            if step == 0:
                initial_observation_max_abs = float(np.max(np.abs(cpu_frame - isaac_frame)))
                initial_action_max_abs = float(np.max(np.abs(cpu_action - isaac_action)))
                term_names = ("angular_velocity", "projected_gravity", "joint_position", "joint_velocity", "previous_action", "command_velocity", "command_gait", "command_misc")
                term_errors = {}
                offset = 0
                for name, width in zip(term_names, TERM_WIDTHS):
                    term_errors[name] = float(np.max(np.abs(cpu_frame[offset : offset + width] - isaac_frame[offset : offset + width])))
                    offset += width
                initial_alignment = {
                    "term_max_abs_errors": term_errors,
                    "cpu_motor_positions": np.asarray(cpu_data.qpos[cpu_qpos]).tolist(),
                    "isaac_motor_positions": robot.data.joint_pos.torch[0, motor_indices].detach().cpu().numpy().tolist(),
                    "isaac_default_motor_positions": robot.data.default_joint_pos.torch[0, motor_indices].detach().cpu().numpy().tolist(),
                    "motor_names": MOTOR_JOINTS,
                    "cpu_root_pose": np.asarray(cpu_data.qpos[:7]).tolist(),
                    "isaac_root_pose": robot.data.root_state_w.torch[0, :7].detach().cpu().numpy().tolist(),
                }
            observation_max_abs.append(float(np.max(np.abs(cpu_frame - isaac_frame))))
            action_mean_abs.append(float(np.mean(np.abs(cpu_action - isaac_action))))
            cpu_data.ctrl[:] = DEFAULT + cpu_action
            shadow_applied = cpu_action if args.control_mode == "shared_cpu" else shadow_action
            wp.to_torch(shadow_data.ctrl)[0].copy_(torch.as_tensor(DEFAULT + shadow_applied, device=simulation.device))
            isaac_applied_action = cpu_action if args.control_mode == "shared_cpu" else isaac_action
            full_target = robot.data.default_joint_pos.torch.clone()
            full_target[:, motor_indices] = torch.as_tensor(DEFAULT + isaac_applied_action, device=simulation.device)
            robot.set_joint_position_target_index(target=full_target)
            for _ in range(DECIMATION):
                scene.write_data_to_sim()
                simulation.step()
                scene.update(0.005)
                mujoco.mj_step(cpu_model, cpu_data)
            wp.capture_launch(shadow_graph)
            cpu_previous = cpu_action
            isaac_previous = isaac_applied_action
            shadow_previous = shadow_applied
            shadow_qpos = shadow_data.qpos.numpy()[0]
            shadow_tilt = tilt_from_quat(shadow_qpos[3:7])
            shadow_fell |= bool(shadow_qpos[2] < 0.065 or shadow_tilt > math.radians(45))
            shadow_max_separation = max(shadow_max_separation, float(np.linalg.norm(shadow_qpos[:2] - cpu_data.qpos[:2])))
            isaac_root_pos = robot.data.root_pos_w.torch[0].detach().cpu().numpy()
            isaac_root_quat = robot.data.root_quat_w.torch[0].detach().cpu().numpy()
            cpu_tilt = tilt_from_quat(np.asarray(cpu_data.qpos[3:7]))
            isaac_tilt = tilt_from_isaac_quat(isaac_root_quat)
            finite = finite and bool(
                np.isfinite(cpu_data.qpos).all()
                and np.isfinite(cpu_data.qvel).all()
                and torch.isfinite(robot.data.joint_pos.torch).all()
                and torch.isfinite(robot.data.joint_vel.torch).all()
                and torch.isfinite(robot.data.root_pos_w.torch).all()
                and np.isfinite(isaac_root_quat).all()
                and np.isfinite(shadow_qpos).all()
                and np.isfinite(shadow_data.qvel.numpy()).all()
                and np.isfinite(cpu_action).all()
                and np.isfinite(isaac_action).all()
                and np.isfinite(shadow_action).all()
            )
            cpu_fell = cpu_fell or bool(cpu_data.qpos[2] < 0.065 or cpu_tilt > math.radians(45))
            isaac_fell = isaac_fell or bool(isaac_root_pos[2] < 0.065 or isaac_tilt > math.radians(45))
            if step < 10 or step % 25 == 0 or step == args.steps - 1:
                snapshots.append(
                    {
                        "step": step,
                        "time_s": (step + 1) * 0.02,
                        "command_x": float(command[0]),
                        "cpu_root_xyz": np.asarray(cpu_data.qpos[:3]).tolist(),
                        "isaac_root_xyz": isaac_root_pos.tolist(),
                        "cpu_tilt_deg": math.degrees(cpu_tilt),
                        "isaac_tilt_deg": math.degrees(isaac_tilt),
                        "observation_max_abs": observation_max_abs[-1],
                        "action_mean_abs": action_mean_abs[-1],
                        "source_gpu_root_xyz": shadow_qpos[:3].tolist(),
                        "source_gpu_tilt_deg": math.degrees(shadow_tilt),
                        "source_gpu_joint_error": float(np.max(np.abs(shadow_qpos[cpu_qpos] - cpu_data.qpos[cpu_qpos]))),
                        "isaac_joint_error": float(np.max(np.abs(robot.data.joint_pos.torch[0, motor_indices].detach().cpu().numpy() - cpu_data.qpos[cpu_qpos]))),
                        "cpu_frame": cpu_frame.tolist() if step < 10 else None,
                        "gpu_frame": shadow_frame.tolist() if step < 10 else None,
                        "isaac_frame": isaac_frame.tolist() if step < 10 else None,
                        "cpu_action": cpu_action.tolist() if step < 10 else None,
                        "gpu_action": shadow_action.tolist() if step < 10 else None,
                    }
                )
        cpu_xy = np.asarray(cpu_data.qpos[:2])
        isaac_xy = robot.data.root_pos_w.torch[0, :2].detach().cpu().numpy()
        metrics = {
            "initial_observation_max_abs": initial_observation_max_abs,
            "initial_action_max_abs": initial_action_max_abs,
            "mean_observation_max_abs": float(np.mean(observation_max_abs)),
            "mean_action_abs_difference": float(np.mean(action_mean_abs)),
            "final_xy_separation_m": float(np.linalg.norm(cpu_xy - isaac_xy)),
            "cpu_final_xy_m": cpu_xy.tolist(),
            "isaac_final_xy_m": isaac_xy.tolist(),
            "cpu_fell": cpu_fell,
            "isaac_fell": isaac_fell,
            "finite": finite,
            "source_gpu_fell": shadow_fell,
            "source_gpu_final_xy_m": shadow_qpos[:2].tolist(),
            "source_gpu_max_separation_m": shadow_max_separation,
            "source_gpu_final_separation_m": float(np.linalg.norm(shadow_qpos[:2] - cpu_xy)),
        }
        gates = {
            "finite": finite,
            "both_upright": not cpu_fell and not isaac_fell,
            "initial_observation_close": initial_observation_max_abs is not None and initial_observation_max_abs <= 0.02,
            "initial_action_close": initial_action_max_abs is not None and initial_action_max_abs <= 0.05,
            "mean_action_close": float(np.mean(action_mean_abs)) <= 0.15,
            "final_path_close": float(np.linalg.norm(cpu_xy - isaac_xy)) <= 0.25,
        }
        report = {
            "status": "v89-replay-complete",
            "policy_sha256": "1693fd8608037a6b73a128ba737c57adc7bcb6228cac2cc8603719f4be740c3a",
            "control_mode": args.control_mode,
            "sensor_source": args.sensor_source,
            "deterministic_requested": args.deterministic,
            "case": args.case,
            "runtime_tolerances": runtime_tolerances,
            "steps": args.steps,
            "scenario": {"duration_s": args.steps * 0.02, "idle_s": min(args.steps, 100) * 0.02, "launch_s": max(0, min(args.steps, 250) - 100) * 0.02, "command_x": 0.8, "stop_s": max(0, args.steps - 250) * 0.02, "policy_controls": args.case == "policy"},
            "runtime": {"gpu": torch.cuda.get_device_name(0), "physics_device": str(simulation.device), "policy_provider": session.get_providers(), "mujoco_version": mujoco.__version__, "torch_version": torch.__version__, "replay_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            "physics": {"dt_s": 0.005, "control_dt_s": 0.02, "contact_friction": 1.0, "wheel_frictionloss": 0.003, "motor_limit_nm": 0.9625},
            "metrics": metrics,
            "initial_alignment": initial_alignment,
            "gates": gates,
            "behavior_equivalent": all(gates.values()) and args.case == "policy" and args.control_mode == "independent" and args.steps == 500,
            "snapshots": snapshots,
            "qualification": "matched dynamics diagnostic; no release promotion",
        }
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(json.dumps({"metrics": metrics, "gates": gates, "behavior_equivalent": report["behavior_equivalent"]}))


main()
