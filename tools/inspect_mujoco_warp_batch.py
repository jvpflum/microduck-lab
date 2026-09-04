#!/usr/bin/env python3
"""Inspect direct batched MuJoCo Warp state/contact layouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

import search_frontflip_native_oc as oc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=4)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    nodes = np.asarray(reference["full_nodes"], dtype=np.float64)
    oc.make_context(str(args.scene.resolve()), nodes[0].tolist(), 1.8)
    context = oc._TLS.context
    model: mujoco.MjModel = context["model"]
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    qa, va = context["qpos_adr"], context["qvel_adr"]
    data.qpos[qa : qa + 7] = [0.0, 0.0, 0.1385, 1.0, 0.0, 0.0, 0.0]
    data.qvel[va] = 1.54
    for dof in context["wheel_dofs"]:
        data.qvel[dof] = 1.54 / oc.WHEEL_RADIUS
    data.qpos[context["actuator_qpos"]] = context["default"]
    data.ctrl[:] = context["default"]
    mujoco.mj_forward(model, data)
    wp.set_device("cuda:0")
    warp_model = mjw.put_model(model)
    warp_data = mjw.put_data(
        model,
        data,
        nworld=args.worlds,
        nconmax=64,
        nccdmax=64,
        njmax=256,
        njmax_nnz=6144,
        nvmax=model.nv,
    )
    mjw.step(warp_model, warp_data)
    wp.synchronize_device("cuda:0")
    for name, value in (
        ("qpos", warp_data.qpos),
        ("qvel", warp_data.qvel),
        ("ctrl", warp_data.ctrl),
        ("nacon", warp_data.nacon),
        ("contact.geom", warp_data.contact.geom),
        ("contact.worldid", warp_data.contact.worldid),
        ("contact.dist", warp_data.contact.dist),
    ):
        tensor = wp.to_torch(value)
        print(name, tuple(tensor.shape), tensor.dtype, tensor.flatten()[:32])


if __name__ == "__main__":
    main()
