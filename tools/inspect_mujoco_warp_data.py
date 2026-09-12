#!/usr/bin/env python3
"""Print MuJoCo Warp data/contact array layout for one MicroDuck world."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

import search_frontflip_native_oc as oc
from validate_mujoco_warp_parity import reset_data


def describe(name: str, value: object) -> None:
    print(f"{name}: type={type(value)!r}")
    for attribute in ("shape", "dtype", "device"):
        if hasattr(value, attribute):
            print(f"  {attribute}={getattr(value, attribute)!r}")
    if hasattr(value, "numpy"):
        array = np.asarray(value.numpy())
        print(f"  numpy shape={array.shape} dtype={array.dtype}")
        print(f"  sample={array.reshape(-1)[:32].tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--start-speed", type=float, default=1.54)
    parser.add_argument("--steps", type=int, default=240)
    args = parser.parse_args()

    wp.set_device("cuda:0")
    if args.reference:
        reference = json.loads(args.reference.read_text())
        nodes = np.asarray(reference["full_nodes"], dtype=np.float64)
        default = nodes[0].tolist()
    else:
        nodes = np.zeros((len(oc.KNOT_TIMES), oc.SERVO_COUNT), dtype=np.float64)
        default = nodes[0].tolist()
    oc.make_context(str(args.scene.resolve()), default, 0.10)
    context = oc._TLS.context
    model: mujoco.MjModel = context["model"]
    host_data = mujoco.MjData(model)
    reset_data(context, host_data, args.start_speed)
    device_model = mjw.put_model(model)
    device_data = mjw.put_data(
        model,
        host_data,
        nworld=1,
        nconmax=256,
        nccdmax=256,
        njmax=512,
        njmax_nnz=8192,
        naconmax=256,
        naccdmax=256,
        nvmax=int(model.nv),
    )
    first_active = None
    maximum_nacon = 0
    maximum_ncollision = 0
    for step in range(args.steps):
        target = oc.target_at(step * oc.PHYSICS_DT, nodes)
        device_data.ctrl.assign(np.asarray(target, dtype=np.float32)[None, :])
        mjw.step(device_model, device_data)
        wp.synchronize_device("cuda:0")
        active = int(np.asarray(device_data.nacon.numpy()).reshape(-1)[0])
        collisions = int(np.asarray(device_data.ncollision.numpy()).reshape(-1)[0])
        maximum_nacon = max(maximum_nacon, active)
        maximum_ncollision = max(maximum_ncollision, collisions)
        if active and first_active is None:
            first_active = step + 1
            geom = np.asarray(device_data.contact.geom.numpy())[:active]
            world = np.asarray(device_data.contact.worldid.numpy())[:active]
            print(
                f"FIRST ACTIVE step={first_active} nacon={active} "
                f"ncollision={collisions} geom={geom.tolist()} "
                f"worldid={world.tolist()}"
            )
    print(
        f"COUNTS steps={args.steps} first_active={first_active} "
        f"max_nacon={maximum_nacon} max_ncollision={maximum_ncollision}"
    )

    print("DATA ATTRIBUTES")
    print([name for name in dir(device_data) if not name.startswith("_")])
    for name in (
        "qpos", "qvel", "ncon", "nacon", "ncollision", "nefc", "overflow", "contact"
    ):
        if hasattr(device_data, name):
            describe(name, getattr(device_data, name))

    contact = device_data.contact
    print("CONTACT ATTRIBUTES")
    print([name for name in dir(contact) if not name.startswith("_")])
    for name in ("geom", "worldid", "dist", "pos", "frame"):
        if hasattr(contact, name):
            describe(f"contact.{name}", getattr(contact, name))


if __name__ == "__main__":
    main()
