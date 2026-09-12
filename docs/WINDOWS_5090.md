# Windows RTX 5090 worker

Use Windows as a separate GPU training worker, controlled from your Mac. Keep a
separate clone on the Spark for the dashboard and simulator. Do not share one
working directory over SSH or a network drive.

The supported path is **Ubuntu in WSL2**, not native Windows Python. WSL2 gives
the trainer Linux CUDA, MuJoCo, Warp, and the same shell scripts used on the
Spark.

## One-time setup

1. Install the current NVIDIA Windows driver that supports the RTX 5090.
2. In an elevated PowerShell window, install WSL2 and Ubuntu if needed:

   ```powershell
   wsl --install -d Ubuntu-24.04
   ```

   Reboot if Windows asks, then open **Ubuntu** from the Start menu.

3. In Ubuntu, verify that the Windows driver is visible. Do not install a Linux
   NVIDIA driver inside WSL2:

   ```bash
   nvidia-smi
   ```

   The command must show the RTX 5090 before continuing.

4. Install the basic Linux tools and clone a fresh worker checkout:

   ```bash
   sudo apt update
   sudo apt install -y git make python3 python3-venv build-essential
   git clone --recurse-submodules https://github.com/jvpflum/microduck-lab.git
   cd microduck-lab
   make bootstrap
   make preflight
   ```

`make preflight` now accepts both the Spark's ARM64 architecture and x86_64
workers. It must report CUDA available and name the RTX 5090.

## Bring the V11 training donor over privately

The public release includes V11 as an ONNX inference policy for evaluation and
simulator playback. PPO resumption needs the separate raw `.pt` donor, which is
deliberately not published because trainer state and generated metadata do not
belong in the public repository.

From the Windows WSL checkout, copy it over your authenticated private network:

```bash
mkdir -p checkpoints
scp <ssh-user>@<spark-address>:~/projects/microduck-lab/upstream/microduck_rl/logs/rsl_rl/velocity_race5/2026-08-31_03-06-10_ducklab-race5-v11-drag-launch-i10-s42/model_10.pt checkpoints/v11-model_10.pt
sha256sum checkpoints/v11-model_10.pt
```

Keep `checkpoints/` local. It is ignored by Git and should never be committed.

## First 5090 race run

Start conservatively so the new worker is proven before using its full capacity:

```bash
DUCKLAB_RACE5_WARMSTART_CHECKPOINT="$PWD/checkpoints/v11-model_10.pt" \
DUCKLAB_ENVS=4096 \
DUCKLAB_ITERATIONS=4000 \
DUCKLAB_SEED=5090 \
DUCKLAB_RACE5_RUN_NAME=ducklab-race5-v14-5090-s5090 \
./scripts/train-race5-v2.sh
```

Use a different seed and run name from the Spark; keep the resulting logs and
raw checkpoints on the worker. When a candidate is ready, export its ONNX and
copy only that ONNX plus the scrubbed evaluation summary back to the Spark or
publish it as a reviewed release artifact.

## Worker operating rules

- Run one large trainer per GPU; do not run the Spark trainer and a 5090 trainer
  in the same checkout.
- Keep the dashboard/simulator on the Spark. It can evaluate a selected ONNX
  after you copy it back.
- Use Git for source changes. Commit source and documentation only; never add
  `policy-bench/`, `reports/`, `checkpoints/`, raw `.pt` files, TensorBoard, or
  W&B data.
- If `nvidia-smi` fails in WSL2, update the Windows driver first. Installing a
  separate Linux GPU driver in WSL2 is not the fix.
