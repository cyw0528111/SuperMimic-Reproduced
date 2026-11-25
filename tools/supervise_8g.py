"""训练守护进程：崩溃自动续训，并保证总时长不超过全局 deadline。

Windows + 8GB 长跑会偶发 CUDA illegal instruction（Warp mempool / torch 分配器），
单靠一次拉起的进程熬不到 36 小时，所以要能自动从 ckpt_latest.pt 续上。

用法：
    ./.venv/Scripts/python.exe tools/supervise_8g.py --hours 36 --ckpt-interval 500
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv" / "Scripts" / "python.exe"
RUN = "lafan_8g_v1"
CKPT = ROOT / "logs" / RUN / "ckpt_latest.pt"

BASE = [
    str(PY), "-u", "-m", "supermimic.scripts.train",
    "--run-name", RUN,
    "--env.num-envs", "512",
    "--env.anchor-obs",
    "--policy-trainer.batch-size", "512",
    "--policy-trainer.horizon", "16",
    # 世界模型 backward 是崩溃现场（CUDA 715）：默认 10×2048 窗口是全流程最大单次分配，
    # 在 8GB 上减半到 1024，牺牲一点梯度精度换稳定性。
    "--wm-trainer.batch-size", "1024",
]

ENV = {
    **os.environ,
    "PYTORCH_CUDA_ALLOC_CONF": "garbage_collection_threshold:0.8,max_split_size_mb:512",
    "MUJOCO_GL": "glfw",
}


def keep_awake() -> None:
    """进程级阻止系统空闲睡眠（挡不住合盖，但能防自动休眠）。"""
    try:
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    except Exception:
        pass


def prune_ckpts(keep_every: int = 5000, keep_recent: int = 3) -> None:
    """存档每 500 迭代一个、每个 215MB；长跑会把磁盘吃穿。

    只保留：每个 keep_every 的整数倍 + 最近 keep_recent 个 + ckpt_latest.pt。
    """
    d = ROOT / "logs" / RUN
    files = sorted(
        (p for p in d.glob("ckpt_*.pt") if p.stem != "ckpt_latest"),
        key=lambda p: int(p.stem.split("_")[1]) if p.stem.split("_")[1].isdigit() else 0,
    )
    if len(files) <= keep_recent:
        return
    keep = set(files[-keep_recent:])
    for p in files:
        it = int(p.stem.split("_")[1])
        if it % keep_every == 0:
            keep.add(p)
    removed = 0
    for p in files:
        if p not in keep:
            try:
                p.unlink()
                removed += 1
            except OSError:
                pass
    if removed:
        log(f"pruned {removed} old checkpoints, kept {len(keep)}")


def log(msg: str) -> None:
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(ROOT / "supervisor.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=36.0, help="总预算（跨重启累计）")
    ap.add_argument("--ckpt-interval", type=int, default=500)
    ap.add_argument("--max-restarts", type=int, default=40)
    ap.add_argument("--cooldown", type=float, default=20.0)
    ap.add_argument("--no-prune", action="store_true", help="不清理旧存档（安全钩子会拦批量删除）")
    args = ap.parse_args()

    deadline = time.time() + args.hours * 3600
    keep_awake()
    log(f"supervisor start: budget {args.hours} h, ckpt every {args.ckpt_interval} it")

    for attempt in range(1, args.max_restarts + 1):
        remaining = (deadline - time.time()) / 3600
        if remaining <= 0.02:
            log("budget exhausted, stop")
            return
        cmd = BASE + ["--schedule.max-hours", f"{remaining:.2f}",
                      "--schedule.checkpoint-interval", str(args.ckpt_interval)]
        resumed = CKPT.exists()
        if resumed:
            cmd += ["--resume", str(CKPT)]
        if not args.no_prune:
            prune_ckpts()
        log(f"attempt {attempt}/{args.max_restarts}: {'resume' if resumed else 'fresh'}, "
            f"remaining {remaining:.2f} h")
        with open(ROOT / f"train_{RUN}.log", "a", encoding="utf-8") as f:
            f.write(f"\n===== supervisor attempt {attempt} {time.strftime('%m-%d %H:%M:%S')} =====\n")
            rc = subprocess.run(cmd, cwd=ROOT, env=ENV, stdout=f, stderr=subprocess.STDOUT).returncode
        log(f"attempt {attempt} exit rc={rc}")
        if rc == 0:
            log("clean exit (max-hours reached), done")
            return
        time.sleep(args.cooldown)

    log("max restarts reached, giving up")


if __name__ == "__main__":
    sys.exit(main())
