"""训练进度快照：对最新 checkpoint 做评测 + 录像 + 渲染，产出可横向比较的视频与指标表。

用法（在仓库根目录）：
    ./.venv/Scripts/python.exe tools/snapshot_8g.py                # 取最新 ckpt
    ./.venv/Scripts/python.exe tools/snapshot_8g.py --ckpt logs/lafan_8g_v1/ckpt_0004000.pt

产物：
    snapshots/iter_<NNNNNN>/       每档一个目录
        bench.txt  rollouts.npz  <clip>.mp4  compare.mp4
    snapshots/progress.md          累积的指标表（新档追加在末尾）
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv" / "Scripts" / "python.exe"
RUN = "lafan_8g_v1"
CKPT_DIR = ROOT / "logs" / RUN
SNAP = ROOT / "snapshots"
CLIPS = ["walk1_subject1", "run1_subject5", "jumps1_subject1"]
# 每条 clip 里动作真正开始的位置（秒）——LAFAN1 开头有静止站位，从 0 秒录会全是站着
ACTION_START = {"walk1_subject1": 44.0, "run1_subject5": 4.0, "jumps1_subject1": 62.6}
WINDOW = 8.0  # 每个片段展示秒数

ENV = {
    **os.environ,
    "MUJOCO_GL": "glfw",
    "PYTORCH_CUDA_ALLOC_CONF": "garbage_collection_threshold:0.8,max_split_size_mb:512",
    "PATH": str(ROOT / ".bin") + os.pathsep + os.environ.get("PATH", ""),
}


def newest_ckpt() -> Path:
    cands = sorted(CKPT_DIR.glob("ckpt_*.pt"))
    numbered = [c for c in cands if re.fullmatch(r"ckpt_\d{7}\.pt", c.name)]
    if numbered:
        return max(numbered, key=lambda p: int(p.stem.split("_")[1]))
    latest = CKPT_DIR / "ckpt_latest.pt"
    if latest.exists():
        return latest
    sys.exit(f"没有找到 checkpoint：{CKPT_DIR}")


def iter_of(ckpt: Path) -> int:
    m = re.fullmatch(r"ckpt_(\d{7})\.pt", ckpt.name)
    if m:
        return int(m.group(1))
    # ckpt_latest.pt：从训练日志里抓最后一行的迭代数
    log = ROOT / f"train_{RUN}.log"
    if log.exists():
        hits = re.findall(r"^it (\d+) \|", log.read_text(errors="ignore"), re.M)
        if hits:
            return int(hits[-1])
    return 0


def run(cmd: list[str], log_path: Path) -> int:
    with open(log_path, "w", encoding="utf-8", errors="ignore") as f:
        p = subprocess.run(cmd, cwd=ROOT, env=ENV, stdout=f, stderr=subprocess.STDOUT)
    return p.returncode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--force", action="store_true", help="已存在同档快照时也重跑")
    # eval_benchmark 在本机必崩（CUDA 715），默认跳过；升级驱动后可加 --eval 重新启用
    ap.add_argument("--eval", action="store_true", help="同时跑 40 条全量评测（当前环境会崩）")
    args = ap.parse_args()

    ckpt = args.ckpt or newest_ckpt()
    if not ckpt.is_absolute():
        ckpt = ROOT / ckpt
    it = iter_of(ckpt)
    out = SNAP / f"iter_{it:06d}"
    if (out / "benchmark.json").exists() and not args.force:
        print(f"[snapshot] iter {it} 已有快照，跳过（用 --force 强制重跑）")
        return
    out.mkdir(parents=True, exist_ok=True)
    print(f"[snapshot] ckpt={ckpt.name} iter={it} -> {out}")

    # ---- 1) Appendix-F 基准（仅 --eval 时跑）--------------------------------
    n_clips, n_fail, mean_ret = "—", "—", "跳过"
    rc = 0
    if args.eval:
        rc = run(
            [str(PY), "-u", "scripts/eval_benchmark.py", "--ckpt", str(ckpt),
             "--npz-dir", "data/npz/lafan1", "--num-envs", str(args.num_envs),
             "--out", str(out / "benchmark.json")],
            out / "bench.txt",
        )
        txt = (out / "bench.txt").read_text(errors="ignore")
        m = re.search(r"(\d+) clips \| failed (\d+) \| mean norm return ([\d.]+) \| step-weighted ([\d.]+)", txt)
        if m:
            n_clips, n_fail, mean_ret = m.group(1), m.group(2), m.group(3)
        else:  # 评测进程被 CUDA 715 打断：不写假数据
            n_clips, n_fail, mean_ret = "—", "—", "评测失败"
    print(f"[snapshot] eval rc={rc} clips={n_clips} failed={n_fail} return={mean_ret}")

    # ---- 2) 录像 -----------------------------------------------------------
    roll = out / "rollouts.npz"
    rc_rec = run(
        [str(PY), "-u", "scripts/media/record_rollout.py", "--ckpt", str(ckpt),
         "--clips", *CLIPS, "--npz-dir", "data/npz/lafan1", "--out", str(roll)],
        out / "record.txt",
    )
    print(f"[snapshot] record rc={rc_rec}")

    # record_rollout 会逐条打印 "steps 已跑/总步数 return 回报"，比 eval_benchmark 稳，
    # 拿它当可靠指标（eval 常被 CUDA 715 打断）。
    rec_txt = (out / "record.txt").read_text(errors="ignore") if (out / "record.txt").exists() else ""
    per = {}
    for name, done, total, ret in re.findall(r"^(\S+)\s+steps\s+(\d+)/(\d+)\s+return ([\d.]+)", rec_txt, re.M):
        per[name] = (int(done), int(total), float(ret))
    three = (
        f"{sum(v[2] for v in per.values()) / len(per):.3f}（{len(per)}条）" if per else "未生成"
    )
    completed = sum(1 for v in per.values() if v[0] >= v[1]) if per else 0
    print(f"[snapshot] record 3-clip mean return {three}, 完整跑完 {completed}/{len(per)}")

    # ---- 3) 渲染：从动作真正开始处截 8 秒 ----------------------------------
    segs: list[str] = []
    note = "完整"
    if roll.exists():
        import numpy as np

        d = np.load(roll, allow_pickle=True)
        lens = {str(n): int(d["lengths"][i]) for i, n in enumerate(d["names"])}
        for name in CLIPS:
            if name not in lens:
                continue
            L = lens[name]
            start = ACTION_START[name]
            if L / 50.0 > start + WINDOW:
                a, b = start, start + WINDOW
            elif L > 100:  # 走不到动作段：录最后 2 秒，标注为「提前摔倒」
                a, b = max(0.0, L / 50.0 - 2.0), L / 50.0
                note = "提前摔倒"
            else:
                a, b = 0.0, max(0.5, L / 50.0)
                note = "倒地"
            segs.append(f"{name}:{a:.2f}:{b:.2f}")
        # tyro 对 list[str] 只认最后一个 --segments，必须一次把所有窗口给它
        rc_ren = run(
            [str(PY), "-u", "scripts/media/render_rollout.py", "--traj", str(roll),
             "--out-dir", str(out), "--fps", "25", "--ghost",
             "--concat", str(out / "compare.mp4"), "--segments", *segs],
            out / "render.txt",
        )
        print(f"[snapshot] render rc={rc_ren} state={note}")

    # ---- 4) 累积进度表 -----------------------------------------------------
    SNAP.mkdir(exist_ok=True)
    prog = SNAP / "progress.md"
    if not prog.exists():
        prog.write_text(
            "# lafan_8g_v1 训练进度\n\n"
            "实体机器人=策略，半透明蓝=参考动作。每档视频在该迭代的目录下（`compare.mp4` 是三条拼一起）。\n\n"
            "| 迭代 | 时间 | 归一化回报 | 失败条数 | 录像状态 | 目录 |\n"
            "|---|---|---|---|---|---|\n",
            encoding="utf-8",
        )
    import time

    stamp = time.strftime("%m-%d %H:%M")
    with open(prog, "a", encoding="utf-8") as f:
        f.write(
            f"| {it} | {stamp} | {three} | {completed}/{len(per)} | {mean_ret} | {n_fail} | {note} | `iter_{it:06d}` |\n"
        )
    print(f"[snapshot] done -> {prog}")


if __name__ == "__main__":
    main()
