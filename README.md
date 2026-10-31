# SuperMimic-Reproduced

> **English summary.** An independent from-scratch reproduction of *SuperMimic* (arXiv:2610.28480): a learned differentiable world model plus short-horizon BPTT actor-critic for Unitree G1 whole-body motion tracking, built on [mjlab](https://github.com/mujocolab/mjlab) / MuJoCo Warp. Includes the data pipeline (LAFAN1, BONES-SEED 129.9k clips), differentiable FK and analytic reward heads, the world model and imagination trainer, torchrun data-parallel training, the Appendix-F benchmark, and an interactive replay viewer. On LAFAN1 it reaches the paper's 0.82 tracking score in 1.66 h on one RTX 5090 (with anchor-relative observations; paper-observation runs are in progress). Documentation is in Chinese; see `docs/` for the paper spec, plan, review log and experiment log.

从零复现论文 **SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model**（[arXiv:2610.28480](https://arxiv.org/abs/2610.28480)，CoRL 2026）：一个学习得到的可微世界模型 + 短视野 BPTT actor-critic，用于 Unitree G1 人形全身动作跟踪。底座是 [mjlab](https://github.com/mujocolab/mjlab)（MuJoCo Warp），固定到 commit `8bb2bc9`；不使用 rsl_rl，世界模型、BPTT 训练器、多 clip 动作库、基于回报的重置采样、评测全部自己实现。

> 状态（2026-10-21）：LAFAN1 上达到论文量级（M3），BONES-SEED 全量数据管线与单卡训练完成，多卡训练已跑通；正式结果正在按论文观测口径重跑（e6/e7）。详见 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)。

## 结果速览

评测协议为论文 Appendix F：每条 clip 从首帧整条确定性播放、无噪声无域随机化，**归一化回报** = Σ每步奖励 / (0.1 × 步数)，1.0 为完美跟踪；"失败" = 训练终止条件触发。

| 实验 | 数据 | 硬件 | 观测口径 | 训练时长 | 评测回报 | 失败 |
|---|---|---|---|---|---|---|
| 论文 Fig. 4a | LAFAN1 40 clip | 1×H100 | 论文 | 1.2 h | 0.82 | – |
| e2 | LAFAN1 40 clip | 1×5090 | **论文** | 1.8 h | 0.56 | 9/40 |
| e3 | LAFAN1 40 clip | 1×5090 | +锚点相对位姿 | 1.66 h / 3.3 h | 0.824 / **0.904** | 5/40 / 3/40 |
| 论文 Table A.8 | BONES-SEED 129.5k clip | 1×H100 | 论文 | 3 h | >0.90 | 25/129,514 |
| e5 | BONES-SEED 129.9k clip | 1×5090 | +锚点相对位姿 | 4.2 h / 5.8 h | 0.934 / 0.935（2000 条子集） | 6/2000 / 5/2000 |
| e6 / e7 | LAFAN1 40 clip | 3×5090 / 1×5090 | **论文** | 进行中 | – | – |

- 世界模型 8 步开环归一化误差 0.0065–0.0099（论文 0.007–0.011）。
- 仿真吞吐 4096 环境约 2×10⁵ env-steps/s（5090）；`torch.compile` 后一次想象 fwd+bwd（批 2048，H=32）0.33 s。
- 四卡数据并行（显式梯度 all-reduce）效率 91%。

**最重要的偏离项**：论文 Table A.1 的策略观测只有 93 维本体项（无状态估计）。按此口径的 e2 评测回报封顶 0.56，根部位置/朝向奖励几乎拿不到；加入 BeyondMimic 式锚点相对观测（goal 组追加 9 维）后 e3 达到 0.904。e3/e5 的数字因此**不与论文同口径**，正式结果以论文口径的 e6/e7 及后续 BONES-SEED 全量 run 为准。

## 文档

| 文件 | 内容 |
|---|---|
| [docs/PAPER_SPEC.md](docs/PAPER_SPEC.md) | 论文全部数值、公式、协议的逐条转录；"未指明"清单 U1–U20 |
| [docs/REPRODUCTION_PLAN.md](docs/REPRODUCTION_PLAN.md) | 复现计划：目标、环境、设计决策 D1–D20、repo 骨架、分阶段步骤、测试、预算、风险 |
| [docs/REVIEW_LOG.md](docs/REVIEW_LOG.md) | 计划的自审与两轮对抗审查记录 |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | 实验日志：每个实验的配置、曲线、评测、偏离计划的决定、踩坑记录 |
| [docs/WIN_8G_TRAINING.md](docs/WIN_8G_TRAINING.md) | Windows / 8GB 显存训练方案与实测实录（RTX 5060 Laptop，含 7 条 Windows 专属坑清单） |
| [docs/talk/](docs/talk) | 讲解课件 PPT、30 分钟视频完整口播台词、TTS 分句与分镜规划（见下方"讲解资料"） |
| 论文原文 | [arXiv:2610.28480](https://arxiv.org/abs/2610.28480)（不随仓库分发 PDF） |

## 代码结构

```
src/supermimic/
  config.py              TrainConfig（tyro CLI：data / env / world_model / policy / wm_trainer / policy_trainer / schedule）
  data/                  CSV/BVH → 50 Hz npz 转换、replay buffer、仿真记录器
  envs/                  mjlab ManagerBasedRlEnv 配置：多 clip 动作命令、回报采样重置、完整状态读写（state_io）、sim context（域随机化参数）
    mdp/                 motion_library（含紧凑 fp16 库 + 在线 FK）、commands、observations、rewards、terminations
  kinematics/            G1 可微位置 + 速度 FK（torch），与 MuJoCo 对拍到 1e-10（fp64）
  world_model/           状态 71 维 MLP 动力学（速度增量 + 显式积分）、终止头、解析奖励/观测头、训练器
  imagination/           想象 rollout、Eq. 2 目标 + GAE critic 目标、BPTT 策略训练器、torch.compile 封装
  policy/                MotionEncoder(680→32) + 策略/critic MLP
  training/              主循环（Algorithm A.1）、cosine 学习率、数据并行（torchrun，显式 all-reduce）
  eval/                  Appendix F benchmark
  scripts/train.py       训练入口（sm-train）
scripts/
  data/                  csv_to_npz_batch.py、filter_bones_seed.py、build_library.py
  eval_benchmark.py      整 clip 评测；eval_push.py 推力测试；play.py 交互式 replay（含参考 ghost）
  bench_sim_speed.py / bench_imagination.py   吞吐基准
tests/                   FK / 解析头 / 环境 / 世界模型 / 想象 / 动作转换 / 紧凑库 的数值对拍与冒烟测试
```

## 环境

- Python 3.12，`uv`；torch 2.14.1+cu130、mujoco 3.16、warp 1.18、mjlab `8bb2bc9`（见 `pyproject.toml` 的 `[tool.uv.sources]`）。
- 已验证硬件：机器 A = 1×RTX 5090（驱动 580 开源内核模块）；机器 B = 4×RTX 5090（Threadripper PRO，PCIe，无 NVLink）。
- 系统 `PYTHONPATH` 若带有 ROS 的 python3.10 包会让 pytest 崩溃，所有命令前加 `env -u PYTHONPATH`。

```bash
uv sync --group dev
env -u PYTHONPATH uv run pytest tests -q
```

## 数据

- **LAFAN1**（40 条 G1 重定向 CSV，HF `lvhaidong/LAFAN1_Retargeting_Dataset`）→ `data/npz/lafan1/`，2.451 h（论文 2.45 h）。
- 国内下载备选：ModelScope `unitreerobotics/LAFAN1_Retargeting_Dataset`（宇树官方镜像，内容与 HF 版逐位一致），脚本 `download_lafan_ms.py`（需 `pip install modelscope`）。
- **BONES-SEED**（HF `bones-studio/seed`，`g1.tar.gz`，142,220 条 120 fps CSV，厘米/角度）→ 50 Hz npz → 名字过滤 + 可行性过滤 → **129,943 条 / 263.75 h**（论文 129,514 / 264.56）→ 紧凑 fp16 库 `data/library/bones_seed_compact.pt`（6.7 GB，常驻显存，目标特征与刚体状态在线 FK）。

```bash
uv run python scripts/data/csv_to_npz_batch.py --help
uv run python scripts/data/filter_bones_seed.py --help
uv run python scripts/data/build_library.py --help
```

`data/` 与 `logs/` 不入库。

## 训练

```bash
# LAFAN1，论文观测口径，单卡
env -u PYTHONPATH uv run sm-train --run-name e7 --data.npz-dir data/npz/lafan1 \
  --env.num-envs 4096 --policy-trainer.batch-size 4096 --schedule.max-hours 3

# BONES-SEED 全量，紧凑库
env -u PYTHONPATH uv run sm-train --run-name e5 --data.library data/library/bones_seed_compact.pt \
  --env.num-envs 4096 --policy-trainer.batch-size 2048 --schedule.max-hours 20

# 多卡：torchrun，一进程一卡（每 rank 自己的仿真与 buffer，梯度 all-reduce，lr × sqrt(world)）
env -u PYTHONPATH uv run torchrun --nproc_per_node 4 -m supermimic.scripts.train --run-name e6 ...
# 跳过坏卡：CUDA_VISIBLE_DEVICES=0,2,3 ... --nproc_per_node 3

# 可选：--env.anchor-obs（BeyondMimic 锚点相对观测，偏离论文），--resume <ckpt>
```

默认超参按论文 Table A.5/A.6：4096 环境、H=32、UTD 0.4、γ=0.99、critic 5 epoch × 4 minibatch、WM 每次更新 10 批。日志为 TensorBoard（`logs/<run>/tb`），每 `checkpoint-interval` 迭代存 `ckpt_XXXXXXX.pt`。

## Windows / 8GB 显存训练（实测）

整套管线已在 **RTX 5060 Laptop 8GB / Windows 11** 上完整跑通，方案与逐步实录见 [docs/WIN_8G_TRAINING.md](docs/WIN_8G_TRAINING.md)。方法本身网络很小（世界模型 8.82M + 策略 8.91M + critic 0.237M ≈ 18M 参数），8GB 显存绰绰有余，显存大头是并行仿真与 BPTT 激活：

| 配置 | 值 | 说明 |
|---|---|---|
| `--env.num-envs` | 512 | 仿真阶段显存 1.35 GB |
| `--policy-trainer.batch-size` | 512 | |
| `--policy-trainer.horizon` | 16 | 论文：配 critic 时 16 步 ≈ 无 critic 的 32 步 |
| `--env.anchor-obs` | 开 | 锚点相对观测（与 e3 同口径） |
| torch.compile | 关 | Windows 缺 MSVC/triton 编译链，纯 eager |
| 实测吞吐 | 0.6–0.8 it/s | 峰值显存 2.64 GB / 8 GB（32%） |

```bash
uv run python -m supermimic.scripts.train --run-name lafan_8g --data.npz-dir data/npz/lafan1 \
  --env.num-envs 512 --env.anchor-obs --policy-trainer.batch-size 512 \
  --policy-trainer.horizon 16 --schedule.max-hours 36 --schedule.checkpoint-interval 2000
```

Windows 专属注意（完整坑清单见 [docs/WIN_8G_TRAINING.md](docs/WIN_8G_TRAINING.md) §5）：

- 必设 `PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.8,max_split_size_mb:512"`（WDDM 假 OOM）；
- 用 `python -m supermimic.scripts.train` 代替 `sm-train`（uv console wrapper 在 Windows 上会被安全策略 SIGTERM）；
- 训练期间防系统睡眠：`train_wrapper_8g.py` 进程内 `SetThreadExecutionState`，且勿合盖（合盖仍会睡）；
- LAFAN1 国内走 ModelScope（见"数据"节），hf-mirror 不稳定时用它，5 秒下完 40 clips。

`logs/e009_pretrained/ckpt_0040000.pt`（预训练起步权重，221 MB）超 Gitee/GitHub 单文件 100 MB 限制不入库；自行训练数小时即可得到同量级 checkpoint。

## 讲解资料（课件与视频台词）

| 文件 | 内容 |
|---|---|
| [docs/talk/SuperMimic透彻讲解.pptx](docs/talk/SuperMimic透彻讲解.pptx) | 30 分钟视频配套课件：从任务定义、可微世界模型到策略训练与复现全过程的透彻拆解 |
| [docs/talk/台词_v1.md](docs/talk/台词_v1.md) | 视频完整口播台词稿（82 段，约 30 分钟） |
| [docs/talk/tts_chunks.json](docs/talk/tts_chunks.json) | 实际送 CosyVoice3 合成语音的最终分句（含 "SuperMimic→Super Mimic"、缩写拆字母等转写规则） |
| [docs/talk/分镜规划.md](docs/talk/分镜规划.md) | 分镜方案：8 类素材（仿真渲染 / 代码卡 / 真实曲线动画 / 终端卡 / 大数字卡等）逐段对应 |

> 成片 mp4（352 MB）超单文件 100 MB 限制，未随仓库分发；画面铁律与素材构成见分镜规划。

## 评测与可视化

```bash
# Appendix F 整 clip benchmark
env -u PYTHONPATH uv run python scripts/eval_benchmark.py --ckpt logs/e7/ckpt_latest.pt --npz-dir data/npz/lafan1
env -u PYTHONPATH uv run python scripts/eval_benchmark.py --ckpt logs/e5/ckpt_latest.pt \
  --npz-dir data/npz/bones_seed --clip-list data/npz/bones_seed/subset2000.txt --num-envs 1024

# 交互式 replay（原生 MuJoCo 窗口；无显示时 --viewer viser → http://localhost:8080）
env -u PYTHONPATH uv run python scripts/play.py --ckpt logs/e7/ckpt_latest.pt --clips dance1_subject1
```

## 与论文的对应及已知差异

- 实现了 Algorithm A.1 全流程、Table A.1–A.6 的观测/奖励/终止/网络/超参、Appendix F 评测协议、D 节域随机化（sim context ξ）。
- 论文未指明之处（U1–U20）的取舍记录在 `REPRODUCTION_PLAN.md` 第 4 节；实测后改动的决定记录在 `EXPERIMENTS.md`。
- 未做：速度接口与蒸馏、PPO-Sim/PPO-WM 基线、H×critic 消融、ONNX 导出、真机。
- mjlab 的 `csv_to_npz` 四元数 slerp 在相邻帧几乎相同时会提前返回（姿态滞后一帧），本仓库用自己的向量化 slerp（与 scipy 一致到 1e-9）。
- MuJoCo Warp 的 solver 溢出 printf 是编译期常量，必须在 `put_model` 后、图捕获前关闭，否则早期策略每步触发、仿真慢 10×。
- torchrun 下 MuJoCo Warp 总在第一张可见卡上，需按 rank 设置 `CUDA_VISIBLE_DEVICES`（`scripts/train.py` 自动处理）。
