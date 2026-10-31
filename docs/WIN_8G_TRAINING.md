# 方案设计：8G 显存训练 SuperMimic（RTX 5060 Laptop）

日期：2026-10-27
硬件：本机 RTX 5060 Laptop 8GB（Blackwell sm_120，Windows 11）
目标：绕开 SONIC 发布权重 "SuperMimic tracker 禁止提取" 的 LICENSE 限制，**自训一份 G1 motion-tracking policy**（50Hz，与 SONIC decoder 同构，可直接接入现有 50Hz decoder + 500Hz PD 闭环管线）。

---

## 1. 背景与可行性论证

SuperMimic（arXiv:2610.28480，CoRL 2026 Oral，Berkeley）= 单个可微世界模型 + 短视野 BPTT actor-critic。论文在 1×H100 上 3 小时超过 SONIC 释放检查点（0.90），2×H100 9.8h 全库零失败。官方代码未开源，但有高质量独立复现：

- **CuboiLeo/SuperMimic-Reproduced**（2026-10-21 起）：mjlab / MuJoCo Warp 实现，含数据管线、可微 FK、世界模型、想象训练器、Appendix-F 基准、回放查看器。单卡 5090 复现 LAFAN1 0.904（加锚相对观测）/ 论文口径 0.56–0.59。

**为什么 8G 可行**：整个方法的网络全是小 MLP——
- 世界模型 = heading-canonical 特征上的 MLP 动力学 + 终止头 + 经可微 FK 的解析奖励/观测头
- 策略 = MotionEncoder (680→32) + policy/critic MLPs

参数量合计不到几百 MB。显存大头是三个**可线性缩放**的项：

| 显存项 | 论文/复现配置 | 8G 裁剪 | 估计占用 |
|---|---|---|---|
| 动作库 fp16 常驻 | BONES-SEED 264h = **6.7GB** | **LAFAN1 2.45h ≈ 70MB** | ~0.1GB |
| MuJoCo Warp 并行仿真 | 4096 envs | **1024 envs** | ~1.0–1.5GB |
| 想象 BPTT 激活（H=32） | batch 2048 | **batch 1024** | ~1–2GB |
| critic GAE/TD 缓冲 | — | 同步缩 | ~0.5GB |
| 模型参数+优化器 | MLP | 不变 | ~0.2GB |
| CUDA context + WDDM 保留 | — | Windows 固有 | ~1.0–1.5GB |
| **合计** | 24G 卡 | — | **≈ 4.5–6.5GB / 8GB ✓** |

换 LAFAN1 同时也是论文口径的正统实验：论文自己就在 LAFAN1 40 条 clip 上跑出 0.82（1.2h@H100）。

## 2. 关键配置定版（8G 档）

| 参数 | 复现默认 | 8G 配置 | 说明 |
|---|---|---|---|
| 数据集 | BONES-SEED 129k clips | **LAFAN1 40 clips** | HF `lvhaidong/LAFAN1_Retargeting_Dataset` 的 `g1/` |
| `env.num-envs` | 4096 | **1024** | Warp 仿真显存近似线性 |
| `policy-trainer.batch-size` | 4096 | **1024** | OOM 再降到 512 |
| BPTT H | 32 | **32**（不动） | 论文：配 critic 时 16 步≈无 critic 32 步，OOM 备选 H=16 |
| UTD | 0.4 | 0.4 | 不动 |
| γ / TD(λ) | 0.99 / 0.95 | 不动 | |
| 学习率 | 余弦→0.03×@250k | 不动，但实际跑不到那么久 | 绝对值查 `src/supermimic/config.py` |
| 迭代数 | 100k+ | **max-hours 24–40** | Laptop 吞吐决定，checkpoint 按小时滚存 |
| torch.compile | 开 | 视 Windows triton 而定 | 见 §5 坑 3 |

## 3. 分阶段实施

### 阶段 0：环境（约 30 分钟）

```bash
cd D:/codes
git clone https://github.com/CuboiLeo/SuperMimic-Reproduced.git
cd SuperMimic-Reproduced
# uv 管理，Python 3.12 独立环境（不动系统 Python313）
uv sync --group dev
# RTX 5060 = sm_120，仓库固定 torch 2.14.1+cu130，原生支持 Blackwell，无需改动
env -u PYTHONPATH uv run pytest tests -q   # 数值一致性冒烟
```

Windows 注意：bash 里给 uv/python 传路径用 `D:/` 形式；系统 `PYTHONPATH` 污染时命令前加 `env -u PYTHONPATH`。

### 阶段 1：数据（约 20 分钟）

```bash
# 国内走 hf-mirror
export HF_ENDPOINT=https://hf-mirror.com
# 下载 LAFAN1_Retargeting_Dataset 的 g1/ 文件夹（40 条 30fps CSV，共 2.45h）
uv run python scripts/data/csv_to_npz_batch.py --help   # CSV→50Hz npz
# 产物落 data/npz/lafan1/
```

不需要 BONES-SEED 的过滤/建库步骤（那是 6.7GB 常驻的来源）。

### 阶段 2：冒烟（先跑 30–60 分钟验证不 OOM、reward 上升）

```bash
export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.8,max_split_size_mb:512"
uv run python scripts/train.py sm-train \
  --env.num-envs 512 \
  --policy-trainer.batch-size 512 \
  --schedule.max-hours 0.5 \
  --output runs/smoke_8g
```

跑通且 reward 上行后升正式档。

### 阶段 3：正式训练（约 24–48h，2–3 天）

```bash
uv run python scripts/train.py sm-train \
  --env.num-envs 1024 \
  --policy-trainer.batch-size 1024 \
  --schedule.max-hours 40 \
  --output runs/lafan_8g
```

- 开跑前进程内调 `SetThreadExecutionState(0x80000001)` 防系统睡眠（合盖挡不住，保持开盖或改电源计划）。
- 每 1–2 小时滚存 checkpoint；**长任务以磁盘产物为准**，中断后从 checkpoint resume。
- 显存若还有余量 → envs 升到 2048；若 OOM → batch 512 + H=16（论文证明配 critic 等效）。

### 阶段 4：评估

```bash
# Appendix-F 基准：40 条 LAFAN1 clip 各播一次，统计 tracking score / 失败数
uv run python scripts/eval/...            # 仓库 eval 模块
# 渲染目检：回放查看器 / 媒体录制脚本
```

**验收线**：tracking score ≥0.70 及格、≥0.85 良好（复现 5090 拿 0.904）；失败 clip ≤6/40。8G+Laptop 算力下取 0.7–0.9 区间为合理预期。

## 4. 吞吐与时长预估

- 复现：5090@4096envs ≈ 2×10⁵ env-steps/s；想象 fwd+bwd（batch2048,H32）0.33s
- 5060 Laptop 算力约 5090 的 1/6–1/7，envs 减到 1024 → 估 **1.5–3×10⁴ env-steps/s**
- LAFAN1 复现 0.904 用时 3.3h@5090 → 5060 约 **20–40h**。可接受，且可中途 checkpoint 直接投入使用。

**提速备选**：172.16.200.10（A100×8）同配置训练约 2–5h 出结果。本机 8G 方案的价值在零成本随时跑 + 管线验证；冲指标时把同一份代码 rsync 到 200.10 跑即可（Linux 下注意 `setsid env ... </dev/null &` 后台惯例、NO_PROXY）。

## 5. 8G / Windows 专属坑清单（按命中概率排序）

1. **WDDM 假 OOM**：必设 `PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.8,max_split_size_mb:512"`（本机已实锤有效）。
2. **系统自动睡眠**：训练停顿≠卡死，先查睡眠。`SetThreadExecutionState(0x80000001)` 进程级防睡。
3. **torch.compile 依赖 triton**：Windows 原生没有 triton，先 `uv pip install triton-windows`；仍失败则关闭 compile（想象封装在 `src/supermimic/imagination/`，慢 2–3 倍但能跑）。
4. **MuJoCo XML 编译 qhull 临时堆 ~800MB 不归还**：吃的是**系统内存**，本机 commit 余量仅 ~3.7GB——训练机不要同时开多进程渲染/服务。
5. **HF 下载**：`HF_ENDPOINT=https://hf-mirror.com`；VPN 残留代理时 curl 加 `--noproxy '*'`。
6. **后台长驻 stdout 禁接管道**：日志直接 `> train.log 2>&1`，长任务以 checkpoint/log 磁盘产物为准。
7. **50Hz 参考帧约定**：一条 50Hz 控制步 = Warp 内 4×5ms 子步，评估和部署时别把物理频率和控制频率搞混。

## 6. 与本项目的衔接

- 产出 = 50Hz tracking policy，与 sonic_decoder 闭环同构（50Hz decoder + 500Hz PD）。接管 `D:/codes/g1_step_move` / 乒乓球管线时，沿用三大已知坑：position 执行器禁用须 gain+bias 全清；free joint qvel[3:6] 本就是体系角速度；decoder obs 需重排序（`--jperm m2i` 对齐自训权重的 obs 顺序需自行导出）。
- LAFAN1 主要是舞蹈/走跑，对乒乓球步态（SLIDE 横移、HOP 小跳）覆盖有限。二期可把 g1_step_move 的步态轨迹重定向为 G1 npz 追加进动作库，用「基于回报的重置采样（1s 分桶）」机制做增量 finetune。

## 7. 风险与对策

| 风险 | 概率 | 对策 |
|---|---|---|
| 1024envs+batch1024 仍 OOM | 中 | batch 512、H=16+critic、envs 512 逐级降 |
| 复现论文口径只有 0.56–0.59 | 高（复现作者实测） | 加锚相对观测（其 0.904 配置，标注不可比）；或接受 tracking 质量，转二期 finetune |
| Laptop 散热降频，时长翻倍 | 中 | max-hours 上限放宽到 60h；或直接切 A100 |
| Windows triton/compile 不兼容 | 中 | 关 compile，纯 eager 跑 |
| 复现仓库还在快速迭代 | 低 | 锁定当前 commit；报告问题给作者 |

---

## 8. 实施实录（2026-10-27 晚，全部实测）

### 环境实况
- 仓库 `D:/codes/SuperMimic-Reproduced`（commit d9be385，单 commit 新仓库）
- uv 0.13.6 + Python 3.12.14 venv，torch 2.14.1+cu130 **原生识别 RTX 5060（sm_120）**，mujoco 3.16.0 / warp 1.18.0 / mjlab@8bb2bc9
- pytest：7 passed / 17 skipped / 0 failed

### 网络规模实锤（比预估更小）
- world model **8.82M** + policy **8.91M**（anchor-obs 版）+ critic **0.237M** ≈ 18M 参数
- 512 envs 仿真阶段显存 **1.35GB**；正式训练（512 envs + BPTT）峰值 **2.64GB / 8.15GB（32%）**

### 数据
- hf-mirror 当晚挂了（直连/代理全超时），改用 **ModelScope `unitreerobotics/LAFAN1_Retargeting_Dataset`**（宇树官方），5 秒下完 40 clips / 87MB
- 与 HF lvhaidong 版**数值逐位一致（max diff = 0，仅换行符差异）**；csv_to_npz 40/40 成功，2.451h 与 README 精确一致

### eager 决策（Windows 无 VS Build Tools）
- torch.compile 依赖 triton → 装 `triton-windows 3.8.0` 后仍需 MSVC 编译 kernel，本机无 cl.exe → **改 eager**
- tyro 坑：`--wm-trainer.compile none` 会把 "none" 当字符串传给 torch.compile 报错 → 直接改 `wm_trainer.py` / `policy_trainer.py` 默认值为 `None`（两处，带注释可还原）

### 吞吐实测（eager）
| 配置 | it/s | 说明 |
|---|---|---|
| 256 envs / batch 256 / H32 | 0.5 | 微型冒烟，训练循环全通 |
| 512 envs / batch 512 / H16 / anchor-obs | **0.6–0.8** | 稳态；36h ≈ 7–8.5 万 iters |
| 5090 compiled（作者） | ~8.4 | 0.904@100k it 用 3.3h |

- 每迭代时间分布：sim 0.04s / wm 0.7s / pi(BPTT) 1.4s（H16 batch512）；GPU util 9–18%（CPU bound）
- **学习信号正常**：wm nse1 0.19→0.06、nse8 1.11→0.19；imag_ret 0.09→0.40；surv 0→0.7（前 350 it）

### 正式训练（已启动）
- 启动方式：**explorer.exe 启动 run_lafan_8g_v1.bat**（Job Object 放行路径，PowerShell 工具调）
- 防睡眠：train_wrapper_8g.py 进程内 SetThreadExecutionState(0x80000001)——**合盖仍会睡，勿合盖**
- 命令：`--env.num-envs 512 --env.anchor-obs --policy-trainer.batch-size 512 --policy-trainer.horizon 16 --schedule.max-hours 36 --schedule.checkpoint-interval 2000`
- 产物：`logs/lafan_8g_v1/`（每 2000 it 存 ckpt_latest.pt + ckpt_0002000.pt），日志 `train_lafan_8g_v1.log`
- 恢复：`--resume logs/lafan_8g_v1/ckpt_latest.pt`

### 已知问题
- it 250→300 出现 45 分钟墙钟空档（疑似系统睡眠，合盖所致）；恢复后速率正常。若长期跑务必保持开盖或改电源计划（需管理员）。
- sm-train.exe（uv console wrapper）被安全策略 SIGTERM，一律用 `python -m supermimic.scripts.train` 代替。
