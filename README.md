# π0.5 on Ascend NPU：四卡适配与训练加速

基于 [Physical Intelligence / OpenPI](https://github.com/Physical-Intelligence/openpi) 的 π0.5 PyTorch 实现，完成 **4 × Ascend 910B1** 上的全参数微调通过性验证、完整保存与新进程恢复，并研究训练加速。固定上游版本：[`215abfb217dbac7d5f1273282331b9b1866c0479`](https://github.com/Physical-Intelligence/openpi/commit/215abfb217dbac7d5f1273282331b9b1866c0479)。本项目为社区适配，保留上游 [Apache-2.0 LICENSE](LICENSE)；原项目说明见 [README_UPSTREAM.md](README_UPSTREAM.md)。

**已验证：** 完整 3.617B 模型、四卡 HCCL/DDP、真实 LIBERO 小样本更新、模型/优化器/RNG 保存恢复，以及有界动作推理。推荐的重计算优化通过全量有效梯度对照，短测耗时 **1.3038 → 1.1715 秒/步（−10.15%）**，对应吞吐 **+11.29%**。

**验证范围：** 2026-10-11 的小样本实验，不是完整 LIBERO 收敛训练或仿真成功率评测。加速针对训练；仓库不包含模型权重、数据集或运行环境。

## 适配了什么

| 文件 / 组件 | 改动及作用 |
|---|---|
| `src/openpi/models_pytorch/device.py` | 在设备探测前加载 torch_npu；统一 CPU/CUDA/NPU、HCCL/NCCL/Gloo、seed 和显存相关路由 |
| `scripts/train_pytorch.py` | DDP 使用 NPU device IDs 与 HCCL；恢复 safetensors 时经 CPU 加载；保留 CPU/CUDA 路径 |
| `src/openpi/models_pytorch/pi0_pytorch.py` | 不支持的 FP64 时间正弦嵌入在 CPU 保持原精度计算，按调用方精度传回；FP32 输出逐元素对照一致 |
| `examples/convert_jax_model_to_pytorch.py`、`npu/convert_base.py` | 复用官方 JAX→PyTorch 转换，延迟 CLI 依赖导入；检查缺失/多余权重键，仅允许共享输出层和未参与 flow head 的 expert LM head |
| `npu/prepare_libero.py` | 复用已有 LIBERO 缓存，仅取 8 个示教窗口；保留真实 RGB、文本、动作、相机 mask 与归一化约定 |
| `npu/validate_four_cards.py` | 完整模型真实更新、梯度同步抽样、参数变化、全状态保存、新进程恢复及动作输出验证 |
| `npu/profile_training.py`、`npu/optimize_training.py` | 热点分析、全量梯度诊断、checkpoint / AdamW / DDP 桶实验 |

实测入口为 `npu/validate_four_cards.py` 等有界脚本。原始 `scripts/train_pytorch.py` 已做设备适配，其完整 LeRobot/W&B/DataLoader 链路未作为本次端到端验收入口。

## 实测环境与数据契约

| 项目 | 配置 |
|---|---|
| 硬件 / 分布式 | 单节点 4 × 910B1，物理卡 0–3，HCCL/DDP |
| 运行时 | Python 3.11.10、torch 2.6.0、torch_npu 2.6.0、CANN 8.2.RC1 |
| 模型 | PaliGemma 2B + Gemma 300M action expert + SigLIP；3,616,757,520 参数全部 requires_grad |
| 精度 | 上游选定 BF16 主体和 FP32 稳定性参数，不额外 autocast |
| batch | 每卡 1，梯度累积 1，global batch 4 |
| 输入 | LIBERO spatial 缓存前 8 个 episode，每个取第 16 帧与后续 10 个动作 |
| 特征 | action horizon 10，动作填充维度 32，文本长度 200；不额外 delta 转换 |
| 归一化 | 仅由上述 8 个训练 episode 计算 q01/q99，用于通过性验证 |
| 优化器 | AdamW，betas=(0.9, 0.95)，eps=1e-8，weight_decay=1e-10，clip_norm=1，原 warmup |

原始数据来自服务器已有 OBS 资产缓存，**没有下载全量数据集**。保留平台 torch/torch_npu/CANN 配套；这些版本是实测记录，不保证适用于其他环境。

## 训练加速结果

统一 `HCCL_DETERMINISTIC=true`，每项 3 步 warmup + 8 步测量；每步取四个 rank 最慢值，再取中位数。计时包含输入搬运、前向、反向/HCCL、裁剪和优化器更新；排除加载、数值审计和 profiler。各测量块连续更新权重，不在每块重置。数据已预处理并驻留 CPU 内存。

| 配置 | 秒/步（中位数） | 比基线耗时减少 | 峰值 allocated GiB/卡 | 判定 |
|---|---:|---:|---:|---|
| 原始 checkpoint + AdamW + 25 MiB 桶 | 1.3038 | — | 32.743 | 基线 |
| **减少外层 checkpoint + AdamW + 25 MiB** | **1.1715** | **10.15%** | **32.743** | **推荐；全量有效梯度对照通过** |
| 减少外层 checkpoint + fused AdamW + 25 MiB | 1.0725 | 17.74% | 30.393 | 实验；参数/一阶动量对照未通过 |
| 减少外层 checkpoint + AdamW + 50 MiB | 1.1553 | 11.39% | 32.743 | 实验；梯度对照未通过 |
| 减少外层 checkpoint + AdamW + 100 MiB | 1.1570 | 11.26% | 32.743 | 实验；梯度对照未通过 |

推荐方案复测 1.1723 秒/步，基线复测 1.3026 秒/步。所有慢样本均保留；这是短程实测，不承诺长训练或完整视频解码链路具有同等收益。

### 重计算与梯度差异

上游逐层 checkpoint 外还有整段 Gemma 的 checkpoint 包装。实验只去掉整段外层包装，保留逐层及视觉 checkpoint。原始路径重复计算也出现梯度差异；`DDP.no_sync()` 对照与新进程启用确定性 HCCL 的对照，将主要差异定位到 HCCL 归约非确定性。

固定权重、输入和各 rank RNG，覆盖 **805 个张量、3,248,572,176 个有效梯度元素**：减少重计算与原始路径的最大绝对差 **1.3970e-9**、相对 L2 **1.6750e-8**，四卡通过原有 `atol=1e-7, rtol=1e-5`，未放宽阈值。这是严格容差内一致，不是逐位相同。

推荐组合：**确定性 HCCL + 仅移除 Gemma 整段外层 checkpoint + 标准 AdamW + 25 MiB 桶**。已通过 `--pytorch-npu-training-optimization` 接入常规 `scripts/train_pytorch.py`；模型原生支持 `gradient_checkpointing_enable(checkpoint_whole_stack=False)`。实验脚本保留 `layer_only` / `layer25` 作为历史对照。[推荐配置 JSON](npu/recommended_training.json) 是说明文件，不会自动加载。开关默认关闭，显式开启后生效。

### 在常规训练入口启用

准备好常规训练需要的 LeRobot 数据、对应 norm stats、tokenizer 及 CLI 依赖后，在原有训练命令上增加 `--pytorch-npu-training-optimization`。例如四卡、global batch=4 的两步有界运行：

```bash
source .venv/bin/activate
source npu/environment.sh
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/train_pytorch.py pi05_libero \
  --exp-name pi05_npu_optimized_001 \
  --pytorch-weight-path /path/to/converted-pi05-weights \
  --batch-size 4 --num-train-steps 2 --num-workers 0 \
  --no-wandb-enabled --pytorch-npu-training-optimization
```

`pi05_libero` 的默认数据 repo 是 `physical-intelligence/libero`；请在配置中或通过 `--data.repo-id` 指定自己的本地 LeRobot 资产对应 repo，并准备该配置要求的归一化统计。上面的命令不把 `samples.pt` 当作 LeRobot 数据集；小样本缓存回归使用文末独立脚本。平台 overlay 依赖列表用于有界验证，常规 CLI 还需要上游的 tyro、wandb、LeRobot 等依赖；安装时保留现有 torch/torch_npu/CANN。

开关会在 NPU/HCCL 初始化前设置 `HCCL_DETERMINISTIC=true`，显式使用 25 MiB DDP 桶，保留标准 AdamW。冲突的 HCCL 环境值、已初始化的设备/进程组、非 NPU 或非 π0.5 模型会明确报错。启动日志会打印实际优化策略，W&B 配置包含该参数。恢复运行时继续传同一开关；它不改变模型参数名或优化器格式。删除开关或使用 `--no-pytorch-npu-training-optimization` 可回到原始 checkpoint 路径。

常规入口数据链路的端到端吞吐和保存恢复需要在对应环境另行验收；前述 10.15% 是相同模型策略在缓存小样本实验中的历史测量，不是新 CLI 运行的保证值。

### 融合优化器与大桶的边界

- 当前 `NpuFusedAdamW` 拒绝 BF16；`torch.optim.AdamW(fused=True)` 在该环境支持混合 BF16/FP32 原生 NPU 执行。后者额外降低耗时约 8.45%，节省约 2.35 GiB/卡，但同梯度三步探针的参数和一阶动量未通过严格阈值，尚无长期收敛或 fused 完整保存恢复结论。
- 50/100 MiB 桶仅比推荐方案额外降低耗时约 1.38% / 1.24%，却未通过严格梯度对照。确定性 HCCL 不保证不同桶配置采用相同浮点归约顺序，因此保留 25 MiB。

## 复现入口

### 1. 源码与环境

```bash
git clone https://github.com/Wanch9/pi05-npu.git
cd pi05-npu
# 使用已验证 torch/torch_npu/CANN 配套的 Python 3.11；不要用 uv sync 替换平台 torch。
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --no-deps -r npu/requirements-overlay.txt
python npu/install_transformers_overlay.py
source npu/environment.sh
```

`--no-deps` 保留平台运行时，需要已有基础依赖。完整实测版本见 [validation_summary.json](npu/validation_summary.json) 的 `runtime.versions`（例如 numpy 1.26.4、safetensors 0.5.3、pyarrow 20.0.0、av 15.1.0、sentencepiece 0.2.0）。缺失项按目标机环境补齐。如果 transformers 来自共享 site-packages，先用 `python -m pip install --ignore-installed --no-deps transformers==4.53.2` 安装到当前 venv。overlay 仅修改当前隔离 venv，源文件随仓库提供。

### 2. 准备已有资产和 8 个窗口

替换下列路径。基座来源为官方 `gs://openpi-assets/checkpoints/pi05_base`；tokenizer 使用官方 `paligemma_tokenizer.model`。无需下载完整 LIBERO 数据，准备 episode 000000–000007 的 parquet、双相机视频和 `meta/tasks.jsonl` 即可。

```bash
export PI05_BASE=/path/to/pi05_base
export PI05_LIBERO=/path/to/libero_spatial_no_noops_lerobot
export PI05_TOKENIZER=/path/to/paligemma_tokenizer.model
export PI05_WORK=/path/to/new-pi05-work
mkdir -p "$PI05_WORK"
python npu/convert_base.py --source "$PI05_BASE" --output "$PI05_WORK/weights"
python npu/prepare_libero.py --root "$PI05_LIBERO" \
  --tokenizer "$PI05_TOKENIZER" --output "$PI05_WORK/samples"
```

### 3. 四卡真实更新与保存恢复

确认目标卡 0–3 空闲。每次尝试用全新输出目录；完整模型约 14.47 GB，每份模型及优化器 checkpoint 约 22.00 GB，还需足够主存与缓存磁盘空间。

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  npu/validate_four_cards.py --weights "$PI05_WORK/weights/model.safetensors" \
  --data "$PI05_WORK/samples" --output "$PI05_WORK/train001" --max-steps 1
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  npu/validate_four_cards.py --weights "$PI05_WORK/weights/model.safetensors" \
  --data "$PI05_WORK/samples" --resume "$PI05_WORK/train001/checkpoint.pt" \
  --output "$PI05_WORK/resume001" --max-steps 2 --inference
```

实测首次平均 loss 0.144705，新进程恢复更新后 0.124039；805 组优化器状态恢复，参数实际变化，跨卡抽样梯度差为 0。动作输出 `[1, 10, 32]` 且有限。两步 loss 只证明通过性，不能证明效果提升。

### 4. 重跑训练优化实验

```bash
export HCCL_DETERMINISTIC=true
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  npu/optimize_training.py --mode all \
  --weights "$PI05_WORK/weights/model.safetensors" \
  --data "$PI05_WORK/samples" --output "$PI05_WORK/optimize001"
```

`--mode all` 包含诊断及全部候选，完成 77 次真实更新/卡，不保存训练 checkpoint、不执行推理。`--mode diagnose` 只诊断，`--mode benchmark` 只测速。热点采集可用同样路径参数运行 `npu/profile_training.py`。脱离终端运行可用 `npu/run_recorded.py start --help` 查看监督器，读取其 `result.json` 中的真实退出码。

## 报告、证据与未测范围

- [NPU_PORTING.md](NPU_PORTING.md)：完整适配记录、首轮 profile、失败诊断、最终优化结果。
- [validation_summary.json](npu/validation_summary.json)：训练、保存恢复、推理和视觉数值对照。
- [optimization_summary.json](npu/optimization_summary.json)：逐步计时及数值比较；[evidence](npu/evidence) 保留脱敏后的原始回执和失败记录。
- `python -m unittest npu.test_training_optimization -v`：CPU 可运行的初始化顺序与回退检查；`npu/check_training_optimization.py`：四卡完整模型的新开关对历史实验实现的梯度/真实更新回归，接收 `--weights`、`--data`、`--output`。
- `python npu/summarize_optimization.py` / `python npu/summarize_profile.py`：在 CPU 上从随仓库历史证据重建汇总，不启动训练。
- [PUBLICATION.md](npu/PUBLICATION.md)：原始运行哈希与发布修订的区别；[publication_manifest.json](npu/publication_manifest.json) 记录发布文件哈希。
- [NPU 适配 skill](https://github.com/Wanch9/ascend-npu-porting)：通用方法和 π0.5 案例。

未验证：完整 DataLoader/视频解码吞吐、长程收敛、LIBERO 仿真成功率、完整 CUDA 梯度对照、fused 优化器全状态恢复，以及其他硬件/卡数。原始保存恢复验收使用基线路径；加速路径完成数值对照和短程真实更新，未另做加速路径 checkpoint 恢复验收。首次发布仅调整路径参数；后续常规入口接入另做四卡缓存样本回归，见 NPU_PORTING.md。历史加速比例未作为新 CLI 的测速结果。历史 `/cache` 产物不等于持久备份，本轮没有配置 OBS 输出备份。
