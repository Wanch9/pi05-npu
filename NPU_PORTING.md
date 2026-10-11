# π0.5 四卡 NPU 适配

本目录基于 Physical Intelligence 官方 OpenPI，固定源码提交为
`215abfb217dbac7d5f1273282331b9b1866c0479`。
本轮目标是完整模型的全参数微调通过性验证，不是长训练或 LIBERO 成功率评估。
2026-10-11 已通过四卡训练、完整保存、新进程恢复更新和动作推理验证。
机器可读结果见 `npu/validation_summary.json`，发布用脱敏证据见 `npu/evidence/final001.json`。
发布副本的路径参数、证据脱敏及历史哈希边界见 [PUBLICATION.md](npu/PUBLICATION.md)；可移植运行命令见 [README.md](README.md)。

## 实测结果

| 阶段 | 结果 |
|---|---|
| 官方基座转换 | `convert001` 退出码 0；PyTorch 权重 14,467,165,872 字节 |
| 四卡通信 | FP32、BF16 HCCL all-reduce 均正确，`hccl001` 退出码 0 |
| 首次更新 | `train001` 四个 rank 均通过，平均 loss 0.144705，退出码 0 |
| 恢复更新 | 新的四个进程严格恢复模型和 805 组优化器状态，计数从 1 到 2；平均 loss 0.124039，`resume001` 退出码 0 |
| 更新证据 | 两步梯度和优化器矩均有限；跨 rank 抽样梯度最大差为 0；均测到参数实际变化 |
| 保存 | 两个完整 checkpoint 各 22,000,460,514 字节 |
| 推理 | 四个 rank 均完成 10 步去噪，输出 `[1, 10, 32]` 且全部有限 |
| 显存 | 每卡峰值分配 32.7444 GiB；验证后 NPU 0–3 无运行进程 |

模型共有 3,616,757,520 个参数，全部设置为可训练；实际参与动作损失的 805 个参数张量产生梯度。
两步使用不同样本，loss 用于有限性检查，不代表任务效果提升。
原始 warmup 学习率约为 `5e-9`、`1e-8`；部分 BF16 注意力权重抽样变化为零，
FP32 动作投影、归一化和视觉 patch embedding 等参数测得实际更新。
短步耗时约 16–17 秒，包含本次运行开销，不作为稳态吞吐基准。

真实图像的 SigLIP 对照采用 CPU FP32 参考，统计量在 CPU FP64 汇总：

| 路径 | 相对 L2 误差 | 余弦相似度 |
|---|---:|---:|
| NPU FP32 | 0.0006183305 | 0.9999998089 |
| NPU 选定 BF16 | 0.0231799446 | 0.9997313619 |

时间嵌入转为 FP32 后逐元素一致。`numeric002` 退出码 0，替代初次报告的 FP32 累加统计。
训练快照为 `source008.tar.gz`，其 manifest SHA-256 为
`3d893f7c25705428d3ac38f669def0b72081c3398d6b7f40a3c2446386214893`。
原始工作区曾核对训练、模型、数据准备和环境代码与该实测快照一致；发布副本的少量路径参数调整未重新进行 NPU 实测。

## 本轮配置

- worker-0，物理 NPU 0、1、2、3，HCCL/DDP。
- 原始 π0.5 基座完整加载，保持 PaliGemma 2B、Gemma 300M 动作专家和 SigLIP。
- 与官方 PyTorch 实现一致的 BF16 主体及 FP32 稳定性参数，不使用 autocast；全参数可训练。
- 每卡 1 个样本，累积 1，全局 batch 4；第一次执行完成 1 次更新，独立新进程恢复后完成第 2 次更新。
- 官方 LIBERO 模型配置：动作长度 10、填充动作维度 32、文本长度 200、`discrete_state_input=False`、不额外做 delta 转换。
- 使用服务器现有 OBS-backed LIBERO spatial 缓存的前 8 个示教，各取第 16 帧和后续 10 个动作。
  仅用这 8 个示教计算归一化统计；未下载数据集，也不遍历全部训练数据。
- 采用官方 `pi05_libero` AdamW 和学习率 warmup 配置。很小的首步学习率可能使 BF16 参数更新低于舍入间隔，验收单独测量真实参数变化。

完整配置见 `npu/contract.json`。

## 路径

| 内容 | 服务器位置 |
|---|---|
| 代码与监督记录 | `/home/ma-user/pi05-npu-20261011/` |
| 独立环境 | `/home/ma-user/pi05-npu-20261011/venv` |
| 源码 | `/home/ma-user/pi05-npu-20261011/source` |
| 小样本 | `/cache/pi05-npu-20261011/libero-eight-v2` |
| 原始基座 | `/cache/pi05-npu-20261011/base-assets/checkpoints/pi05_base` |
| 转换权重 | `/cache/pi05-npu-20261011/pi05-base-pytorch/model.safetensors` |
| 首次完整 checkpoint | `/cache/pi05-npu-20261011/train001/checkpoint.pt` |
| 恢复后完整 checkpoint | `/cache/pi05-npu-20261011/resume001/checkpoint.pt` |
| 推理输出 | `/cache/pi05-npu-20261011/resume001/rank-{0,1,2,3}-actions.pt` |
| 官方来源 | `gs://openpi-assets/checkpoints/pi05_base` |
| LIBERO 原始缓存（只读） | `/cache/fastwam-assets/data/libero/libero_spatial_no_noops_lerobot` |

本轮输出保存到服务器；没有配置 OBS 输出备份，`/cache` 不应被当作持久备份。

## 环境与改动

保留平台 Python 3.11、torch 2.6.0、torch_npu 2.6.0、CANN 8.2.RC1。
独立 venv 复用已有运行时，只安装 `npu/requirements-overlay.txt` 中的固定依赖。
官方 `transformers_replace` 仅复制到本轮 venv 中的 transformers 4.53.2。

- `device.py`：显式 CPU/CUDA/NPU 路由，并在探测设备前导入 torch_npu。
- `train_pytorch.py`：增加 HCCL、NPU device IDs 和设备相关内存接口；NPU 恢复 safetensors 时先由 CPU 加载。
- `pi0_pytorch.py`：NPU 不支持的 FP64 时间正弦嵌入在 CPU 保持原始计算，再按照原调用处的 dtype 转换后传回。
- Beta 时间采样使用 torch_npu 原生 CPU 回退；本轮保持该行为，尚未做性能优化。
- `convert_base.py`：调用上游 JAX→PyTorch 转换，检查缺失/多余键；只允许共享输出层或不参与 flow head 的 expert lm_head 缺失。
- `prepare_libero.py`：真实 RGB、动作和文本小样本准备，保持官方相机与缺失腕部图像掩码约定。
- `validate_four_cards.py`：记录有效更新、DDP 同步梯度抽样、参数变化、完整 checkpoint 和恢复后的更新。

原始 CLI 的完整 LeRobot/W&B 依赖链未作为本轮验收入口。本轮可重建、实测的入口是下面的小样本验证脚本。

## 执行入口

在服务器源码目录执行；正式运行前需确认四张目标卡空闲，权重转换已成功，并为每次尝试选择全新输出目录。

```bash
source npu/environment.sh
python npu/convert_base.py \
  --source /cache/pi05-npu-20261011/base-assets/checkpoints/pi05_base \
  --output /cache/pi05-npu-20261011/pi05-base-pytorch

python -m torch.distributed.run --standalone --nproc_per_node=4 \
  npu/validate_four_cards.py \
  --weights /cache/pi05-npu-20261011/pi05-base-pytorch/model.safetensors \
  --data /cache/pi05-npu-20261011/libero-eight-v2 \
  --output /cache/pi05-npu-20261011/train001 --max-steps 1

python -m torch.distributed.run --standalone --nproc_per_node=4 \
  npu/validate_four_cards.py \
  --weights /cache/pi05-npu-20261011/pi05-base-pytorch/model.safetensors \
  --data /cache/pi05-npu-20261011/libero-eight-v2 \
  --resume /cache/pi05-npu-20261011/train001/checkpoint.pt \
  --output /cache/pi05-npu-20261011/resume001 --max-steps 2 --inference
```

实际执行使用目标机 `npu/run_recorded.py` 监督器管理上述前台命令，保存日志和真实退出码，脱离 SSH 生命周期。

## 验收边界

每个 rank 的 `identity`、`loaded`、`update`、`complete` 报告记录对应证据。
参数变化仅为指定主要分支的抽样，不能描述为全部参数逐元素审计。
checkpoint 包括模型、优化器、计数、学习率配置、各 rank RNG 和确定性样本游标。
恢复验收必须由新进程严格加载并完成下一次真实更新。

`numerical_check.py` 按实际 batch=1、SDPA 视觉注意力，比较真实图像上 SigLIP 的 CPU FP32、NPU FP32 和选定 BF16 路径，并验证时间嵌入转换后的逐元素一致性。
这些结果不能代替完整 CUDA 对照或 LIBERO 仿真成功率；本轮不宣称任务效果已验证。

## 首轮训练性能分析（2026-10-11，profile002；历史记录）

本节保留首轮未解决的数值差异与后续候选；最终诊断和结论以文末“继续训练优化”为准。

四张 910B1（0–3），全参数训练，global batch=4、每卡 1，保持原 BF16/FP32 参数分配、AdamW 和 warmup。复用已有 LIBERO 八个 episode 的八个缓存窗口，没有下载数据。进程真实退出码为 0，四个 rank 各完成 26 次优化器更新；不保存训练 checkpoint，不执行推理。结束后四卡均无运行进程。

计时包含输入搬运、前向、反向/HCCL、梯度裁剪和优化器更新；前后同步 NPU，每步开始前 barrier 不计入。模型加载约 291 秒、数值检查、warmup、日志、profiler 开销排除。数据已经预处理并驻留 CPU 内存，因此结果不代表完整 DataLoader、视频解码或长时间训练吞吐。

### 四卡稳态计时

先取每一步四个 rank 的最大耗时，再取步骤中位数；没有删除慢样本。吞吐为 global batch 除以中位数，仅表示该中位数对应的速率。基线和候选各 3 步 warmup、5 步测量；基线复测 1+3 步，逐步审计测量 2 步。

| 路径 | 中位数（秒/步） | 测量范围 | 中位数对应样本/秒 | 峰值 allocated GiB/卡 |
|---|---:|---:|---:|---:|
| 原始训练路径 | 1.3211 | 1.3078–2.1292 | 3.028 | 32.746 |
| 原始路径复测 | 1.3150 | 1.3122–1.3435 | 3.042 | 32.746 |
| 仅保留逐层及视觉 checkpoint（实验） | 1.1702 | 1.1652–1.1741 | 3.418 | 32.746 |
| 原始路径加逐步验收审计 | 1.4851 | 1.4326–1.5376 | 2.693 | 32.746 |

候选中位数耗时降低 **11.4%**，对应吞吐增加 **12.9%**。基线第一组有一个 2.129 秒慢步骤，均值 1.4799 秒；复测均值 1.3235 秒。样本数较少，应以约 11% 的短测收益理解，不能宣称长期稳定收益。计时块连续更新权重，没有每块重新加载；数值对比另行固定权重、输入和 RNG。

逐步验收审计使中位数增加约 12.4%，包括全量梯度和优化器矩检查、梯度抽样跨卡一致性、参数变化等；这些不是普通训练每一步都必须执行的操作。前次冷启动验收的 16–17 秒/步不能用作稳态基线。

### 时间花在哪里

基线另测一个阶段同步步骤，以下为各阶段四卡中位数，百分比约值用于定位热点；阶段同步会改变计算/通信重叠，不能当作所有训练步骤的精确分解。

| 阶段 | 耗时 | 约占比 |
|---|---:|---:|
| 输入搬运 | 1.3 ms | 0.1% |
| 前向 | 291 ms | 22% |
| 反向与 HCCL | 821 ms | 62% |
| 梯度裁剪 | 61 ms | 5% |
| AdamW 更新 | 141 ms | 11% |

候选阶段同步步骤出现 1.789 秒异常慢步，某 rank 的前向等待、其他 rank 的反向等待明显增大，原因未定位；保留原始记录，不用这一单步推断候选阶段加速比例。

rank 0 的 CPU+NPU Level1 时间线记录了 25,394 个 device kernel，主要热点如下。下列数值是该次带 profiler 的 kernel 累计时长，包含重叠，不能直接换算为上述稳态 wall time 占比。

| Kernel 类型 | 次数 | 累计耗时 |
|---|---:|---:|
| AllReduce | 123 | 238.4 ms |
| MatMulV2 | 3,362 | 142.3 ms |
| Transpose | 1,643 | 43.5 ms |
| Mul | 2,802 | 37.9 ms |
| Addcdiv | 805 | 37.7 ms |
| Cast | 5,156 | 35.3 ms |

采集窗口约 1.958 秒，报告计算 540 ms、通信 238 ms，其中未与计算重叠约 186 ms，Free 约 1,232 ms。采集显著增加了主机开销，Free 也可能包括等待，**不能据此宣称正常训练有 63% 可直接消除的空闲时间**。前向、反向、裁剪和优化器 scope 均已生成；但 profiler 报告在 RECORD 状态退出，以及两条内存记录不完整。因此该单次时间线仅支持热点定位，不作为无缺失的精确利用率报告；稳态计时不依赖 profiler。

### 加速优先级与数值边界

1. **减少嵌套 checkpoint 重计算：已有约 11% 耗时改善，尚未通过数值验收。** 当前整段 Gemma 外层 checkpoint 与逐层 checkpoint 同时存在。实验只在 profiling 进程中跳过整段外层包装，保留逐层和视觉 checkpoint；峰值显存没有上升。正式模型源码未启用该改动。
2. **融合 AdamW 与裁剪相关小算子：值得下一轮 A/B。** 优化器本身约 141 ms，trace 中有 805 次 Addcdiv 和多组 foreach kernel。官方提供 [NpuFusedAdamW](https://www.hiascend.com/document/detail/zh/Pytorch/700/apiref/apilist/ptaoplist_001378.html)；尚未验证本环境下 BF16/FP32 混合参数、优化器状态恢复与数值一致性，也没有测得收益。
3. **HCCL 通信与 DDP 桶调优：值得测量，收益未知。** 单步观察到 123 次 AllReduce；可单变量扫描 bucket 大小并验证通信重叠。[PyTorch DDP 文档](https://docs.pytorch.org/docs/main/generated/torch.nn.parallel.DistributedDataParallel.html)说明了 bucket 大小与重叠机制。当前有实际未用的 expert LM head，不能直接关闭 unused-parameter 检测；static graph 也需要先验证图结构稳定性。
4. **Gemma 联合注意力融合：后续候选。** 代码中仍使用显式 matmul/softmax 注意力；视觉分支已在本次 trace 出现 FlashAttentionScore/Grad。联合注意力需要保留跨模态 mask、GQA、位置编码和精度语义，不能承诺简单替换即可获得收益。本文没有测量此项。

数值对比在相同权重、输入、各设备 RNG 下执行，无优化器更新夹在比较之间；先运行一次 backward 稳定 DDP，再对照原始、候选、原始复测。每个有效参数张量抽前 64 个梯度值，共 51,488 个元素，并比较全部有效梯度张量的范数：

| 固定状态对比 | loss 差 | 梯度抽样最大绝对差 | 梯度抽样相对 L2 | 严格 allclose |
|---|---:|---:|---:|---|
| 候选 vs 原始 | 0 | 6.1035e-5 | 7.4322e-4 | 未通过 |
| 原始复测 vs 原始 | 0 | 1.5259e-5 | 4.7766e-4 | 未通过 |

阈值保持 atol=1e-7、rtol=1e-5，没有放宽。原始路径自身也有微差，但这不足以证明候选等价；误差来源尚未定位，候选没有通过完整梯度或多步收敛验证，不能作为已验收优化交付。首次 profile001 因严格梯度检查失败退出 1；该记录保留，profile002 的成功仅表示性能采集完成。

### 复现与证据

- 采集脚本：`npu/profile_training.py`，实际执行 v2 的 SHA-256 为 `f66ce67a0b8283d13b07750e51b93e4a26e99b0c8b47f47e7dedaf6ce240c959`。
- 测量约定：`npu/profile_contract.json`；汇总：`npu/profile_summary.json`；原始四卡记录、退出码、卡状态：`npu/evidence/profile002.json`。
- 运行 `python npu/summarize_profile.py` 可从本地原始记录重建汇总。
- 服务端监督记录：`/home/ma-user/pi05-npu-20261011/profile002/`；脚本：`/home/ma-user/pi05-npu-20261011/profile-training-v2.py`。
- 原始 trace：`/cache/pi05-npu-20261011/profile002/trace/ascend-worker-0_2589450_20261011112557841_ascend_pt/ASCEND_PROFILER_OUTPUT/`，含 `trace_view.json`、kernel/operator CSV 和通信报告。
- 实测沿用之前交付的源码及环境；本轮没有修改模型训练实现或旧 checkpoint。

## 继续训练优化（diagnose001 / optimize001）

本轮保持四张 910B1、global batch=4、原 BF16/FP32 参数精度、全参数训练和八个 LIBERO 缓存窗口。按用户要求先诊断重计算梯度差异，再测试融合 AdamW 和 25/50/100 MiB DDP 通信桶。完整实验入口为 `npu/optimize_training.py`；没有下载新数据、执行推理或写训练 checkpoint。

### 梯度差异的定位

同一权重、输入、各 rank RNG 下，比较原始路径、原始重复、减少整段 Gemma checkpoint 的候选及候选重复；每次没有优化器更新。此次检查 **805 个有效梯度张量、3,248,572,176 个梯度元素**，不再沿用上轮的 51,488 元素抽样。容差仍为 atol=1e-7、rtol=1e-5。

| 条件 | 对比 | 四卡最大绝对差 | 最大相对 L2 | 严格检查 |
|---|---|---:|---:|---|
| 普通 HCCL | 原始重复 vs 原始 | 2.4414e-4 | 2.0299e-3 | 未通过 |
| 普通 HCCL | 减少重计算 vs 原始 | 2.4414e-4 | 1.9821e-3 | 未通过 |
| `DDP.no_sync()`，关闭梯度通信 | 原始重复 vs 原始 | 2.7940e-9 | 1.8835e-8 | 四卡通过 |
| `DDP.no_sync()`，关闭梯度通信 | 减少重计算 vs 原始 | 2.7940e-9 | 1.9449e-8 | 四卡通过 |

去掉通信后，候选与原始的全量梯度误差已落在原始重复波动范围内。随后在新进程设置 `HCCL_DETERMINISTIC=true`，重新做同样的固定状态比较。这是对归约非确定性的独立干预，不是放宽误差阈值。该开关控制归约算子的确定性；见 [昇腾 HCCL_DETERMINISTIC 文档](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/maintenref/envvar/envref_07_0099.html)，本报告的有效性以当前 CANN 8.2.RC1 实测为准。

首次 diagnose001 在进一步打开 `torch.use_deterministic_algorithms(True)` 时失败：PyTorch 的插值分解路径请求了当前 NPU 不支持的内存布局。失败日志和退出码 1 已保留。optimize001 保持原来的非训练参数图像增强路径，只在后续模型计算中启用这一诊断开关；计时阶段关闭 torch 全局确定性开关，所有配置统一保留确定性 HCCL。

### 融合优化器与通信桶的边界

当前安装的 `torch_npu.optim.NpuFusedAdamW` 只接受 FP32/FP16，真实 BF16 小测试明确报错。本轮使用能处理现有 BF16/FP32 混合参数的 `torch.optim.AdamW(fused=True)`；已核实当前运行时注册了 `aten::_fused_adamw_` 的 PrivateUse1 原生实现，小测试没有 CPU 回退警告。

融合优化器的数值对比使用真实模型权重和裁剪后的梯度，每个有效张量取前 4,096 个元素，共 2,448,160 个，保持原 dtype。标准与 fused AdamW 均从零动量开始，在 peak LR=5e-5 下使用相同梯度连续更新三次。该探针独立于计时训练，不改变模型权重。它检查参数、一阶和二阶动量，不是完整模型收敛或优化器 checkpoint 恢复验证。

通信桶固定状态对比在确定性 HCCL 下进行，恢复相同 RNG，并覆盖全部有效梯度。确定性保证相同配置的可复现性，不保证不同桶大小采用相同的浮点归约顺序。改变到 50/100 MiB 后 loss 保持一致，但梯度未通过原有严格阈值；重新回到 25 MiB 后通过。因此更大桶的速度收益与数值差异必须一起报告。

### 最终结果与推荐配置

optimize001 实际退出码为 **0**，四个 rank 均完成全部诊断和 **77 次真实优化器更新**。全部计时方案均验证了有限 loss/梯度范数、805 份有效优化器状态的有限动量、主要分支参数变化和跨卡梯度抽样一致性。四张卡在结束后均无运行进程。

开启确定性 HCCL 后，减少重计算 vs 原始的全量梯度最大绝对差为 **1.3970e-9**，相对 L2 为 **1.6750e-8**，四卡均通过原来的严格阈值。进一步开启模型算子确定性检查时，最大差为 9.3132e-10，仍通过。所有比较的 loss 差为 0；这里是严格容差内一致，不是逐位完全相同。结合关闭通信和单独启用确定性 HCCL 的对照，可将此前的主要差异定位到 HCCL 归约的非确定性，未发现减少外层 checkpoint 引入额外的显著梯度误差。

每项 3 步 warmup + 8 步计时，统一确定性 HCCL；先取每步最慢 rank，再取中位数。基线和减少重计算的 25 MiB 方案均复测；计时不含模型加载、数值检查或结果审计，不开 profiler。各块连续更新权重并保留优化器状态，未在每块重置训练；数据和有效 batch 固定，仍属于短程缓存数据对照。

| 配置 | 中位秒/步 | 相对原始耗时减少 | 峰值 allocated GiB/卡 | 数值结论 |
|---|---:|---:|---:|---|
| 原始 checkpoint + 标准 AdamW + 25 MiB | 1.3038 | — | 32.743 | 基线 |
| 减少外层 checkpoint + 标准 AdamW + 25 MiB | **1.1715** | **10.15%** | 32.743 | 全量梯度严格检查通过 |
| 减少外层 checkpoint + fused AdamW + 25 MiB | 1.0725 | 17.74% | **30.393** | 参数/一阶动量严格检查未通过 |
| 减少外层 checkpoint + 标准 AdamW + 50 MiB | 1.1553 | 11.39% | 32.743 | 桶变更后梯度严格检查未通过 |
| 减少外层 checkpoint + 标准 AdamW + 100 MiB | 1.1570 | 11.26% | 32.743 | 桶变更后梯度严格检查未通过 |
| 减少外层 checkpoint + 标准 AdamW + 25 MiB 复测 | 1.1723 | 10.08% | 32.743 | 性能复测 |
| 原始配置复测 | 1.3026 | — | 32.743 | 性能复测 |

原始两组各有一个约 1.8 秒慢步骤，均未删除；均值分别为 1.3632、1.3645 秒。减少重计算两组均值为 1.1700、1.1703 秒。中位数对应吞吐从约 3.068 提升到 **3.414 样本/秒（+11.29%）**。

融合 AdamW 在减少重计算的基础上，额外降低耗时 **8.45%**，峰值 allocated 显存减少约 **2.35 GiB/卡**。其三步参数抽样最大差依次为 3.0518e-5、6.1035e-5、9.1553e-5；第三步参数相对 L2 为 1.4896e-6，一阶动量相对 L2 为 7.1339e-4。虽然误差较小且所有短程更新有限，但参数与一阶动量没有通过既定阈值，不能把 fused 路径称为已验证等价。它保持实验选项，不替换标准 AdamW 默认配置。切换优化器时保留了已有状态，并核对 fused 标志与 step tensor 设备，避免测到实际上未开启融合的路径。

50/100 MiB 桶的额外耗时改善分别只有 **1.38% / 1.24%**；实际 bucket 数从 119 降到 88 / 46。相对 25 MiB 的全量梯度最大绝对差分别为 6.1035e-5 / 2.4414e-4，相对 L2 为 1.6385e-3 / 2.8257e-3。25 MiB 回切对照最大差仅 8.1491e-10，并通过严格检查。大桶收益有限且引入额外数值差异，本轮保留 **25 MiB**。

当前推荐组合是 **`HCCL_DETERMINISTIC=true` + 只移除整段 Gemma 外层 checkpoint + 保留逐层/视觉 checkpoint + 标准 AdamW + 25 MiB 桶**。这是已完成全量梯度检查及真实更新测速的组合；本轮修改在实验训练进程内生效，未自动改写原始训练 CLI 默认行为。融合优化器和大桶仅保留实验记录，未与推荐配置混合宣称收益。

### 本轮复现与交付

- 实际运行脚本：服务器 `/home/ma-user/pi05-npu-20261011/optimize-training-v3.py`，SHA-256 `787c1df891f76884149bed31c3b8e9ffb61c31f18a7703a0dbc8dacf283e50ce`；发布版 `npu/optimize_training.py` 增加了权重和数据路径参数，原始运行哈希保留用于追溯。
- 监督记录与完整日志：`/home/ma-user/pi05-npu-20261011/optimize001/`；四卡 JSON：`/cache/pi05-npu-20261011/optimize001/`。
- 约定：`npu/optimization_contract.json`；原始证据：`npu/evidence/optimization001.json`；汇总：`npu/optimization_summary.json`；推荐配置：`npu/recommended_training.json`。
- 运行 `python npu/summarize_optimization.py` 可重建汇总，全部慢步骤和失败记录均保留。
- 未覆盖完整 DataLoader/视频解码、长期收敛、任务成功率、fused 优化器的完整 checkpoint 恢复，也未扩大卡数或下载全量数据。

复现实验时先确认四张卡空闲，使用全新输出目录：

```bash
cd /home/ma-user/pi05-npu-20261011/source
source npu/environment.sh
export HCCL_DETERMINISTIC=true
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  /home/ma-user/pi05-npu-20261011/optimize-training-v3.py \
  --mode all --output /cache/pi05-npu-20261011/NEW_UNIQUE_RUN
```
