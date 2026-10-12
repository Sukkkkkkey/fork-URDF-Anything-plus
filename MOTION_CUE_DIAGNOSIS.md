# Motion cue 对比实验诊断与收尾报告

日期：2026-10-12。状态：按用户决定暂停本项工作，保留代码、权重和诊断产物。
本轮完成文档归档与代码提交，不继续修复 DDP、调参、训练或扩大评估。

## 结论及适用范围

真实 part 的去噪 loss 处于平台，尚未证明 motion cue 带来稳定的几何收益。
EOT 和部分关节参数有改善，不能称为完全没有学习，也不能仅凭 MSE 约 0.93
认定模型未收敛或已经高度拟合。

训练代码绕过 DDP forward，导致各 rank 独立更新。已有四卡训练及 LR sweep
不能作为同步全局 batch 12 的有效消融实验；W&B 的聚合曲线也不是同一组权重
在全部数据上的 loss。此前仅凭进程存活和日志聚合认定“四卡训练正常”的结论撤回。
这个实现问题确定存在，但尚未通过修复后的重训练量化它对平台的贡献。

## 实验与修复记录

- 对比 1：geometry history + motion cues（10D cue、256D GRU、21 层 AdaLN）；
  保持 autoregressive。geometry-only baseline 后续评估使用作者权重，没有另训。
- 对比 2：5 slots simultaneous set denoising，使用 part-order head，未采用
  register/global-attention 方案。仅完成单对象最小训练，没有正式收敛结论。
- 2026-09-15：两项设计均完成最小训练并已提交；最新设计提交为 `fd698e2`。
- 2026-09-17：旧 motion run `fr0x7b1u` 在 epoch 8 的 `783/2779` 处中断。
  日志突然 EOF、W&B exit code 1，无 Python/CUDA/DDP traceback；当时没有
  OOM 或系统重启证据。PTY 生命周期终止是推测，未获得内核日志排除其他原因。
- 2026-09-18：修复空 history 在 AdaLN bias 更新后不再恒等的问题；保留零占位
  GRU 计算图。修复 AR 训练 W&B 多次 log 引起的 step 偏移，跨 rank 聚合指标。
  这些修改没有修复梯度同步。
- 以作者权重在独立 tmux `urdf_motion_context_fixed` 中重新启动
  `pl8qobrc`；本地日志确认完成全部 10 epochs，保存 epoch 5 和 epoch 10。
- 2026-10-12：完成固定噪声 checkpoint 对照、motion 开关对照与 DDP 最小复现，
  按用户决定暂停工作。

W&B 历史 run：
[旧 run](https://wandb.ai/liusq-77-ucas/urdf%2B/runs/fr0x7b1u)、
[重新启动的 run](https://wandb.ai/liusq-77-ucas/urdf%2B/runs/pl8qobrc)。
本报告的完成状态以本地训练结束日志和 checkpoint 为依据。

## 实际训练配置

| 设置 | 值 |
| --- | --- |
| 初始权重 | 作者 `epoch_200.pth` |
| 模式 | `geometry_motion` / `autoregressive` |
| GPU / 每卡 batch | 4–7 / 3 |
| 配置声明的 global batch | 12；实际各 rank 独立以 batch 3 更新 |
| optimizer | AdamW，LR `5e-6`，weight decay `0.01`，betas `(0.9, 0.999)` |
| schedule / epoch / seed | constant / 10 / 42 |
| train / validation samples | 33339 / 2459 |
| 预测目标 | v-prediction，1000 timesteps |
| URDF loss | `t < 300`，各项权重 `0.01` |
| checkpoint | 每 5 epochs 保存；不含 optimizer state |

优化范围包含 DiT、DINO adapter、SoT 等，不是仅微调 AdaLN。模型总参数
1,652,197,582，新增 GRU/AdaLN 22,311,936（约 1.35%）。共有主干 state 的
整体 L2 变化相对作者权重约 3.36%；新增 AdaLN 权重已非零。
`feature3d_adapter` 虽被设置为可训练，但未在当前训练前向中使用。

## DDP 问题：确定存在，尚未修复

`urdf_anything/training/trainer.py` 先包装 `DDP(self.model, ...)`，随后在
`train_step` 中取 `self.model.module`，直接调用子模块
`model_to_use.DiTRunner.model(...)`。未经过 DDP forward 的 reducer 准备逻辑。
set 模式的 `_compute_set_losses` 同样绕过 wrapper，因此也不能视作已验证的
多卡训练路径。

`scripts/diagnose_ddp_sync.py` 使用安装环境中的 PyTorch、两个 CPU/Gloo rank，
相同初值、不同输入，得到：

| 路径 | 两 rank 梯度 | SGD 一步后的两 rank 权重 |
| --- | --- | --- |
| 直接调用 `.module` | 1、2 | 0.9、0.8 |
| 调用 DDP wrapper | 1.5、1.5 | 0.85、0.85 |

这是同类调用方式的最小复现，不是恢复历史 rank 1–3 权重做的比较。
结合训练源码可以确定当前路径缺少梯度同步：各 rank 模型分叉，保存 rank 0，
但训练/验证 loss 聚合来自不同模型。标称 global batch 和吞吐量不能证明同步。

如果未来恢复工作，需让完整训练前向经过 DDP，处理 unused/条件分支参数，并在
多步后直接断言各 rank 梯度与参数一致。空 GRU 保持计算图不等于 DDP 同步修复。

## 固定条件 checkpoint 对照

种子 `20261012`；按 cache 类别、train/test、root/first/later/EOT 分层随机选
65 个样本，每个使用 `50,150,...,950` 共 10 个 timestep。不同 checkpoint
使用完全相同的缓存、噪声和 timestep，eval 模式、无参数更新、batch 4。
这是等样本权重的诊断子集，不是按原数据分布加权的全量验证。

| 指标 | 作者权重 | Epoch 5 | Epoch 10 |
| --- | ---: | ---: | ---: |
| Train non-EOT MSE | 0.928616 | 0.926526 | 0.926069 |
| Validation non-EOT MSE | 0.925477 | 0.923095 | 0.923021 |
| Validation EOT MSE | 0.003625 | 0.002585 | 0.000760 |
| Validation origin MAE | 0.100834 | 0.082413 | 0.076961 |
| Validation axis MAE | 0.064030 | 0.058513 | 0.070261 |
| Validation limits MAE | 0.199991 | 0.137730 | 0.129160 |

关节项仅有效 movable 样本且 `t < 300`，origin 遵循训练的轴向维度 mask。
真实 part 的验证 MSE 改善约 0.27%，后半程几乎平台；关节项改善不一致。
同一验证子集的恒零 v 预测 MSE 为 0.990347，仅作量纲参考，不是理论最优下界。

## Motion 分支敏感性分析

保持 epoch 10 权重不变，将 motion history lengths 置零，通过现有 mask 关闭
AdaLN 调制。它衡量分支贡献，不等价于重新训练一个 geometry-only 模型，
也不能单独证明模型使用了 cue 的语义信息而非其他相关信号。

最初分层子集只有 3 个带历史验证 part，关闭 motion 后 loss 略降；样本不足，
随后扩展到验证集中全部 37 个去重后续 part（34 个对象，每个 part 取一个 image，
10 个固定 timestep，共 370 次评估）：

| 设置 | 后续真实 part MSE |
| --- | ---: |
| Epoch 10，motion 开启 | 0.913030924 |
| Epoch 10，motion 关闭 | 0.913188957 |

均值改善约 0.017%，37 个 part 中 25 个改善。按对象 cluster bootstrap
（seed 42，5000 次重采样），开启减关闭的差值 95% 区间为
`[-0.000904, 0.000716]`，跨零；尚无稳定几何收益证据。该区间仅反映当前
对象样本的不确定性，没有覆盖训练种子和不同噪声种子。

在最初分层子集上，关闭 motion 使验证 EOT MSE 从 0.000760 升至 0.003449，
表明可见收益更偏向 EOT。不能据此宣称 motion 设计普遍无效。

## 其他影响因素与超参数判断

- 训练阶段计数：root 10083、first 10062、later 3111、EOT 10083。
  第三个及以后真实 part 仅占全部训练样本 9.3%；这是 part 阶段计数，不保证
  每个 later 样本都有完整有效 cue。前两步没有 motion history，稀释整体指标。
- 验证 later 为 164/2459（含不同 image）；StorageFurniture 验证样本为零。
  去重后的后续 part 仅来自 Dishwasher、Faucet、Refrigerator。
- 训练 loss 为 `mean(non-EOT) + 0.1 * mean(EOT) + URDF terms`；验证 geometry
  loss 为全部样本的 MSE。训练约 0.90 与验证约 0.64 不可直接比较。小 batch
  的 EOT 构成和随机 timestep 也会引起抖动。
- validation 每轮重新采样噪声和 timestep；日志本身难以分辨千分之几的变化。
- 之前的 LR sweep 主要是约 100 steps，且受同一 DDP 问题影响。它只能说明
  当时能运行，不能证明 `5e-6` 是最优超参数。
- 全模型统一 LR、没有 warmup、关节项稀疏且低权重，都值得未来独立检查；
  当前没有证据把不收敛简单归因于 LR 太大或太小。
- 作者初始 loss 已接近平台，支持边际改善有限的解释；没有证据证明“过度拟合”
  是根因。真实几何质量仍需生成/评估指标，当前仅完成 teacher-forced loss 诊断。

## 产物与重现命令

所有权重与原始诊断输出留在服务器，不纳入 Git；原数据只读。

```bash
export URDF_ROOT=/data1/LiuShuqi/code/articulation/baselines/URDF-Anything-plus
export URDF_OUT=/data2/LiuShuqi/output/URDF-Anything-plus
export ENV_PREFIX=/home/LiuShuqi/.conda/envs/urdf-anything-plus
export PYTHONPATH="$URDF_ROOT/TripoSG:$URDF_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export RUN_DIR="$URDF_OUT/history-tuning/formal/motion-context-fixed/lr5e-6_bs3_ep10_eot_urdf-params_motion-history"
cd "$URDF_ROOT"

OMP_NUM_THREADS=1 "$ENV_PREFIX/bin/python" scripts/diagnose_ddp_sync.py

# 输出目录必须不存在；重复运行时换一个后缀，保留旧结果。
CUDA_VISIBLE_DEVICES=0 "$ENV_PREFIX/bin/python" -u scripts/diagnose_motion_loss.py \
  --run-dir "$RUN_DIR" \
  --output-dir "$URDF_OUT/history-tuning/diagnostics/convergence-20261012"
CUDA_VISIBLE_DEVICES=0 "$ENV_PREFIX/bin/python" -u scripts/diagnose_motion_loss.py \
  --run-dir "$RUN_DIR" --history-only --models epoch_10 epoch_10_no_motion \
  --output-dir "$URDF_OUT/history-tuning/diagnostics/convergence-history-only-20261012"
```

以上三项诊断实际执行成功；两组 loss 评估均 exit 0。各输出目录包含
`selection.json`、`protocol.json`、`paired_losses.csv` 和 checkpoint 分组汇总 JSON。

主要绝对路径：

- 作者权重：`/data2/LiuShuqi/output/URDF-Anything-plus/checkpoints/urdf-anything-plus/epoch_200.pth`
- Epoch 5：`/data2/LiuShuqi/output/URDF-Anything-plus/history-tuning/formal/motion-context-fixed/lr5e-6_bs3_ep10_eot_urdf-params_motion-history/epoch_5.pth`
- Epoch 10：`/data2/LiuShuqi/output/URDF-Anything-plus/history-tuning/formal/motion-context-fixed/lr5e-6_bs3_ep10_eot_urdf-params_motion-history/epoch_10.pth`
- 正式日志：`/data2/LiuShuqi/output/URDF-Anything-plus/history-tuning/formal/motion-context-fixed/train.log`
- 完整训练配置：`/data2/LiuShuqi/output/URDF-Anything-plus/history-tuning/formal/motion-context-fixed/lr5e-6_bs3_ep10_eot_urdf-params_motion-history/config.json`
- Checkpoint 对照：`/data2/LiuShuqi/output/URDF-Anything-plus/history-tuning/diagnostics/convergence-20261012/`
- 全部后续 part 对照：`/data2/LiuShuqi/output/URDF-Anything-plus/history-tuning/diagnostics/convergence-history-only-20261012/`
- 收尾验证日志：`/data2/LiuShuqi/output/URDF-Anything-plus/history-tuning/diagnostics/closeout-20261012/`

收尾回归（2026-10-12 实测）：

```bash
export TMPDIR="$URDF_OUT/history-tuning/diagnostics/closeout-20261012/tmp"
mkdir -p "$TMPDIR"
"$ENV_PREFIX/bin/python" tests/test_design_comparison.py
"$ENV_PREFIX/bin/python" tests/test_reproduction_utils.py
OMP_NUM_THREADS=1 "$ENV_PREFIX/bin/python" scripts/diagnose_ddp_sync.py
git diff --check
```

设计测试 8 项、复现测试 3 项全部通过；DDP 复现 exit 0（预期复现未同步缺陷，
不代表训练 DDP 修复）；8 个相关 Python 文件 AST 解析与 diff 空白检查通过。
日志分别为 `design-tests.log`、`reproduction-tests.log`、`ddp-diagnostic.log`。
本轮未重跑耗时的 checkpoint 对照，复用同日上一轮已成功产出的固定噪声结果。

## 留待未来的工作

DDP 同步修复及参数一致性验证、修复后的调参/重训练、全量生成质量评估、set
模式正式训练均未完成。暂停状态不应记作“消融成功”或“模型有效性已验证”。
本次提交保存已有空 history 修复、AR 日志改进、调参辅助选项、测试及诊断工具；
不包含 DDP 同步修复。日志 step 修复目前针对 AR，set 日志没有同等完整验收。
