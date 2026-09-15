# URDF-Anything+ 设计对比

本文记录两项彼此独立、可切换的消融设计，以及最小训练的固定参数和运行命令。
默认配置保持作者的 geometry-only 自回归实现，两个改进分别验证，不把它们合并成
第三种模型。实际跑通结果和产物路径在每项完成后同步写入 `REPRODUCE.md`。

## 共同设置

| 项目 | 固定值 |
| --- | --- |
| 数据 | 作者 Microwave `7201`，3 parts（base + 2 revolute） |
| latent | 每个 part `512 x 64` |
| 图像 | `images/0.png`（归档实际成员） |
| batch / epoch | `1 / 1` |
| seed | `42` |
| 初始化 | 作者 `epoch_200.pth`，新增层按各设计初始化 |
| 主模型 timestep | 1000；URDF 参数 loss 仅在 `t < 300` |
| 日志 | 禁用 wandb，不保存 optimizer |
| 处理数据 | `/data2/LiuShuqi/data/processed/URDF-Anything-plus-design-smoke/` |
| 缓存与输出 | `/data2/LiuShuqi/output/URDF-Anything-plus/design-comparison/` |

数据从只读作者归档中只抽取 `test.urdf`、`whole.obj`、`images/0.png`、
`body.obj`、`door_1.obj`、`door_2.obj` 和 `info.json`。单对象 train/val 重叠只用于
链路 smoke test，不用于衡量泛化。该对象的 `info.json` 仍写着不存在的
`whole_image: body.png`；缓存构建器以 `images/` 中实际文件为准，所以本次使用
`0.png`。

## 对比 1：历史信息传递

### Baseline

保持作者的自回归顺序。生成当前 part 时，`encode_pre` 表示此前已生成 geometry；
root 使用可学习 `SoT`，后续步骤使用此前所有 part mesh 的联合 VAE encoding。

### Motion-history 改进

- 模式开关：`--history_mode geometry`（默认）或
  `--history_mode geometry_motion`。
- geometry context 不变；另把此前已生成关节组成顺序敏感的 motion history。
- 单个 cue 为 10 维：`origin(3) + unit_axis(3) + normalized_limits(2) +
  motion_type_one_hot(2)`。axis 使用 L2 normalize；limits 使用 `tanh` 映射到
  `[-1, 1]`，避免 revolute/prismatic 量纲和异常预测值直接放大 GRU 输入。
- motion cues 由单层 GRU 编码，`input_size=10`、`hidden_size=256`；空序列条件为
  全零向量。
- root 与第一个 movable part 没有历史 motion；第三个 part 开始包含此前 movable
  parts；EOT 包含所有已生成关节。
- 训练用此前关节的 GT cue（teacher forcing），推理用此前步骤的预测 cue。
- 每个 DiT block 在 geometry-context cross-attention 之后、FFN 之前，用 GRU 的最终
  hidden state 做 AdaLN：`LN(x) * (1 + scale) + shift`。每层独立的
  `SiLU + Linear(256, 2 * 2048)` 产生 scale/shift，Linear 权重和 bias 均零初始化，
  因而载入作者 checkpoint 时新增分支初始严格退化为原 block。
- geometry、URDF 参数与 EOT loss 保持作者训练目标不变，唯一变量是额外历史条件。

已验证 smoke 命令：

```bash
export URDF_ROOT=/data1/LiuShuqi/code/articulation/baselines/URDF-Anything-plus
export URDF_OUT=/data2/LiuShuqi/output/URDF-Anything-plus/design-comparison
export DESIGN_DATA=/data2/LiuShuqi/data/processed/URDF-Anything-plus-design-smoke
export ENV_PREFIX=/home/LiuShuqi/.conda/envs/urdf-anything-plus
export PYTHONPATH="$URDF_ROOT/TripoSG:$URDF_ROOT"

CUDA_VISIBLE_DEVICES=0 "$ENV_PREFIX/bin/python" -u scripts/train.py \
  --cache_path "$URDF_OUT/cache/microwave_7201_token512" \
  --history_mode geometry_motion \
  --batch_size 1 --learning_rate 1e-5 --max_epochs 1 --save_interval 1 \
  --init_mode resume_from_ckpt \
  --checkpoint_path /data2/LiuShuqi/output/URDF-Anything-plus/checkpoints/urdf-anything-plus/epoch_200.pth \
  --train_urdf_params True --train_eot True --use_wandb False \
  --save_optimizer False --save_checkpoint_dir "$URDF_OUT/motion-history" \
  --seed 42 --deterministic
```

2026-09-15 实测：cache 的 4 个 AR steps（3 parts + EOT）对应 motion-history 长度
`[0, 0, 1, 2]`；一轮训练完成，`Train Loss=0.690475`、
`Val Loss=0.641541`。日志和 checkpoint 分别为：

```text
/data2/LiuShuqi/output/URDF-Anything-plus/design-comparison/motion-history-train.log
/data2/LiuShuqi/output/URDF-Anything-plus/design-comparison/motion-history/lr1e-5_bs1_ep1_eot_urdf-params_motion-history/epoch_1.pth
```

## 对比 2：模型架构

### Baseline

保持 geometry-only 自回归：按论文的空间顺序逐个预测 part，`encode_pre` 传递此前
geometry，最后以 zero latent 作为 EOT 监督和停止信号。

### Simultaneous set denoising 改进

- 模式开关：`--generation_mode autoregressive`（默认）或
  `--generation_mode set`。
- 固定 5 slots（作者数据最大 5 parts，可用 `--num_part_slots` 覆盖）；对象超过槽数
  直接报错，不静默截断。
- 一个对象的 5 个 slots 同时送入原始 DiT，计算形状为
  `[batch * 5, 512, 64]`。对象内共享 timestep，各槽使用独立 Gaussian noise。
- 槽间不新增 attention，不使用曾讨论过的 register/global-attention 方案。每个槽只
  依赖 DINO image、`encode_whole`、自身 noisy latent 和共享 timestep；不使用
  `SoT/encode_pre`。
- 训练时将真实 parts 随机分配到 5 slots，其余 slots 使用 zero-latent padding。
- 保留 presence head；把原 root head 替换为 5 类 part-order head。所有 heads 都从
  geometry tokens 做 attention pooling 后预测。
- 空间顺序复用论文 Appendix A.4 的顺序先验：固定/base/outer frame 为 order 0；
  movable parts 根据 AABB minimum 按 `Z -> X -> Y` 字典序排序。
- 推理先按 presence threshold 过滤；如果全部低于阈值，保留 presence 最高的槽。
  对保留槽的 order logits 做一对一最优分配以消除重复 order，再按 order 排序并输出
  root-first URDF。
- set 模式训练开启 gradient checkpointing。

训练损失：

```text
L_geometry_real
+ 0.1 L_geometry_padding
+ 0.01 * (
    L_presence
    + L_part_order
    + L_origin
    + L_axis
    + L_limits
    + L_motion_type
  )
```

其中 geometry loss 按 real/padding 分开求均值；presence 和 part-order 在全部 timestep
训练，但 padding 不参与 part-order；motion 参数仅真实、非 root 且 `t < 300` 的槽
参与。presence 使用全部 5 slots。

预定 smoke 命令（实现完成后以实际 CLI 为准）：

```bash
CUDA_VISIBLE_DEVICES=0 "$ENV_PREFIX/bin/python" -u scripts/train.py \
  --cache_path "$URDF_OUT/cache/microwave_7201_token512" \
  --generation_mode set --num_part_slots 5 \
  --batch_size 1 --learning_rate 1e-5 --max_epochs 1 --save_interval 1 \
  --init_mode resume_from_ckpt \
  --checkpoint_path /data2/LiuShuqi/output/URDF-Anything-plus/checkpoints/urdf-anything-plus/epoch_200.pth \
  --train_urdf_params True --train_eot False --use_wandb False \
  --save_optimizer False --save_checkpoint_dir "$URDF_OUT/set-denoising" \
  --seed 42 --deterministic
```

## 执行与验收状态

- [x] 对比 1：代码和单元测试
- [x] 对比 1：Microwave `7201` 一 epoch 最小训练
- [x] 对比 1：更新 `REPRODUCE.md` 并提交 `add motion-aware autoregressive history`
- [ ] 对比 2：代码和单元测试
- [ ] 对比 2：Microwave `7201` 一 epoch 最小训练
- [ ] 对比 2：更新 `REPRODUCE.md` 并提交 `add simultaneous part set denoising`
