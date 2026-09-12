# URDF-Anything+ 运行说明

> **这是一份活文档。** 当前代码版本和复现状态须随工作进展更新。

## 固定路径

- 代码：`/data1/LiuShuqi/code/articulation/baselines/URDF-Anything-plus`
- Conda：`/home/LiuShuqi/miniconda3/bin/conda`
- 环境：`/home/LiuShuqi/.conda/envs/urdf-anything-plus`
- 原始数据：`/data2/LiuShuqi/data/raw/PartNetMobility`（严格只读）
- 处理数据：`/data2/LiuShuqi/data/processed/PartNetMobility_URDF-Anything-plus`
- 权重、缓存、日志与结果：`/data2/LiuShuqi/output/URDF-Anything-plus`

## 当前版本

- URDF-Anything+ 远程：`git@github.com:Sukkkkkkey/fork-URDF-Anything-plus.git`
- URDF-Anything+ 分支与初始提交：`main`，`816e2834cc29d2beeba4852229916ac527a9f46a`
- TripoSG 远程：`https://github.com/VAST-AI-Research/TripoSG.git`
- TripoSG 分支与提交：`main`，`fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c`

## 进入环境

```bash
source /home/LiuShuqi/miniconda3/etc/profile.d/conda.sh
conda activate /home/LiuShuqi/.conda/envs/urdf-anything-plus
cd /data1/LiuShuqi/code/articulation/baselines/URDF-Anything-plus
```

主要已验证版本为 Python 3.10.21、PyTorch 2.6.0+cu124、torchvision 0.21.0+cu124、torch-cluster 1.6.3+pt26cu124、diso 0.1.4、diffusers 0.35.1、transformers 4.57.2 和 trimesh 4.7.4。

### NVIDIA 用户态/内核版本不一致时的局部兼容方式

2026-09-11 的 unattended upgrade 将用户态 NVIDIA 库从 `580.173.02`
更新到 `580.178.04`，但运行中的内核模块仍为 `580.173.02`。无需覆盖
系统文件，可将 Ubuntu 官方旧版 `libnvidia-compute-580` 解压到本工作输出：

```bash
export URDF_ANYTHING_DRIVER_COMPAT=/data2/LiuShuqi/output/URDF-Anything-plus/driver-compat/580.173.02/root/usr/lib/x86_64-linux-gnu
env LD_LIBRARY_PATH=$URDF_ANYTHING_DRIVER_COMPAT nvidia-smi
```

本地包为
`driver-compat/580.173.02/packages/libnvidia-compute-580_580.173.02-0ubuntu0.22.04.1_amd64.deb`，
SHA-256 为
`01327514a8fde543dc092b79016f92a51dc7d72aa59db193c2405bde7d371503`。
运行 PyTorch 或推理时同样在命令前加
`env LD_LIBRARY_PATH=$URDF_ANYTHING_DRIVER_COMPAT`；这只影响该子进程。

TripoSG 的完整 VAE 编码器需要启用 `torch_cluster.fps`，克隆或更新 TripoSG 后运行以下幂等脚本：

```bash
python scripts/patch_triposg.py --triposg-root TripoSG
```

## 必要环境变量

```bash
export URDF_ANYTHING_ROOT=/data1/LiuShuqi/code/articulation/baselines/URDF-Anything-plus
export URDF_ANYTHING_OUTPUT=/data2/LiuShuqi/output/URDF-Anything-plus
export URDF_ANYTHING_DATA=/data2/LiuShuqi/data/processed/PartNetMobility_URDF-Anything-plus
export HF_HOME=$URDF_ANYTHING_OUTPUT/huggingface
export PIP_CACHE_DIR=$URDF_ANYTHING_OUTPUT/cache/pip
export TORCH_HOME=$URDF_ANYTHING_OUTPUT/cache/torch
export URDF_ANYTHING_TRIPOSG_VAE_PATH=$URDF_ANYTHING_OUTPUT/checkpoints/triposg/vae
export URDF_ANYTHING_DRIVER_COMPAT=$URDF_ANYTHING_OUTPUT/driver-compat/580.173.02/root/usr/lib/x86_64-linux-gnu
export PYTHONPATH=$URDF_ANYTHING_ROOT/TripoSG:$URDF_ANYTHING_ROOT
```

在当前内核仍为 `580.173.02` 时，所有 CUDA 命令均应使用
`env LD_LIBRARY_PATH=$URDF_ANYTHING_DRIVER_COMPAT` 前缀；不要回退到当前系统的
`/usr/lib/x86_64-linux-gnu`，其中的 `580.178.04` 用户态库会再次触发
driver/library version mismatch。机器重启并确认内核模块升级后，才应重新检查是否
还需要该兼容前缀。

## 模型权重

已通过无代理的 `https://hf-mirror.com` 下载并校验以下文件：

- 主模型：`/data2/LiuShuqi/output/URDF-Anything-plus/checkpoints/urdf-anything-plus/epoch_200.pth`，6,519,815,555 bytes，SHA-256 `4542c33f53c479633485c7fde116cf3686eab6e4e3bbc3a7e2b39ceee589faba`
- TripoSG VAE：`/data2/LiuShuqi/output/URDF-Anything-plus/checkpoints/triposg/vae/diffusion_pytorch_model.safetensors`，970,685,468 bytes，SHA-256 `a2e667c24927a5a35e5f19fcb4c75890e9399aa966b6db8131d7df733a750c8b`

DINOv3 已从 ModelScope 无代理直连下载，无需登录或验证码，并通过 Safetensors 与 Transformers 本地加载校验：

- 目录：`/data2/LiuShuqi/output/URDF-Anything-plus/checkpoints/dinov3-vith16plus-pretrain-lvd1689m/`
- `config.json`：744 bytes
- `preprocessor_config.json`：585 bytes
- `model.safetensors`：3,362,432,800 bytes，SHA-256 `3e1d4d18b9bfa9f28fad8e9de6a783f1313532d3460efa4cd0b12521d81d1a4d`
- Transformers 识别为 `DINOv3ViTModel`，`hidden_size=1280`

复现下载命令如下，显式移除 HTTP(S) 代理：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  -u http_proxy -u https_proxy -u all_proxy \
  modelscope download facebook/dinov3-vith16plus-pretrain-lvd1689m \
  --local-dir $URDF_ANYTHING_OUTPUT/checkpoints/dinov3-vith16plus-pretrain-lvd1689m
```

## PartNet-Mobility 输入准备

使用 Particulate 的同一 77 个测试对象及已验证归一化网格和 GT，并直接从只读 PartNet ZIP 中重建无红色高亮的整物体图像：

```bash
python scripts/prepare_partnet_inputs.py \
  --raw-root /data2/LiuShuqi/data/raw/PartNetMobility \
  --particulate-assets-root /data2/LiuShuqi/data/processed/PartNetMobility_test_particulate/assets \
  --particulate-gt-root /data2/LiuShuqi/data/processed/PartNetMobility_test_particulate/gt \
  --split-file /data2/LiuShuqi/data/processed/PartNetMobility_test_particulate/data_split.json \
  --output-root $URDF_ANYTHING_DATA --split test
```

Particulate 网格最大边长为 1，而 URDF-Anything+ 示例和 TripoSG 归一化最大边长为 2，因此准备脚本将网格坐标乘以 2，导出评测格式时再乘以 0.5 与原 GT 对齐，并将解码器 `±1.005` 搜索边界产生的最多 0.5% 越界裁回 `[-0.5,0.5]`。

原始 ZIP 的 `parts_render_after_merging` 图像只改变红色高亮部件，准备脚本逐像素选择绿色与蓝色之和最大的版本以恢复中性整物体图像。

样本 `103425` 的原始 ZIP 不含任何渲染图，因此使用 `xvfb-run` 和 Blender 2.90 生成固定相机的 512×512 白底回退图像，并在 `info.json` 中单独标记。

验证全部输入、GT 字段和归一化尺度：

```bash
python scripts/validate_partnet_inputs.py \
  --root $URDF_ANYTHING_DATA --split test --expected-count 77
```

## 单样本推理

完成 DINOv3 校验后，先在当时空闲的 GPU 上运行两部件样本 `12252`：

```bash
env LD_LIBRARY_PATH=$URDF_ANYTHING_DRIVER_COMPAT \
CUDA_VISIBLE_DEVICES=7 python scripts/inference_partnet.py \
  --input-root $URDF_ANYTHING_DATA/test \
  --output-root $URDF_ANYTHING_OUTPUT/inference/PartNetMobility-test \
  --model-path $URDF_ANYTHING_OUTPUT/checkpoints/urdf-anything-plus/epoch_200.pth \
  --dino-path $URDF_ANYTHING_OUTPUT/checkpoints/dinov3-vith16plus-pretrain-lvd1689m \
  --triposg-vae-path $URDF_ANYTHING_OUTPUT/checkpoints/triposg/vae \
  --ids 12252 --seed 42 --max-links 16
```

成功后样本目录应包含 `generated.urdf`、`link_*.obj`、`prediction_raw.npz`、`eval/pred.obj`、`eval/pred.npz` 与 `complete.json`。

## 完整测试集推理

单样本成功后运行全部 77 个对象，`--resume` 会跳过已有 `complete.json` 的样本：

```bash
env LD_LIBRARY_PATH=$URDF_ANYTHING_DRIVER_COMPAT \
CUDA_VISIBLE_DEVICES=7 python scripts/inference_partnet.py \
  --input-root $URDF_ANYTHING_DATA/test \
  --output-root $URDF_ANYTHING_OUTPUT/inference/PartNetMobility-test \
  --model-path $URDF_ANYTHING_OUTPUT/checkpoints/urdf-anything-plus/epoch_200.pth \
  --dino-path $URDF_ANYTHING_OUTPUT/checkpoints/dinov3-vith16plus-pretrain-lvd1689m \
  --triposg-vae-path $URDF_ANYTHING_OUTPUT/checkpoints/triposg/vae \
  --seed 42 --max-links 16 --resume
```

批处理只加载一次主模型、VAE 和 DINOv3，每个对象单独保存完成标记与 Particulate 兼容评测输入，并以最大 16 个 link 防止 EoT 失效导致无界生成。

## 修正朝向后的完整测试集推理

README 要求 in-the-wild 物体朝向模型的正 Z 方向。PartNet-Mobility 网格采用
`+Z` 向上、`-X` 向前，而模型采用 `+Y` 向上、`+Z` 向前，因此在推理输入上使用
`(x,y,z)_PartNet -> (-y,z,-x)_model`：

```bash
python comparison/prepare_orientation_probe.py \
  --input-root $URDF_ANYTHING_DATA/test \
  --output-root /data2/LiuShuqi/data/processed/PartNetMobility_URDF-Anything-plus-orientation-probe/test \
  --resume
```

用独立目录推理，避免覆盖旧朝向结果；以下单 GPU 命令可复现全部 77 个对象，实际
全量运行将 ID 列表分为四组并行放在 GPU 4--7：

```bash
env LD_LIBRARY_PATH=$URDF_ANYTHING_DRIVER_COMPAT \
CUDA_VISIBLE_DEVICES=7 python scripts/inference_partnet.py \
  --input-root /data2/LiuShuqi/data/processed/PartNetMobility_URDF-Anything-plus-orientation-probe/test \
  --output-root $URDF_ANYTHING_OUTPUT/inference/PartNetMobility-test-oriented \
  --model-path $URDF_ANYTHING_OUTPUT/checkpoints/urdf-anything-plus/epoch_200.pth \
  --dino-path $URDF_ANYTHING_OUTPUT/checkpoints/dinov3-vith16plus-pretrain-lvd1689m \
  --triposg-vae-path $URDF_ANYTHING_OUTPUT/checkpoints/triposg/vae \
  --seed 42 --max-links 16 --resume
```

四个实际运行日志为
`$URDF_ANYTHING_OUTPUT/logs/inference_partnet_oriented_shard{0,1,2,3}.log`；
77/77 个 `complete.json` 的状态均为 `ok`。

## Evaluation

批量推理会直接生成 Particulate `evaluate.py` 所需的 `*/eval/pred.obj` 和 `*/eval/pred.npz`，因此在 Particulate 环境中运行：

```bash
conda activate /home/LiuShuqi/.conda/envs/particulate
cd /data1/LiuShuqi/code/articulation/baselines/particulate
env LD_LIBRARY_PATH=$URDF_ANYTHING_DRIVER_COMPAT \
CUDA_VISIBLE_DEVICES=7 python evaluate.py \
  --gt_dir /data2/LiuShuqi/data/processed/PartNetMobility_URDF-Anything-plus/gt \
  --result_dir /data2/LiuShuqi/output/URDF-Anything-plus/inference/PartNetMobility-test \
  --output_dir /data2/LiuShuqi/output/URDF-Anything-plus/evaluation/PartNetMobility-test \
  --num_points 100000
```

该评测复用 Particulate 协议以便在相同 77 个对象上横向比较，但不是 URDF-Anything+ 论文公开的官方评测脚本，因为当前仓库没有附带 evaluation 实现。

Particulate 的 `evaluate.py` 使用未固定随机种子的 Trimesh 表面采样；下述数值对应已保存的 77 个逐对象 JSON，清空评测目录后重跑可能在末位出现轻微波动。

### 双 IoU URDF metric

`comparison/evaluate_author_metrics.py` 的 v4 协议默认同时输出 surface IoU 和
volume IoU。两者共享 `[-1.01,1.01]³` 中的一次 128³ 表面栅格化；volume IoU
再以三个轴向都被表面体素双向包围为条件填充实体内部，Whole occupancy 是各
part occupancy 的并集。论文对比固定使用 volume IoU。每个对象 JSON 写入
`comparison-local-v4-dual-iou`，`--resume` 遇到旧协议缓存时会自动重算。

运行修正朝向的 URDF-Anything+：

```bash
conda activate /home/LiuShuqi/.conda/envs/particulate
cd $URDF_ANYTHING_ROOT
env LD_LIBRARY_PATH=$URDF_ANYTHING_DRIVER_COMPAT \
python -m comparison.evaluate_author_metrics \
  --method urdf-anything-plus \
  --prediction-root $URDF_ANYTHING_OUTPUT/inference/PartNetMobility-test-oriented \
  --gt-root /data2/LiuShuqi/data/processed/PartNetMobility_test_particulate/gt \
  --asset-root /data2/LiuShuqi/data/processed/PartNetMobility_test_particulate/assets \
  --output-dir $URDF_ANYTHING_OUTPUT/comparison/PartNetMobility-test/author_metrics_dual_iou/urdf-anything-plus \
  --urdf-coordinate world --resume
```

运行 Particulate：

```bash
python -m comparison.evaluate_author_metrics \
  --method particulate \
  --prediction-root /data2/LiuShuqi/output/particulate/PartNetMobility-test-evaluation/predictions \
  --gt-root /data2/LiuShuqi/data/processed/PartNetMobility_test_particulate/gt \
  --asset-root /data2/LiuShuqi/data/processed/PartNetMobility_test_particulate/assets \
  --output-dir $URDF_ANYTHING_OUTPUT/comparison/PartNetMobility-test/author_metrics_dual_iou/particulate \
  --resume
```

生成两方法横向报告：

```bash
python -m comparison.build_metric_comparison \
  --candidate-summary $URDF_ANYTHING_OUTPUT/comparison/PartNetMobility-test/author_metrics_dual_iou/urdf-anything-plus/summary.json \
  --baseline-summary $URDF_ANYTHING_OUTPUT/comparison/PartNetMobility-test/author_metrics_dual_iou/particulate/summary.json \
  --candidate-label 'URDF-Anything+' --baseline-label 'Particulate' \
  --title 'URDF-Anything+ vs Particulate under dual-IoU URDF metrics' \
  --output-dir $URDF_ANYTHING_OUTPUT/comparison/PartNetMobility-test/author_metrics_dual_iou/method_comparison
```

| Metric | URDF-Anything+ | Particulate |
|---|---:|---:|
| Parts surface IoU | 0.321733 | 0.970942 |
| Parts volume IoU | 0.320077 | 0.955790 |
| Parts F-score | 0.621243 | 0.987162 |
| Parts CD | 0.065410 | 0.004489 |
| Whole surface IoU | 0.357589 | 1.000000 |
| Whole volume IoU | 0.322978 | 0.977165 |
| Whole F-score | 0.685852 | 0.989819 |
| Whole CD | 0.008336 | 0.000144 |
| Axis error (rad) | 0.446832 | 0.000907 |
| Origin error（归一化坐标） | 0.143116 | 0.024016 |
| Limit error (rad) | 1.317631 | 0.139491 |
| 匹配部件 / GT 部件 | 158/243 | 243/243 |
| Revolute coverage | 62/103 | 103/103 |

主表为 matched-only；报告同时写出两种 IoU、未匹配部件压力诊断和 joint
coverage。Particulate 保留并分割输入 whole mesh，而 URDF-Anything+ 从相同
GT mesh 条件解码新的 link geometry，因此几何指标差距包含方法输出表示差异，
不能视为纯关节推理能力差距。

## 当前验证状态

- 77/77 个输入具备网格、图像、元数据和 GT，所有网格非空且最大边长为 2。
- 主检查点、TripoSG VAE 和 DINOv3 已在 GPU 7 完整加载；DINOv3 本地配置为 `dinov3_vit`、`hidden_size=1280`，Safetensors 含 615 个键。
- 单样本 `12252` 推理成功，生成 2 个 link，耗时 55.564 秒；全量推理完成 77/77 个对象，共 120 个 link，范围 1–3 个/对象，导出 42 个 revolute 和 1 个 prismatic 可动关节。
- 全量产物审计无缺失、无无效 URDF 引用且所有 NPZ 数值有限；`pip check` 无依赖冲突，`python tests/test_reproduction_utils.py` 的 3 个测试全部通过。
- Particulate 协议评测完成 77/77 个对象，所有逐对象指标均为有限数值。
- 修正朝向推理完成 77/77 个对象，共生成 169 个 link；第一版 metric 完成
  77/77 个对象，62/103 个 GT revolute joints 获得条件式 joint error。
- 双 IoU v4 对 URDF-Anything+ 和 Particulate 均完成 77/77 个对象；一次运行
  同时输出 surface/volume IoU，四个聚合值与此前两套独立实现精确一致。
- Particulate 匹配 243/243 个 GT 部件并覆盖 103/103 个 GT revolute joints；
  URDF-Anything+ 对应为 158/243 和 62/103。

## 当前复现结果

惩罚未匹配部件的整体指标：

- `rest_per_part_avg_chamfer`: 0.1802
- `rest_per_part_avg_giou`: 0.0921
- `rest_per_part_avg_mIoU`: 0.4028
- `fully_per_part_articulated_avg_chamfer`: 0.2797
- `fully_per_part_articulated_avg_giou`: 0.0885
- `fully_articulated_overall_chamfer_distances`: 0.0321

不惩罚未匹配部件的整体指标：

- `rest_per_part_avg_chamfer_nopunish`: 0.0489
- `rest_per_part_avg_giou_nopunish`: 0.4209
- `rest_per_part_avg_mIoU_nopunish`: 0.5304
- `fully_per_part_articulated_avg_chamfer_nopunish`: 0.1587
- `fully_per_part_articulated_avg_giou_nopunish`: 0.4171
- `fully_articulated_overall_chamfer_distances_nopunish`: 0.0321

主要输出：

- 推理：`/data2/LiuShuqi/output/URDF-Anything-plus/inference/PartNetMobility-test`（约 7.2 GiB）
- 评测：`/data2/LiuShuqi/output/URDF-Anything-plus/evaluation/PartNetMobility-test`
- 单样本日志：`/data2/LiuShuqi/output/URDF-Anything-plus/logs/inference_12252.log`
- 全量推理日志：`/data2/LiuShuqi/output/URDF-Anything-plus/logs/inference_all_77.log`
- 评测日志：`/data2/LiuShuqi/output/URDF-Anything-plus/logs/evaluate_particulate.log`
- 修正朝向推理：`/data2/LiuShuqi/output/URDF-Anything-plus/inference/PartNetMobility-test-oriented`（约 7.1 GiB）
- 修正朝向第一版 metric：`/data2/LiuShuqi/output/URDF-Anything-plus/comparison/PartNetMobility-test/author_metrics_oriented/urdf-anything-plus`
- 论文对比报告：`/data2/LiuShuqi/output/URDF-Anything-plus/comparison/PartNetMobility-test/author_metrics_oriented/paper_comparison`
- 修正朝向评测日志：`/data2/LiuShuqi/output/URDF-Anything-plus/logs/evaluate_author_metrics_oriented.log`
- 双 IoU v4 指标：`/data2/LiuShuqi/output/URDF-Anything-plus/comparison/PartNetMobility-test/author_metrics_dual_iou/{urdf-anything-plus,particulate}`
- 双 IoU v4 横向报告：`/data2/LiuShuqi/output/URDF-Anything-plus/comparison/PartNetMobility-test/author_metrics_dual_iou/method_comparison`
- 双 IoU v4 日志：`/data2/LiuShuqi/output/URDF-Anything-plus/logs/evaluate_author_metrics_{urdf,particulate}_dual_iou.log`
