# URDF-Anything+ 简洁复现指南

本文覆盖新服务器上的环境、作者数据、缓存、单样本训练、推理、本地评估和可视化。
已在 8×RTX 4090、驱动 580.178.04 上验证；单样本只证明链路可运行，不代表模型
收敛或论文指标复现。作者未公开官方评估代码，本文的 PartNet-Mobility 指标是仓库中
明确标注协议差异的本地实现。

## 1. 路径与代码

所有原始数据保持只读；缓存、权重和日志不要写进代码仓库。

```bash
export URDF_ROOT=/data1/LiuShuqi/code/articulation/baselines/URDF-Anything-plus
export URDF_OUT=/data2/LiuShuqi/output/URDF-Anything-plus
export AUTHOR_DATA=/data2/LiuShuqi/data/processed/URDF-Anything-plus-author
export SMOKE_DATA=/data2/LiuShuqi/data/processed/URDF-Anything-plus-smoke
export ENV_PREFIX=/home/LiuShuqi/.conda/envs/urdf-anything-plus

git clone https://github.com/Sukkkkkkey/fork-URDF-Anything-plus.git "$URDF_ROOT"
cd "$URDF_ROOT"
git clone https://github.com/VAST-AI-Research/TripoSG.git TripoSG
git -C TripoSG checkout fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c
```

新服务器应检出包含本文和缓存/训练修复的仓库版本。检查 GPU 时不要沿用旧机器的
`LD_LIBRARY_PATH`：

```bash
cat /proc/driver/nvidia/version
readlink -f /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1
nvidia-smi
```

内核模块与用户态库版本应一致。

## 2. Conda 环境

```bash
conda create --prefix "$ENV_PREFIX" python=3.10 -y
conda activate "$ENV_PREFIX"
python -m pip install torch==2.6.0 torchvision==0.21.0
python -m pip install -r requirements.txt
python -m pip install torch-cluster==1.6.3 \
  -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
python -m pip install diso==0.1.4 --no-build-isolation
python scripts/patch_triposg.py --triposg-root TripoSG
python -m pip check
```

设置缓存和 Python 路径：

```bash
mkdir -p "$URDF_OUT"/{cache,checkpoints,logs,training-smoke}
export HF_HOME="$URDF_OUT/huggingface"
export PIP_CACHE_DIR="$URDF_OUT/cache/pip"
export TORCH_HOME="$URDF_OUT/cache/torch"
export PYTHONPATH="$URDF_ROOT/TripoSG:$URDF_ROOT"
```

## 3. 权重

需要 URDF-Anything+ 主模型、TripoSG VAE 和 DINOv3。可直连 Hugging Face 时去掉
`HF_ENDPOINT`；本机验证过无代理镜像和 ModelScope：

```bash
HF_ENDPOINT=https://hf-mirror.com hf download \
  URDF-Anything-plus/URDF-Anything-Plus-Model epoch_200.pth \
  --local-dir "$URDF_OUT/checkpoints/urdf-anything-plus"

HF_ENDPOINT=https://hf-mirror.com hf download VAST-AI/TripoSG \
  --include 'vae/*' --local-dir "$URDF_OUT/checkpoints/triposg"

python -m pip install modelscope
modelscope download facebook/dinov3-vith16plus-pretrain-lvd1689m \
  --local-dir "$URDF_OUT/checkpoints/dinov3-vith16plus-pretrain-lvd1689m"
```

已验证文件校验值：

```text
epoch_200.pth  4542c33f53c479633485c7fde116cf3686eab6e4e3bbc3a7e2b39ceee589faba
TripoSG VAE    a2e667c24927a5a35e5f19fcb4c75890e9399aa966b6db8131d7df733a750c8b
DINOv3         3e1d4d18b9bfa9f28fad8e9de6a783f1313532d3460efa4cd0b12521d81d1a4d
```

从头预训练还需要 `VAST-AI/TripoSG` 的 `transformer/*`，存入输出目录后设置：

```bash
export URDF_ANYTHING_TRIPOSG_DIT_PATH="$URDF_OUT/checkpoints/triposg/transformer/diffusion_pytorch_model.safetensors"
```

## 4. 作者数据与单样本缓存

数据受 CC BY-NC 4.0 和 ShapeNet Terms 约束。归档约 8.51 GiB：

```bash
mkdir -p "$AUTHOR_DATA"
curl -L -C - -o "$AUTHOR_DATA/data_normalized.tar.gz" \
  'https://hf-mirror.com/datasets/URDF-Anything-plus/Dataset/resolve/main/data_normalized.tar.gz?download=true'
echo 'dc2e2ba9d4a76d10cb273ba46c27b3a868c16446a6086dc784cda06c191b2acb  '"$AUTHOR_DATA/data_normalized.tar.gz" | sha256sum -c -
```

抽取包含完整关节标注的 Laptop `11070`：

```bash
mkdir -p "$SMOKE_DATA/Laptop_urdf"
tar --occurrence=1 -xzf "$AUTHOR_DATA/data_normalized.tar.gz" \
  -C "$SMOKE_DATA/Laptop_urdf" --strip-components=2 \
  data_normalized/Laptop_urdf/11070/test.urdf \
  data_normalized/Laptop_urdf/11070/whole.obj \
  data_normalized/Laptop_urdf/11070/images/screen.png \
  data_normalized/Laptop_urdf/11070/images/base.png \
  data_normalized/Laptop_urdf/11070/screen.obj \
  data_normalized/Laptop_urdf/11070/base.obj \
  data_normalized/Laptop_urdf/11070/info.json

CUDA_VISIBLE_DEVICES=0 python scripts/build_cache.py \
  --data_root "$SMOKE_DATA/Laptop_urdf" \
  --cache_name laptop_11070_smoke --allow_single_object_overlap \
  --cache_dir "$URDF_OUT/cache/training-smoke" \
  --dino_model_path "$URDF_OUT/checkpoints/dinov3-vith16plus-pretrain-lvd1689m" \
  --triposg_vae_path "$URDF_OUT/checkpoints/triposg/vae" \
  --token_length 512 --seed 42
```

`--allow_single_object_overlap` 只供 smoke test：同一对象进入 train 和 val，不能用该
val loss 衡量泛化。全量数据则先解压，再去掉该开关：

```bash
tar -xzf "$AUTHOR_DATA/data_normalized.tar.gz" -C "$AUTHOR_DATA"
CUDA_VISIBLE_DEVICES=0 python scripts/build_cache.py \
  --data_normalized_root "$AUTHOR_DATA/data_normalized" \
  --cache_dir "$URDF_OUT/cache/full" \
  --dino_model_path "$URDF_OUT/checkpoints/dinov3-vith16plus-pretrain-lvd1689m" \
  --triposg_vae_path "$URDF_OUT/checkpoints/triposg/vae" \
  --token_length 512 --seed 42
```

## 5. 单样本训练

用作者发布权重继续训练最适合链路验证；`batch_size` 是每 GPU batch：

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py \
  --cache_path "$URDF_OUT/cache/training-smoke/laptop_11070_smoke_token512" \
  --batch_size 1 --learning_rate 1e-5 --max_epochs 1 --save_interval 1 \
  --init_mode resume_from_ckpt \
  --checkpoint_path "$URDF_OUT/checkpoints/urdf-anything-plus/epoch_200.pth" \
  --train_urdf_params True --train_eot True --use_wandb False \
  --save_optimizer False --save_checkpoint_dir "$URDF_OUT/training-smoke/final" \
  --seed 42 --deterministic
```

检查点位于
`$URDF_OUT/training-smoke/final/lr1e-5_bs1_ep1_eot_urdf-params/epoch_1.pth`。
`--deterministic` 是尽力确定性；PyTorch 的 memory-efficient attention 仍会提示其
反向算法非严格确定。

全量多 GPU 训练用 `torchrun --nproc_per_node=<GPU数> scripts/train.py ...`，把
`--cache_path` 后替换为十个 `$URDF_OUT/cache/full/*_eot_token512` 目录。预训练用
`--init_mode train_from_scratch`；随后以预训练检查点运行
`--init_mode resume_from_ckpt --train_urdf_params True --train_eot True`。

## 6. 推理、评估与可视化

仅验证作者样本推理时可直接运行：

```bash
export SMOKE_CKPT="$URDF_OUT/training-smoke/final/lr1e-5_bs1_ep1_eot_urdf-params/epoch_1.pth"
CUDA_VISIBLE_DEVICES=0 python scripts/inference.py \
  --image_path "$SMOKE_DATA/Laptop_urdf/11070/images/base.png" \
  --whole_mesh_path "$SMOKE_DATA/Laptop_urdf/11070/whole.obj" \
  --model_path "$SMOKE_CKPT" \
  --dino_path "$URDF_OUT/checkpoints/dinov3-vith16plus-pretrain-lvd1689m" \
  --triposg_vae_path "$URDF_OUT/checkpoints/triposg/vae" \
  --output_dir "$URDF_OUT/training-smoke/inference/author-11070" \
  --seed 42 --max_links 4 --vae_deterministic
```

可量化评估需要本地 PartNet-Mobility raw ZIP，以及 Particulate 可再生的 `assets/gt`
预处理结果。先按 [RUNBOOK.md](RUNBOOK.md) 准备单样本 `12252` 并转换朝向，然后：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/inference_partnet.py \
  --input-root /data2/LiuShuqi/data/processed/PartNetMobility_URDF-Anything-plus-orientation-probe/test \
  --output-root "$URDF_OUT/training-smoke/inference/PartNetMobility-test-oriented-final" \
  --model-path "$SMOKE_CKPT" \
  --dino-path "$URDF_OUT/checkpoints/dinov3-vith16plus-pretrain-lvd1689m" \
  --triposg-vae-path "$URDF_OUT/checkpoints/triposg/vae" \
  --ids 12252 --seed 42 --max-links 4

python -m comparison.evaluate_author_metrics \
  --method urdf-anything-plus \
  --prediction-root "$URDF_OUT/training-smoke/inference/PartNetMobility-test-oriented-final" \
  --gt-root /data2/LiuShuqi/data/processed/PartNetMobility_test_particulate/gt \
  --asset-root /data2/LiuShuqi/data/processed/PartNetMobility_test_particulate/assets \
  --output-dir "$URDF_OUT/training-smoke/evaluation/12252-final" \
  --ids 12252 --urdf-coordinate world --points-per-part 10000 \
  --points-whole 10000 --matching-points 1024 --voxel-resolution 64

xvfb-run -a blender -b --python scripts/blender_render_obj.py -- \
  --input "$URDF_OUT/training-smoke/inference/PartNetMobility-test-oriented-final/12252/eval/pred.obj" \
  --output "$URDF_OUT/training-smoke/visualization/12252_prediction_final.png" \
  --samples 16
```

smoke metric 的 10k 点/64³ 体素仅检查执行链路。论文级本地对比使用 100k 点和
128³ 体素；协议限制、全量命令与历史 77 对象结果见 [RUNBOOK.md](RUNBOOK.md)。

本机参考结果：缓存 train/val 均为 6 条；训练 `Train Loss=0.632314`、
`Val Loss=0.604450`；12252 推理生成 2 个 link，约 46.2 秒。低分辨率评估的
parts volume IoU 为 0.521684、whole volume IoU 为 0.257426、axis error 为
0.377708 rad。末位可能受表面采样和 CUDA 算法影响。

## 7. 最小验收

```bash
python -m compileall -q scripts urdf_anything comparison
python tests/test_reproduction_utils.py
python -m pip check
test -f "$SMOKE_CKPT"
test -f "$URDF_OUT/training-smoke/inference/PartNetMobility-test-oriented-final/12252/complete.json"
test -f "$URDF_OUT/training-smoke/evaluation/12252-final/summary.json"
test -f "$URDF_OUT/training-smoke/visualization/12252_prediction_final.png"
```
