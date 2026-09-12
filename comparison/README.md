# PartNet-Mobility evaluation

`evaluate_author_metrics.py` implements the local URDF-Anything+ metric family
for the shared 77-object PartNet-Mobility test split. Protocol
`comparison-local-v4-dual-iou` emits surface and volume IoU together by default.
The paper comparison always selects volume IoU.

The authors have not released their Table 1 evaluator, so local choices remain
explicit in every summary and must be reported with the results.

## Inputs and coordinates

- Both methods are evaluated against the same PartNet GT meshes and motion data.
- URDF-Anything+ receives the PartNet GT whole mesh and rendered image, following
  the author's clarification for the quantitative experiment.
- PartNet `+Z`-up/`-X`-front is converted to model `+Y`-up/`+Z`-front before
  inference by `(x,y,z) -> (-y,z,-x)`.
- `--urdf-coordinate world` applies the generated fixed root joint and returns
  URDF-Anything+ geometry and axes to the PartNet GT frame.
- Evaluation uses the existing approximately `[-1,1]^3` coordinates without
  result-dependent post-alignment.

## Metrics

Geometry sampling is deterministic with seed 42: 100,000 surface points per
part and 100,000 for each whole object. One-to-one part matching uses Hungarian
assignment on bidirectional squared-L2 CD estimated from 2,048 points per part.

- `iou_surface`: intersection over union of occupied surface voxels.
- `iou_volume`: intersection over union after orthographic solid filling; a
  voxel is interior when bounded by surface voxels in both directions along X,
  Y, and Z. This is the primary IoU used against the paper.
- Both IoUs share one surface rasterization at resolution 128 in
  `[-1.01,1.01]^3`; Whole occupancy is the union of part occupancy.
- F-score uses strict nearest-neighbour distance `<0.02` and `2PR/(P+R)`.
- CD is the sum of the two directional mean squared-L2 distances.

Parts metrics first average matched pairs within each object and then average
77 object values. Whole metrics average the 77 object values directly. The
primary table is matched-only and must be accompanied by part counts. A local
non-paper stress test also sets both IoUs and F-score to zero and CD to 12 for
unmatched parts.

Joint correspondence reuses the geometry matching. Errors are pooled over GT
revolute joints with a matched part and predicted revolute type:

- axis: `acos(abs(dot(unit(axis_pred), unit(axis_gt))))`;
- origin: shortest distance between infinite Pluecker axis lines;
- limit: sign-invariant Articulate-Anything motion-vector range error.

Coverage is always reported. Origin error is in normalized object coordinates,
not the metres used by the paper.

## Cache and output schema

Each object JSON carries `protocol_version=comparison-local-v4-dual-iou`.
`--resume` only reuses cache entries whose version, sample ID, and method all
match; older single-IoU JSON is automatically recomputed.

The explicit geometry keys are:

```text
geometry.parts.{matched_only,penalized}.{iou_surface,iou_volume,fscore,cd}
geometry.whole.{iou_surface,iou_volume,fscore,cd}
```

## Run both methods

URDF-Anything+:

```bash
export URDF_ANYTHING_DRIVER_COMPAT=/data2/LiuShuqi/output/URDF-Anything-plus/driver-compat/580.173.02/root/usr/lib/x86_64-linux-gnu
env LD_LIBRARY_PATH=$URDF_ANYTHING_DRIVER_COMPAT \
/home/LiuShuqi/.conda/envs/particulate/bin/python \
  -m comparison.evaluate_author_metrics \
  --method urdf-anything-plus \
  --prediction-root /data2/LiuShuqi/output/URDF-Anything-plus/inference/PartNetMobility-test-oriented \
  --gt-root /data2/LiuShuqi/data/processed/PartNetMobility_test_particulate/gt \
  --asset-root /data2/LiuShuqi/data/processed/PartNetMobility_test_particulate/assets \
  --output-dir /data2/LiuShuqi/output/URDF-Anything-plus/comparison/PartNetMobility-test/author_metrics_dual_iou/urdf-anything-plus \
  --urdf-coordinate world --resume
```

Particulate:

```bash
/home/LiuShuqi/.conda/envs/particulate/bin/python \
  -m comparison.evaluate_author_metrics \
  --method particulate \
  --prediction-root /data2/LiuShuqi/output/particulate/PartNetMobility-test-evaluation/predictions \
  --gt-root /data2/LiuShuqi/data/processed/PartNetMobility_test_particulate/gt \
  --asset-root /data2/LiuShuqi/data/processed/PartNetMobility_test_particulate/assets \
  --output-dir /data2/LiuShuqi/output/URDF-Anything-plus/comparison/PartNetMobility-test/author_metrics_dual_iou/particulate \
  --resume
```

Cross-method report:

```bash
/home/LiuShuqi/.conda/envs/particulate/bin/python \
  -m comparison.build_metric_comparison \
  --candidate-summary /data2/LiuShuqi/output/URDF-Anything-plus/comparison/PartNetMobility-test/author_metrics_dual_iou/urdf-anything-plus/summary.json \
  --baseline-summary /data2/LiuShuqi/output/URDF-Anything-plus/comparison/PartNetMobility-test/author_metrics_dual_iou/particulate/summary.json \
  --candidate-label 'URDF-Anything+' --baseline-label 'Particulate' \
  --title 'URDF-Anything+ vs Particulate under dual-IoU URDF metrics' \
  --output-dir /data2/LiuShuqi/output/URDF-Anything-plus/comparison/PartNetMobility-test/author_metrics_dual_iou/method_comparison
```

The report directory contains `REPORT.md`, `paper_comparison.csv/json`, and the
explicit four-row `iou_comparison.csv`.

## Current result

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
| Origin error (normalized) | 0.143116 | 0.024016 |
| Limit error (rad) | 1.317631 | 0.139491 |
| Matched parts / GT parts | 158/243 | 243/243 |
| Revolute coverage | 62/103 | 103/103 |

Particulate preserves and segments the provided whole mesh, whereas
URDF-Anything+ decodes new per-link geometry while conditioned on that GT mesh.
The geometry gap therefore includes the different output representations and
is not a pure articulation-reasoning comparison.
