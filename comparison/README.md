# PartNet-Mobility evaluation

This directory contains the frozen first-version local metric used for the
corrected-orientation URDF-Anything+ result. The authors have not released the
Table 1 evaluation implementation, so every unpublished protocol choice below
must remain explicit when quoting the result.

## Inputs and coordinates

- Test split: the same 77 PartNet-Mobility objects used by Particulate.
- Quantitative mesh input: PartNet ground-truth whole-object mesh, following
  the URDF-Anything+ author's clarification; the image is also provided.
- Inference orientation: PartNet `+Z`-up/`-X`-front is converted to model
  `+Y`-up/`+Z`-front by `(x,y,z) -> (-y,z,-x)`.
- Evaluation frame: `--urdf-coordinate world` applies the generated URDF fixed
  root joint, returning prediction geometry and axes to the PartNet GT frame.
- Evaluation normalization: GT and prediction use the existing approximately
  `[-1,1]^3` coordinates; no result-dependent post-alignment is applied.

Prepare all corrected inputs:

```bash
/home/LiuShuqi/.conda/envs/urdf-anything-plus/bin/python \
  comparison/prepare_orientation_probe.py \
  --input-root /data2/LiuShuqi/data/processed/PartNetMobility_URDF-Anything-plus/test \
  --output-root /data2/LiuShuqi/data/processed/PartNetMobility_URDF-Anything-plus-orientation-probe/test \
  --resume
```

## Geometry metrics

- Sampling: deterministic mesh-surface sampling with seed 42, 100,000 points
  per part and 100,000 points for the whole object.
- Part matching: one-to-one Hungarian assignment; its cost is bidirectional
  squared-L2 Chamfer distance estimated from 2,048 points per part.
- Reported Parts metrics: compute each metric for matched pairs, average pairs
  within each object, then average the 77 object values. Missing and extra parts
  are not inserted into the reported matched-only values; counts are mandatory.
- Reported Whole metrics: compute on each complete predicted/GT surface and
  average the 77 object values.
- F-score: precision and recall use strict nearest-neighbour distance `<0.02`;
  `F=2PR/(P+R)`.
- CD: `mean_pred min_gt ||p-g||^2 + mean_gt min_pred ||g-p||^2`.
- IoU: intersection over union of occupied surface voxels at resolution 128 in
  `[-1.01,1.01]^3`; this is a local surface proxy, not a confirmed reproduction
  of the paper's solid/volume occupancy implementation.

The evaluator also retains a clearly non-paper missing-part stress-test branch
(`IoU=0`, `F=0`, `CD=12` for unmatched parts), but paper comparisons use only
the matched-only values above.

## Revolute-joint metrics

Joint correspondence comes from the same Hungarian part assignment. Errors are
pooled over GT revolute joints whose part is matched and whose predicted type is
also revolute; the numerator and denominator must always be reported.

- Axis error: `acos(abs(dot(unit(axis_pred), unit(axis_gt))))` in radians.
- Origin error: shortest Euclidean distance between the two infinite 3D axis
  lines, recovered from their Pluecker coordinates.
- Limit error: sign-invariant Articulate-Anything motion-vector error,
  `min(||a_p * range_p - a_g * range_g||,
  ||-a_p * range_p - a_g * range_g||)`.
- Aggregation: pool all eligible joints across 77 objects and take the mean.
- Coverage: `correct predicted revolute / all GT revolute`; wrong type and
  missing-part joints receive no fabricated numeric error.

The local origin distance is in normalized object coordinates, whereas Table 1
labels origin error in metres. These values are not directly comparable.

## Run and report

The current NVIDIA kernel/user-library workaround is required until the host is
rebooted or the versions are reconciled:

```bash
export URDF_ANYTHING_DRIVER_COMPAT=/data2/LiuShuqi/output/URDF-Anything-plus/driver-compat/580.173.02/root/usr/lib/x86_64-linux-gnu
env LD_LIBRARY_PATH=$URDF_ANYTHING_DRIVER_COMPAT \
/home/LiuShuqi/.conda/envs/particulate/bin/python \
  -m comparison.evaluate_author_metrics \
  --method urdf-anything-plus \
  --prediction-root /data2/LiuShuqi/output/URDF-Anything-plus/inference/PartNetMobility-test-oriented \
  --gt-root /data2/LiuShuqi/data/processed/PartNetMobility_test_particulate/gt \
  --asset-root /data2/LiuShuqi/data/processed/PartNetMobility_test_particulate/assets \
  --output-dir /data2/LiuShuqi/output/URDF-Anything-plus/comparison/PartNetMobility-test/author_metrics_oriented/urdf-anything-plus \
  --urdf-coordinate world --resume
```

Build the old-orientation/corrected-orientation/paper comparison:

```bash
/home/LiuShuqi/.conda/envs/particulate/bin/python \
  -m comparison.build_oriented_paper_comparison \
  --oriented-summary /data2/LiuShuqi/output/URDF-Anything-plus/comparison/PartNetMobility-test/author_metrics_oriented/urdf-anything-plus/summary.json \
  --unoriented-summary /data2/LiuShuqi/output/URDF-Anything-plus/comparison/PartNetMobility-test/author_metrics/urdf-anything-plus/summary.json \
  --output-dir /data2/LiuShuqi/output/URDF-Anything-plus/comparison/PartNetMobility-test/author_metrics_oriented/paper_comparison
```

## Current result

| Metric | Paper Table 1 | Old orientation | Corrected orientation |
|---|---:|---:|---:|
| Parts IoU | 0.879 | 0.195182 | 0.321733 |
| Parts F-score | 0.721 | 0.427521 | 0.621243 |
| Parts CD | 0.033 | 0.180844 | 0.065410 |
| Whole IoU | 0.930 | 0.316774 | 0.357589 |
| Whole F-score | 0.742 | 0.633097 | 0.685852 |
| Whole CD | 0.009 | 0.019815 | 0.008336 |
| Axis error (rad) | 0.129 | 1.230936 | 0.446832 |
| Origin error | 0.062 m | 0.621544 normalized | 0.143116 normalized |
| Limit error (rad) | 0.225 | 2.450820 | 1.317631 |

Corrected-orientation counts are 169 predicted parts, 243 GT parts, and 158
matched pairs. Joint errors cover 62/103 GT revolute joints. Whole CD is close
to the paper value; IoU and origin remain protocol/unit-incomparable as noted
above.
