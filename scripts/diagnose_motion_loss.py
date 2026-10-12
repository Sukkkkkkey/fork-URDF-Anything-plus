"""Read-only checkpoint comparison with paired samples/noise/timesteps.

Writes diagnostics only; never trains or changes checkpoints. Category/part-stage
stratification is deliberately not a dataset-weighted validation estimate.
"""
import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
from diffusers import DDPMScheduler

from urdf_anything.data.cache import load_cache_data
from urdf_anything.data.dataset import CachedDataset, collate_fn
from urdf_anything.model.urdf_model import URDFModel
from urdf_anything.model.utils import load_model_config


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--history-only', action='store_true',
                        help='All unique validation object/part pairs with prior movable parts; one image per pair')
    parser.add_argument('--models', nargs='+', default=['pretrained', 'epoch_5', 'epoch_10', 'epoch_10_no_motion'],
                        choices=['pretrained', 'epoch_5', 'epoch_10', 'epoch_10_no_motion'])
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    config = json.loads((Path(args.run_dir) / 'config.json').read_text())
    random.seed(20261012)
    torch.manual_seed(20261012)
    torch.set_num_threads(4)
    selected, metadata, counts = [], [], {}
    for split in (('test',) if args.history_only else ('train', 'test')):
        for cache in config['cache_paths']:
            index, meta = load_cache_data(cache, split=split)
            groups = defaultdict(list)
            for entry in index:
                stage = ('eot' if entry['link_name'] == 'eot' else
                         'root' if entry['link_idx'] == 0 else
                         'first' if entry['link_idx'] == 1 else 'later')
                groups[stage].append(entry)
            counts[split + '/' + meta['cache_name']] = dict(Counter(
                {k: len(v) for k, v in groups.items()}))
            chosen = [random.choice(v) for v in groups.values()]
            if args.history_only:
                unique = {}
                for entry in groups['later']:
                    unique.setdefault((entry['obj_id'], entry['link_idx']), entry)
                chosen = list(unique.values())
            dataset = CachedDataset(chosen, {meta['cache_name']: meta['data_root']},
                                    split=split, train_eot=True)
            for entry, sample in zip(chosen, dataset):
                if sample is None:
                    raise RuntimeError(f'Unreadable sample: {entry}')
                selected.append(sample)
                metadata.append(dict(split=split, category=meta['cache_name'],
                                     obj_id=entry['obj_id'], link_idx=entry['link_idx'],
                                     eot=entry['link_name'] == 'eot',
                                     history=len(sample['motion_history']),
                                     file_path=entry['file_path'],
                                     image=entry['whole_image_name']))
    (out / 'selection.json').write_text(json.dumps(dict(counts=counts, samples=metadata), indent=2))
    mc = load_model_config(config['model_config_path'])
    mc['ditrunner']['history_mode'] = 'geometry_motion'
    first = args.models[0].removesuffix('_no_motion')
    initial_path = config['checkpoint_path'] if first == 'pretrained' else str(Path(args.run_dir)/(first+'.pth'))
    model = URDFModel(config_dict=mc, device='cuda', init_mode='resume_from_ckpt',
                      checkpoint_path=initial_path).cuda().eval()
    scheduler = DDPMScheduler(**{k: mc['ditrunner']['noise_scheduler'][k] for k in
                               ('num_train_timesteps', 'beta_start', 'prediction_type', 'clip_sample')})
    parameter_counts = dict(total=sum(p.numel() for p in model.parameters()),
                            motion=sum(p.numel() for n, p in model.named_parameters()
                                       if 'motion_adaln' in n or 'motion_history_encoder' in n))
    all_rows = []
    loaded = first
    for label in args.models:
        checkpoint = label.removesuffix('_no_motion')
        if checkpoint != loaded:
            if checkpoint == 'pretrained':
                raise ValueError('pretrained must precede fine-tuned models to preserve zero-initialized motion layers')
            model.load_checkpoint(str(Path(args.run_dir) / (checkpoint + '.pth')))
            loaded = checkpoint
        start = len(all_rows)
        with torch.inference_mode():
            for offset in range(0, len(selected), 4):
                batch = {k: v.cuda() if torch.is_tensor(v) else v
                         for k, v in collate_fn(selected[offset:offset + 4]).items()}
                x = batch['target_labels']
                pre = torch.where((batch['link_indices'] == 0)[:, None, None],
                                  model.SoT.expand(x.shape[0], -1, -1), batch['encode_pres'])
                image = model.dino_adapter(batch['dino_features'])
                lengths = batch['motion_history_lengths']
                if label.endswith('no_motion'):
                    lengths = torch.zeros_like(lengths)
                for t in range(50, 1000, 100):
                    generator = torch.Generator(device='cuda').manual_seed(20261012 + offset * 1000 + t)
                    noise = torch.randn(x.shape, device='cuda', generator=generator)
                    times = torch.full((x.shape[0],), t, device='cuda', dtype=torch.long)
                    xt = scheduler.add_noise(x, noise, times)
                    v = scheduler.get_velocity(x, noise, times)
                    pred = model.DiTRunner.model(hidden_states=xt, timestep=times,
                        encoder_hidden_states=image,
                        encoder_hidden_states_2=torch.cat((pre, batch['encode_wholes']), dim=1),
                        motion_history=batch['motion_histories'], motion_history_lengths=lengths)
                    mse = (pred['latent'] - v).square().mean((1, 2))
                    zero_mse = v.square().mean((1, 2))
                    # Conditional Gaussian baseline: independent unit-normal x0/noise -> E[v|xt]=0.
                    for j in range(x.shape[0]):
                        row = dict(model=label, sample=offset+j, timestep=t, **metadata[offset+j],
                                   mse=mse[j].item(), zero_mse=zero_mse[j].item(),
                                   pred_rms=pred['latent'][j].square().mean().sqrt().item(),
                                   target_rms=x[j].square().mean().sqrt().item())
                        if t < 300 and batch['has_urdf'][j] and not batch['is_eot'][j]:
                            axis = batch['urdf_axes'][j]
                            mask = torch.ones_like(axis, dtype=torch.bool)
                            top = axis.abs().topk(2).values
                            if top[0] > 0.9 and top[1] < 0.1:
                                mask[axis.abs().argmax()] = False
                            row.update(origin_mae=(pred['param1'][j]-batch['urdf_origins'][j]).abs()[mask].mean().item(),
                                       axis_mae=(pred['param2'][j]-axis).abs().mean().item(),
                                       limits_mae=(pred['param3'][j]-batch['lower_upper_limits'][j]).abs().mean().item())
                        all_rows.append(row)
        rows = all_rows[start:]
        summary = {}
        for split in ('train', 'test'):
            for group in ('non_eot', 'eot', 'non_eot_history'):
                chosen = [r for r in rows if r['split']==split and
                          (r['eot'] if group=='eot' else not r['eot']) and
                          (r['history']>0 if group=='non_eot_history' else True)]
                if not chosen:
                    continue
                summary[split+'/'+group] = dict(n=len(chosen), **{
                    k: sum(r[k] for r in chosen)/len(chosen)
                    for k in ('mse', 'zero_mse', 'pred_rms', 'target_rms')})
        print(label, json.dumps(summary), flush=True)
        (out / (label + '.json')).write_text(json.dumps(summary, indent=2))
        fields = sorted(set().union(*(r.keys() for r in all_rows)))
        with (out / 'paired_losses.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_rows)
    (out / 'protocol.json').write_text(json.dumps(dict(seed=20261012, timesteps=list(range(50,1000,100)),
        parameter_counts=parameter_counts, run_dir=args.run_dir, config=config,
        history_only=args.history_only, models=args.models,
        note=('All unique validation later-part pairs, one image each.' if args.history_only else
              'One random sample per category/split/stage; equal sample weights, not full validation.')), indent=2))


if __name__ == '__main__':
    main()
