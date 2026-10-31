"""Summarize checkpoint-separated fixed-path stability evidence."""
import json
from pathlib import Path


def summarize(directory):
    rows = []
    for path in sorted(directory.glob('*.json')):
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or 'checkpoints' not in data:
            continue
        for checkpoint in data['checkpoints']:
            for window in checkpoint['windows']:
                modes = {m['grad_steps']: m for m in window['modes']}
                full = modes[64]
                norm = full['gradients']['loss']['norm64']
                short = modes[8]['gradients']['loss']['norm64']
                row = dict(
                    source=path.name, checkpoint=checkpoint['checkpoint'], identity=window['identity'],
                    full_norm64=norm, last8_norm64=short, last1_norm64=modes[1]['gradients']['loss']['norm64'],
                    full_to_last8=norm / short if norm is not None and short else None,
                    endpoint_rms=(full['endpoint_gradient'] or {}).get('rms'),
                    gradient_elements_finite=all(m['gradients']['loss']['finite'] for m in modes.values()),
                    max_forward_difference=max(v for m in modes.values() for v in m['forward_differences'].values()),
                    decoder=full['decoder_gradients'] if 'decoder_gradients' in full else None,
                )
                if full.get('joint_recurrence_vjp'):
                    row['joint_recurrence_vjp'] = full['joint_recurrence_vjp']
                rows.append(row)
    return rows


if __name__ == '__main__':
    directory = Path('results/lasp_stability_v2')
    rows = summarize(directory)
    target = directory / 'panel_summary.json'
    target.write_text(json.dumps(rows, indent=2, allow_nan=False))
    print(f'{len(rows)} checkpoint-window observations -> {target}')
    for checkpoint in sorted({r['checkpoint'] for r in rows}):
        subset = [r for r in rows if r['checkpoint'] == checkpoint]
        ratios = [r['full_to_last8'] for r in subset if r['identity']['horizon'] == 64 and r['full_to_last8'] is not None]
        print(checkpoint, 'n=', len(subset), 'h64 full/last8 range=', (min(ratios), max(ratios)) if ratios else None,
              'finite=', all(r['gradient_elements_finite'] for r in subset),
              'forward max=', max(r['max_forward_difference'] for r in subset))
