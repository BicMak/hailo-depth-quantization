"""Qualitative comparison: ONNX FP32 vs Hailo INT8 (emulator) on the best / worst images.

Images are ranked by per-image a1 from an evaluate.py CSV (default: onnx_vs_quantized.csv).
Each row shows: input | FP32 depth | INT8 depth | relative error of the scale-matched INT8 output.
"""
import argparse
import csv
import os

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from common import preproc, run_har, run_onnx

A1_THRESHOLD = 0.25  # a1 counts a pixel as correct when relative error < 25%
TEXT = '#1f2328'
MUTED = '#59636e'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', default='MIDAS_model/model-small.onnx')
    parser.add_argument('--har', required=True, help='quantized HAR')
    parser.add_argument('--csv', required=True, help='per-image metrics CSV from evaluate.py (e.g. eval/onnx_vs_quantized.csv)')
    parser.add_argument('--num', type=int, default=5, help='images per group')
    parser.add_argument('--out-dir', default='.')
    return parser.parse_args()


def load_ranked(csv_path):
    """[(image_path, a1), ...] sorted by a1 descending; ties (many images reach a1 = 100%) broken by lower abs_rel"""
    with open(csv_path) as f:
        rows = [(r['image'], float(r['a1']), float(r['abs_rel'])) for r in csv.DictReader(f)]
    rows.sort(key=lambda r: (-r[1], r[2]))
    return [(path, a1) for path, a1, _ in rows]


def relative_error(fp, q):
    """Per-pixel relative error after the same median scaling used in the metrics."""
    mask = fp > 0
    scaled = q * np.median(fp[mask]) / np.median(q[mask])
    err = np.full(fp.shape, np.nan)
    err[mask] = np.abs(scaled[mask] - fp[mask]) / fp[mask]
    return err


def plot_group(title, paths, a1s, images, fp_out, q_out, out_path):
    n = len(paths)
    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n + 0.8), constrained_layout=True)
    axes = np.atleast_2d(axes)
    fig.suptitle(title, fontsize=14, color=TEXT)

    cmap_err = plt.get_cmap('Oranges').copy()
    cmap_err.set_bad('#d0d7de')  # excluded pixels (reference = 0)
    for i in range(n):
        vmin, vmax = np.percentile(fp_out[i], [1, 99])  # same range for FP32 and INT8
        err = relative_error(fp_out[i], q_out[i])
        panels = [
            (images[i], None, 'Input'),
            (fp_out[i], dict(cmap='Blues', vmin=vmin, vmax=vmax), 'FP32 (ONNX)'),
            (q_out[i] * np.median(fp_out[i][fp_out[i] > 0]) / np.median(q_out[i][fp_out[i] > 0]),
             dict(cmap='Blues', vmin=vmin, vmax=vmax), 'INT8 (Hailo, scale-matched)'),
            (err, dict(cmap=cmap_err, vmin=0, vmax=A1_THRESHOLD), 'Relative error (0-25%)'),
        ]
        for j, (img, kw, name) in enumerate(panels):
            ax = axes[i, j]
            im = ax.imshow(img, **(kw or {}))
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if i == 0:
                ax.set_title(name, fontsize=11, color=TEXT)
            if j == 3:
                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, format=lambda x, _: f'{x:.0%}')
                cb.ax.tick_params(labelsize=8, colors=MUTED)
                cb.outline.set_visible(False)
        category = paths[i].split('/images/')[-1].split('/')[0]
        axes[i, 0].set_ylabel(f'{category}\na1 {a1s[i]:.2f}%', fontsize=10, color=TEXT)

    fig.text(0.5, -0.01, 'Depth: darker = closer (relative inverse depth). Error: gray = excluded pixels; '
             f'saturated = at or above the a1 threshold ({A1_THRESHOLD:.0%}).',
             ha='center', fontsize=9, color=MUTED)
    fig.savefig(out_path, dpi=110, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ranked = load_ranked(args.csv)
    groups = {
        'top': ranked[:args.num],
        'bottom': ranked[-args.num:][::-1],  # worst first
    }
    selected = [p for rows in groups.values() for p, _ in rows]

    images = np.stack([preproc(np.array(Image.open(p))) for p in selected])
    fp_out = run_onnx(args.onnx, images)
    q_out = run_har(args.har, images, 'quantized')

    for offset, (name, rows) in enumerate(groups.items()):
        sl = slice(offset * args.num, (offset + 1) * args.num)
        label = 'highest' if name == 'top' else 'lowest'
        plot_group(f'FP32 vs INT8: {args.num} images with the {label} a1',
                   [p for p, _ in rows], [a for _, a in rows],
                   images[sl], fp_out[sl], q_out[sl],
                   os.path.join(args.out_dir, f'qualitative_{name}{args.num}.png'))


if __name__ == '__main__':
    main()
