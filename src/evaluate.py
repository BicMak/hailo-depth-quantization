"""Compare ONNX FP32 and HAR emulator outputs on the same images.

ONNX output is the reference. For every requested HAR context we report
  - ONNX vs context            : total difference up to that stage
  - previous context vs context: loss added by that stage only
      native       <- ONNX          : parsing (ONNX -> HAR)
      fp_optimized <- native        : FP optimizations (equalization, ...)
      quantized    <- fp_optimized  : quantization
A parsed-only HAR supports native / fp_optimized; a quantized HAR supports all three.
"""
import argparse
import os

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import numpy as np

from common import ALIGN_MODES, HAR_CONTEXTS, METRIC_NAMES, evaluate_pair, load_dataset, run_har, run_onnx, write_metrics_csv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', default='MIDAS_model/model-small.onnx')
    parser.add_argument('--har', default='Midas_quantize_model.har')
    parser.add_argument('--contexts', nargs='+', choices=HAR_CONTEXTS, default=list(HAR_CONTEXTS))
    parser.add_argument('--images-root', default='DA-2K/DA-2K/images')
    parser.add_argument('--num-images', type=int, default=None, help='use only the first N images (default: all)')
    parser.add_argument('--out-dir', default='eval')
    parser.add_argument('--align', nargs='+', choices=ALIGN_MODES, default=list(ALIGN_MODES),
                        help='median: scale only (KITTI-style), lsq: least-squares scale + shift (MiDaS-style)')
    parser.add_argument('--save-npy', action='store_true', help='also save raw outputs as <out-dir>/<name>.npy')
    return parser.parse_args()


def build_comparisons(contexts):
    """ONNX vs every context, then each context vs the one before it."""
    comparisons = [('onnx', ctx) for ctx in contexts]
    comparisons += list(zip(contexts, contexts[1:]))
    return comparisons


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    contexts = [c for c in HAR_CONTEXTS if c in args.contexts]  # keep pipeline order

    paths, dataset = load_dataset(args.images_root, args.num_images)

    outputs = {'onnx': run_onnx(args.onnx, dataset)}
    for ctx in contexts:
        outputs[ctx] = run_har(args.har, dataset, ctx)
    if args.save_npy:
        for name, out in outputs.items():
            np.save(f'{args.out_dir}/{name}.npy', out)

    print(f"\nONNX: {args.onnx}\nHAR : {args.har}\nimages: {len(paths)}")
    for align in args.align:
        # median keeps the original file names; other modes get a suffix
        suffix = '' if align == 'median' else f'_{align}'
        print(f"\n[align: {align}]")
        print(f"{'reference -> target':30s}" + ''.join(f"{k:>10s}" for k in METRIC_NAMES)
              + f"{'a1 med':>10s}{'a1 p5':>10s}{'scale med':>10s}{'shift med':>10s}")
        for ref, tgt in build_comparisons(contexts):
            metrics, params = evaluate_pair(outputs[ref], outputs[tgt], align)
            write_metrics_csv(f'{args.out_dir}/{ref}_vs_{tgt}{suffix}.csv', metrics, paths, params)
            a1 = metrics[:, METRIC_NAMES.index('a1')]
            print(f"{ref + ' -> ' + tgt:30s}" + ''.join(f"{v:10.4f}" for v in np.nanmean(metrics, axis=0))
                  + f"{np.nanmedian(a1):10.2f}{np.nanpercentile(a1, 5):10.2f}"
                  + f"{np.median(params[:, 0]):10.4f}{np.median(params[:, 1]):10.3f}")
    print(f"\nPer-image CSVs saved to {args.out_dir}/")


if __name__ == '__main__':
    main()
