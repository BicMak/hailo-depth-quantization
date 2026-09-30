"""HAR (full precision) -> quantized HAR."""
import argparse
import os

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

from hailo_sdk_client import ClientRunner

from common import load_dataset

DEFAULT_ALLS = """\
model_optimization_flavor(optimization_level=4, compression_level=0, batch_size=32)
model_optimization_config(calibration, batch_size=32, calibset_size=1024)
model_optimization_config(checker_cfg, policy=disabled)
pre_quantization_optimization(equalization, policy=enabled)
pre_quantization_optimization(weights_clipping, layers={*}, mode=mmse)
post_quantization_optimization(finetune, policy=enabled, learning_rate=0.00001, epochs=5, dataset_size=1024)
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--har-in', default='Midas_hailo_model_normalize.har')
    parser.add_argument('--har-out', default='Midas_quantize_model.har')
    parser.add_argument('--images-root', default='DA-2K/DA-2K/images')
    parser.add_argument('--alls', default=None, help='model script file (default: DEFAULT_ALLS)')
    return parser.parse_args()


def main():
    args = parse_args()

    alls = open(args.alls).read() if args.alls else DEFAULT_ALLS
    print("Model script:\n" + alls)

    _, calib_dataset = load_dataset(args.images_root)

    runner = ClientRunner(har=args.har_in)
    runner.load_model_script(alls)
    runner.optimize(calib_dataset)
    runner.save_har(args.har_out)
    print(f"Saved {args.har_out}")


if __name__ == '__main__':
    main()
