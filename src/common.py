"""Shared code: preprocessing, dataset loading, ONNX/HAR inference, depth metrics."""
import csv
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.python.eager.context import eager_mode

IMAGE_SIZE = 256
RESIZE_SIDE = 300
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
REL_ERR_THRESHOLD = 0.001  # 0.1%
METRIC_NAMES = ['abs_rel', 'sq_rel', 'rms', 'log_rms', 'a1', 'a2', 'a3', 'rel<0.1%']
HAR_CONTEXTS = ('native', 'fp_optimized', 'quantized')


# -----------------------------------------
# Data
# -----------------------------------------
def list_images(root_dir):
    """root_dir 아래 모든 이미지 경로 (정렬)"""
    return sorted(str(p) for p in Path(root_dir).rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def preproc(image):
    """Aspect-preserving resize (short side 300) -> center crop 256 -> /255. Model normalizes internally."""
    with eager_mode():
        h, w = image.shape[0], image.shape[1]
        scale = tf.cond(tf.less(h, w), lambda: RESIZE_SIDE / h, lambda: RESIZE_SIDE / w)
        resized_image = tf.image.resize(tf.expand_dims(image, 0), [int(h * scale), int(w * scale)])
        cropped_image = tf.image.resize_with_crop_or_pad(resized_image, IMAGE_SIZE, IMAGE_SIZE)
        return tf.squeeze(tf.cast(cropped_image, tf.float32) / 255.0).numpy()


def load_dataset(images_root, num_images=None):
    """Returns (paths, NHWC float32 array)"""
    paths = list_images(images_root)[:num_images]
    dataset = np.zeros((len(paths), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
    for idx, path in enumerate(paths):
        dataset[idx] = preproc(np.array(Image.open(path)))
    print(f"Loaded {len(paths)} images from {images_root}")
    return paths, dataset


# -----------------------------------------
# Inference
# -----------------------------------------
def run_onnx(onnx_path, dataset):
    """ONNX FP32 inference (NHWC dataset -> NCHW input). Returns (N, H, W)."""
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name
    return np.stack([
        np.squeeze(sess.run(None, {input_name: np.transpose(x, (2, 0, 1))[None]})[0])
        for x in dataset
    ])


def run_har(har_path, dataset, context):
    """Hailo emulator inference. context: native | fp_optimized | quantized. Returns (N, H, W)."""
    from hailo_sdk_client import ClientRunner, InferenceContext

    contexts = {
        'native': InferenceContext.SDK_NATIVE,
        'fp_optimized': InferenceContext.SDK_FP_OPTIMIZED,
        'quantized': InferenceContext.SDK_QUANTIZED,
    }
    runner = ClientRunner(har=har_path)
    with runner.infer_context(contexts[context]) as ctx:
        results = runner.infer(ctx, dataset)
    return np.squeeze(np.asarray(results), axis=-1)


# -----------------------------------------
# Metrics
# -----------------------------------------
ALIGN_MODES = ('median', 'lsq')


def align_prediction(gt, pred, mask, align='median'):
    """Map pred onto gt's range. Returns (aligned_pred, scale, shift).

    median: pred * s, s = median(gt) / median(pred)                 (KITTI-style, scale only)
    lsq   : pred * s + t, (s, t) = least squares fit of pred -> gt  (MiDaS-style scale + shift)
    """
    if align == 'median':
        scale = np.median(gt[mask]) / np.median(pred[mask])
        return pred * scale, scale, 0.0
    if align == 'lsq':
        g, p = gt[mask].astype(np.float64), pred[mask].astype(np.float64)
        scale = np.sum((p - p.mean()) * (g - g.mean())) / np.sum((p - p.mean()) ** 2)
        shift = g.mean() - scale * p.mean()
        return pred * scale + shift, scale, shift
    raise ValueError(f"unknown align mode: {align}")


def compute_metrics(gt_data, pred_data, align='median'):
    """GT 대비 pred의 depth 지표. 반환: (METRIC_NAMES 순서의 값 리스트, scale, shift)

    align='median' 은 기존 계산과 완전히 같음 (log_rms도 기존처럼 정렬 전 pred 사용).
    align='lsq' 는 정렬된 pred를 모든 지표에 사용하고, 0 이하로 내려간 값은 eps로 자름.
    """
    gt_data = np.squeeze(gt_data)
    pred_data = np.squeeze(pred_data)

    mask = gt_data > 0

    pred_data_scaled, scale, shift = align_prediction(gt_data, pred_data, mask, align)

    eps = 1e-6
    if align == 'lsq':
        pred_data_scaled = np.maximum(pred_data_scaled, eps)
        pred_for_log = pred_data_scaled
    else:
        pred_for_log = pred_data

    abs_rel = np.mean(np.abs(gt_data[mask] - pred_data_scaled[mask]) / gt_data[mask])
    sq_rel = np.mean(np.square(gt_data[mask] - pred_data_scaled[mask]) / np.square(gt_data[mask]))
    rms = np.sqrt(np.mean(np.square((gt_data[mask] - pred_data_scaled[mask]))))

    log_rms = np.sqrt(np.mean(np.square(np.log(gt_data[mask] + eps) - np.log(pred_for_log[mask] + eps))))

    with np.errstate(divide='ignore'):
        ratio = np.maximum(pred_data_scaled[mask] / gt_data[mask], gt_data[mask] / pred_data_scaled[mask])
    a1 = np.mean(ratio < 1.25) * 100
    a2 = np.mean(ratio < 1.25**2) * 100
    a3 = np.mean(ratio < 1.25**3) * 100
    # a1과 같은 ratio 기준, 픽셀별 오차가 REL_ERR_THRESHOLD 미만인 비율 (%)
    within_rel = np.mean(ratio < 1 + REL_ERR_THRESHOLD) * 100

    return [abs_rel, sq_rel, rms, log_rms, a1, a2, a3, within_rel], scale, shift


def evaluate_pair(gt_batch, pred_batch, align='median'):
    """Per-image metrics for two (N, H, W) batches.
    Returns (metrics (N, len(METRIC_NAMES)), align_params (N, 2) = [scale, shift])."""
    results = [compute_metrics(g, p, align) for g, p in zip(gt_batch, pred_batch)]
    metrics = np.array([r[0] for r in results], dtype=np.float64)
    params = np.array([[r[1], r[2]] for r in results], dtype=np.float64)
    return metrics, params


def write_metrics_csv(csv_file, metrics, paths=None, align_params=None):
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['idx'] + (['image'] if paths else []) + METRIC_NAMES
                        + (['scale', 'shift'] if align_params is not None else []))
        for idx, row in enumerate(metrics):
            writer.writerow([idx] + ([paths[idx]] if paths else []) + list(row)
                            + (list(align_params[idx]) if align_params is not None else []))
