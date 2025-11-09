# General imports used throughout the tutorial
# file operations
import json
import os

import numpy as np
import tensorflow as tf
from IPython.display import SVG
from matplotlib import patches
from matplotlib import pyplot as plt
from PIL import Image
from tensorflow.python.eager.context import eager_mode
from pathlib import Path
import random



# import the hailo sdk client relevant classes
from hailo_sdk_client import ClientRunner, InferenceContext

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

IMAGES_TO_VISUALIZE = 5
# -----------------------------------------
# Pre processing (prepare the input images)
# -----------------------------------------
def preproc(image, output_height=256, output_width=256, resize_side=300, rescale=False):
    """imagenet-standard: aspect-preserving resize to 256px smaller-side, then central-crop to 224px"""
    with eager_mode():
        h, w = image.shape[0], image.shape[1]
        scale = tf.cond(tf.less(h, w), lambda: resize_side / h, lambda: resize_side / w)
        resized_image = tf.image.resize(tf.expand_dims(image, 0), [int(h * scale), int(w * scale)])
        cropped_image = tf.image.resize_with_crop_or_pad(resized_image, output_height, output_width)
        cropped_image = tf.cast(cropped_image, tf.float32)

        if rescale:
            cropped_image = cropped_image / 255.0

        return tf.squeeze(cropped_image)


# -----------------------------------------------------
# Image names
# -----------------------------------------------------
def get_all_image_files(root_dir='DA-2K/DA-2K/images'):
    """
    images 폴더의 모든 하위 폴더에서 이미지 파일명을 수집합니다.

    Args:
        root_dir: 이미지가 있는 루트 디렉토리 경로

    Returns:
        list: 모든 이미지 파일의 전체 경로 리스트
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    image_files = []

    # Path 객체로 변환
    root_path = Path(root_dir)

    # 모든 하위 폴더를 재귀적으로 탐색
    for file_path in root_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_files.append(str(file_path))

    return sorted(image_files)


def visualize_results(
    images,
    image_depth,
    first_title="Full Precision",
    second_title="Other",
):
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    # 첫 번째 행: Full Precision 이미지 5장
    for i in range(5):
        axes[0, i].imshow(images[i])
        axes[0, i].axis('off')
        if i == 2:  # 중앙에만 제목 표시
            axes[0, i].set_title(first_title, fontsize=14, pad=10)
    
    # 두 번째 행: Other 이미지 5장
    for i in range(5):
        axes[1, i].imshow(image_depth[i])
        axes[1, i].axis('off')
        if i == 2:  # 중앙에만 제목 표시
            axes[1, i].set_title(second_title, fontsize=14, pad=10)
    
    plt.tight_layout()
    plt.show()
    
model_name = "Har_file/Midas_hailo_model_normalize.har"
runner = ClientRunner(har=model_name)
# By default it uses the hw_arch that is saved on the HAR. For overriding, use the hw_arch flag.


images_list = get_all_image_files()
print(f"Total images found: {len(images_list)}")

select_images_list = random.sample(images_list, 5)

# 이미지 전수 검사진행
# Create an un-normalized dataset for visualization
image_dataset = np.zeros((len(select_images_list), 256, 256, 3))
original_dataset = np.zeros((len(select_images_list), 256, 256, 3))

# Create a normalized dataset to feed into the Native emulator
for idx, img_name in enumerate(select_images_list):
    img = np.array(Image.open(img_name))
    original_dataset[idx, :, :, :] = preproc(img,rescale=False)
    img_preproc = preproc(img,rescale=True)
    image_dataset[idx, :, :, :] = img_preproc.numpy()
print(f"Total images size: {image_dataset.shape}")


quantize_dataset = np.zeros((len(images_list), 256, 256, 3))
for idx, img_name in enumerate(images_list):
    img = np.array(Image.open(img_name))
    img_preproc = preproc(img,rescale=True)
    quantize_dataset[idx, :, :, :] = img_preproc.numpy()

alls_lines = [
    # Batch size is 8 by default; 2 was used for stability on PCs with low amount of RAM / VRAM
    "model_optimization_flavor(optimization_level=4, compression_level=2, batch_size=32)\n",
    # The following line is needed because resnet_v1_18 is a really small model, and the compression_level is always reverted back to 0.'
    # To force using compression_level with small models, the following line should be used (compression level=4 equals to 80% 4-bit):
    "model_optimization_config(calibration, batch_size=32, calibset_size=1024)\n",
    # The application of the compression could be seen by the [info] messages: "Assigning 4bit weight to layer .."
    "pre_quantization_optimization(equalization, policy=enabled)\n",
    "pre_quantization_optimization(weights_clipping, layers={*}, mode=mmse)\n",
    
    "post_quantization_optimization(bias_correction, policy=enabled)\n",
    "post_quantization_optimization(finetune, dataset_size=1024, policy=enabled,epochs=5)\n"
]
# -- Reduces weights memory by 80% !



print("start optimize")
runner.load_model_script("".join(alls_lines))
runner.optimize(quantize_dataset)

names = "Midas"
hailo_model_har_name = f"{names}_quantize_model.har"
runner.save_har(hailo_model_har_name)



print("after quantize infer")
# Notice that the original images are used, because normalization is present in the model
with runner.infer_context(InferenceContext.SDK_QUANTIZED) as ctx:
    depth_results = runner.infer(ctx, image_dataset)

visualize_results(
    images=image_dataset,
    image_depth=depth_results,
    first_title="Original Images (5 Random)",
    second_title="Depth Estimation Results"
)

hailo_model_har_name = f"{names}_quantize_model_analyzenoise.har"
runner.analyze_noise(dataset = image_dataset, 
                     batch_size=1, 
                     data_count=16,
                     analyze_mode = "advanced")  # Batch size is 1 by default
runner.save_har(hailo_model_har_name)