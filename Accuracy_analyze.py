import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image
from tensorflow.python.eager.context import eager_mode


from hailo_sdk_client import ClientRunner

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# -----------------------------------------
# Pre processing (prepare the input images)
# -----------------------------------------
def preproc(image, output_height=256, output_width=256, resize_side=260, normalize=False):
    """imagenet-standard: aspect-preserving resize to 256px smaller-side, then central-crop to 224px"""
    with eager_mode():
        h, w = image.shape[0], image.shape[1]
        scale = tf.cond(tf.less(h, w), lambda: resize_side / h, lambda: resize_side / w)
        resized_image = tf.image.resize(tf.expand_dims(image, 0), [int(h * scale), int(w * scale)])
        cropped_image = tf.image.resize_with_crop_or_pad(resized_image, output_height, output_width)
        cropped_image = tf.image.convert_image_dtype(cropped_image, tf.uint8) 

        if normalize:
            # Default normalization parameters for ImageNet
            cropped_image = (cropped_image - [123.675, 116.28, 103.53]) / [58.395, 57.12, 57.375]

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


#2. Load the dataset
images_list = get_all_image_files()
print(f"Total images found: {len(images_list)}")
image_dataset = np.zeros((len(images_list), 256, 256, 3))
# Create a normalized dataset to feed into the Native emulator
for idx, img_name in enumerate(images_list):
    img = np.array(Image.open(img_name))
    img_preproc = preproc(img)
    image_dataset[idx, :, :, :] = img_preproc.numpy()

har_path = "Midas_quantize_model.har"

#3. check model input
if len(tf.config.list_physical_devices("GPU")) == 0:
    print("Warning: you are running the accuracy analysis tool without a GPU, expect long running time.")

runner = ClientRunner()
runner.load_har(har_path)
runner.analyze_noise(dataset = image_dataset, 
                     batch_size=16, 
                     data_count=64,
                     analyze_mode = "advanced")  # Batch size is 1 by default
runner.save_har(f"test_Midas_quantize_model_analyzenoise.har")
