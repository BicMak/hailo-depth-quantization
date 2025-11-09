import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path

# 파일 경로 설정
depth_results_dir = 'depth_results'
gt_dir = f'{depth_results_dir}/depth_data'
pred_dir = f'{depth_results_dir}/pred_depth'

# GT 파일 5개 가져오기 (정렬해서 인덱스 맞추기)
gt_files = sorted(glob.glob(f'{gt_dir}/*.npy'))[:5]
pred_files = sorted(glob.glob(f'{pred_dir}/*.npy'))[:5]

print(f"GT files: {len(gt_files)}")
print(f"Pred files: {len(pred_files)}")

# 데이터 로드
gt_depths = []
pred_depths = []

for gt_file in gt_files:
    gt_data = np.load(gt_file)
    # 모든 차원 1 제거해서 (256, 256)으로 만들기
    gt_data = np.squeeze(gt_data)
    gt_depths.append(gt_data)

for pred_file in pred_files:
    pred_data = np.load(pred_file)
    # 모든 차원 1 제거해서 (256, 256)으로 만들기
    pred_data = np.squeeze(pred_data)
    pred_depths.append(pred_data)

# 시각화
fig, axes = plt.subplots(2, 5, figsize=(20, 8))

# 첫 번째 행: GT depth
for i in range(5):
    im1 = axes[0, i].imshow(gt_depths[i], cmap='viridis')
    axes[0, i].set_title(f'GT Depth {i}', fontsize=12)
    axes[0, i].axis('off')
    plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)

# 두 번째 행: Pred depth
for i in range(5):
    im2 = axes[1, i].imshow(pred_depths[i], cmap='viridis')
    axes[1, i].set_title(f'Pred Depth {i}', fontsize=12)
    axes[1, i].axis('off')
    plt.colorbar(im2, ax=axes[1, i], fraction=0.046, pad=0.04)

plt.suptitle('GT vs Predicted Depth Maps', fontsize=16, y=1.00)
plt.tight_layout()
plt.show()

# 차이 계산 (MSE)
print("\n=== Depth Comparison ===")
for i in range(5):
    mse = np.mean((gt_depths[i] - pred_depths[i]) ** 2)
    mae = np.mean(np.abs(gt_depths[i] - pred_depths[i]))
    print(f"Image {i}: MSE={mse:.6f}, MAE={mae:.6f}")