import numpy as np
import matplotlib.pyplot as plt
import glob
import csv
from pathlib import Path

# 파일 경로 설정
depth_results_dir = 'depth_results'
gt_dir = f'{depth_results_dir}/depth_data'
pred_dir = f'{depth_results_dir}/pred_depth'

# GT 파일 가져오기 (정렬해서 인덱스 맞추기)
gt_files = sorted(glob.glob(f'{gt_dir}/*.npy'))
pred_files = sorted(glob.glob(f'{pred_dir}/*.npy'))

# CSV 파일 열기
csv_file = 'depth_evaluation_results.csv'
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    # 헤더 쓰기
    writer.writerow(['idx', 'abs_rel', 'sq_rel', 'rms', 'log_rms', 'a1', 'a2', 'a3'])
    
    for idx, (gt_np, pred_np) in enumerate(zip(gt_files, pred_files)):
        gt_data = np.load(gt_np)
        pred_data = np.load(pred_np)
        
        
        # Shape 정규화
        gt_data = np.squeeze(gt_data)      # (1, 1, 256, 256) -> (256, 256)
        pred_data = np.squeeze(pred_data)  # 같은 방식

        mask = gt_data > 0

        print(f"GT min: {gt_data[mask].min()}, max: {gt_data[mask].max()}")
        print(f"Pred min: {pred_data[mask].min()}, max: {pred_data[mask].max()}")

        # Scale matching (KITTI 방식)
        scale = np.median(gt_data[mask]) / np.median(pred_data[mask])
        pred_data_scaled = pred_data * scale

        # 그 다음에 메트릭 계산
        abs_rel = np.mean(np.abs(gt_data[mask] - pred_data_scaled[mask]) / gt_data[mask])
        sq_rel = np.mean(np.square(gt_data[mask] - pred_data_scaled[mask]) / np.square(gt_data[mask]))
        rms = np.sqrt(np.mean(np.square((gt_data[mask] - pred_data_scaled[mask]))))

        eps = 1e-6
        log_rms = np.sqrt(np.mean(np.square(np.log(gt_data[mask] + eps) - np.log(pred_data[mask] + eps))))

        ratio = np.maximum(pred_data_scaled[mask] / gt_data[mask], gt_data[mask] / pred_data_scaled[mask])
        a1 = np.mean(ratio < 1.25) * 100
        a2 = np.mean(ratio < 1.25**2) * 100
        a3 = np.mean(ratio < 1.25**3) * 100

        writer.writerow([idx, abs_rel, sq_rel, rms, log_rms, a1, a2, a3])
        print(f"{idx} / a1 = {a1:.2f}, a2 = {a2:.2f}, a3 = {a3:.2f}")
        
print(f"\nResults saved to {csv_file}")