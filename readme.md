# MiDaS v2.1 Small Model Quantization for Hailo NPU

## 1. Project Overview

### 1.1 Objectives
* Quantize a Depth Estimation model to optimize for Hailo NPU deployment
* Absence of state-of-the-art depth estimation models in the existing Hailo model zoo
* Fill the gap in quantization research for depth estimation tasks in Hailo NPU application (most prior work focuses on object detection and segmentation)

## 2. Dataset
![alt text](image.png)
* **Dataset**: DA-2K dataset used for depth estimation evaluation
* **Rationale**: Selected because it aligns with state-of-the-art models like Depth Anything
* **Preprocessing**: Images resized to [3, 256, 256] format only. No additional preprocessing applied as the model includes normalization in its input layer
* **Total samples**: 1,033 images

## 3. Model Architecture
* **Model**: MiDaS v2.1 Small (256x256)
* **Input shape**: [1, 3, 256, 256]
* **Output shape**: [1, 1, 256, 256]

### 3.1 Model Selection Rationale
* MiDaS v2.1 Small uses exclusively CNN-based layers throughout the architecture
* CNN-based models are more quantization-friendly compared to Transformers on NPU devices like Hailo
* Current quantization performance for Transformers on Hailo is suboptimal, making CNN architectures more suitable for this application

## 4. Quantization Method

### 4.1 Configuration
* **Batch size**: 32
* **Calibration set size**: 1,024 samples
* **Quantization precision**: INT8 (all layers quantized to 8-bit integers due to parameter constraints)

### 4.2 Pre-Processing Techniques
* **Layer Equalization**: Applied uniformly across all layers (default Hailo parameters for other settings)
* **Weights Clipping**: NMSE-based clipping applied across all layers (default Hailo parameters for other settings)

### 4.3 Post-Processing Techniques
* **Adaround**: 320 epochs (default Hailo parameters for other settings)
* **Fine-tuning**: 5 epochs (default Hailo parameters for other settings)

## 5. Experimental Results

### 5.1 Quantitative Evaluation
![alt text](image-1.png)
Evaluation metrics computed on 1,024 test images using scale-matched comparison (following KITTI benchmark standards).

| Metric |  Quantized (INT8) |
|--------|---|
| abs_rel | 0.218 |
| sq_rel | 4959.37 |
| rms | 70.91 |
| log_rms | 1.438 |
| **a1** | **81.94%** |
| **a2** | **93.02%** | 
| **a3** | **95.84%** |

**Note**: Pixels with zero depth values (invalid regions) were excluded from evaluation calculations.

### 5.2 Model Size Comparison
| Metric | Full Precision | Quantized (INT8) | Compression Ratio |
|--------|---|---|---|
| Model Size | 63.67 MB | 19.90 MB | **68.8%** |

### 5.3 Qualitative Evaluation
* Visual comparison of depth maps: Full Precision vs. Quantized (INT8)

## 6. Conclusions

* **Task Complexity**: Depth estimation presents greater challenges compared to other vision tasks (detection, segmentation) due to its regression nature and per-pixel evaluation requirements
* **Model Efficiency**: Achieved significant compression (68.8% size reduction) while maintaining reasonable accuracy, enabling deployment on edge devices
* **Practical Impact**: The 19.90 MB quantized model is deployable on resource-constrained edge devices

## 7. Future Work

* Deploy quantized model on Hailo-8 NPU device to measure actual inference performance:
  - Latency (milliseconds)
  - Throughput (FPS)
  - Memory utilization
* Test on-device deployment on Raspberry Pi 5 for end-to-end inference evaluation


## 8. Reference 
- Midas model : https://github.com/isl-org/MiDaS
- Estimation method : https://github.com/FangGet/tf-monodepth2
- DA 2K dataset : https://huggingface.co/datasets/depth-anything/DA-2K


**Author**: [Jiwon Kim]  
**Date**: [9 Nov 2025]  
