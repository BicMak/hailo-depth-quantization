# MiDaS v2.1 Small Quantization for Hailo NPU

[한국어](README.md)

## 1. Project Overview

### 1.1 Objective
* Quantize a depth estimation model (MiDaS v2.1 Small) and optimize it for deployment on a Hailo NPU

## 2. Dataset
![alt text](assets/DA2K_sample_images.png)
* **Dataset**: DA-2K, a depth estimation evaluation set. Only the images are used; the annotations are not
* **Why DA-2K**:
    - It is the evaluation set used by Depth Anything, one of the state-of-the-art models for monocular depth estimation
    - It covers scenes from 8 categories (indoor, outdoor, underwater, aerial, object, transparent/reflective, non-real, adverse style), which reduces the risk of results being skewed toward a particular type of scene (67-195 images per category)
* **Preprocessing**: aspect-preserving resize (short side 300) → center crop 256×256 → scale to [0, 1]. ImageNet mean/std normalization is already inside the ONNX graph (Sub/Div), so it is not applied separately
* **Total samples**: 1,033 images

## 3. Model Architecture
* **Model**: MiDaS v2.1 Small (256x256)
* **Input shape**: [1, 3, 256, 256]
* **Output shape**: [1, 256, 256] (relative inverse depth)

### 3.1 Why MiDaS v2.1 Small
* MiDaS v2.1 Small was designed from the start as a lightweight model for on-device deployment
* Its architecture is CNN-based, which was judged to be favorable for quantization. The Hailo Dataflow Compiler documentation is also written around CNNs, so quantizing it for this hardware was expected to be straightforward

## 4. Quantization Method

### 4.1 Pipeline
| Step | Script | Description |
|---|---|---|
| Parse | `src/parse.py` | ONNX → HAR (FP). Sets the opset-10 `Resize` to an explicit `asymmetric` mode before parsing (see 5.2) |
| Quantize | `src/quantize.py` | HAR → INT8 HAR (using the model script below) |
| Evaluate | `src/evaluate.py` | Compares ONNX FP32 with each Hailo emulator stage (`native`, `fp_optimized`, `quantized`) |

```bash
./run_pipeline.sh                  # parse -> quantize -> evaluate (outputs in runs/<timestamp>/)
./run_pipeline.sh parse evaluate   # parsing check only
./run_pipeline.sh -h               # all options
```

### 4.2 Configuration (final model)
| Item | Setting |
|---|---|
| Optimization level | 4 (`compression_level=0`, so all layers are 8-bit) |
| Calibration | 1,024 images, batch size 32 |
| Pre-quantization | Equalization; weights clipping (MMSE) on all layers |
| Post-quantization | Adaround (`train_all`, 320 epochs, 1,024 images, level-4 default) → fine-tuning (5 epochs, 1,024 images, lr 1e-5) |
| Disabled | Layer noise analysis checker (diagnostics only) |

```
model_optimization_flavor(optimization_level=4, compression_level=0, batch_size=32)
model_optimization_config(calibration, batch_size=32, calibset_size=1024)
model_optimization_config(checker_cfg, policy=disabled)
pre_quantization_optimization(equalization, policy=enabled)
pre_quantization_optimization(weights_clipping, layers={*}, mode=mmse)
post_quantization_optimization(finetune, policy=enabled, learning_rate=0.00001, epochs=5, dataset_size=1024)
```

## 5. Experimental Results

### 5.1 Evaluation Protocol
* **Reference**: output of the original ONNX FP32 model (onnxruntime), with the same preprocessing applied to the 1,033 DA-2K images.
    - The scores therefore measure **agreement between the FP32 model and the INT8 model**, not accuracy against real depth.
* **Scale matching**: for each image, only the scale of the prediction is adjusted so that its median matches the reference median (KITTI-style median scaling). The bias (shift) is not adjusted
* **Stage-wise comparison** (`src/evaluate.py`): each Hailo emulator stage is compared with the stage before it
  * ONNX → HAR `native`: loss from parsing (ONNX → HAR conversion)
  * `native` → `fp_optimized`: loss from full-precision optimizations (equalization, etc.)
  * `fp_optimized` → `quantized`: loss from INT8 quantization
* The calibration set and the evaluation set are the same 1,033 images.

### 5.2 Parsing Issue: Resize Coordinate Mode Mismatch
The first version of the pipeline scored a1 = 81.94% (ONNX FP32 vs. INT8). Breaking this down by stage showed that most of the loss occurred **before quantization**, when the ONNX model was parsed into a HAR.

* The MiDaS ONNX was exported with opset 10. In opset 10, `Resize` has no `coordinate_transformation_mode` attribute and always behaves as `asymmetric`.
* When this attribute is missing, the Hailo parser assumes the opset-11 default, `half_pixel`.
* As a result, the five 2x bilinear upsamplings in the decoder sampled at shifted positions. The shift accumulated over the five stages, and the output drifted by several pixels.
* **Fix** (`src/parse.py`): before parsing, the model is converted to opset 11 with an explicit `asymmetric` mode. The converted ONNX produces exactly the same output as the original, and Hailo parses its resize layers as `disabled` (= asymmetric).

| ONNX FP32 → HAR `native` (FP) | a1 | a1 median | a1 5th percentile |
|---|---|---|---|
| Before fix (`half_pixel`) | 83.28% | 86.55% | 56.60% |
| After fix (`asymmetric`) | **99.95%** | 100.00% | 99.76% |

### 5.3 Quantitative Evaluation (Final INT8 Model)
Final configuration: Hailo `optimization_level=4` (Adaround `train_all`, 320 epochs, 1,024 images), equalization, MMSE weights clipping, fine-tuning for 5 epochs (lr 1e-5), `compression_level=0` (all layers 8-bit).

| Metric | HAR FP32 → INT8<br>(quantization only) | ONNX FP32 → INT8<br>(end-to-end) |
|---|---|---|
| abs_rel | 0.082 | 0.085 |
| rms | 18.25 | 18.24 |
| **a1** (δ < 1.25) | **96.34%** | **96.34%** |
| **a2** (δ < 1.25²) | **98.46%** | **98.46%** |
| **a3** (δ < 1.25³) | **98.96%** | **98.96%** |
| a1 median (per image) | 97.98% | 97.99% |
| a1 5th percentile (per image) | 87.21% | 87.24% |

* rms is in the model's output (relative inverse depth) units.
* sq_rel and log_rms are omitted because the current implementation deviates from the standard definitions (sq_rel divides by gt², and log_rms uses the unscaled prediction).

**Before vs. after (ONNX FP32 → INT8, end-to-end)**

| | a1 | a1 5th percentile |
|---|---|---|
| Initial pipeline (resize mismatch) | 81.94% | 56.10% |
| Current pipeline | **96.34%** | **87.24%** |

### 5.4 Adaround Comparison
All runs below use the fixed parsing, equalization, MMSE weights clipping, and 5-epoch fine-tuning. Scores are HAR FP → INT8.

| Adaround (epochs / images) | Fine-tune lr | a1 | a1 median | a1 5th percentile | Adaround time |
|---|---|---|---|---|---|
| none | 1e-4 | 75.90% | 79.99% | 40.76% | – |
| 50 / 256 | 1e-4 | 83.97% | 88.16% | 51.89% | 29 min |
| 200 / 256 | 1e-4 | 93.78% | 96.58% | 79.55% | 1 h 10 min |
| 320 / 256 | 1e-4 | 94.34% | 96.72% | 80.38% | 1 h 44 min |
| 320 / 1,024 (level-4 default) | 1e-5 | **96.34%** | **97.98%** | **87.21%** | 6 h 19 min |

* The first four rows differ only in Adaround epochs. Adaround has a large effect, and the gain nearly saturates after 200 epochs (+0.56%p from 200 to 320).
* In the last row, the Adaround image count and the fine-tuning lr changed at the same time, so the +2.0%p gain cannot be attributed to either one.
* Timings are on an RTX 4090 without NVIDIA DALI.

### 5.5 Model Size Comparison
| Metric | Full Precision (ONNX) | Quantized (INT8 HEF) | Size reduction |
|--------|---|---|---|
| File size | 63.67 MB | 19.81 MB | **68.9%** |

* Measured on the HEF compiled from the final model (`runs/20260929_232724_addADAoptiLV4_addFine/Midas_quantize_model.hef`). 1 MB = 1024² bytes.

### 5.6 Qualitative Evaluation
* FP32 (ONNX) and INT8 (Hailo emulator) depth maps are compared on the 5 lowest and 5 highest images by per-image a1. Each row shows: input | FP32 | INT8 | relative error (0-25%).
    - In the two middle columns (FP32, INT8), darker blue means closer to the camera.
    - In the relative error map, gray marks pixels excluded from evaluation because the original (FP32) output is 0, and the darkest red marks pixels with an error of 25% or more, which fall outside the a1 threshold.

**Bottom 5**
![Bottom 5](assets/qualitative_bottom5.png)
* The bottom 5 are scenes with large distant areas, such as night, fog, sky, and underwater. All except the one underwater image are outdoor scenes. Most of the error is concentrated in these distant areas.
* MiDaS outputs inverse depth, so values approach 0 as distance increases. Since a1 is based on relative error, quantization error on these small values is amplified, which is likely why scenes with large distant areas score lower.

**Top 5**
![Top 5](assets/qualitative_top5.png)
* The top 5 include both indoor and outdoor scenes, so being indoor or outdoor does not appear to be related to the score.
* Instead, most of each frame is filled with nearby objects or walls, with few distant areas. Scenes with more cues for judging relative depth, such as nearby objects, appear to score higher in a1.

**Summary**
1. Outdoor scenes with large distant areas get lower a1 scores. However, for applications that only need to know that something is far away, this is unlikely to matter much in practice.
2. For judging relative depth, the difference between the depth maps before and after quantization is hardly visible to the eye, regardless of the benchmark score.

## 6. Conclusions

* Because the opset-10 `Resize` layers were parsed with a different coordinate mode (`half_pixel` instead of `asymmetric`), the Hailo model's output drifted by several pixels even in full precision (a1 83.28%). Fixing this raised ONNX vs. HAR FP agreement to 99.95%.
* Comparing the emulator stages one by one (ONNX → native → fp_optimized → quantized) separated parsing loss from quantization loss. Looking only at the end-to-end score would have pointed to the wrong cause.
* Among the settings tested, Adaround had the largest effect. Increasing only the Adaround epochs from 0 to 320 raised a1 from 75.90% to 94.34%.
* **Limitations**: the reference is the FP32 model output, not ground-truth depth. The calibration and evaluation images are the same. The final +2.0%p (level-4 run) could not be attributed to a single cause, because two settings changed at once.


## 7. References
- MiDaS model: https://github.com/isl-org/MiDaS
- Evaluation method: https://github.com/FangGet/tf-monodepth2
- DA-2K dataset: https://huggingface.co/datasets/depth-anything/DA-2K


**Author**: [Jiwon Kim]  
**Date**: [30 Sep 2026]  
