# Multi-view Depth Estimation Based on Diffusion Models  

This repository contains the implementation of the undergraduate thesis **"Multi-view Depth Estimation Based on Diffusion Models"** at Zhejiang University.  
The project is based on the open-source implementation of **Effi-MVS (CVPR2022)**, with improvements that introduce a **Conditional Diffusion Model** and a multi-stage GRU network into the multi-view stereo framework.  

By modeling the coarse-to-fine depth refinement process as an **iterative denoising process**, our method improves the accuracy of depth estimation while maintaining the efficiency and lightweight characteristics of Effi-MVS, and also mitigates the risk of local minima in depth prediction.  

![](imgs/structure_teaser.jpg)

---

## Differences from Effi-MVS  

- **Effi-MVS**:  
  - Proposed an efficient multi-view stereo framework with iterative dynamic cost volume and lightweight 3D CNN.  
  - Focused on efficiency and low computation cost, showing strong performance on DTU and Tanks & Temples datasets.  

- **This project (Effi-MVS + Diffusion)**:  
  - Introduces **diffusion models** into Effi-MVS, treating the depth refinement stage as a denoising process.  
  - Conditional inputs include reference image context features, current depth estimates, and dynamic cost volumes.  
  - A GRU structure is adopted to accelerate convergence, yielding higher accuracy in depth prediction.  
  - Achieves significant improvements on Tanks & Temples F-score, while preserving inference efficiency.  

---

## Experimental Results  

### DTU Dataset  

| Method                 | Acc. (mm) ↓ | Comp. (mm) ↓ | Overall (mm) ↓ |
|------------------------|-------------|--------------|----------------|
| Effi-MVS               | 0.321       | 0.313        | 0.317          |
| **Effi-MVS + Diffusion (Ours)** | 0.314       | 0.317        | 0.316          |

---

### Tanks & Temples Dataset  

| Method                 | Intermediate F-score ↑ | Advanced F-score ↑ |
|------------------------|------------------------|---------------------|
| Effi-MVS               | 56.88                 | 34.39              |
| **Effi-MVS + Diffusion (Ours)** | **60.19 (+3.3)**       | **37.47 (+3.1)**  |

➡️ On the Tanks & Temples advanced set, our approach improves F-score by nearly **9%**, delivering finer reconstruction quality.  

---

## Original Effi-MVS Description (Preserved)  

### Introduction  
An efficient framework for high-resolution multi-view stereo. This work aims to improve the accuracy and reduce the consumption at the same time.  
If you find this project useful for your research, please cite:  

```
@inproceedings{wang2022efficient,
  title={Efficient Multi-View Stereo by Iterative Dynamic Cost Volume},
  author={Wang, Shaoqian and Li, Bo and Dai, Yuchao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8655--8664},
  year={2022}
}
```  

---

## Installation  

### Requirements  
* python 3.8  
* CUDA >= 11.1  

```bash
pip install -r requirements.txt
```  

---

## Reproducing Results  

* Download pre-processed datasets (provided by PatchmatchNet):  
  [DTU's evaluation set](https://drive.google.com/file/d/1jN8yEQX0a-S22XwUjISM8xSJD39pFLL_/view?usp=sharing),  
  [Tanks & Temples](https://drive.google.com/file/d/1gAfmeoGNEFl9dL4QcAU4kF0BAyTd-r8Z/view?usp=sharing)  

(The dataset folder organization, `cam.txt`, and `pair.txt` formats are the same as Effi-MVS. Details preserved from the original documentation, omitted here for brevity.)  

---

## Training & Evaluation  

- In `train.sh`, set `MVS_TRAINING` or `BLEND_TRAINING` as the dataset root directory;  
- In `test.sh`, set `DTU_TESTING` or `TANK_TESTING` as the test dataset root directory;  
- Use `--OUT_DIR` to specify where to store output point clouds;  
- Run `sh train.sh` or `sh test.sh` for training/testing.  

Outputs are point clouds in `.ply` format.  

---

## Acknowledgement  

- [Effi-MVS](https://github.com/) for providing the base framework  
- [MVSNet](https://github.com/YoYo000/MVSNet) for the multi-view stereo baseline  
- [Diffusion Models](https://arxiv.org/abs/2006.11239) for inspiring our denoising-based refinement design  
