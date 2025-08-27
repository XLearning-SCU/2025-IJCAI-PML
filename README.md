# PML
Peng Hu, Yang Qin, Yuanbiao Gou, Yunfan Li, Mouxing Yang and Xi Peng*,[Probabilistic Multimodal Learning with von Mises-Fisher Distributions](https://ijcai-preprints.s3.us-west-1.amazonaws.com/2025/2595.pdf), The 34th International Joint Conference on Artificial Intelligence (IJCAI-25), 2025. (PyTorch Code)

## Abstract
Multimodal learning is pivotal for the advancement of artificial intelligence, enabling machines to integrate complementary information from diverse data sources for holistic perception and understanding. Despite significant progress, existing methods struggle with challenges such as noisy inputs, noisy correspondence, and the inherent uncertainty of multimodal data, limiting their reliability and robustness. To address these issues, this paper presents a novel Probabilistic Multimodal Learning framework (PML) that models each data point as a von Mises-Fisher (vMF) distribution, effectively capturing intrinsic uncertainty and enabling robust fusion. Unlike traditional Gaussian-based models, PML learns directional representation with a concentration parameter to quantify reliability directly, enhancing stability and interpretability. To enhance discrimination, we propose a von Mises-Fisher Prototypical Contrastive Learning paradigm (vMF-PCL), which projects data onto a hypersphere by pulling within-class samples closer to their class prototype while pushing between-class prototypes apart, adaptively learning the reliability estimations. Building upon the estimated reliability, we develop a Reliable Multimodal Fusion mechanism (RMF) that dynamically adjusts the contribution and conflict of each modality, ensuring robustness against noisy data, noisy correspondence, and uncertainty. Extensive experiments on nine benchmarks demonstrate the superiority of PML, consistently outperforming 14 state-of-the-art methods. Code is available at https://github.com/XLearning-SCU/2025-IJCAI-PML.

## Framework
<h4>Figure 1. Overview of our Probabilistic Multimodal Learning (PML) framework. First, PML utilizes modality-specific backbones to project the data into a latent space. Then, PML projects each point into a distribution described by a mean direction (µ) and a concentration parameter (κ), enabling intrinsic uncertainty estimation. To obtain discriminative directional representations, PML exploits vMF-based prototype contrastive learning to maximize the agreement between the data and the corresponding class prototype in the latent hypersphere.
</h4> 
<img src=img/framework.png class='center' \>

## Usage
To train a model on Scene, just run main_vMF.py:
```bash
python main_vMF.py --dataset Scene
```

You can get outputs as follows:
```
...
Epoch:185 ====> loss: 0.2188 best acc: 0.8194 acc: 0.8049
Epoch:190 ====> loss: 0.2505 best acc: 0.8194 acc: 0.8161
Epoch:195 ====> loss: 0.2406 best acc: 0.8194 acc: 0.8105
Epoch:200 ====> loss: 0.2476 best acc: 0.8194 acc: 0.8138
***************************************
***************************************
2 0.8439
1 0.8395
4 0.8328
8 0.8317
5 0.8294
7 0.8294
3 0.8283
0 0.8227
6 0.8216
9 0.8194
Acc : 82.99 ± 0.73
***************************************
***************************************
Namespace(num_layer=3, batch_size=32, epochs=200, lr=0.0001, times=10, mid_dim=1024, dataset='Scene', bert_model='bert-base-uncased', data_path='./dataset/', drop_img_percent=0.0, max_seq_len=512, n_workers=8, noise_rate=0.0, nc_rate=0.0, task_type='classification', backbone=False, eval=False, beta=0.2, interval=5, num_views=3, dims=array([[20],
       [59],
       [40]]), num_classes=15)
```

## Comparison with the State-of-the-Art
Table 1. Accuracy (\%) performance on normal test sets. The best and second-best results are in bold and underlined, respectively.
| Methods                                |         Handwritten        |           MSRC-V1          |           NUSOBJ           |         Fashion-MV         |           Scene15          |           LandUse          |          Leaves100         |
|----------------------------------------|:--------------------------:|:--------------------------:|:--------------------------:|:--------------------------:|:--------------------------:|:--------------------------:|:--------------------------:|
| DUA-Nets <span style="color: #888888">(AAAI’21)</span> |      98.10 $\pm$ 0.32      |      84.67 $\pm$ 3.03      |      27.75 $\pm$ 0.00      |      91.08 $\pm$ 0.17      |      65.01 $\pm$ 1.55      |      45.24 $\pm$ 1.85      |      90.31 $\pm$ 1.25      |
| TMC <span style="color: #888888">(ICLR’21)</span>    |      98.51 $\pm$ 0.15      |      91.70 $\pm$ 2.70      |      38.77 $\pm$ 0.81      |      95.40 $\pm$ 0.40      |      67.71 $\pm$ 0.30      |      31.69 $\pm$ 3.93      |      86.81 $\pm$ 2.20      |
| ETMC <span style="color: #888888">(TPAMI’22)</span>  |      98.75 $\pm$ 0.00      |      92.86 $\pm$ 3.01      |      44.23 $\pm$ 0.76      |      96.21 $\pm$ 0.36      |      71.61 $\pm$ 0.28      |      43.52 $\pm$ 3.19      |      91.44 $\pm$ 2.39      |
| TMDL-OA <span style="color: #888888">(AAAI’22)</span>   |      98.55 $\pm$ 0.45      |      95.00 $\pm$ 1.67      |      27.88 $\pm$ 0.67      |      86.52 $\pm$ 0.04      |      75.57 $\pm$ 0.02      |      25.02 $\pm$ 2.10      |      75.28 $\pm$ 3.57      |
| DFTMC <span style="color: #888888">(CVPR’22)</span>  |      98.75 $\pm$ 0.39      |      96.90 $\pm$ 2.14      |              -             |              -             |      63.10 $\pm$ 3.60      |      34.95 $\pm$ 1.69      |      69.92 $\pm$ 2.54      |
| DCP-CV <span style="color: #888888">(TPAMI’22)</span> |      97.91 $\pm$ 0.59      |      92.86 $\pm$ 2.61      |      32.19 $\pm$ 9.48      |      97.96 $\pm$ 0.16      |      76.70 $\pm$ 2.15      |      71.71 $\pm$ 2.09      |      95.62 $\pm$ 1.38      |
| DCP-CG <span style="color: #888888">(TPAMI’22)</span>  | <u>99.00 $\pm$ 0.47</u> |      95.24 $\pm$ 3.69      |      43.65 $\pm$ 1.10      |      98.11 $\pm$ 0.23      | <u>77.79 $\pm$ 1.73</u> | <u>75.74 $\pm$ 0.98</u> | <u>98.19 $\pm$ 0.46</u> |
| UIMC <span style="color: #888888">(CVPR’23)</span>     |      98.25 $\pm$ 0.00      | <u>98.81 $\pm$ 1.19</u> |      43.42 $\pm$ 0.12      |      98.13 $\pm$ 0.13      |      77.70 $\pm$ 0.00      |      57.95 $\pm$ 0.61      |      95.31 $\pm$ 0.71      |
| QMF <span style="color: #888888">(ICML’23)</span>     |      98.72 $\pm$ 0.48      |      97.86 $\pm$ 1.28      |      38.13 $\pm$ 0.73      |      98.93 $\pm$ 0.32      |      68.58 $\pm$ 1.49      |      47.86 $\pm$ 2.55      |      95.69 $\pm$ 1.25      |
| ECML <span style="color: #888888">(AAAI’24)</span>    |      98.72 $\pm$ 0.39      |      94.05 $\pm$ 1.60      |      39.10 $\pm$ 0.74      |      95.25 $\pm$ 0.46      |      76.19 $\pm$ 0.12      |      60.10 $\pm$ 2.01      |      92.53 $\pm$ 1.94      |
| TMNR <span style="color: #888888">(IJCAI’24)</span>  |      97.20 $\pm$ 0.63      |      94.05 $\pm$ 3.24      |      34.52 $\pm$ 0.85      |      94.10 $\pm$ 0.50      |      68.10 $\pm$ 1.15      |      27.38 $\pm$ 1.88      |      90.13 $\pm$ 1.53      |
| CCML <span style="color: #888888">(MM’24)</span>    |      97.60 $\pm$ 0.62      |      96.90 $\pm$ 2.39      |      41.43 $\pm$ 0.71      |      95.16 $\pm$ 0.41      |      73.02 $\pm$ 1.44      |      44.86 $\pm$ 2.03      |      97.72 $\pm$ 0.92      |
| PDF <span style="color: #888888">(ICML’24)</span>  |      98.40 $\pm$ 0.37      |      97.14 $\pm$ 1.78      | <u>46.78 $\pm$ 0.33</u> | <u>98.95 $\pm$ 0.19</u> |      70.25 $\pm$ 1.21      |      45.17 $\pm$ 2.66      |      98.03 $\pm$ 0.71      |
| PML <span style="color: #888888">(Ours)</span>   |      99.32 $\pm$ 0.45      |      99.52 $\pm$ 0.95      |      49.16 $\pm$ 0.40      |      99.10 $\pm$ 0.22      |      82.70 $\pm$ 0.86      |      82.05 $\pm$ 1.36      |      99.91 $\pm$ 0.14      |


Table 2. Accuracy ($\%$) performance on noisy test sets. The best and second-best results are in bold and underlined, respectively.
| Methods                                |         Handwritten        |           MSRC-V1          |           NUSOBJ           |         Fashion-MV         |           Scene15          |           LandUse          |          Leaves100         |
|----------------------------------------|:--------------------------:|:--------------------------:|:--------------------------:|:--------------------------:|:--------------------------:|:--------------------------:|:--------------------------:|
| DUA-Nets <span style="color: #888888">(AAAI’21)</span> |      98.10 $\pm$ 0.32      |      84.67 $\pm$ 3.03      |      27.75 $\pm$ 0.00      |      91.08 $\pm$ 0.17      |      65.01 $\pm$ 1.55      |      45.24 $\pm$ 1.85      |      90.31 $\pm$ 1.25      |
| TMC <span style="color: #888888">(ICLR’21)</span>      |      98.51 $\pm$ 0.15      |      91.70 $\pm$ 2.70      |      38.77 $\pm$ 0.81      |      95.40 $\pm$ 0.40      |      67.71 $\pm$ 0.30      |      31.69 $\pm$ 3.93      |      86.81 $\pm$ 2.20      |
| ETMC <span style="color: #888888">(TPAMI’22)</span>    |      98.75 $\pm$ 0.00      |      92.86 $\pm$ 3.01      |      44.23 $\pm$ 0.76      |      96.21 $\pm$ 0.36      |      71.61 $\pm$ 0.28      |      43.52 $\pm$ 3.19      |      91.44 $\pm$ 2.39      |
| TMDL-OA <span style="color: #888888">(AAAI’22)</span>  |      98.55 $\pm$ 0.45      |      95.00 $\pm$ 1.67      |      27.88 $\pm$ 0.67      |      86.52 $\pm$ 0.04      |      75.57 $\pm$ 0.02      |      25.02 $\pm$ 2.10      |      75.28 $\pm$ 3.57      |
| DFTMC <span style="color: #888888">(CVPR’22)</span>    |      98.75 $\pm$ 0.39      |      96.90 $\pm$ 2.14      |              -             |              -             |      63.10 $\pm$ 3.60      |      34.95 $\pm$ 1.69      |      69.92 $\pm$ 2.54      |
| DCP-CV <span style="color: #888888">(TPAMI’22)</span>  |      97.91 $\pm$ 0.59      |      92.86 $\pm$ 2.61      |      32.19 $\pm$ 9.48      |      97.96 $\pm$ 0.16      |      76.70 $\pm$ 2.15      |      71.71 $\pm$ 2.09      |      95.62 $\pm$ 1.38      |
| DCP-CG <span style="color: #888888">(TPAMI’22)</span>  | <u>99.00 $\pm$ 0.47</u> |      95.24 $\pm$ 3.69      |      43.65 $\pm$ 1.10      |      98.11 $\pm$ 0.23      | <u>77.79 $\pm$ 1.73</u> | <u>75.74 $\pm$ 0.98</u> | <u>98.19 $\pm$ 0.46</u> |
| UIMC <span style="color: #888888">(CVPR’23)</span>     |      98.25 $\pm$ 0.00      | <u>98.81 $\pm$ 1.19</u> |      43.42 $\pm$ 0.12      |      98.13 $\pm$ 0.13      |      77.70 $\pm$ 0.00      |      57.95 $\pm$ 0.61      |      95.31 $\pm$ 0.71      |
| QMF <span style="color: #888888">(ICML’23)</span>      |      98.72 $\pm$ 0.48      |      97.86 $\pm$ 1.28      |      38.13 $\pm$ 0.73      |      98.93 $\pm$ 0.32      |      68.58 $\pm$ 1.49      |      47.86 $\pm$ 2.55      |      95.69 $\pm$ 1.25      |
| ECML <span style="color: #888888">(AAAI’24)</span>     |      98.72 $\pm$ 0.39      |      94.05 $\pm$ 1.60      |      39.10 $\pm$ 0.74      |      95.25 $\pm$ 0.46      |      76.19 $\pm$ 0.12      |      60.10 $\pm$ 2.01      |      92.53 $\pm$ 1.94      |
| TMNR <span style="color: #888888">(IJCAI’24)</span>    |      97.20 $\pm$ 0.63      |      94.05 $\pm$ 3.24      |      34.52 $\pm$ 0.85      |      94.10 $\pm$ 0.50      |      68.10 $\pm$ 1.15      |      27.38 $\pm$ 1.88      |      90.13 $\pm$ 1.53      |
| CCML <span style="color: #888888">(MM’24)</span>       |      97.60 $\pm$ 0.62      |      96.90 $\pm$ 2.39      |      41.43 $\pm$ 0.71      |      95.16 $\pm$ 0.41      |      73.02 $\pm$ 1.44      |      44.86 $\pm$ 2.03      |      97.72 $\pm$ 0.92      |
| PDF <span style="color: #888888">(ICML’24)</span>      |      98.40 $\pm$ 0.37      |      97.14 $\pm$ 1.78      | <u>46.78 $\pm$ 0.33</u> | <u>98.95 $\pm$ 0.19</u> |      70.25 $\pm$ 1.21      |      45.17 $\pm$ 2.66      |      98.03 $\pm$ 0.71      |
| PML <span style="color: #888888">(Ours)</span>         |      99.32 $\pm$ 0.45      |      99.52 $\pm$ 0.95      |      49.16 $\pm$ 0.40      |      99.10 $\pm$ 0.22      |      82.70 $\pm$ 0.86      |      82.05 $\pm$ 1.36      |      99.91 $\pm$ 0.14      |

## Citation
If you find PML useful in your research, please consider citing:
```
@inproceedings{hu2025PML,
   title={Probabilistic Multimodal Learning with von Mises-Fisher Distributions},
   author={Hu, Peng and Qin, Yang and Gou, Yuanbiao and Li, Yunfan and Yang, Mouxing and Peng, Xi},
   booktitle={The 34th International Joint Conference on Artificial Intelligence (IJCAI-25)},
   year={2025},
}
```
