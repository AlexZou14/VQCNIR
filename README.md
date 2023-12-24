# AAAI'24 VQCNIR: Clearer Night Image Restoration with Vector-Quantized Codebook

This is the office implementation of ***VQCNIR: Clearer Night Image Restoration with Vector-Quantized Codebook, AAAI2024.***

***Wenbin Zou, Hongxia Gao, Tian Ye, Liang Chen, Weipeng Yang, Shasha Huang, Hongshen Chen, Sixiang Chen***

***SCUT｜ HKUST(GZ)｜FJNU***

[Arxiv](https://arxiv.org/pdf/2312.08606), [Supplementary](Figures/VQCNIR_supp.pdf)

<hr />

![fig1.png](Figures/Fig1.png)
![fig2.png](Figures/Fig2.png)

> **Abstract:** *Night photography often struggles with challenges like low light and blurring, stemming from dark environments and prolonged exposures. Current methods either disregard priors and directly fitting end-to-end networks, leading to inconsistent illumination, or rely on unreliable handcrafted priors to constrain the network, thereby bringing the greater error to the final result. We believe in the strength of data-driven high-quality priors and strive to offer a reliable and consistent prior, circumventing the restrictions of manual priors.
In this paper, we propose Clearer Night Image Restoration with Vector-Quantized Codebook (VQCNIR) to achieve remarkable and consistent restoration outcomes on real-world and synthetic benchmarks. To ensure the faithful restoration of details and illumination, we propose the incorporation of two essential modules: the Adaptive Illumination Enhancement Module (AIEM) and the Deformable Bi-directional Cross-Attention (DBCA) module. The AIEM leverages the inter-channel correlation of features to dynamically maintain illumination consistency between degraded features and high-quality codebook features. Meanwhile, the DBCA module effectively integrates texture and structural information through bi-directional cross-attention and deformable convolution, resulting in enhanced fine-grained detail and structural fidelity across parallel decoders.Extensive experiments validate the remarkable benefits of VQCNIR in enhancing image quality under low-light conditions, showcasing its state-of-the-art performance on both synthetic and real-world datasets.* 

<hr />


## TODO List
- [x] Testing Code&Checkpoint
- [x] Model.py
- [ ] Train.py

## Installation
Our VQCNIR is built in Pytorch1.12.0, we train and test it ion Ubuntu20.04 environment (Python3.8, cuda11.3).
For installing, please follow these intructions.

```
conda create -n vqcnir python=3.8
conda activate vqcnir
conda install pytorch=1.12 
pip install opencv-python ....
```

## Model Testing
**You can find the model weights under the model_wight folder: VQCNIR_LOLBlur_G.pth.**
Run the following commands:
```
python3  inference_vqlol.py -i dataset_path -w model_weight  -o output_dir 
```

## More Results
![fig3.png](Figures/Fig3.png)
![fig4.png](Figures/fig4.png)

## Contact
**Wenbin Zou: alexzou14@foxmail.com** 


## Citation
```bibtex
@article{zou2023vqcnir,
  title={VQCNIR: Clearer Night Image Restoration with Vector-Quantized Codebook},
  author={Zou, Wenbin and Gao, Hongxia and Ye, Tian and Chen, Liang and Yang, Weipeng and Huang, Shasha and Chen, Hongsheng and Chen, Sixiang},
  journal={arXiv preprint arXiv:2312.08606},
  year={2023}
}
```