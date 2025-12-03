<img src="figures/ImAge.jpg" width="1000px">

This is the official repository for the NeurIPS 2025 paper "[Towards Implicit Aggregation: Robust Image Representation for Place Recognition in the Transformer Era](https://openreview.net/pdf?id=uVYqwEgIpE)".

ImAge is an implicit aggregation method to get robust global image descriptors for visual place recognition, which neither modifies the backbone nor needs an extra aggregator. It only adds some aggregation tokens before a specific block of the transformer backbone, leveraging the inherent self-attention mechanism to implicitly aggregate patch features. Our method provides a novel perspective different from the previous paradigm, effectively and efficiently achieving SOTA performance. 

The difference between ImAge and the previous paradigm is shown in this figure:

<img src="figures/pipeline.jpg" width="800px">

To quickly use our model, you can use Torch Hub:
```
import torch
model = torch.hub.load("Lu-Feng/ImAge", "ImAge")
```

## Getting Started

This repo follows the framework of [GSV-Cities](https://github.com/amaralibey/gsv-cities) for training, and the [Visual Geo-localization Benchmark](https://github.com/gmberton/deep-visual-geo-localization-benchmark) for evaluation. You can download the GSV-Cities datasets [HERE](https://www.kaggle.com/datasets/amaralibey/gsv-cities), and refer to [VPR-datasets-downloader](https://github.com/gmberton/VPR-datasets-downloader) to prepare test datasets.

The test dataset should be organized in a directory tree as such:

```
├── datasets_vg
    └── datasets
        └── pitts30k
            └── images
                ├── train
                │   ├── database
                │   └── queries
                ├── val
                │   ├── database
                │   └── queries
                └── test
                    ├── database
                    └── queries
```

Before training, you should download the pre-trained foundation model DINOv2-register(ViT-B/14) [HERE](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth).

## Train
```
python3 train.py --eval_datasets_folder=/path/to/your/datasets_vg/datasets --eval_dataset_name=pitts30k --backbone=dinov2 --freeze_te=8 --num_learnable_aggregation_tokens=8 --train_batch_size=120 --lr=0.00005 --epochs_num=20 --patience=20 --initialization_dataset=msls_train --training_dataset=gsv_cities --foundation_model_path=/path/to/pre-trained/dinov2_vitb14_reg4_pretrain.pth
```

If you don't have the MSLS-train dataset, you can also set `--initialization_dataset=gsv_cities`.

## Test
```
python3 eval.py --eval_datasets_folder=/path/to/your/datasets_vg/datasets --eval_dataset_name=pitts30k --backbone=dinov2 --freeze_te=8 --num_learnable_aggregation_tokens=8 --resume=/path/to/trained/model/ImAge_GSV.pth
```

## Trained Model

<table style="margin: auto">
  <thead>
    <tr>
      <th>Training set</th>
      <th>Pitts30k</th>
      <th>MSLS-val</th>
      <th>Nordland</th>
      <th>Download</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">GSV-Cities</td>
      <td align="center">94.0</td>
      <td align="center">93.0</td>
      <td align="center">93.2</td>
      <td><a href="https://drive.google.com/file/d/1ui9K8TMXXyURlgy55LiVXfs_9DAoJtWX/view?usp=drive_link">LINK</a></td>
    </tr>
    <tr>
      <td align="center">Unified dataset</td>
      <td align="center">94.1</td>
      <td align="center">94.5</td>
      <td align="center">97.7</td>
      <td><a href="https://drive.google.com/file/d/1k3m0yMMRB20B0aU4CaH_HIMO64k9epWx/view?usp=drive_link">LINK</a></td>
    </tr>
  </tbody>
</table>

！！！The code for merging previous VPR datasets to get the unified (merged) dataset is still being refined and will be released alongside the code of SelaVPR++. Please wait patiently.

## Others

This repository also supports training NetVLAD, SALAD, and BoQ on the GSV-Cities dataset with PyTorch (not pytorch-lightning in other repos) and using Automatic Mixed Precision.

## Acknowledgements

Parts of this repo are inspired by the following repositories:

[GSV-Cities](https://github.com/amaralibey/gsv-cities)

[Visual Geo-localization Benchmark](https://github.com/gmberton/deep-visual-geo-localization-benchmark)

[DINOv2](https://github.com/facebookresearch/dinov2)

## Citation

If you find this repo useful for your research, please consider leaving a star⭐️ and citing the paper

```
@inproceedings{ImAge,
title={Towards Implicit Aggregation: Robust Image Representation for Place Recognition in the Transformer Era},
author={Feng Lu and Tong Jin and Canming Ye and Xiangyuan Lan and Yunpeng Liu and Chun Yuan},
booktitle={The Annual Conference on Neural Information Processing Systems},
year={2025}
}
```
