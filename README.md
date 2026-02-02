<img src="figures/ImAge.jpg" width="1000px">

This is the official repository for the NeurIPS 2025 paper "Towards Implicit Aggregation: Robust Image Representation for Place Recognition in the Transformer Era".
[Paper on ArXiv](https://arxiv.org/pdf/2511.06024) | [Paper on HF](https://huggingface.co/papers/2511.06024) | [Model on HF](https://huggingface.co/fenglu96/ImAge4VPR)

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
      <td><a href="https://cas-bridge.xethub.hf.co/xet-bridge-us/697f7fd51397352ac04c136e/0fe073271cfe89cc89a2ff74dd257205f713faa92bc3d9f64ac520689d49fbb9?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=cas%2F20260202%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260202T085344Z&X-Amz-Expires=3600&X-Amz-Signature=82b238970ee4bcb8c3b0fc934ff8502e4342a19014f1f5299e069bd823095a3e&X-Amz-SignedHeaders=host&X-Xet-Cas-Uid=643bae2fb409fef15e05cf5f&response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27ImAge_GSV.pth%3B+filename%3D%22ImAge_GSV.pth%22%3B&x-id=GetObject&Expires=1770026024&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc3MDAyNjAyNH19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2FzLWJyaWRnZS54ZXRodWIuaGYuY28veGV0LWJyaWRnZS11cy82OTdmN2ZkNTEzOTczNTJhYzA0YzEzNmUvMGZlMDczMjcxY2ZlODljYzg5YTJmZjc0ZGQyNTcyMDVmNzEzZmFhOTJiYzNkOWY2NGFjNTIwNjg5ZDQ5ZmJiOSoifV19&Signature=dXwzqGAk2tpvtZgm9uIb405b5%7Ewvdu4455wWoOzCY-g8XpoxaEIwM89BPrn1wzI8r-Q4cPs%7EsxyLw4uDLBNS26mUZQfQ8lZ4Gg2qt7trOCJikNNorHfrsRZSmBT3f16NMXeU4IGAZuTJuMCoQcJL8gl6RdkC98MHu6Qzj7MfjQxvrrnIHhB9rtCfo7OIl-2D9I2ABfZCKm-WNQrXvxLUrd%7E6v0rUsUuKMF%7EgQWQa%7ED79MO7tjRwDlEBH6d2x%7E%7EU8hXjQL36tg9pHvujujznedA7SaWtL-wlBOYpGknCyqo4CvWrZClo2QBK%7EZCxI0fbxwCofc7Y7txMa-0N091mMFA__&Key-Pair-Id=K2L8F4GPSG1IFC">LINK</a></td>
    </tr>
    <tr>
      <td align="center">Unified dataset</td>
      <td align="center">94.1</td>
      <td align="center">94.5</td>
      <td align="center">97.7</td>
      <td><a href="https://cas-bridge.xethub.hf.co/xet-bridge-us/697f7fd51397352ac04c136e/9b83c50d594a535f6a888d7ee62127fa174d8c26c3d40e482c6a5a61f4844c57?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=cas%2F20260202%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260202T085412Z&X-Amz-Expires=3600&X-Amz-Signature=e2d31ef9278f9ab577b28e3defcb829570d7b3a422de2a301d179604141cbe57&X-Amz-SignedHeaders=host&X-Xet-Cas-Uid=643bae2fb409fef15e05cf5f&response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27ImAge_Merged.pth%3B+filename%3D%22ImAge_Merged.pth%22%3B&x-id=GetObject&Expires=1770026052&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc3MDAyNjA1Mn19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2FzLWJyaWRnZS54ZXRodWIuaGYuY28veGV0LWJyaWRnZS11cy82OTdmN2ZkNTEzOTczNTJhYzA0YzEzNmUvOWI4M2M1MGQ1OTRhNTM1ZjZhODg4ZDdlZTYyMTI3ZmExNzRkOGMyNmMzZDQwZTQ4MmM2YTVhNjFmNDg0NGM1NyoifV19&Signature=Sykfkr7YYy4TkE8dBJAkYMiClkjLBcNSEw5eVCv8PhqUn0mMlOrPUkOhHhO73JWP%7E4IVdJimgXOsUjbHynM%7EZDa-hECsi19hXxG98gRmQu%7EFcN8TGKQV1XeYHODxbNtoqBkB3aRJ23OpFMjgUUcMJHJPkVjf2z7tFKHOAwLNfMM41L30FXGDJIi2tAFZW6JAmCmudSuhwYhDbZey-7kMqMdm7Mayndq6pA21V24FmIELWpEPPhajcnj6bjdjCjithCA-LTYRXP-7sXzlmAZqCbXQrSTAQAVZk38RfmxP3GIPzNipVbt7XuYlnRob30rtvqGT%7Ekk4hQE3hJgFZGnZow__&Key-Pair-Id=K2L8F4GPSG1IFC">LINK</a></td>
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
