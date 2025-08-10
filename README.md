# X-DECODE: EXtreme Deblurring with Curriculum Optimization and Domain Equalization

## Evaluation - Train/Test on Extreme-GOPRO::
![](images/GoPro-test-comparision-baseline.png)

| BL | Metric | DeblurGAN | DeblurGANv2 | NAFNet | Ours |
|----|--------|-----------|-------------|--------|------|
| 19 | SSIM | 0.282 | 0.668 | 0.642 | **0.764** |
|  | PSNR | 19.51 | 23.94 | 22.99 | **24.90** |
| 25 | SSIM | 0.258 | 0.635 | 0.577 | **0.674** |
|  | PSNR | 19.09 | 22.88 | 21.50 | **23.64** |
| 29 | SSIM | 0.240 | **0.618** | 0.545 | 0.613 |
|  | PSNR | 18.84 | 22.33 | 20.76 | **22.68** |

## Evaluation - Train on EXTREME-GOPRO and Test on EXTREME-KITTI:
![](images/kitti-test-comparision-baseline.png)

| Method | SSIM | PSNR |
|--------|------|------|
| DeblurGAN [5] | 0.256 | 16.24 |
| DeblurGANv2 [6] | 0.466 | 18.45 |
| NAFNet [8] | 0.428 | 18.18 |
| Ours (Without CL) | 0.386 | 18.15 |
| **Ours (With CL)** | **0.549** | **20.71** |


## Training and Testing on Extreme GoPRO Dataset

First install all requirements:
```bash
pip install -r requirements.txt
```

To train the model on the Extreme GoPRO dataset, use the following command:

```bash
python TrainTestSameDomainGoPro/main.py --dataset="<dataset_name>"
```

To test the model:

```bash
python TrainTestSameDomainGoPro/test.py
```

## Training on Extreme Cityscapes and PascalVOC, Testing on KITTI Dataset 

* Supported dataset options: `Cityscapes`, `GoPRO`, `PascalVOC`. 
* Use this to train the model using different curriculum learning strategies:

```bash
python TrainTestOnDiffDomain/main.py --dataset="dataset_dir/" --curr_lear="<curriculum_type>"
```

Supported values for `--curr_lear`:
- `linear` : Linear curriculum learning
- `stepwise` : Step-wise curriculum learning
- `slow-stepwise` : Slower step-wise curriculum learning
- `expo` : Exponential curriculum learning
- `sigmoid` : Sigmoid-based curriculum learning
- `none` : Training without curriculum learning

* Supported datasets: `Cityscapes`, `GoPRO`, `PascalVOC`.

To test the model:

```bash
python TrainTestOnDiffDomain/test_new_range_of_blur.py --dataset_dir="dataset_dir/" --model_dir="model_dir/"
```
It also generates a `CSV` file containing the computed SSIM and PSNR for each image pair and the final mean SSIM and PNSR.


## Dataset

The following datasets were used for training and evaluation:

| Dataset        | Purpose                | Description |
|---------------|------------------------|-------------|
| GoPRO         | Primary training & evaluation | Each blurred image has a corresponding sharp image. Extreme motion blur levels (BL19 to BL29) are simulated using the Albumentations library. |
| PascalVOC-2012 | Cross-domain training   | Standard object recognition dataset adapted for deblurring experiments. |
| Cityscapes    | Cross-domain training   | Urban street scene dataset adapted for deblurring tasks. |
| KITTI         | Cross-domain testing    | Major test dataset for evaluating performance across domains. |

### Sample Images

Extreme-GoPRO - Samples (First column sharp image and rest others is obtained after applying Blur Level 19, 25 and 29):
![](images/DiffBlurLevelComparisionGoPro.png)

KITTI - Samples (First column sharp image and rest others is obtained after applying Blur Level 19, 25 and 29):
![](images/DiffBlurLevelComparisionKitti.png)

### Dataset Download Links

| Dataset        | Download Link |
|---------------|--------------|
| Extrene GoPRO        | [Download](https://mstate-my.sharepoint.com/:u:/g/personal/jdc1258_msstate_edu/EdkBLlfW7QBOmmzcMApydPoBjjiAq3vCW3ULe-0vjX2fDw?e=dkGqwt) |
| Extreme PascalVOC-2012 | [Download](https://mstate-my.sharepoint.com/:u:/g/personal/jdc1258_msstate_edu/EexX5aV6nkFLpFT4hXlQ4PsB0xrqPp7PrXQIWl9vTSrdNw?e=yHQetn) |
| Extreme Cityscapes  | [Download](https://mstate-my.sharepoint.com/:u:/g/personal/jdc1258_msstate_edu/Ee_CKv7accNFjDo708ePvRUBplLuovdAm2_jTmqzfvfwKA?e=A0mAqT) |
| Extreme KITTI         | [Download](https://mstate-my.sharepoint.com/:u:/g/personal/jdc1258_msstate_edu/Ec0OqRZVGwZHq_wg0xryPaYBCnbZKOL3szrQgwc_EAJCOg?e=C2oqgc) |

## Pretrained Models

Pretrained models for different datasets:

| Trained on | Learning type | Download Link |
|-------------------|--------------------|---------------|
| GoPRO extreme Blurred | Step-wise (Train/Test on GoPRO)| [Download](https://mstate-my.sharepoint.com/:u:/g/personal/jdc1258_msstate_edu/ER9XuWU1C61KkxLAMlLICvkBJbE-L8ePX73A6VTjB78-JA?e=qAVg2d) |
| GoPRO extreme Blurred | Linear Curriculum Learning (Test on KITTI) | [Download](https://mstate-my.sharepoint.com/:u:/g/personal/jdc1258_msstate_edu/ERl1otNsVilMkcKXed_4DFoBnGhSadTcpokpHAwVw_Je_Q?e=jER3gC) |
| Cityscapes extreme Blurred | Linear Curriculum Learning (for KITTI testing) | [Download](https://mstate-my.sharepoint.com/:u:/g/personal/jdc1258_msstate_edu/EZG2L7Yegn1FlFDqJR1nEhwBtvqamuI1zoN3nGRB_rrvDw?e=ohU40C) |
| Pascal VOC extreme Blurred | Linear Curriculum Learning (for KITTI testing) | [Download](https://mstate-my.sharepoint.com/:u:/g/personal/jdc1258_msstate_edu/ETZdge3-N7pNlrn6Ugs6VS8BChOnoMGB9EXMI8SWZKJu2g?e=ktiOTL) |

Pretrained models for different curriculum learning. All Tested on Kitti Dataset:

| Model Description | Curriculum Learning | Download Link |
|-------------------|--------------------|---------------|  
| Cityscapes Extreme Blurred  | Step-wise | [Download](https://mstate-my.sharepoint.com/:u:/g/personal/jdc1258_msstate_edu/EUfaumAiE9NDs-09iIcjPDEByn87EPmsAq7Uieu_PKc5vg?e=MEzsx4) | 
| Cityscapes Extreme Blurred | Slower Step-wise | [Download](https://mstate-my.sharepoint.com/:u:/g/personal/jdc1258_msstate_edu/ESxBLDS8w7tOqILuZCUqr9gBIrFivLToabMB9xu9ZYC6yQ?e=bXXyDA) |
| Cityscapes Extreme Blurred  | Linear | [Download](https://mstate-my.sharepoint.com/:u:/g/personal/jdc1258_msstate_edu/EWCZYdFzyXBMncRlWr6R9KYBy-Jg6hdrtGIk9qzMoDFmyA?e=CqXVVP) |
| Cityscapes Extreme Blurred  | Exponential | [Download](https://mstate-my.sharepoint.com/:u:/g/personal/jdc1258_msstate_edu/EbU4Ktz_e3FGkbCZkEEb8FsBtyOy_BIc9hnS9lebuUyh7A?e=FJNfWA) |
| Cityscapes Extreme Blurred  | Sigmoid | [Download](https://mstate-my.sharepoint.com/:u:/g/personal/jdc1258_msstate_edu/EbuwnsPwQKBLmOt2Q3hsr6AB6Pi0LOQl2GS718LBwQFfPw?e=r646dK) |


### Comparison of Curriculum Learning Techniques on Extreme-KITTI Dataset, Trained on Extrem-Cityscapes Dataset Results

| Curriculum Learning Method | SSIM | PSNR |
|----------------------------|------|------|
| Step-wise | 0.619 | 21.97 |
| Slower Step-wise | 0.468 | 18.91 |
| **Linear** | **0.628** | **22.20** |
| Sigmoid | 0.405 | 18.02 |
| Exponential | 0.614 | 21.79 |

# Citation
If you use the X-DECODE data or code please cite:
```bash
@article{xdecode2025,
  title = {X-DECODE: EXtreme Deblurring with Curriculum Optimization and Domain Equalization},
  author = {Gautam Sushant and Chen Jingdao},
  journal = {ArXiv e-prints},
  eprint = {2504.08072},
  year = {2025},
  url={https://arxiv.org/abs/2504.08072}
}


```
