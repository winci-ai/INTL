<style>
table
{
    margin: auto;
}
</style>

# Modulate Your Spectrum in Self-Supervised Learning

This is a PyTorch implementation of the paper.

## Requirements
- Install PyTorch ([pytorch.org](http://pytorch.org))
- Install wandb for Logging ([wandb.ai](https://wandb.ai/)) 

## Experiments on Standard SSL Benchmark
The code includes experiments in section 5. 

### Evaluation for Classification
The datasets include ImageNet, CIFAR-10, CIFAR-100 and ImageNet-100.

The unsupervised pretraining scripts for small and medium datasets are shown in `scripts/base.sh`

The results are shown in the following table:

| Method  |CIFAR-10 | CIFAR-100 |STL-10 | Tiny-ImageNet |
| :--------:  |:-------------:| :--: | :--: | :--: |
|   | **top-1** &nbsp; **5-nn** |**top-1** &nbsp; **5-nn**  |**top-1** &nbsp; **5-nn** | **top-1** &nbsp; **5-nn** |
| CW-RGP 2|  91.92 &nbsp;   89.54 |  67.51 &nbsp;   57.35  |90.76 &nbsp;   87.34|  49.23 &nbsp;   34.04 |
| CW-RGP 4|  92.47 &nbsp; 90.74| 68.26 &nbsp;  58.67 |92.04 &nbsp; 88.95| 50.24 &nbsp;  35.99 |

### Evaluation on ImageNet

#### Pre-trained Models
Our pretrained ResNet-50 models (using multi-crop and EMA):

<table>
  <tr>
    <th>epochs</th>
    <th>batch size</th>
    <th>top-1 acc</th>
    <th colspan="5">download</th>
  </tr>
  <tr>
    <td>100</td>
    <td>256</td>
    <td>73.5%</td>
    <td><a href="scripts/intl_ep100_multi-crop_ema.sh">script</a></td>
    <td><a href="https://drive.google.com/file/d/1DZVKlqWaRJ7Xkq9g4rDoVOfClRtIa0Uk/view?usp=drive_link">ResNet-50</a></td>
    <td><a href="https://drive.google.com/file/d/1zemhf-UbzcmpteAB5nKEv7r4dWbYINLP/view?usp=drive_link">full checkpoint</a></td>
    <td><a href="https://drive.google.com/file/d/1khGuJ37B6yEl1ME4Bc9WYikO3Hh8a7vK/view?usp=drive_links">lincls checkpoint</a></td>
    <td><a href="https://drive.google.com/file/d/1y1HEOvlQxkqTfQXikBKOogM-df_EeRQs/view?usp=drive_link">lincls logs</a></td>
  </tr>
  <tr>
    <td>200</td>
    <td>256</td>
    <td>75.2%</td>
    <td><a href="scripts/intl_ep200_multi-crop_ema.sh">script</a></td>
    <td><a href="https://drive.google.com/file/d/1H6i__9IYkX4VYcMILY-8JgHQY1m_aUlP/view?usp=drive_link">ResNet-50</a></td>
    <td><a href="https://drive.google.com/file/d/1MQlwD1Ep6oMCpDrz3T7Ih4DCRe9fwGg3/view?usp=drive_link">full checkpoint</a></td>
    <td><a href="https://drive.google.com/file/d/1WmtWxULXPiTPq_NWTv_ceXGouUCoVAX-/view?usp=drive_link">lincls checkpoint</a></td>
    <td><a href="https://drive.google.com/file/d/1dEflhK2K79GoqPfizDxBUUoNwTDgSAMs/view?usp=drive_link">lincls logs</a></td>
  </tr>
  <tr>
    <td>400</td>
    <td>256</td>
    <td>76.1%</td>
    <td><a href="scripts/intl_ep400_multi-crop_ema.sh">script</a></td>
    <td><a href="https://drive.google.com/file/d/1CsowRCBNL6zTvjXe2PKhVOiDIGP-2to1/view?usp=drive_link">ResNet-50</a></td>
    <td><a href="https://drive.google.com/file/d/1PUoGL0fr-WbtWkbk9gopSr7vO_kA2S9z/view?usp=drive_link">full checkpoint</a></td>
    <td><a href="https://drive.google.com/file/d/1M40oIQFvMYXZCeOMba23fqIUxmYDCQTg/view?usp=drive_link">lincls checkpoint</a></td>
    <td><a href="https://drive.google.com/file/d/1Zl-nQPVzc-MRbs6u26nIaM4YA904blym/view?usp=drive_link">lincls logs</a></td>
  </tr>
  <tr>
    <td>800</td>
    <td>256</td>
    <td>76.6%</td>
    <td><a href="scripts/intl_ep800_multi-crop_ema.sh">script</a></td>
    <td><a href="https://drive.google.com/file/d/1zHZPpHjMKnzwHyOD93cuRO0o8QMbWnxV/view?usp=drive_link">ResNet-50</a></td>
    <td><a href="https://drive.google.com/file/d/1wLN1I4kXJtbmuRH1HKHE5snp6S-MtNLR/view?usp=drive_link">full checkpoint</a></td>
    <td><a href="https://drive.google.com/file/d/1wKeEKcGojfHYhdLh24f8VhmY6ZXuD8Fw/view?usp=drive_link">lincls checkpoint</a></td>
    <td><a href="https://drive.google.com/file/d/1lEDwwr5qbPpQM71loIxXbDx-biCQsDii/view?usp=drive_link">lincls logs</a></td>
  </tr>
</table>
You can choose to download either the weights of the pretrained ResNet-50 network or the full checkpoint, which also contains the weights of the projection and the state of the optimizer.

#### INTL Training

Install PyTorch and download ImageNet by following the instructions in the [requirements](https://github.com/pytorch/examples/tree/master/imagenet#requirements) section of the PyTorch ImageNet training example. The code has been developed for PyTorch version 1.7.1 and torchvision version 0.8.2, but it should work with other versions just as well. 

Our best model is obtained by running the following command:

```
python main.py --data_path /path/to/imagenet/ 
```

Training time is approximately 7 days on 16 v100 GPUs.

### Evaluation: Linear Classification

Train a linear probe on the representations learned by Barlow Twins. Freeze the weights of the resnet and use the entire ImageNet training set.

```
python evaluate.py /path/to/imagenet/ /path/to/checkpoint/resnet50.pth --lr-classifier 0.3
```

### Evaluation: Semi-supervised Learning

Train a linear probe on the representations learned by Barlow Twins. Finetune the weights of the resnet and use a subset of the ImageNet training set.

```
python evaluate.py /path/to/imagenet/ /path/to/checkpoint/resnet50.pth --weights finetune --train-perc 1 --epochs 20 --lr-backbone 0.005 --lr-classifier 0.5 --weight-decay 0 --checkpoint-dir ./checkpoint/semisup/
```

### Transferring to Object Detection
Same as [MoCo](https://github.com/facebookresearch/moco) for object detection transfer, please see [moco/detection](https://github.com/facebookresearch/moco/tree/master/detection).

Transfer learning results of CW-RGP (200-epochs pretrained on ImageNet):
| downstream task |$AP_{50}$| $AP$ | $AP_{75}$ |ckpt|log|
| :----:  |:------:| :--: | :--: | :--: | :--: |
| VOC 07+12 detection  | $82.2_{±0.07}$|$57.2_{±0.10}$ | $63.8_{±0.11}$| [voc_ckpt](https://drive.google.com/file/d/1yUnBCCqcjBRhFJMi8R-cvnTIgqCUh7YB/view?usp=sharing)|[voc_log](https://drive.google.com/file/d/1tKUmBHUQiNZauiZ3Oe4-6YMsRG9iqILp/view?usp=sharing)|
| COCO detection| $60.5_{±0.28}$|$40.7_{±0.14}$ | $44.1_{±0.14}$|[coco_ckpt](https://drive.google.com/file/d/1_QGsK9Uvk60yeAgpMYChUB7QKc9kahTJ/view?usp=sharing) |[coco_log](https://drive.google.com/file/d/1ywNNEHGdX-ecztQV9nDFWN91Mu5cP1h6/view?usp=sharing)|
| COCO instance seg.| $57.3_{±0.16}$|$35.5_{±0.12}$ | $37.9_{±0.14}$|[coco_ckpt](https://drive.google.com/file/d/1_QGsK9Uvk60yeAgpMYChUB7QKc9kahTJ/view?usp=sharing) | [coco_log](https://drive.google.com/file/d/1ywNNEHGdX-ecztQV9nDFWN91Mu5cP1h6/view?usp=sharing)|
