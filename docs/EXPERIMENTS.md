# Experiment setup

## Environment

The checked environment used Python 3.10, PyTorch 2.7.1 and torchvision 0.22.1. Install dependencies from the repository root with `python -m pip install -r requirements.txt`.

## Fashion-MNIST and ResNet-18

The paper evaluates Fashion-MNIST, CIFAR-10 and Tiny-ImageNet. The default scripts select Fashion-MNIST and ResNet-18. Download the four IDX archives from the [Fashion-MNIST dataset repository](https://github.com/zalandoresearch/fashion-mnist#download) and extract each `.gz` file into the layout below. The loader expects both the training and test files.

```text
repository/
  data/fashion-mnist/FashionMNIST/raw/
    train-images-idx3-ubyte
    train-labels-idx1-ubyte
    t10k-images-idx3-ubyte
    t10k-labels-idx1-ubyte
  train_code/
    test_model/resnet18_fash.pth
    importance_score/resnet18_fash.npy
```

`resnet18_fash.pth` is the state dictionary for the one-channel, ten-class ResNet-18 defined in the Fashion-MNIST branch of `Main_embedding.py`. Use the corresponding experiment checkpoint from the lab archive or train a compatible classifier. The `.npy` file is the ascending importance permutation for the two protected layers of that checkpoint. Use the paired archive file or generate it with `python Score_cal.py` from `train_code/` after placing the dataset and checkpoint. The primary ranking is written to `importance_score/resnet18_fash.npy`. The `_gra` and `_wei` variants are saved separately.

`NNWHR_DATA_ROOT` can point to an external directory containing `fashion-mnist/`, `cifar10/` and `tiny-imagenet-200/`. Use the [CIFAR-10 Python archive](https://www.cs.toronto.edu/~kriz/cifar.html) under `cifar10/cifar-10-batches-py/`. For Tiny-ImageNet, keep `wnids.txt`, `train/<wnid>/images/`, `val/images/` and `val/val_annotations.txt` from the [Stanford archive](http://cs231n.stanford.edu/tiny-imagenet-200.zip). For alternate datasets, use matching checkpoints, rankings and the same dataset/model indices in all stages.

## Run

From `train_code/`, run the stages in this order.

```bash
python Main_embedding.py
python Model_attack.py
python Main_recovery.py
```

The scripts write the marked checkpoint and recovery arrays to `Marked_model/`, the tampered checkpoint and mask to `Tampered_model/`, and the recovery evaluation log to `Recovered_model/`. For a short run, set `NNWHR_EVAL_SAMPLES=4` and `NNWHR_ATTACK_TRIALS=1` in the process environment. For a ranking-generation check, set `NNWHR_SCORE_SAMPLES=2` and `NNWHR_SCORE_EPOCHS=1`. The paper-scale settings use the complete datasets and the original number of trials.

See the [validation record](VALIDATION.md).
